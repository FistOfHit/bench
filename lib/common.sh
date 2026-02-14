#!/usr/bin/env bash
# common.sh — Shared functions for HPC bench suite
# Source from scripts: source "$(dirname "$0")/../lib/common.sh"

set -euo pipefail

# ── Locale safety (prevent non-ASCII date formats on zh_TW etc.) ──
export LC_TIME=C LC_NUMERIC=C LC_COLLATE=C LANG="${LANG:-C.UTF-8}"

# ── CUDA PATH (ensure nvcc is discoverable after toolkit install) ──
if [ -d /usr/local/cuda/bin ] && [[ ":$PATH:" != *":/usr/local/cuda/bin:"* ]]; then
    export PATH="/usr/local/cuda/bin:${PATH}"
fi
if [ -d /usr/local/cuda/lib64 ] && [[ ":${LD_LIBRARY_PATH:-}:" != *":/usr/local/cuda/lib64:"* ]]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

# ── Paths ──
export HPC_BENCH_ROOT="${HPC_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ -z "${HPC_RESULTS_DIR:-}" ]; then
    if [ "$(id -u)" -eq 0 ]; then
        export HPC_RESULTS_DIR="/var/log/hpc-bench/results"
    else
        export HPC_RESULTS_DIR="${HOME:-/tmp}/.local/var/hpc-bench/results"
    fi
fi
# Prevent writing into protected system paths (e.g. HPC_RESULTS_DIR=/etc/)
case "$HPC_RESULTS_DIR" in
    /|/etc|/etc/*|/bin|/bin/*|/sbin|/sbin/*|/usr/bin|/usr/bin/*|/usr/sbin|/usr/sbin/*|/boot|/boot/*|/proc|/proc/*|/sys|/sys/*)
        echo "FATAL: HPC_RESULTS_DIR=$HPC_RESULTS_DIR is a protected path" >&2
        exit 1
        ;;
esac
export HPC_LOG_DIR="${HPC_LOG_DIR:-${HPC_RESULTS_DIR}/logs}"
export HPC_WORK_DIR="${HPC_WORK_DIR:-/tmp/hpc-bench-work}"

mkdir -p "$HPC_RESULTS_DIR" "$HPC_LOG_DIR" "$HPC_WORK_DIR"

# ── Logging ──
_log() {
    local level="$1"; shift
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] [$level] $*" | tee -a "${HPC_LOG_DIR}/${SCRIPT_NAME:-unknown}.log" >&2
}
log_info()  { _log INFO  "$@"; }
log_warn()  { _log WARN  "$@"; }
log_error() { _log ERROR "$@"; }
log_ok()    { _log OK    "$@"; }

# ── JSON helpers ──
# Emit a JSON result file for a module
# _HPC_JSON_EMITTED tracks whether this was called, used by the crash-safety trap
_HPC_JSON_EMITTED=false
emit_json() {
    _HPC_JSON_EMITTED=true
    local module="$1" status="$2" file="${HPC_RESULTS_DIR}/${1}.json"
    shift 2
    # Read JSON object from stdin or remaining args
    if [ $# -gt 0 ]; then
        echo "$@" | jq --arg m "$module" --arg s "$status" --arg t "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            '. + {module: $m, status: $s, timestamp: $t}' > "$file"
    else
        jq --arg m "$module" --arg s "$status" --arg t "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            '. + {module: $m, status: $s, timestamp: $t}' > "$file"
    fi
    log_info "Results written to $file"
}

# Validate JSON from stdin and emit; on invalid JSON write error record and return 1.
# Use when piping dynamically built JSON (awk/python) to avoid corrupt output.
emit_json_safe() {
    local module="$1" status="$2"
    local input
    input=$(</dev/stdin)
    if ! echo "$input" | jq . >/dev/null 2>&1; then
        log_error "Invalid JSON for module $module — writing error record"
        jq -n --arg m "$module" --arg e "Invalid JSON output" \
            '{module: $m, status: "error", error: $e}' \
            > "${HPC_RESULTS_DIR}/${module}.json"
        _HPC_JSON_EMITTED=true
        return 1
    fi
    echo "$input" | emit_json "$module" "$status"
}

# Sanitize string for JSON - removes control characters and escapes properly
sanitize_json_str() {
    printf '%s' "$1" | tr -d '\000-\037' | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()), end="")' 2>/dev/null || printf '"%s"' "$1"
}

# Simple JSON string escape (legacy, use sanitize_json_str for user input)
json_str() { printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' 2>/dev/null || printf '"%s"' "$1"; }

# Trim leading/trailing whitespace.
trim_ws() {
    local v="${1-}"
    v="${v#"${v%%[![:space:]]*}"}"
    v="${v%"${v##*[![:space:]]}"}"
    printf '%s' "$v"
}

# Convert potentially non-numeric telemetry values to JSON-safe numbers.
# Returns: numeric literal (int/float) or "null".
json_numeric_or_null() {
    local raw
    raw=$(trim_ws "${1-}")
    case "$raw" in
        ""|"N/A"|"[N/A]"|"[Not Supported]"|"[Unknown]"|"unknown"|"null")
            printf 'null'
            return 0
            ;;
    esac
    if [[ "$raw" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
        printf '%s' "$raw"
    else
        printf 'null'
    fi
}

# ── Command availability ──
has_cmd() { command -v "$1" &>/dev/null; }
require_cmd() {
    for cmd in "$@"; do
        if ! has_cmd "$cmd"; then
            log_error "Required command not found: $cmd"
            return 1
        fi
    done
}

# ── Safe JSON pipe ──
# Runs a command pipeline and returns valid JSON or a fallback.
safe_json_var() {
    local _varname="$1" _fallback="$2"; shift 2
    local _output
    _output=$("$@" 2>/dev/null) || true
    # Sanitize control characters
    _output=$(printf '%s' "$_output" | tr -d '\000-\037')
    if [ -z "$_output" ] || ! echo "$_output" | jq . >/dev/null 2>&1; then
        _output="$_fallback"
    fi
    eval "$_varname=\$(cat <<'SAFE_JSON_EOF'
$_output
SAFE_JSON_EOF
)"
}

# ── Virtualization detection ──
detect_virtualization() {
    local virt_type="none"
    local virt_details=""

    # Check systemd-detect-virt first
    if has_cmd systemd-detect-virt; then
        virt_type=$(systemd-detect-virt 2>/dev/null || echo "none")
    fi

    # Fallback checks
    if [ "$virt_type" = "none" ]; then
        if [ -d /proc/xen ]; then
            virt_type="xen"
        elif grep -q "hypervisor" /proc/cpuinfo 2>/dev/null; then
            virt_type="hypervisor"
        elif [ -f /sys/class/dmi/id/product_name ]; then
            local product
            product=$(cat /sys/class/dmi/id/product_name 2>/dev/null || echo "")
            case "$product" in
                *VMware*|*ESXi*) virt_type="vmware" ;;
                *VirtualBox*) virt_type="virtualbox" ;;
                *KVM*|*QEMU*) virt_type="kvm" ;;
                *Xen*) virt_type="xen" ;;
                *Hyper-V*) virt_type="hyperv" ;;
                Google*) virt_type="gcp" ;;
                *Amazon*|*AWS*) virt_type="aws" ;;
                *Azure*) virt_type="azure" ;;
            esac
        fi
    fi

    # Check if running in container
    if [ -f /.dockerenv ] || grep -qE 'docker|containerd' /proc/1/cgroup 2>/dev/null; then
        virt_details="container"
    fi

    # Strip control characters so jq never sees invalid JSON (e.g. systemd-detect-virt or DMI can emit stray bytes)
    virt_type=$(printf '%s' "$virt_type" | tr -d '\000-\037')
    virt_details=$(printf '%s' "$virt_details" | tr -d '\000-\037')

    echo "{\"type\": \"$virt_type\", \"details\": \"$virt_details\"}"
}

is_virtualized() {
    local virt
    virt=$(detect_virtualization)
    [ "$(echo "$virt" | jq -r '.type')" != "none" ]
}

# ── Privilege helpers ──
is_root() { [ "$(id -u)" -eq 0 ]; }
try_sudo() {
    if is_root; then
        "$@"
    elif sudo -n true 2>/dev/null; then
        sudo "$@"
    else
        "$@"  # Try anyway, may fail with permission error
    fi
}

# ── Timeout wrapper ──
run_with_timeout() {
    local secs="$1" desc="$2"; shift 2
    log_info "Running: $desc (timeout: ${secs}s)"
    if timeout --signal=KILL "$secs" "$@" 2>>"${HPC_LOG_DIR}/${SCRIPT_NAME:-unknown}.log"; then
        return 0
    else
        local rc=$?
        if [ $rc -eq 137 ]; then
            log_error "$desc: KILLED (timeout after ${secs}s)"
        else
            log_warn "$desc: exited with code $rc"
        fi
        return $rc
    fi
}

# ── Package manager detection ──
detect_pkg_manager() {
    if has_cmd apt-get; then echo "apt"
    elif has_cmd dnf; then echo "dnf"
    elif has_cmd yum; then echo "yum"
    else echo "unknown"; fi
}

pkg_install() {
    local mgr
    mgr=$(detect_pkg_manager)
    case "$mgr" in
        apt) DEBIAN_FRONTEND=noninteractive apt-get install -y "$@" ;;
        dnf) dnf install -y "$@" ;;
        yum) yum install -y "$@" ;;
        *) log_error "Unknown package manager"; return 1 ;;
    esac
}

pkg_update() {
    local mgr
    mgr=$(detect_pkg_manager)
    case "$mgr" in
        apt) apt-get update -y ;;
        dnf) dnf makecache ;;
        yum) yum makecache ;;
    esac
}

# ── Cleanup registration ──
CLEANUP_ITEMS=()
register_cleanup() { CLEANUP_ITEMS+=("$@"); }

do_cleanup() {
    local exit_code=$?

    # Safety net: if the module exited without calling emit_json, write a crash record
    # so the report doesn't silently skip this module
    if [ "$_HPC_JSON_EMITTED" = false ] && [ -n "${SCRIPT_NAME:-}" ] && [ "$SCRIPT_NAME" != "run-all" ] && [ "$SCRIPT_NAME" != "report" ]; then
        local crash_file="${HPC_RESULTS_DIR}/${SCRIPT_NAME}.json"
        if [ ! -f "$crash_file" ]; then
            log_warn "Module '$SCRIPT_NAME' exited (code=$exit_code) without writing results — emitting crash record"
            jq -n \
                --arg m "$SCRIPT_NAME" \
                --arg t "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                --argjson rc "$exit_code" \
                '{module: $m, status: "error", timestamp: $t, error: "Module exited unexpectedly", exit_code: $rc}' \
                > "$crash_file" 2>/dev/null || true
        fi
    fi

    # Original cleanup
    if [ "${HPC_KEEP_TOOLS:-0}" = "1" ]; then
        log_info "HPC_KEEP_TOOLS=1, skipping cleanup"
        return 0
    fi
    for item in "${CLEANUP_ITEMS[@]:-}"; do
        if [ -n "$item" ]; then
            rm -rf "$item" && log_info "Cleaned up: $item" || true
        fi
    done
    return 0
}
trap do_cleanup EXIT

# ── GPU requirement (skip module when no NVIDIA GPU) ──
require_gpu() {
    local module="${1:?require_gpu: module name required}"
    local note="${2:-no GPU}"
    if ! has_cmd nvidia-smi; then
        log_warn "nvidia-smi not found — skipping $module"
        jq -n --arg note "$note" '{note: $note}' | emit_json "$module" "skipped"
        exit 0
    fi
    if ! nvidia-smi &>/dev/null; then
        log_warn "nvidia-smi failed — skipping $module"
        jq -n --arg note "nvidia-smi failed" '{note: $note}' | emit_json "$module" "skipped"
        exit 0
    fi
}

# ── Nvidia helpers ──
gpu_count() {
    nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]' || echo 0
}
gpu_model() {
    local raw
    raw=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    # Aggressive sanitization: strip control chars, trim, remove "unknown" suffix artifacts
    echo "$raw" | tr -d '[:cntrl:]' | sed 's/unknown$//; s/[[:space:]]*$//' || echo "unknown"
}

# ── NVLink detection ──
nvlink_status() {
    if ! has_cmd nvidia-smi; then
        echo '{"available": false, "links": []}'
        return
    fi

    local nvlink_info
    nvlink_info=$(nvidia-smi nvlink --status_query 2>/dev/null || echo "")

    if [ -z "$nvlink_info" ] || echo "$nvlink_info" | grep -q "not supported"; then
        echo '{"available": false, "links": []}'
        return
    fi

    # Parse nvlink status - count active links per GPU
    local link_count=0
    local gpu_count
    gpu_count=$(gpu_count)

    for ((i=0; i<gpu_count; i++)); do
        local gpu_links
        gpu_links=$(nvidia-smi nvlink -i "$i" --status_query 2>/dev/null | grep -c "NVLink" || echo 0)
        link_count=$((link_count + gpu_links))
    done

    echo "{\"available\": true, \"total_links\": $link_count, \"gpus\": $gpu_count}"
}
