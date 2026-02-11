#!/usr/bin/env bash
# common.sh — Shared functions for HPC bench suite (V1.1)
# Source this from every script: source "$(dirname "$0")/../lib/common.sh"

set -euo pipefail

# ── Locale safety (prevent non-ASCII date formats on zh_TW etc.) ──
export LC_TIME=C LC_NUMERIC=C LC_COLLATE=C LANG="${LANG:-C.UTF-8}"

# ── Paths ──
export HPC_BENCH_ROOT="${HPC_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HPC_RESULTS_DIR="${HPC_RESULTS_DIR:-/var/log/hpc-bench/results}"
export HPC_LOG_DIR="${HPC_LOG_DIR:-${HPC_RESULTS_DIR}/logs}"
export HPC_WORK_DIR="${HPC_WORK_DIR:-/tmp/hpc-bench-work}"
export HPC_SPECS_FILE="${HPC_BENCH_ROOT}/specs/hardware-specs.json"

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

# Sanitize string for JSON - removes control characters and escapes properly
sanitize_json_str() {
    printf '%s' "$1" | tr -d '\000-\037' | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()), end="")' 2>/dev/null || printf '"%s"' "$1"
}

# Simple JSON string escape (legacy, use sanitize_json_str for user input)
json_str() { printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' 2>/dev/null || printf '"%s"' "$1"; }

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

# ── GPU spec lookup ──
lookup_gpu_spec() {
    local model="$1"
    if [ ! -f "$HPC_SPECS_FILE" ]; then
        log_warn "Hardware specs file not found: $HPC_SPECS_FILE"
        echo '{}'
        return
    fi
    # Sanitize: strip leading/trailing whitespace and control chars
    model=$(echo "$model" | tr -d '[:cntrl:]' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    if [ -z "$model" ] || [ "$model" = "unknown" ]; then
        echo '{}'
        return
    fi

    # Try exact match first
    local result
    result=$(jq --arg m "$model" '.gpus[$m] // empty' "$HPC_SPECS_FILE" 2>/dev/null)
    if [ -n "$result" ]; then
        echo "$result"
        return
    fi

    # Scored fuzzy match using python3 for robustness
    # Prefer: longest spec key that is a substring of the model name (nvidia-smi names
    # are typically more specific, e.g. "NVIDIA A100-SXM4-80GB" vs spec key "NVIDIA A100 SXM4 80GB").
    # Normalise both sides (lowercase, strip hyphens/underscores) before comparing.
    result=$(jq -r '.gpus | to_entries[] | "\(.key)\t\(.value | @json)"' "$HPC_SPECS_FILE" 2>/dev/null | \
        python3 -c "
import sys, re

model = sys.argv[1]
model_norm = re.sub(r'[-_/\s]+', ' ', model).lower().strip()

best_key = None
best_score = -1
best_val = None

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    key, val = line.split('\t', 1)
    key_norm = re.sub(r'[-_/\s]+', ' ', key).lower().strip()

    score = 0
    # Check if spec key words are all in the model name
    key_words = key_norm.split()
    model_words = model_norm.split()

    if key_norm in model_norm:
        # Spec key is substring of model — strong match, score by key length
        score = len(key_norm) * 10
    elif model_norm in key_norm:
        # Model is substring of spec key — weaker match
        score = len(model_norm) * 5
    else:
        # Word overlap scoring
        common = sum(1 for w in key_words if w in model_words)
        if common >= 2:  # Need at least 2 words in common
            score = common * 3

    if score > best_score:
        best_score = score
        best_key = key
        best_val = val

if best_score > 0 and best_val:
    print(best_val)
else:
    print('{}')
" "$model" 2>/dev/null) || result='{}'

    # Ensure valid JSON output
    if [ -z "$result" ] || [ "$result" = "null" ]; then
        result='{}'
    fi
    echo "$result"
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
