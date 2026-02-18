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

# ── Load central defaults (thresholds, tunables) ──
_defaults="${HPC_BENCH_ROOT}/conf/defaults.sh"
if [ -f "$_defaults" ]; then
    # shellcheck source=../conf/defaults.sh
    source "$_defaults"
fi
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

# Suite version (for module JSON traceability); run-all/report set it, else read VERSION
if [ -z "${HPC_BENCH_VERSION:-}" ] && [ -f "${HPC_BENCH_ROOT}/VERSION" ]; then
    export HPC_BENCH_VERSION=$(tr -d '[:space:]' < "${HPC_BENCH_ROOT}/VERSION" 2>/dev/null || echo "unknown")
fi

# ── ASCII-safe status display (for HPC_ASCII_OUTPUT=1) ──
# Returns Unicode symbol or ASCII label for PASS/WARN/FAIL/SKIP. Use in report and run-all checklist.
status_display_string() {
    local score="${1:-}"
    if [ "${HPC_ASCII_OUTPUT:-0}" = "1" ]; then
        case "$score" in
            PASS) echo "[OK]" ;;
            WARN) echo "[WARN]" ;;
            FAIL) echo "[FAIL]" ;;
            SKIP) echo "[SKIP]" ;;
            *) echo "[?]" ;;
        esac
    else
        case "$score" in
            PASS) echo "✅" ;;
            WARN) echo "⚠️" ;;
            FAIL) echo "❌" ;;
            SKIP) echo "⏭️" ;;
            *) echo "❓" ;;
        esac
    fi
}

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
# Emit a JSON result file for a module.
# _HPC_JSON_MARKER tracks whether emit_json was called in this process.
# We use a marker FILE (not a variable) because emit_json is often called on the
# right side of a pipe (e.g. echo "$json" | emit_json ...), which runs in a
# subshell — variable changes inside the subshell are invisible to the parent.
_HPC_JSON_MARKER="${HPC_WORK_DIR}/.emit_marker_${SCRIPT_NAME:-unknown}_$$"

# Standard module completion: emit JSON, log summary, print compact output.
# Usage: finish_module "module-name" "status" '{...json...}' '.{key1, key2}'
#   json_body  — full JSON payload (piped to emit_json)
#   jq_summary — optional jq filter for compact stdout (e.g. '{status, score}')
finish_module() {
    local module="$1" status="$2" json_body="$3" jq_summary="${4:-.}"
    echo "$json_body" | emit_json "$module" "$status"
    log_ok "$module complete (status: $status)"
    echo "$json_body" | jq "$jq_summary" 2>/dev/null || true
}

emit_json() {
    touch "$_HPC_JSON_MARKER" 2>/dev/null || true
    local module="$1" status="$2" file="${HPC_RESULTS_DIR}/${1}.json"
    shift 2
    local ts suite_ver
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    suite_ver="${HPC_BENCH_VERSION:-unknown}"
    # Read JSON object from stdin or remaining args
    if [ $# -gt 0 ]; then
        echo "$@" | jq --arg m "$module" --arg s "$status" --arg t "$ts" --arg v "$suite_ver" \
            '. + {module: $m, status: $s, timestamp: $t, suite_version: $v}' > "$file"
    else
        jq --arg m "$module" --arg s "$status" --arg t "$ts" --arg v "$suite_ver" \
            '. + {module: $m, status: $s, timestamp: $t, suite_version: $v}' > "$file"
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
        touch "$_HPC_JSON_MARKER" 2>/dev/null || true
        return 1
    fi
    echo "$input" | emit_json "$module" "$status"
}

# Sanitize string for safe embedding in JSON — strips control characters
# (U+0000–U+001F), then JSON-encodes with proper escaping.
# Uses python3 when available; falls back to jq, then to pure-bash escaping
# so the suite never silently breaks without python3.
sanitize_json_str() {
    local input
    input=$(printf '%s' "$1" | tr -d '\000-\037')
    if has_cmd python3; then
        printf '%s' "$input" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()), end="")' 2>/dev/null && return
    fi
    if has_cmd jq; then
        printf '%s' "$input" | jq -Rs '.' 2>/dev/null && return
    fi
    # Last resort: manual escaping (handles quotes and backslashes)
    input="${input//\\/\\\\}"
    input="${input//\"/\\\"}"
    printf '"%s"' "$input"
}

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

# Normalize a value to an integer string, else emit default (0 by default).
int_or_default() {
    local raw="${1-}" def="${2:-0}"
    raw=$(trim_ws "$raw")
    case "$raw" in
        ''|*[!0-9]*)
            printf '%s' "$def"
            ;;
        *)
            printf '%s' "$raw"
            ;;
    esac
}

# Count grep matches safely; never emits "0\n0".
# Usage:
#   count_grep_re 'FAULTY' "$text"
#   printf '%s\n' "$text" | count_grep_re 'FAULTY'
count_grep_re() {
    local re="${1:?count_grep_re: regex required}"
    local count
    if [ $# -ge 2 ]; then
        count=$(printf '%s\n' "${2-}" | grep -cE -- "$re" 2>/dev/null || true)
    else
        count=$(grep -cE -- "$re" 2>/dev/null || true)
    fi
    int_or_default "${count:-0}" 0
}

# Compact JSON if valid, else emit fallback (default: {}).
# NOTE: Do NOT use ${2:-{}} — bash mis-parses the nested braces, yielding
#       the caller's $2 + a stray "}" (e.g. "[]}" instead of "[]").
json_compact_or() {
    local json="${1-}"
    local fallback="${2-}"
    [ -z "$fallback" ] && fallback='{}'
    if [ -n "$json" ] && printf '%s' "$json" | jq -c . >/dev/null 2>&1; then
        printf '%s' "$json" | jq -c . 2>/dev/null || printf '%s' "$fallback"
        return 0
    fi
    printf '%s' "$fallback"
}

# Read newline-delimited text from stdin and return a JSON array of strings.
# Empty lines are dropped.
json_array_from_lines() {
    local fallback="${1:-[]}"
    local out
    out=$(jq -R 'select(length>0)' 2>/dev/null | jq -cs '.' 2>/dev/null) || true
    json_compact_or "${out:-}" "$fallback"
}

# Read newline-delimited JSON objects from stdin and return a JSON array.
json_slurp_objects_or() {
    local fallback="${1:-[]}"
    local out
    out=$(jq -cs '.' 2>/dev/null) || true
    json_compact_or "${out:-}" "$fallback"
}

# Write a JSON blob to a temp file (validated + compacted), register cleanup, echo path.
# If invalid/empty, writes the fallback (default: {}).
json_tmpfile() {
    local prefix="${1:?json_tmpfile: prefix required}"
    local json="${2-}"
    local fallback="${3-}"
    [ -z "$fallback" ] && fallback='{}'
    local tmp
    tmp=$(mktemp -p "${HPC_WORK_DIR}" "${prefix}.XXXXXX")
    if [ -n "$json" ]; then
        if ! printf '%s' "$json" | jq -c . >"$tmp" 2>/dev/null; then
            printf '%s' "$fallback" >"$tmp"
        fi
    else
        printf '%s' "$fallback" >"$tmp"
    fi
    register_cleanup "$tmp"
    printf '%s' "$tmp"
}

# Prefer this over which(1); returns empty on not found.
cmd_path() { command -v "$1" 2>/dev/null || true; }

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
# Usage: local myvar; myvar=$(safe_json_pipe '{}' some_command --args)
# Replaces the old eval-based safe_json_var(); callers use command substitution instead.
safe_json_pipe() {
    local _fallback="$1"; shift
    local _output
    _output=$("$@" 2>/dev/null) || true
    # Sanitize control characters
    _output=$(printf '%s' "$_output" | tr -d '\000-\037')
    if [ -z "$_output" ] || ! echo "$_output" | jq . >/dev/null 2>&1; then
        _output="$_fallback"
    fi
    printf '%s' "$_output"
}

# ── Virtualization detection (cached) ──
# Result is cached in _HPC_VIRT_CACHE after the first call to avoid repeated
# subprocess spawning (systemd-detect-virt, grep /proc/cpuinfo, DMI reads).
_HPC_VIRT_CACHE=""

detect_virtualization() {
    if [ -n "$_HPC_VIRT_CACHE" ]; then
        echo "$_HPC_VIRT_CACHE"
        return
    fi

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
            product=$(</sys/class/dmi/id/product_name 2>/dev/null) || product=""
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

    # Strip control characters so jq never sees invalid JSON
    virt_type=$(printf '%s' "$virt_type" | tr -d '\000-\037')
    virt_details=$(printf '%s' "$virt_details" | tr -d '\000-\037')

    _HPC_VIRT_CACHE="{\"type\": \"$virt_type\", \"details\": \"$virt_details\"}"
    echo "$_HPC_VIRT_CACHE"
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
    # so the report doesn't silently skip this module.
    # We check the marker file (not a variable) because emit_json may run in a
    # pipe subshell where variable changes are invisible to the parent.
    if [ ! -f "${_HPC_JSON_MARKER:-/nonexistent}" ] && [ -n "${SCRIPT_NAME:-}" ] && [ "$SCRIPT_NAME" != "run-all" ] && [ "$SCRIPT_NAME" != "report" ]; then
        local crash_file="${HPC_RESULTS_DIR}/${SCRIPT_NAME}.json"
        log_warn "Module '$SCRIPT_NAME' exited (code=$exit_code) without writing results — emitting crash record"
        jq -n \
            --arg m "$SCRIPT_NAME" \
            --arg t "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            --argjson rc "$exit_code" \
            '{module: $m, status: "error", timestamp: $t, error: "Module exited unexpectedly", exit_code: $rc}' \
            > "$crash_file" 2>/dev/null || true
    fi
    # Clean up the marker file
    rm -f "${_HPC_JSON_MARKER:-/nonexistent}" 2>/dev/null || true

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

# ── Skip module helpers ──
# Usage: skip_module "module-name" "reason text"
# Emits a standard "skipped" JSON record and exits cleanly.
skip_module() {
    local module="${1:?skip_module: module name required}"
    local reason="${2:-not applicable}"
    log_warn "Skipping $module: $reason"
    jq -n --arg note "$reason" '{note: $note, skip_reason: $note}' | emit_json "$module" "skipped"
    exit 0
}

# Usage: skip_module_with_data "module-name" "reason" '{"extra_key": "value", ...}'
# Like skip_module but merges caller-supplied JSON fields into the skip record.
# The extra_json argument must be a valid JSON object string.
skip_module_with_data() {
    local module="${1:?skip_module_with_data: module name required}"
    local reason="${2:-not applicable}"
    local extra_json="${3-}"
    [ -z "$extra_json" ] && extra_json='{}'
    log_warn "Skipping $module: $reason"
    printf '%s' "$extra_json" \
        | jq --arg note "$reason" '. + {note: $note, skip_reason: $note}' \
        | emit_json "$module" "skipped"
    exit 0
}

# ── GPU requirement (skip module when no NVIDIA GPU) ──
require_gpu() {
    local module="${1:?require_gpu: module name required}"
    if ! has_cmd nvidia-smi; then
        skip_module "$module" "nvidia-smi not found"
    fi
    if ! nvidia-smi &>/dev/null; then
        skip_module "$module" "nvidia-smi failed"
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
        gpu_links=$(nvidia-smi nvlink -i "$i" --status_query 2>/dev/null | count_grep_re 'NVLink')
        link_count=$((link_count + gpu_links))
    done

    echo "{\"available\": true, \"total_links\": $link_count, \"gpus\": $gpu_count}"
}

# ── GPU compute capability detection (shared by gpu-burn, nccl-tests) ──
# Returns the numeric compute capability string (e.g. "80" for sm_80) or empty.
detect_compute_capability() {
    local cc
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -1 | tr -d '[:space:].')
    printf '%s' "${cc:-}"
}

# ── CUDA home detection (shared by gpu-inventory, nccl-tests, nvbandwidth) ──
# Returns the best-guess CUDA_HOME path. Checks: $CUDA_HOME env, nvcc in PATH,
# common install locations. Falls back to /usr/local/cuda.
detect_cuda_home() {
    if [ -d "${CUDA_HOME:-}" ]; then echo "$CUDA_HOME"; return; fi
    local p
    p=$(cmd_path nvcc)
    if [ -n "$p" ]; then dirname "$(dirname "$p")"; return; fi
    local d
    for d in /usr/local/cuda /usr/local/cuda-*/; do
        [ -d "$d/bin" ] && { echo "$d"; return; }
    done
    echo "/usr/local/cuda"
}

# ── Container runtime detection (shared by hpl-cpu, hpl-mxp) ──
# Prints "docker" or "podman" if a usable runtime is found, else prints nothing.
detect_container_runtime() {
    if has_cmd docker && docker info &>/dev/null; then echo "docker"
    elif has_cmd podman && podman info &>/dev/null; then echo "podman"
    fi
}

# ── HPL.dat template generation (shared by hpl-cpu, hpl-mxp) ──
# Generates a standard HPL.dat input file. Each caller computes its own N/NB/P/Q.
# Usage: generate_hpl_dat <output_path> <N> <NB> <P> <Q>
generate_hpl_dat() {
    local output_path="$1" n="$2" nb="$3" p="$4" q="$5"
    cat > "$output_path" <<HPLEOF
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
$n           Ns
1            # of NBs
$nb          NBs
0            PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
$p           Ps
$q           Qs
16.0         threshold
1            # of panel fact
2            PFACTs (0=left, 1=Crout, 2=Right)
1            # of recursive stopping criterium
4            NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
1            # of recursive panel fact.
1            RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
1            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
1            DEPTHs (>=0)
2            SWAP (0=bin-exch,1=long,2=mix)
64           swapping threshold
0            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
1            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
HPLEOF
}

# ── Clone-or-copy source helper (shared by gpu-burn, nccl-tests) ──
# Usage: clone_or_copy_source <dest_dir> <git_url> <bundled_src_dir> <label>
# Clones from git_url; on failure copies bundled source. Returns 1 if neither works.
clone_or_copy_source() {
    local dest="$1" git_url="$2" bundled="$3" label="${4:-source}"
    rm -rf "$dest"
    if git clone "$git_url" "$dest" 2>/dev/null; then
        log_info "Using $label from upstream (git clone)"
        return 0
    fi
    if [ -f "${bundled}/Makefile" ]; then
        log_info "Using bundled $label (online clone failed or offline)"
        cp -r "$bundled" "$dest"
        return 0
    fi
    log_error "Failed to obtain $label: git clone failed and no bundled source"
    return 1
}
