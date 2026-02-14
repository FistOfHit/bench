#!/usr/bin/env bash
# nvbandwidth.sh — GPU bandwidth testing (H2D, D2H, D2D)
SCRIPT_NAME="nvbandwidth"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== nvbandwidth ==="

require_gpu "nvbandwidth" "no GPU"

NVB_DIR="${HPC_WORK_DIR}/nvbandwidth"
NVB_BIN="${NVB_DIR}/nvbandwidth"

# ── Find or build nvbandwidth ──
if [ ! -x "$NVB_BIN" ]; then
    # 1. Check PATH
    if has_cmd nvbandwidth; then
        NVB_BIN=$(command -v nvbandwidth)
        log_info "Found nvbandwidth in PATH: $NVB_BIN"
    else
        # 2. Check common CUDA locations
        NVB_FOUND=""
        for candidate in \
            "${CUDA_HOME:-/usr/local/cuda}/bin/nvbandwidth" \
            "/usr/local/cuda/bin/nvbandwidth" \
            "/usr/bin/nvbandwidth" \
            ; do
            if [ -x "$candidate" ]; then
                NVB_FOUND="$candidate"
                break
            fi
        done
        # Also try versioned CUDA installs
        if [ -z "$NVB_FOUND" ]; then
            for candidate in /usr/local/cuda-*/bin/nvbandwidth; do
                if [ -x "$candidate" ]; then
                    NVB_FOUND="$candidate"
                    break
                fi
            done
        fi

        if [ -n "$NVB_FOUND" ]; then
            NVB_BIN="$NVB_FOUND"
            log_info "Found nvbandwidth at: $NVB_BIN"
        else
            # 3. Check if cmake is available before attempting build
            if ! has_cmd cmake; then
                log_warn "nvbandwidth not found and cmake not available — skipping"
                echo '{"note":"nvbandwidth not found, cmake unavailable for build","skip_reason":"no cmake"}' | emit_json "nvbandwidth" "skipped"
                exit 0
            fi

            # 4. Last resort: try git clone (nvbandwidth is CMake-based, too complex to bundle)
            log_warn "nvbandwidth not found in PATH or common locations, attempting git clone..."
            rm -rf "$NVB_DIR"
            if git clone https://github.com/NVIDIA/nvbandwidth.git "$NVB_DIR" 2>/dev/null; then
                cd "$NVB_DIR"
                mkdir -p build && cd build
                CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
                if cmake .. -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" 2>&1 | tail -5 && \
                   make -j$(nproc) 2>&1 | tail -5; then
                    NVB_BIN="${NVB_DIR}/build/nvbandwidth"
                else
                    log_warn "nvbandwidth build failed — skipping (install CUDA toolkit with nvbandwidth or use bare metal)"
                    echo '{"note":"build failed","skip_reason":"nvbandwidth not available — install CUDA toolkit with nvbandwidth or ensure internet and build deps"}' | emit_json "nvbandwidth" "skipped"
                    exit 0
                fi
            else
                log_warn "nvbandwidth not available offline (not bundled). Install via CUDA toolkit or ensure internet for git clone."
                echo '{"note":"nvbandwidth not available offline","skip_reason":"Install CUDA toolkit >= 12.x which includes nvbandwidth, or ensure internet access for git clone"}' | emit_json "nvbandwidth" "skipped"
                exit 0
            fi
        fi
    fi
fi
register_cleanup "$NVB_DIR"

# ── Run tests ──
declare -A test_results

# Quick mode: shorter timeout per test
NVB_TIMEOUT=${HPC_QUICK:+60}
NVB_TIMEOUT=${NVB_TIMEOUT:-300}
P2P_STATUS="unknown"
if has_cmd nvidia-smi; then
    p2p_out=$(nvidia-smi topo -p2p r 2>/dev/null || true)
    if [ -n "$p2p_out" ]; then
        if echo "$p2p_out" | grep -q "NS"; then
            P2P_STATUS="not_supported"
        elif echo "$p2p_out" | grep -q "OK"; then
            P2P_STATUS="supported"
        fi
    fi
fi

run_nvb_test() {
    local test_name="$1"
    log_info "Running nvbandwidth: $test_name"
    local output
    output=$(run_with_timeout "$NVB_TIMEOUT" "nvbandwidth-$test_name" "$NVB_BIN" -t "$test_name" 2>&1) || true

    # nvbandwidth output format:
    #   Running <test_name>.
    #   memcpy CE GPU(row) -> GPU(column) bandwidth (GB/s)
    #          GPU 0  GPU 1  GPU 2 ...
    #   GPU 0  XX.XX  YY.YY  ZZ.ZZ
    #   GPU 1  ...
    # OR for H2D/D2H:
    #   SUM for all gpus: XX.XX GB/s
    #
    # Strategy: extract only lines that start with "GPU" or "SUM" and pull
    # numeric BW values from the data rows. Ignore header/label lines.
    local bw_values
    bw_values=$(echo "$output" | python3 -c "
import sys, json

lines = sys.stdin.read().strip().split('\n')
values = []
sum_bw = None

for line in lines:
    line = line.strip()
    # 'SUM' line (e.g. H2D/D2H single-value results)
    if line.startswith('SUM'):
        parts = line.split()
        for p in parts:
            try:
                v = float(p)
                if v > 0:
                    sum_bw = v
            except ValueError:
                continue
    # Data row: starts with 'GPU N' followed by bandwidth values
    elif line.startswith('GPU') and not line.startswith('GPU('):
        parts = line.split()
        # Skip 'GPU' and the index, take remaining floats
        for p in parts[2:]:
            try:
                v = float(p)
                if v > 0:
                    values.append(v)
            except ValueError:
                continue

if sum_bw is not None:
    print(json.dumps({'sum_gbps': round(sum_bw, 2), 'samples': 1}))
elif values:
    mean_v = sum(values) / len(values)
    max_v = max(values)
    min_v = min(values)
    print(json.dumps({'mean_gbps': round(mean_v, 2), 'max_gbps': round(max_v, 2), 'min_gbps': round(min_v, 2), 'samples': len(values)}))
else:
    print('{}')
" 2>/dev/null || echo '{}')
    # Avoid ${var:-{}} brace-matching bug — trailing } becomes literal, corrupting JSON
    [ -z "$bw_values" ] && bw_values='{}'
    echo "$bw_values"
}

# Test categories
h2d=$(run_nvb_test "host_to_device_memcpy_ce" 2>/dev/null || echo '{}')
d2h=$(run_nvb_test "device_to_host_memcpy_ce" 2>/dev/null || echo '{}')
d2d=$(run_nvb_test "device_to_device_memcpy_read_ce" 2>/dev/null || echo '{}')
d2d_write=$(run_nvb_test "device_to_device_memcpy_write_ce" 2>/dev/null || echo '{}')

# Also try bidirectional if available
d2d_bidir=$(run_nvb_test "device_to_device_bidirectional_memcpy_read_ce" 2>/dev/null || echo '{}')

# Validate JSON before passing to --argjson (avoid crash on empty/invalid output)
for _var in h2d d2h d2d d2d_write d2d_bidir; do
    eval "_val=\$$_var"
    if [ -z "$_val" ] || ! echo "$_val" | jq . >/dev/null 2>&1; then
        eval "$_var='{}'"
    fi
done
RESULT=$(jq -n \
    --argjson h2d "$h2d" \
    --argjson d2h "$d2h" \
    --argjson d2d "$d2d" \
    --argjson d2d_w "$d2d_write" \
    --argjson bidir "$d2d_bidir" \
    --arg p2p "$P2P_STATUS" \
    '{
        host_to_device: $h2d,
        device_to_host: $d2h,
        device_to_device_read: $d2d,
        device_to_device_write: $d2d_w,
        device_to_device_bidirectional: $bidir,
        p2p_status: $p2p
    }')

finish_module "nvbandwidth" "ok" "$RESULT"
