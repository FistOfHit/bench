#!/usr/bin/env bash
# nccl-tests.sh — Build and run NCCL collective tests (single-node)
SCRIPT_NAME="nccl-tests"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== NCCL Tests ==="

require_gpu "nccl-tests" "no GPU"

NGPUS=$(gpu_count)
if [ "$NGPUS" -lt 2 ]; then
    log_warn "NCCL tests require >=2 GPUs, found $NGPUS"
    echo "{\"note\":\"single GPU, skipping multi-GPU NCCL\",\"gpu_count\":$NGPUS}" | emit_json "nccl-tests" "skipped"
    exit 0
fi

# Check for NCCL
NCCL_CHECK=$(ldconfig -p 2>/dev/null | grep libnccl || find /usr -name 'libnccl.so*' 2>/dev/null | head -1 || echo "")
if [ -z "$NCCL_CHECK" ]; then
    log_warn "NCCL library not found — skipping NCCL tests"
    echo "{\"note\":\"NCCL not installed — required for multi-GPU collective tests\",\"gpu_count\":$NGPUS}" | emit_json "nccl-tests" "skipped"
    exit 0
fi

NCCL_DIR="${HPC_WORK_DIR}/nccl-tests"
NCCL_BUILD="${NCCL_DIR}/build"

# ── Build nccl-tests if needed ──
if [ ! -x "${NCCL_BUILD}/all_reduce_perf" ]; then
    log_info "Building nccl-tests..."

    # Find CUDA path
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    if [ ! -d "$CUDA_HOME" ]; then
        CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null))) 2>/dev/null || true
    fi

    # Prefer latest from online; fallback to bundled source in src/nccl-tests/
    BUNDLED_NCCL="${HPC_BENCH_ROOT}/src/nccl-tests"
    rm -rf "$NCCL_DIR"
    if git clone https://github.com/NVIDIA/nccl-tests.git "$NCCL_DIR" 2>/dev/null; then
        log_info "Using nccl-tests from upstream (git clone)"
    elif [ -f "${BUNDLED_NCCL}/Makefile" ]; then
        log_info "Using bundled nccl-tests source (online clone failed or offline)"
        rm -rf "$NCCL_DIR"
        cp -r "$BUNDLED_NCCL" "$NCCL_DIR"
    else
        log_error "Failed to clone nccl-tests and no bundled source available"
        echo '{"error":"source unavailable"}' | emit_json "nccl-tests" "error"
        exit 1
    fi

    cd "$NCCL_DIR"

    # Find MPI if available
    MPI_FLAG=""
    if has_cmd mpirun && [ -d /usr/lib/x86_64-linux-gnu/openmpi ]; then
        MPI_FLAG="MPI=1 MPI_HOME=/usr"
    fi

    if ! make -j$(nproc) CUDA_HOME="$CUDA_HOME" $MPI_FLAG 2>&1 | tail -10; then
        log_error "Failed to build nccl-tests"
        echo '{"error":"build failed"}' | emit_json "nccl-tests" "error"
        exit 1
    fi
fi
register_cleanup "$NCCL_DIR"

# ── Run tests ──
# Quick mode: single test (all_reduce_perf), tiny range 8B–1M, 1 iter — verifies NCCL path in ~20–30s; full: all 5 tests, 8B–8GB
if [ "${HPC_QUICK:-0}" = "1" ]; then
    MIN_BYTES="8"
    MAX_BYTES="1M"
    STEP_FACTOR="2"
    NCCL_ITERS=1
    NCCL_WARMUP=0
    NCCL_TIMEOUT=45
    NCCL_QUICK_TESTS="all_reduce_perf"   # single test to verify functionality
else
    MIN_BYTES="8"
    MAX_BYTES="8G"
    STEP_FACTOR="2"
    NCCL_ITERS=20
    NCCL_WARMUP=5
    NCCL_TIMEOUT=600
    NCCL_QUICK_TESTS=""
fi

declare -A RESULTS

run_nccl_test() {
    local test_name="$1"
    local binary="${NCCL_BUILD}/${test_name}"

    if [ ! -x "$binary" ]; then
        log_warn "Binary not found: $binary"
        return
    fi

    log_info "Running $test_name with $NGPUS GPUs..."
    local output
    output=$(run_with_timeout "${NCCL_TIMEOUT:-600}" "$test_name" \
        "$binary" -b "$MIN_BYTES" -e "$MAX_BYTES" -f "$STEP_FACTOR" -g "$NGPUS" -n "${NCCL_ITERS:-20}" -w "${NCCL_WARMUP:-5}" 2>&1) || true

    # Parse: extract the row with max message size for bus bandwidth
    # Bundled format (7 cols): size  count  type  time(us)  algbw  busbw  error
    # Upstream format (9+ cols): size count type redop root time algbw busbw #wrong
    # Use NF-relative indexing: busbw is always $(NF-1), algbw is always $(NF-2)
    # Fallback: "# Avg bus bandwidth: N.NN" when binary fails before printing rows (e.g. VM NCCL error)
    local parsed
    parsed=$(echo "$output" | awk '
    /^#/ && /Avg bus bandwidth/ { gsub(/,/,"."); avg=$NF+0; next }
    /^#/ {next}
    # Data row: allow leading space (fixed-width) and match lines containing "float" with numeric cols
    NF>=6 && ($1 ~ /^[0-9]/ || ($0 ~ /float/ && $(NF-2) ~ /^[0-9]/)) {
        if ($1 ~ /^[0-9]/) { size=$1; a=$(NF-2)+0; b=$(NF-1)+0 }
        else { for(i=1;i<=NF;i++) if ($i=="float" && i+3<=NF) { size=$(i-2); a=$(NF-2)+0; b=$(NF-1)+0; break } }
        last_size=size; last_algbw=a; last_busbw=b
    }
    END {
        if (last_size != "") printf "{\"max_size_bytes\":\"%s\",\"algbw_gbps\":%.2f,\"busbw_gbps\":%.2f,\"avg_busbw_gbps\":%.2f}", last_size, last_algbw, last_busbw, (avg+0)
        else if (avg > 0) printf "{\"max_size_bytes\":\"\",\"algbw_gbps\":0,\"busbw_gbps\":%.2f,\"avg_busbw_gbps\":%.2f}", avg, avg
        else print "{}"
    }')

    # Extract all data points (same pattern as above)
    local datapoints
    datapoints=$(echo "$output" | awk '
    BEGIN { print "[" ; first=1 }
    NF>=6 && ($1 ~ /^[0-9]/ || ($0 ~ /float/ && $(NF-2) ~ /^[0-9]/)) && !/^#/ {
        if(!first) printf ","
        first=0
        if ($1 ~ /^[0-9]/) printf "{\"size_bytes\":%s,\"algbw_gbps\":%.3f,\"busbw_gbps\":%.3f}", $1, $(NF-2)+0, $(NF-1)+0
        else { for(i=1;i<=NF;i++) if ($i=="float" && i+3<=NF) { printf "{\"size_bytes\":%s,\"algbw_gbps\":%.3f,\"busbw_gbps\":%.3f}", $(i-2), $(NF-2)+0, $(NF-1)+0; break } }
    }
    END { print "]" }')

    echo "{\"test\":\"$test_name\",\"gpus\":$NGPUS,\"summary\":$parsed,\"datapoints\":$datapoints}"
}

# Run the main NCCL collectives (quick mode: one test only)
if [ -n "${NCCL_QUICK_TESTS:-}" ]; then
    NCCL_TEST_LIST="$NCCL_QUICK_TESTS"
    log_info "Quick mode — running single NCCL test: $NCCL_QUICK_TESTS"
else
    NCCL_TEST_LIST="all_reduce_perf all_gather_perf broadcast_perf reduce_scatter_perf reduce_perf"
fi
tests_output="["
first=true
for test in $NCCL_TEST_LIST; do
    result=$(run_nccl_test "$test")
    if [ -n "$result" ] && [ "$result" != "" ]; then
        $first || tests_output+=","
        first=false
        tests_output+="$result"
    fi
done
tests_output+="]"

# ── Compute efficiency ──
gpu_model_name=$(gpu_model)
spec=$(lookup_gpu_spec "$gpu_model_name")
nvlink_bw=$(echo "$spec" | jq '.nvlink_bandwidth_gbps // 0' 2>/dev/null)
theoretical_busbw=$(echo "scale=2; $nvlink_bw * 0.85" | bc 2>/dev/null || echo "0")  # 85% efficiency target

# Get peak all_reduce busbw (prefer busbw_gbps, fallback to avg_busbw_gbps when binary fails after printing avg)
peak_busbw=$(echo "$tests_output" | jq '[.[] | select(.test=="all_reduce_perf") | (.summary.busbw_gbps // .summary.avg_busbw_gbps // 0)] | max // 0' 2>/dev/null)
# Ensure numeric (jq may return null)
peak_busbw=$(echo "$peak_busbw" | grep -E '^[0-9.]+$' || echo "0")

efficiency="N/A"
if [ "$nvlink_bw" != "0" ] && [ "$nvlink_bw" != "null" ] && [ -n "$peak_busbw" ] && [ "${peak_busbw}" != "0" ]; then
    efficiency=$(echo "scale=1; $peak_busbw / $nvlink_bw * 100" | bc 2>/dev/null || echo "N/A")
fi

RESULT=$(jq -n \
    --argjson tests "$tests_output" \
    --arg ngpus "$NGPUS" \
    --arg peak "$peak_busbw" \
    --arg theo "$nvlink_bw" \
    --arg eff "$efficiency" \
    '{
        gpu_count: ($ngpus | tonumber),
        tests: $tests,
        peak_allreduce_busbw_gbps: ($peak | tonumber? // $peak),
        theoretical_nvlink_bw_gbps: ($theo | tonumber? // $theo),
        efficiency_pct: $eff
    }')

echo "$RESULT" | emit_json "nccl-tests" "ok"
log_ok "NCCL tests complete — peak all_reduce busbw: ${peak_busbw} GB/s"
echo "$RESULT" | jq '{gpu_count, peak_allreduce_busbw_gbps, theoretical_nvlink_bw_gbps, efficiency_pct}'
