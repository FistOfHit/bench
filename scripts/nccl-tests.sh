#!/usr/bin/env bash
# nccl-tests.sh -- Build and run NCCL collective tests (single-node)
# Phase: 3 (benchmark)
# Requires: jq, timeout, awk
# Emits: nccl-tests.json
SCRIPT_NAME="nccl-tests"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== NCCL Tests ==="

require_gpu "nccl-tests"

NGPUS=$(gpu_count)
[ "$NGPUS" -lt 2 ] && skip_module "nccl-tests" "requires >=2 GPUs, found $NGPUS"

# Check for NCCL
NCCL_CHECK=$(ldconfig -p 2>/dev/null | grep libnccl || find /usr -name 'libnccl.so*' 2>/dev/null | head -1 || echo "")
[ -z "$NCCL_CHECK" ] && skip_module "nccl-tests" "NCCL library not installed"

# ── CUDA pre-flight: ensure nvidia-uvm is loaded (avoids error 802) ──
if is_root; then modprobe nvidia-uvm 2>/dev/null || true; fi
nvidia-smi -q -d MEMORY >/dev/null 2>&1 || true

NCCL_DIR="${HPC_WORK_DIR}/nccl-tests"
NCCL_BUILD="${NCCL_DIR}/build"

# ── Build nccl-tests if needed ──
if [ ! -x "${NCCL_BUILD}/all_reduce_perf" ]; then
    log_info "Building nccl-tests..."

    # Find CUDA path
    CUDA_HOME=$(detect_cuda_home)

    # Prefer bundled source for repeatability; fallback to online clone.
    # Note: clone_or_copy_source prefers git; we reverse priority here by
    # checking bundled first (repeatability matters for NCCL tests).
    BUNDLED_NCCL="${HPC_BENCH_ROOT}/src/nccl-tests"
    rm -rf "$NCCL_DIR"
    if [ -f "${BUNDLED_NCCL}/Makefile" ]; then
        log_info "Using bundled nccl-tests source"
        cp -r "$BUNDLED_NCCL" "$NCCL_DIR"
    elif ! clone_or_copy_source "$NCCL_DIR" \
            "https://github.com/NVIDIA/nccl-tests.git" \
            "$BUNDLED_NCCL" "nccl-tests"; then
        echo '{"error":"source unavailable"}' | emit_json "nccl-tests" "error"
        exit 1
    fi

    cd "$NCCL_DIR" || exit 1

    # Build without MPI dependency; tests are launched as single process with -g <num_gpus>.
    MPI_FLAG="MPI=0"

    # Auto-detect GPU compute capability to avoid "Unsupported gpu architecture"
    # errors when the installed CUDA version has dropped older arches (e.g. CUDA 13 drops compute_70).
    _detected_cc=$(detect_compute_capability)
    if [ -n "$_detected_cc" ]; then
        NVCC_GENCODE="-gencode=arch=compute_${_detected_cc},code=sm_${_detected_cc}"
        log_info "Auto-detected GPU compute capability: ${_detected_cc} → NVCC_GENCODE=${NVCC_GENCODE}"
    else
        NVCC_GENCODE=""
        log_warn "Could not detect GPU compute capability; using Makefile default NVCC_GENCODE"
    fi

    if ! make -j"$(nproc)" CUDA_HOME="$CUDA_HOME" $MPI_FLAG \
            ${NVCC_GENCODE:+NVCC_GENCODE="$NVCC_GENCODE"} 2>&1 | tail -10; then
        log_error "Failed to build nccl-tests"
        echo '{"error":"build failed"}' | emit_json "nccl-tests" "error"
        exit 1
    fi
fi
register_cleanup "$NCCL_DIR"

# ── Run tests ──
# Quick mode: single test (all_reduce_perf), tiny range 8B–1M, 1 iter — verifies NCCL path in ~20–30s; full: all 5 tests, 8B–8GB
# Test parameters (timeouts from conf/defaults.sh)
if [ "${HPC_QUICK:-0}" = "1" ]; then
    MIN_BYTES="8"
    MAX_BYTES="1M"
    STEP_FACTOR="2"
    NCCL_ITERS=1
    NCCL_WARMUP=0
    NCCL_TIMEOUT=${NCCL_TIMEOUT_QUICK:-45}
    NCCL_QUICK_TESTS="all_reduce_perf"   # single test to verify functionality
else
    MIN_BYTES="8"
    MAX_BYTES="8G"
    STEP_FACTOR="2"
    NCCL_ITERS=20
    NCCL_WARMUP=5
    NCCL_TIMEOUT=${NCCL_TIMEOUT_FULL:-600}
    NCCL_QUICK_TESTS=""
fi

declare -A RESULTS

run_nccl_test() {
    local test_name="$1"
    local binary="${NCCL_BUILD}/${test_name}"
    local run_gpus="$NGPUS"

    if [ ! -x "$binary" ]; then
        log_warn "Binary not found: $binary"
        return
    fi

    log_info "Running $test_name with $NGPUS GPUs..."
    local output
    local run_desc="$test_name"
    output=$(run_with_timeout "${NCCL_TIMEOUT:-600}" "$run_desc" \
        "$binary" -b "$MIN_BYTES" -e "$MAX_BYTES" -f "$STEP_FACTOR" -g "$NGPUS" -n "${NCCL_ITERS:-20}" -w "${NCCL_WARMUP:-5}" 2>&1) || true

    # VM/GPU passthrough environments often fail unless P2P is disabled.
    if echo "$output" | grep -qiE "unhandled cuda error|Test NCCL failure|NCCL failure"; then
        log_warn "$test_name failed in default mode, retrying with NCCL_P2P_DISABLE=1"
        output=$(run_with_timeout "${NCCL_TIMEOUT:-600}" "${run_desc}-p2p-disabled" \
            env NCCL_P2P_DISABLE=1 NCCL_P2P_LEVEL=LOC NCCL_IB_DISABLE=1 \
            "$binary" -b "$MIN_BYTES" -e "$MAX_BYTES" -f "$STEP_FACTOR" -g "$NGPUS" -n "${NCCL_ITERS:-20}" -w "${NCCL_WARMUP:-5}" 2>&1) || true
    fi
    if echo "$output" | grep -qiE "unhandled cuda error|Test NCCL failure|NCCL failure" && [ "$NGPUS" -gt 2 ]; then
        log_warn "$test_name still failing, retrying with 2 GPUs and P2P disabled"
        run_gpus=2
        output=$(run_with_timeout "${NCCL_TIMEOUT:-600}" "${run_desc}-2gpu" \
            env NCCL_P2P_DISABLE=1 NCCL_P2P_LEVEL=LOC NCCL_IB_DISABLE=1 \
            "$binary" -b "$MIN_BYTES" -e "$MAX_BYTES" -f "$STEP_FACTOR" -g "$run_gpus" -n "${NCCL_ITERS:-20}" -w "${NCCL_WARMUP:-5}" 2>&1) || true
    fi

    # Save raw output for debugging (parser issues, format changes)
    echo "$output" > "${HPC_LOG_DIR}/nccl-${test_name}-stdout.log" 2>/dev/null || true

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

    local error_line
    error_line=$(echo "$output" | awk '/unhandled cuda error|NCCL failure|Test NCCL failure/ {print; exit}')
    error_line=$(printf '%s' "$error_line" | tr -d '[:cntrl:]')
    if [ -n "$error_line" ]; then
        jq -n --arg test "$test_name" --argjson g "$run_gpus" --argjson summary "$parsed" --argjson datapoints "$datapoints" --arg err "$error_line" \
            '{test: $test, gpus: $g, summary: $summary, datapoints: $datapoints, error: $err}'
    else
        echo "{\"test\":\"$test_name\",\"gpus\":$run_gpus,\"summary\":$parsed,\"datapoints\":$datapoints}"
    fi
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
        if [ "$first" = false ]; then
            tests_output+=","
        fi
        first=false
        tests_output+="$result"
    fi
done
tests_output+="]"

# Get peak all_reduce busbw (prefer busbw_gbps, fallback to avg_busbw_gbps when binary fails after printing avg)
peak_busbw=$(echo "$tests_output" | jq '[.[] | select(.test=="all_reduce_perf") | (.summary.busbw_gbps // .summary.avg_busbw_gbps // 0)] | max // 0' 2>/dev/null)
# Ensure numeric (jq may return null)
peak_busbw=$(echo "$peak_busbw" | grep -E '^[0-9.]+$' || echo "0")
error_count=$(echo "$tests_output" | jq '[.[] | select(.error != null and .error != "")] | length' 2>/dev/null || echo 0)

if [ "$peak_busbw" = "0" ] && [ "$NGPUS" -ge 2 ]; then
    if [ "${error_count:-0}" -gt 0 ] 2>/dev/null; then
        log_error "NCCL all_reduce produced runtime errors and no bus bandwidth"
    else
        log_warn "NCCL all_reduce busbw=0 with $NGPUS GPUs — possible parsing failure"
        log_warn "Check nccl-tests log for output format changes"
    fi
fi

status="ok"
[ "${error_count:-0}" -gt 0 ] 2>/dev/null && status="warn"
# Zero bandwidth with multiple GPUs and no detected errors is a parsing or
# silent-failure problem — report warn so the suite doesn't silently pass.
if [ "$peak_busbw" = "0" ] && [ "$NGPUS" -ge 2 ]; then
    if [ "${error_count:-0}" -gt 0 ] 2>/dev/null; then
        status="error"
    else
        status="warn"
        log_warn "Marking NCCL as warn: busbw=0 with $NGPUS GPUs — results missing or unparseable"
    fi
fi

RESULT=$(jq -n \
    --argjson tests "$tests_output" \
    --arg ngpus "$NGPUS" \
    --arg peak "$peak_busbw" \
    --arg errc "$error_count" \
    '{
        gpu_count: ($ngpus | tonumber),
        tests: $tests,
        peak_allreduce_busbw_gbps: ($peak | tonumber? // $peak),
        runtime_error_count: ($errc | tonumber)
    }')

finish_module "nccl-tests" "$status" "$RESULT" '{gpu_count, peak_allreduce_busbw_gbps, runtime_error_count}'
if [ "$status" = "error" ]; then
    exit 1
fi
