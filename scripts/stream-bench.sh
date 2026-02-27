#!/usr/bin/env bash
# stream-bench.sh -- STREAM memory bandwidth benchmark
# Phase: 3 (benchmark)
# Requires: jq, timeout, awk
# Emits: stream-bench.json
SCRIPT_NAME="stream-bench"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== STREAM Memory Bandwidth ==="

STREAM_DIR="${HPC_WORK_DIR}/stream"
mkdir -p "$STREAM_DIR"
register_cleanup "$STREAM_DIR"

# ── Download and compile STREAM (or use pre-built in portable mode) ──
STREAM_SRC="${STREAM_DIR}/stream.c"
STREAM_BIN="${STREAM_DIR}/stream"
NPROCS=$(nproc)

# Portable bundle: use pre-built binary if present
if [ "${HPC_PORTABLE:-0}" = "1" ] && [ -x "${HPC_BENCH_ROOT}/bin/stream" ]; then
    STREAM_BIN="${HPC_BENCH_ROOT}/bin/stream"
    log_info "Using pre-built STREAM from bundle (portable mode)"
fi

if [ ! -x "$STREAM_BIN" ]; then
    BUNDLED_SRC="${HPC_BENCH_ROOT}/src/stream.c"
    # Portable mode: no download; use only bundled source. Otherwise try upstream then fallback.
    if [ "${HPC_PORTABLE:-0}" = "1" ]; then
        if [ -f "$BUNDLED_SRC" ]; then
            log_info "Using bundled STREAM source (portable mode, no download)"
            cp "$BUNDLED_SRC" "$STREAM_SRC"
        else
            log_error "Failed to obtain STREAM source (no bundled copy in portable mode)"
            echo '{"error":"STREAM source unavailable"}' | emit_json "stream-bench" "error"
            exit 1
        fi
    else
        if curl -fsSL "https://www.cs.virginia.edu/stream/FTP/Code/stream.c" -o "$STREAM_SRC" 2>/dev/null && [ -s "$STREAM_SRC" ]; then
            log_info "Using STREAM source from upstream (download)"
        elif [ -f "$BUNDLED_SRC" ]; then
            log_info "Using bundled STREAM source (download failed or offline)"
            cp "$BUNDLED_SRC" "$STREAM_SRC"
        else
            log_error "Failed to obtain STREAM source (no bundled copy, download failed)"
            echo '{"error":"STREAM source unavailable"}' | emit_json "stream-bench" "error"
            exit 1
        fi
    fi

    # Compile with optimization
    # Quick mode: small array and few iterations; full: ~4x L3 or 80M elements
    if [ "${HPC_QUICK:-0}" = "1" ]; then
        ARRAY_SIZE=1000000
        NTIMES=3
        log_info "Quick mode — STREAM array size: $ARRAY_SIZE, iterations: $NTIMES"
    else
        ARRAY_SIZE=80000000
        NTIMES=20
        total_mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
        if [ "$total_mem_kb" -gt 67108864 ]; then  # >64GB
            ARRAY_SIZE=200000000
        fi
    fi

    # Use -mcmodel=medium for large arrays (>2GB static data) on x86_64
    MCMODEL=""
    if [ "$(uname -m)" = "x86_64" ] && [ "$ARRAY_SIZE" -gt 100000000 ]; then
        MCMODEL="-mcmodel=medium"
        log_info "Using -mcmodel=medium for large array"
    fi

    log_info "Compiling STREAM (array size: $ARRAY_SIZE)..."
    gcc -O3 -march=native -fopenmp $MCMODEL -DSTREAM_ARRAY_SIZE=$ARRAY_SIZE -DNTIMES=${NTIMES:-20} \
        "$STREAM_SRC" -o "$STREAM_BIN" 2>&1 || {
        # Fallback without OpenMP
        gcc -O3 $MCMODEL -DSTREAM_ARRAY_SIZE=$ARRAY_SIZE "$STREAM_SRC" -o "$STREAM_BIN" 2>&1 || {
            log_error "Failed to compile STREAM"
            echo '{"error":"compile failed"}' | emit_json "stream-bench" "error"
            exit 1
        }
    }
fi

# ── Run STREAM ──
STREAM_TIMEOUT=300
[ "${HPC_QUICK:-0}" = "1" ] && STREAM_TIMEOUT=60
log_info "Running STREAM with OMP_NUM_THREADS=$NPROCS..."
export OMP_NUM_THREADS=$NPROCS

output=$(run_with_timeout "$STREAM_TIMEOUT" "stream" "$STREAM_BIN" 2>&1) || true

# Parse: Function    Best Rate MB/s  (filter out log lines)
copy_mbps=$(echo "$output" | awk '/^Copy:/ {print $2}')
scale_mbps=$(echo "$output" | awk '/^Scale:/ {print $2}')
add_mbps=$(echo "$output" | awk '/^Add:/ {print $2}')
triad_mbps=$(echo "$output" | awk '/^Triad:/ {print $2}')

RESULT=$(jq -n \
    --argjson threads "$NPROCS" \
    --arg copy "${copy_mbps:-0}" \
    --arg scale "${scale_mbps:-0}" \
    --arg add "${add_mbps:-0}" \
    --arg triad "${triad_mbps:-0}" \
    '{
        threads: $threads,
        copy_mbps: (if $copy == "" or $copy == "0" then 0 else ($copy | tonumber) end),
        scale_mbps: (if $scale == "" or $scale == "0" then 0 else ($scale | tonumber) end),
        add_mbps: (if $add == "" or $add == "0" then 0 else ($add | tonumber) end),
        triad_mbps: (if $triad == "" or $triad == "0" then 0 else ($triad | tonumber) end),
        triad_gbps: (if $triad == "" or $triad == "0" then 0 else ($triad | tonumber) / 1000 end)
    }')

echo "$RESULT" | emit_json "stream-bench" "ok"
log_ok "STREAM Triad: ${triad_mbps:-N/A} MB/s"
echo "$RESULT" | jq .
