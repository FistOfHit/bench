#!/usr/bin/env bash
# gpu-burn.sh -- GPU stress test using gpu-burn
# Phase: 3 (benchmark)
# Requires: jq, timeout, awk
# Emits: gpu-burn.json
# Builds from source (online or bundled), runs burn, parses GFLOPS and temps
SCRIPT_NAME="gpu-burn"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== GPU Burn Stress Test ==="

require_gpu "gpu-burn"

# Quick mode (HPC_QUICK=1): short burn to verify suite; VMs: shorter; else 5 min.
# Override with GPU_BURN_DURATION.  Defaults are in conf/defaults.sh.
# Multi-GPU systems (especially large-memory GPUs like A100-80GB) need extra time:
#   - Each GPU allocates memory sequentially (~3-5s per GPU)
#   - gpu-burn only emits GFLOPS in progress lines that require actual burn time
# Scale quick-mode duration: base 10s + 5s per GPU (e.g. 8 GPUs → 50s).
NGPUS=$(gpu_count)
if [ "${HPC_QUICK:-0}" = "1" ]; then
    _base_quick=${GPU_BURN_DURATION_QUICK:-10}
    _scaled=$(( _base_quick + NGPUS * 5 ))
    BURN_DURATION=${GPU_BURN_DURATION:-$_scaled}
elif is_virtualized; then
    BURN_DURATION=${GPU_BURN_DURATION:-${GPU_BURN_DURATION_VM}}
else
    BURN_DURATION=${GPU_BURN_DURATION:-${GPU_BURN_DURATION_FULL}}
fi
BURN_DIR="${HPC_WORK_DIR}/gpu-burn"

# ── CUDA pre-flight: ensure nvidia-uvm module is loaded ──
# Error 802 (CUDA_ERROR_NOT_INITIALIZED / system not yet initialized) commonly
# occurs on VMs or freshly booted machines when the nvidia-uvm kernel module is
# not loaded.  Loading it + a warmup nvidia-smi call resolves the vast majority
# of these failures.
if is_root; then
    modprobe nvidia-uvm 2>/dev/null || true
fi
# Warmup: nvidia-smi forces CUDA driver init; avoids cold-start error 802
nvidia-smi -q -d MEMORY >/dev/null 2>&1 || true

# ── Pre-burn temps ──
pre_temps=$(nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d '[:cntrl:]')

# ── Build gpu-burn if needed ──
if [ ! -x "${BURN_DIR}/gpu_burn" ]; then
    log_info "Building gpu-burn..."

    # Prefer latest from online; fallback to bundled source in src/gpu-burn/
    if ! clone_or_copy_source "$BURN_DIR" \
            "https://github.com/wilicc/gpu-burn.git" \
            "${HPC_BENCH_ROOT}/src/gpu-burn" "gpu-burn"; then
        echo '{"error":"source unavailable"}' | emit_json "gpu-burn" "error"
        exit 1
    fi
    cd "$BURN_DIR" || exit 1
    # Auto-detect GPU compute capability (e.g. "80" for A100) to avoid
    # "Unsupported gpu architecture" with newer CUDA that dropped old arches.
    _detected_cc=$(detect_compute_capability)
    _detected_cc="${_detected_cc:-75}"  # fallback to 75
    log_info "Building with COMPUTE=${_detected_cc}"
    if ! make COMPUTE="$_detected_cc" 2>&1 | tail -5; then
        log_error "Failed to build gpu-burn"
        echo '{"error":"build failed"}' | emit_json "gpu-burn" "error"
        exit 1
    fi
fi
register_cleanup "$BURN_DIR"

# ── Run gpu-burn ──
cd "$BURN_DIR" || exit 1
log_info "Running gpu-burn for ${BURN_DURATION}s..."

# Capture temps during burn in background
TEMP_LOG="${HPC_LOG_DIR}/gpu-burn-temps.csv"
echo "timestamp,gpu,temp_c,power_w" > "$TEMP_LOG"
(
    while true; do
        nvidia-smi --query-gpu=index,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | tr -d '[:cntrl:]' | \
            while IFS=', ' read -r idx temp pwr; do
                echo "$(date +%s),$idx,$temp,$pwr" >> "$TEMP_LOG"
            done
        sleep 5
    done
) &
TEMP_PID=$!

# Run gpu_burn with timeout, capturing stdout for GFLOPS parsing.
# Uses raw timeout (not run_with_timeout) because we need the output in a variable.
burn_output=""
burn_output=$(timeout --signal=KILL $((BURN_DURATION + GPU_BURN_TIMEOUT_GRACE)) ./gpu_burn "$BURN_DURATION" 2>&1) || true
burn_rc=$?

# Kill temp monitoring; allow in-flight write to flush (avoids corrupt CSV/JSON)
kill "$TEMP_PID" 2>/dev/null || true
wait "$TEMP_PID" 2>/dev/null || true
sleep 1

log_info "gpu-burn exited with rc=$burn_rc"
log_info "gpu-burn output length: ${#burn_output} bytes"

# ── Early exit: CUDA init failure (e.g. vGPU without compute, driver mismatch) ──
# Match "cuInit returned <non-zero>" but NOT "cuInit returned 0 (no error)".
if echo "$burn_output" | grep -qiE "Couldn't init CUDA|No CUDA devices" || \
   echo "$burn_output" | grep -qE 'cuInit returned [1-9][0-9]*'; then
    _init_error=$(echo "$burn_output" | grep -iE "Couldn't init CUDA|No CUDA devices|cuInit returned [1-9]" | head -1 | tr -d '[:cntrl:]')
    log_error "CUDA initialization failed: $_init_error"
    RESULT=$(jq -n \
        --arg dur "$BURN_DURATION" \
        --arg err "$_init_error" \
        '{duration_seconds: ($dur | tonumber), status: "error", error: $err, note: "CUDA could not initialize — driver mismatch or vGPU without compute capability"}')
    finish_module "gpu-burn" "error" "$RESULT" '{status, error}'
    exit 1
fi

# ── Post-burn temps ──
post_temps=$(nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d '[:cntrl:]')

# ── Parse results ──
# gpu-burn signals errors with "FAULTY" per GPU. Count only gpu-burn's own output,
# not our log wrapper lines (which contain [ERROR] etc.)
# Reliable patterns: "GPU <N>(<desc>): FAULTY" or "GPU <N>: FAULTY"
errors_found=$(count_grep_re 'FAULTY' "$burn_output")

# gpu-burn prints "OK" or "FAULTY" per GPU in the final summary
gpu_results=$(echo "$burn_output" | grep -E "GPU [0-9]+.*: (OK|FAULTY)" | tail -20 || true)

# Parse GFLOPS per GPU — use POSIX-compatible awk (no named capture groups)
# Strategy: first try per-GPU summary lines (older gpu-burn), then fall back to
# aggregate GFLOPS from progress lines (newer gpu-burn format).
gpu_gflops=$(echo "$burn_output" | awk '
/GPU [0-9]+.*:/ {
    # Extract GPU index
    match($0, /GPU ([0-9]+)/)
    gpu_idx = substr($0, RSTART+4, RLENGTH-4) + 0
    # Extract GFLOPS value
    gflops = 0
    for (i=1; i<=NF; i++) {
        if ($(i+1) == "Gflop/s" || $(i+1) == "GFLOPS") {
            gflops = $i + 0
            break
        }
    }
    if (gflops > 0) {
        status = "ok"
        if ($0 ~ /FAULTY/) status = "fail"
        printf "{\"gpu\":%d,\"gflops\":%.1f,\"status\":\"%s\"},", gpu_idx, gflops, status
    }
}' | sed 's/,$//' | awk '{print "["$0"]"}')
gpu_gflops=$(json_compact_or "$gpu_gflops" "[]")

# ── Fallback: extract GFLOPS from progress lines (newer gpu-burn format) ──
# Newer gpu-burn emits progress lines like:
#   100.0%  proc'd: 281 (36642 Gflop/s)   errors: 0   temps: 47 C
# These have aggregate GFLOPS but no per-GPU breakdown.  Use the LAST
# progress line's value (steady-state) and divide across detected GPUs.
if [ "$gpu_gflops" = "[]" ]; then
    last_progress_gflops=$(echo "$burn_output" | awk '
        /Gflop\/s/ {
            for (i = 1; i <= NF; i++) {
                if ($i ~ /Gflop\/s/) {
                    val = $(i-1)
                    gsub(/[^0-9.]/, "", val)
                    if (val + 0 > 0) last = val + 0
                }
            }
        }
        END { if (last + 0 > 0) printf "%d\n", last }
    ')
    if [ -n "$last_progress_gflops" ]; then
        # Get GPU indices from the final summary ("GPU N: OK" / "GPU N: FAULTY")
        _gpu_idx_list=$(echo "$burn_output" | awk '/GPU [0-9]+: (OK|FAULTY)/ { match($0, /GPU ([0-9]+)/); print substr($0, RSTART+4, RLENGTH-4)+0 }')
        _gpu_count=$(printf '%s\n' "$_gpu_idx_list" | awk 'NF{c++} END{print c}')
        [ "$_gpu_count" -lt 1 ] && _gpu_count=1
        _per_gpu=$(awk "BEGIN { printf \"%.1f\", $last_progress_gflops / $_gpu_count }")
        # Determine per-GPU status from final summary (FAULTY vs OK)
        _gpu_status_map=$(echo "$burn_output" | awk '/GPU [0-9]+: (OK|FAULTY)/ {
            match($0, /GPU ([0-9]+)/); idx=substr($0, RSTART+4, RLENGTH-4)+0
            st="ok"; if ($0 ~ /FAULTY/) st="fail"
            print idx, st
        }')
        # Build JSON array
        gpu_gflops=$(echo "$_gpu_idx_list" | awk -v gf="$_per_gpu" -v smap="$_gpu_status_map" '
            BEGIN {
                # Parse status map (space-separated "idx status" pairs)
                n = split(smap, arr, "\n")
                for (j = 1; j <= n; j++) {
                    split(arr[j], kv, " ")
                    if (kv[1] != "") statmap[kv[1]] = kv[2]
                }
                printf "["
            }
            NR > 1 { printf "," }
            {
                st = (statmap[$1] != "") ? statmap[$1] : "ok"
                printf "{\"gpu\":%s,\"gflops\":%s,\"status\":\"%s\"}", $1, gf, st
            }
            END { printf "]" }
        ')
        gpu_gflops=$(json_compact_or "$gpu_gflops" "[]")
        log_info "GFLOPS from progress lines: ${last_progress_gflops} total (${_per_gpu}/GPU x ${_gpu_count})"
    fi
fi

# ── Third fallback: newer gpu-burn outputs only "GPU N: OK/FAULTY" with no
# ──                 GFLOPS at all (short burns, newer builds). Build a
# ──                 status-only per-GPU array so the JSON result has entries.
if [ "$gpu_gflops" = "[]" ] || [ -z "$gpu_gflops" ]; then
    _status_only=$(echo "$burn_output" | awk '
        /GPU [0-9]+: (OK|FAULTY)/ {
            match($0, /GPU ([0-9]+)/)
            idx = substr($0, RSTART+4, RLENGTH-4) + 0
            st = "ok"; if ($0 ~ /FAULTY/) st = "fail"
            if (seen[idx]++) next  # deduplicate
            entries[idx] = sprintf("{\"gpu\":%d,\"gflops\":0,\"status\":\"%s\"}", idx, st)
            order[++n] = idx
        }
        END {
            printf "["
            for (i = 1; i <= n; i++) {
                if (i > 1) printf ","
                printf "%s", entries[order[i]]
            }
            printf "]"
        }
    ')
    if [ -n "$_status_only" ] && [ "$_status_only" != "[]" ]; then
        gpu_gflops=$(json_compact_or "$_status_only" "[]")
        log_info "No GFLOPS data in gpu-burn output — built status-only per-GPU results"
    fi
fi

# Final safety: ensure gpu_gflops is valid JSON
gpu_gflops=$(json_compact_or "${gpu_gflops:-[]}" "[]")

# Parse temp log for max temps per GPU
max_temps="[]"
max_power="[]"
if [ -f "$TEMP_LOG" ] && [ "$(wc -l < "$TEMP_LOG")" -gt 1 ]; then
    max_temps=$(awk -F, '
        NR>1 && $3+0>0 { if ($3>max[$2]) max[$2]=$3 }
        END {
            first=1; printf "["
            for (g in max) {
                if (!first) printf ","
                first=0
                printf "{\"gpu\":%s,\"max_temp_c\":%s}", g, max[g]
            }
            printf "]"
        }' "$TEMP_LOG" 2>/dev/null) || true
    max_power=$(awk -F, '
        NR>1 && $4+0>0 { if ($4>max[$2]) max[$2]=$4 }
        END {
            first=1; printf "["
            for (g in max) {
                if (!first) printf ","
                first=0
                printf "{\"gpu\":%s,\"max_power_w\":%.1f}", g, max[g]
            }
            printf "]"
        }' "$TEMP_LOG" 2>/dev/null) || true
fi

# Status uses suite-standard values: ok / warn / error / skipped
status="ok"
echo "$burn_output" | grep -qi "FAULTY" && status="error"
[ "$burn_rc" -eq 137 ] && status="error"

max_temps=$(json_compact_or "$max_temps" "[]")
max_power=$(json_compact_or "$max_power" "[]")

# Write JSON arrays to files to avoid --argjson with invalid/large payloads
_tmp_mt=$(json_tmpfile "gpu_burn_mt" "${max_temps:-[]}" "[]")
_tmp_mp=$(json_tmpfile "gpu_burn_mp" "${max_power:-[]}" "[]")
_tmp_gf=$(json_tmpfile "gpu_burn_gf" "${gpu_gflops:-[]}" "[]")

# Save full output to log, then truncate for JSON to avoid ARG_MAX.
# Keep first 20 + last 40 lines in the JSON; full output in the log file.
_tmp_full=$(mktemp -p "${HPC_WORK_DIR}" "gpu_burn_full.XXXXXX")
_tmp_raw=$(mktemp -p "${HPC_WORK_DIR}" "gpu_burn_raw.XXXXXX")
register_cleanup "$_tmp_full" "$_tmp_raw"
printf '%s\n' "$burn_output" > "$_tmp_full"
cp "$_tmp_full" "${HPC_LOG_DIR}/gpu-burn-stdout.log" 2>/dev/null || true
_total_lines=$(wc -l < "$_tmp_full")
if [ "$_total_lines" -gt 80 ]; then
    { head -20 "$_tmp_full"
      echo "... (${_total_lines} lines total — see logs/gpu-burn-stdout.log for full output) ..."
      tail -40 "$_tmp_full"
    } > "$_tmp_raw"
else
    cp "$_tmp_full" "$_tmp_raw"
fi

RESULT=$(jq -n \
    --arg dur "$BURN_DURATION" \
    --arg status "$status" \
    --arg err "$errors_found" \
    --slurpfile max_temps "$_tmp_mt" \
    --slurpfile max_power "$_tmp_mp" \
    --slurpfile gpu_gflops "$_tmp_gf" \
    --arg pre "$pre_temps" \
    --arg post "$post_temps" \
    --arg results "$gpu_results" \
    --rawfile raw_output "$_tmp_raw" \
    '{
        duration_seconds: ($dur | tonumber),
        status: $status,
        errors_detected: ($err | tonumber),
        gpu_performance: $gpu_gflops[0],
        max_temps: $max_temps[0],
        max_power: $max_power[0],
        pre_burn_temps: $pre,
        post_burn_temps: $post,
        gpu_results_summary: $results,
        raw_output: ($raw_output | split("\n") | map(select(length > 0)))
    }')

finish_module "gpu-burn" "$status" "$RESULT" '{status, gpu_performance, errors_detected}'
