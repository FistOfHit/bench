#!/usr/bin/env bash
# gpu-burn.sh — GPU stress test using gpu-burn (V1.1)
# Fixed: Output capture, error handling, cleanup ordering
SCRIPT_NAME="gpu-burn"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== GPU Burn Stress Test ==="

require_gpu "gpu-burn" "no nvidia-smi"

# Quick mode (HPC_QUICK=1): 10s burn to verify suite (≥10s needed for GFLOPS output);
# VMs: shorter; else 5 min default. Override with GPU_BURN_DURATION.
if [ "${HPC_QUICK:-0}" = "1" ]; then
    BURN_DURATION=${GPU_BURN_DURATION:-10}
elif is_virtualized; then
    BURN_DURATION=${GPU_BURN_DURATION:-60}
else
    BURN_DURATION=${GPU_BURN_DURATION:-300}  # 5 minutes default on bare metal
fi
BURN_DIR="${HPC_WORK_DIR}/gpu-burn"

# ── Pre-burn temps ──
pre_temps=$(nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d '[:cntrl:]')

# ── Build gpu-burn if needed ──
if [ ! -x "${BURN_DIR}/gpu_burn" ]; then
    log_info "Building gpu-burn..."

    # Prefer latest from online; fallback to bundled source in src/gpu-burn/
    BUNDLED_BURN="${HPC_BENCH_ROOT}/src/gpu-burn"
    rm -rf "$BURN_DIR"
    if git clone https://github.com/wilicc/gpu-burn.git "$BURN_DIR" 2>/dev/null; then
        log_info "Using gpu-burn from upstream (git clone)"
    elif [ -f "${BUNDLED_BURN}/Makefile" ]; then
        log_info "Using bundled gpu-burn source (online clone failed or offline)"
        rm -rf "$BURN_DIR"
        cp -r "$BUNDLED_BURN" "$BURN_DIR"
    else
        log_error "Failed to clone gpu-burn and no bundled source available"
        echo '{"error":"source unavailable"}' | emit_json "gpu-burn" "error"
        exit 1
    fi
    cd "$BURN_DIR"
    if ! make 2>&1 | tail -5; then
        log_error "Failed to build gpu-burn"
        echo '{"error":"build failed"}' | emit_json "gpu-burn" "error"
        exit 1
    fi
fi
register_cleanup "$BURN_DIR"

# ── Run gpu-burn ──
cd "$BURN_DIR"
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

# Run gpu_burn directly with timeout, capture stdout
burn_output=""
burn_output=$(timeout --signal=KILL $((BURN_DURATION + 120)) ./gpu_burn "$BURN_DURATION" 2>&1) || true
burn_rc=$?

# Kill temp monitoring; allow in-flight write to flush (avoids corrupt CSV/JSON)
kill "$TEMP_PID" 2>/dev/null || true
wait "$TEMP_PID" 2>/dev/null || true
sleep 1

log_info "gpu-burn exited with rc=$burn_rc"
log_info "gpu-burn output length: ${#burn_output} bytes"

# ── Post-burn temps ──
post_temps=$(nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d '[:cntrl:]')

# ── Parse results ──
# gpu-burn signals errors with "FAULTY" per GPU. Count only gpu-burn's own output,
# not our log wrapper lines (which contain [ERROR] etc.)
# Reliable patterns: "GPU <N>(<desc>): FAULTY" or "GPU <N>: FAULTY"
errors_found=$(echo "$burn_output" | grep -c "FAULTY" 2>/dev/null || echo 0)

# gpu-burn prints "OK" or "FAULTY" per GPU in the final summary
gpu_results=$(echo "$burn_output" | grep -E "GPU [0-9]+.*: (OK|FAULTY)" | tail -20)

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
# When no GPU line matched, awk outputs "[]" but sed can leave "[""]" — normalize to valid JSON
# Normalize: ensure gpu_gflops is valid JSON array (empty pipeline → empty string)
if [ -z "$gpu_gflops" ] || ! echo "$gpu_gflops" | jq -e 'type == "array"' >/dev/null 2>&1; then
    gpu_gflops="[]"
fi
gpu_gflops=$(echo "$gpu_gflops" | jq -c . 2>/dev/null) || gpu_gflops="[]"

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
        _gpu_count=$(echo "$_gpu_idx_list" | grep -c . 2>/dev/null || echo 1)
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
        echo "$gpu_gflops" | jq . >/dev/null 2>&1 || gpu_gflops="[]"
        gpu_gflops=$(echo "$gpu_gflops" | jq -c . 2>/dev/null) || gpu_gflops="[]"
        log_info "GFLOPS from progress lines: ${last_progress_gflops} total (${_per_gpu}/GPU x ${_gpu_count})"
    fi
fi

# Parse temp log for max temps per GPU
max_temps="[]"
max_power="[]"
if [ -f "$TEMP_LOG" ] && [ "$(wc -l < "$TEMP_LOG")" -gt 1 ]; then
    max_temps=$(awk -F, 'NR>1 && $3+0>0 {if($3>max[$2]) max[$2]=$3} END {first=1; printf "["; for(g in max) {if(!first) printf ","; first=0; printf "{\"gpu\":%s,\"max_temp_c\":%s}", g, max[g]}; printf "]"}' "$TEMP_LOG" 2>/dev/null) || true
    max_power=$(awk -F, 'NR>1 && $4+0>0 {if($4>max[$2]) max[$2]=$4} END {first=1; printf "["; for(g in max) {if(!first) printf ","; first=0; printf "{\"gpu\":%s,\"max_power_w\":%.1f}", g, max[g]}; printf "]"}' "$TEMP_LOG" 2>/dev/null) || true
fi

status="pass"
echo "$burn_output" | grep -qi "FAULTY" && status="fail"
[ "$burn_rc" -eq 137 ] && status="timeout"

# Validate JSON arrays (avoid jq --argjson on invalid data)
echo "$max_temps" | jq . >/dev/null 2>&1 || max_temps="[]"
echo "$max_power" | jq . >/dev/null 2>&1 || max_power="[]"

# Use --arg for errors so we don't require valid JSON number from shell
case "$errors_found" in ''|*[!0-9]*) errors_found=0 ;; esac

# Write JSON arrays to files to avoid --argjson with invalid/large payloads
_tmp_mt=$(mktemp -p "${HPC_WORK_DIR}" gpu_burn_mt.XXXXXX)
_tmp_mp=$(mktemp -p "${HPC_WORK_DIR}" gpu_burn_mp.XXXXXX)
_tmp_gf=$(mktemp -p "${HPC_WORK_DIR}" gpu_burn_gf.XXXXXX)
printf '%s' "${max_temps:-[]}" > "$_tmp_mt"
printf '%s' "${max_power:-[]}" > "$_tmp_mp"
printf '%s' "${gpu_gflops:-[]}" > "$_tmp_gf"
register_cleanup "$_tmp_mt" "$_tmp_mp" "$_tmp_gf"
# Ensure files contain valid JSON for slurpfile
for _f in "$_tmp_mt" "$_tmp_mp" "$_tmp_gf"; do
    jq -c . < "$_f" > "${_f}.v" 2>/dev/null && mv "${_f}.v" "$_f" || printf '%s' "[]" > "$_f"
done

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
    --arg raw_output "$burn_output" \
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

echo "$RESULT" | emit_json "gpu-burn" "$status"
log_ok "GPU burn: $status (errors: $errors_found)"
echo "$RESULT" | jq '{status, gpu_performance, errors_detected}'
