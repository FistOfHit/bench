#!/usr/bin/env bash
# thermal-power.sh — GPU/CPU thermals, fan speeds, PSU, throttle detection (V1.1)
# Added: Virtualization detection - skips physical sensors in VMs
SCRIPT_NAME="thermal-power"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== Thermal & Power Diagnostics ==="

# ── Check virtualization ──
VIRT_INFO=$(detect_virtualization)
VIRT_TYPE=$(echo "$VIRT_INFO" | jq -r '.type')
VIRT_NOTE=""

if [ "$VIRT_TYPE" != "none" ]; then
    VIRT_NOTE="Virtualized environment - physical thermal sensors not available"
    log_warn "$VIRT_NOTE"
fi

# ── GPU thermals ──
gpu_thermal="[]"
gpu_throttle="[]"
if has_cmd nvidia-smi; then
    gpu_thermal=$(nvidia-smi --query-gpu=index,name,temperature.gpu,temperature.memory,power.draw,power.limit,fan.speed \
        --format=csv,noheader,nounits 2>/dev/null | tr -d '[:cntrl:]' | awk -F', ' '
    BEGIN { print "[" }
    NR>1 { printf "," }
    {
        gsub(/\[Not Supported\]/, "null")
        gsub(/\[Unknown\]/, "null")
        printf "{\"gpu\":%s,\"name\":\"%s\",\"temp_gpu_c\":%s,\"temp_mem_c\":%s,\"power_draw_w\":%s,\"power_limit_w\":%s,\"fan_pct\":%s}", $1,$2,($3=="null"?"null":$3),($4=="null"?"null":$4),($5=="null"?"null":$5),($6=="null"?"null":$6),($7=="null"?"null":"\""$7"\"")
    }
    END { print "]" }
    ')

    # ── Throttle detection ──
    gpu_throttle=$(nvidia-smi --query-gpu=index,clocks_throttle_reasons.active,clocks_throttle_reasons.gpu_idle,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown \
        --format=csv,noheader 2>/dev/null | tr -d '[:cntrl:]' | awk -F', ' '
    BEGIN { print "[" }
    NR>1 { printf "," }
    {
        printf "{\"gpu\":%s,\"active\":\"%s\",\"idle\":\"%s\",\"sw_power_cap\":\"%s\",\"hw_thermal\":\"%s\",\"sw_thermal\":\"%s\"}", $1,$2,$3,$4,$5,$6
    }
    END { print "]" }
    ' 2>/dev/null || echo "[]")

    # ── Event query for historical throttling ──
    log_info "Querying GPU throttle events..."
    throttle_events=$(nvidia-smi --query-gpu=index,clocks_event_reasons.hw_thermal_slowdown --format=csv,noheader 2>/dev/null || echo "")
fi

# ── CPU thermals ──
cpu_temps="[]"
if [ "$VIRT_TYPE" = "none" ] && [ -d /sys/class/thermal ]; then
    _thermal_out=$(find /sys/class/thermal/thermal_zone* -maxdepth 0 2>/dev/null | while read tz; do
        type=$(cat "$tz/type" 2>/dev/null)
        temp=$(cat "$tz/temp" 2>/dev/null)
        [ -n "$temp" ] && echo "{\"zone\":\"$(basename "$tz")\",\"type\":\"$type\",\"temp_c\":$(echo "scale=1; $temp/1000" | bc)}"
    done) || true
    if [ -n "$_thermal_out" ]; then
        cpu_temps=$(echo "$_thermal_out" | jq -s '.' 2>/dev/null) || cpu_temps="[]"
    fi
elif [ "$VIRT_TYPE" != "none" ]; then
    log_warn "Skipping CPU thermal zones - not reliable in VMs"
fi

# ── Fan speeds (from IPMI if available) ──
fan_json="[]"
if [ "$VIRT_TYPE" = "none" ] && has_cmd ipmitool; then
    fan_json=$(ipmitool sensor list 2>/dev/null | awk -F'|' '
    BEGIN { print "["; first=1 }
    /[Ff]an|RPM/ {
        name=$1; val=$2; gsub(/^[ \t]+|[ \t]+$/, "", name); gsub(/^[ \t]+|[ \t]+$/, "", val)
        if(val != "na" && val+0 > 0) {
            if(!first) printf ","
            first=0
            printf "{\"name\":\"%s\",\"rpm\":%s}", name, val
        }
    }
    END { print "]" }
    ' 2>/dev/null || echo "[]")
elif [ "$VIRT_TYPE" != "none" ]; then
    log_warn "Skipping IPMI fan sensors - not available in VMs"
fi

# ── PSU readings ──
psu_json="[]"
if [ "$VIRT_TYPE" = "none" ] && has_cmd ipmitool; then
    psu_json=$(ipmitool sensor list 2>/dev/null | awk -F'|' '
    BEGIN { print "["; first=1 }
    /[Pp]ower|[Ww]att|PSU|[Vv]oltage/ {
        name=$1; val=$2; unit=$3
        gsub(/^[ \t]+|[ \t]+$/, "", name); gsub(/^[ \t]+|[ \t]+$/, "", val); gsub(/^[ \t]+|[ \t]+$/, "", unit)
        if(val != "na") {
            if(!first) printf ","
            first=0
            printf "{\"name\":\"%s\",\"value\":\"%s\",\"unit\":\"%s\"}", name, val, unit
        }
    }
    END { print "]" }
    ' 2>/dev/null || echo "[]")
fi

# ── GPU count from single source (gpu-inventory or nvidia-smi) for report consistency ──
GPU_COUNT_REPORT=0
if has_cmd nvidia-smi; then
    GPU_COUNT_REPORT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d '[:space:]')
fi
if [ -f "${HPC_RESULTS_DIR}/gpu-inventory.json" ]; then
    gpu_inv_count=$(jq -r '.gpu_count // (.gpus | length) // 0' "${HPC_RESULTS_DIR}/gpu-inventory.json" 2>/dev/null) || true
    if [ "${gpu_inv_count:-0}" -gt 0 ] 2>/dev/null; then
        GPU_COUNT_REPORT="$gpu_inv_count"
    fi
fi
# Fallback: length of gpu_thermals (may be limited in some VM/driver setups)
if [ "${GPU_COUNT_REPORT:-0}" -eq 0 ] 2>/dev/null; then
    GPU_COUNT_REPORT=$(echo "$gpu_thermal" | jq 'length' 2>/dev/null) || GPU_COUNT_REPORT=0
fi

# ── Assess thermal health ──
thermal_status="ok"
# Check if any GPU is above 85°C
hot_gpus=$(echo "$gpu_thermal" | jq '[.[] | select(.temp_gpu_c != null and .temp_gpu_c > 85)] | length' 2>/dev/null) || hot_gpus=0
[ "${hot_gpus:-0}" -gt 0 ] 2>/dev/null && thermal_status="warn"
# Check for active throttling
active_throttle=$(echo "$gpu_throttle" | jq '[.[] | select(.hw_thermal != "Not Active" and .hw_thermal != "0x0000000000000000")] | length' 2>/dev/null) || active_throttle=0
[ "${active_throttle:-0}" -gt 0 ] 2>/dev/null && thermal_status="fail"

RESULT=$(jq -n \
    --argjson gpu "$gpu_thermal" \
    --argjson throttle "$gpu_throttle" \
    --argjson cpu "$cpu_temps" \
    --argjson fans "$fan_json" \
    --argjson psu "$psu_json" \
    --arg status "$thermal_status" \
    --arg hot "$hot_gpus" \
    --argjson gpu_count "$GPU_COUNT_REPORT" \
    --argjson virt "$VIRT_INFO" \
    --arg note "$VIRT_NOTE" \
    '{
        thermal_status: $status,
        gpu_thermals: $gpu,
        gpu_throttle_status: $throttle,
        gpu_count: $gpu_count,
        hot_gpus_above_85c: ($hot | tonumber),
        cpu_thermals: $cpu,
        fans: $fans,
        psu_readings: $psu,
        virtualization: $virt,
        note: (if $note != "" then $note else null end)
    }')

echo "$RESULT" | emit_json "thermal-power" "$thermal_status"
log_ok "Thermal assessment: $thermal_status"
echo "$RESULT" | jq '{thermal_status, hot_gpus_above_85c, gpu_count}'
