#!/usr/bin/env bash
# dcgm-diag.sh -- DCGM diagnostics level 3 (fallback to 2, then 1)
# Phase: 3 (benchmark)
# Requires: jq, timeout
# Emits: dcgm-diag.json
SCRIPT_NAME="dcgm-diag"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== DCGM Diagnostics ==="

has_cmd dcgmi || skip_module "dcgm-diag" "dcgmi not available"

# In VMs, DCGM diag often hangs or is unsupported — use shorter timeout and treat failure as skip
# Quick mode (HPC_QUICK=1): run only level 1 (r1) with short timeout to verify suite end-to-end
DCGM_TIMEOUT=1800
if [ "${HPC_QUICK:-0}" = "1" ]; then
    DCGM_TIMEOUT=90
    if is_virtualized; then
        DCGM_TIMEOUT=120
    fi
    log_info "Quick mode — DCGM level 1 only, timeout ${DCGM_TIMEOUT}s"
elif is_virtualized; then
    DCGM_TIMEOUT=120
    log_info "Virtualized environment — using ${DCGM_TIMEOUT}s timeout per level"
fi

# Level(s) to try: quick = 1 only; full = 3 → 2 → 1
[ "${HPC_QUICK:-0}" = "1" ] && DCGM_LEVELS="1" || DCGM_LEVELS="3 2 1"

# Ensure nv-hostengine is running
if ! pgrep -x nv-hostengine &>/dev/null; then
    log_info "Starting nv-hostengine..."
    nv-hostengine 2>/dev/null || true
    sleep 2
fi

# Try level(s)
diag_output=""
diag_level=0
for level in $DCGM_LEVELS; do
    log_info "Attempting DCGM diag level $level..."
    diag_output=$(run_with_timeout "$DCGM_TIMEOUT" "dcgm-diag-r${level}" dcgmi diag -r "$level" 2>&1) && {
        diag_level=$level
        log_ok "DCGM diag level $level completed"
        break
    } || {
        log_warn "DCGM diag level $level failed, trying lower..."
    }
done

if [ "$diag_level" -eq 0 ]; then
    if is_virtualized; then
        skip_module "dcgm-diag" "all attempted DCGM levels failed in VM after diagnostics attempt"
    fi
    log_error "All DCGM diag levels failed"
    echo "{\"error\":\"all levels failed\",\"output\":$(sanitize_json_str "$diag_output")}" | emit_json "dcgm-diag" "error"
    exit 1
fi

# Parse results — look for PASS/FAIL/WARN/SKIP per test (POSIX awk, no gawk match() 3rd arg)
tests_json=$(echo "$diag_output" | awk '
BEGIN { print "["; first=1 }
/PASS|FAIL|WARN|SKIP/ {
    # Lines like: "  Diagnostic                  : PASS"
    if (match($0, /:[[:space:]]*(PASS|FAIL|WARN|SKIP)/)) {
        result = substr($0, RSTART + 1)
        gsub(/^[ \t]+|[ \t\r\n]+$/, "", result)
        name = substr($0, 1, RSTART - 1)
        gsub(/^[ \t]+|[ \t]+$/, "", name)
        gsub(/"/, "\\\"", name)
        if (!first) printf ","
        first = 0
        printf "{\"test\":\"%s\",\"result\":\"%s\"}", name, result
    }
}
END { print "]" }
' 2>/dev/null || echo "[]")
tests_json=$(json_compact_or "$tests_json" "[]")

# Count results
pass_count=$(echo "$tests_json" | jq '[.[] | select(.result=="PASS")] | length' 2>/dev/null || echo 0)
fail_count=$(echo "$tests_json" | jq '[.[] | select(.result=="FAIL")] | length' 2>/dev/null || echo 0)
warn_count=$(echo "$tests_json" | jq '[.[] | select(.result=="WARN")] | length' 2>/dev/null || echo 0)
pass_count=$(int_or_default "${pass_count:-0}" 0)
fail_count=$(int_or_default "${fail_count:-0}" 0)
warn_count=$(int_or_default "${warn_count:-0}" 0)

overall="pass"
[ "$warn_count" -gt 0 ] && overall="warn"
[ "$fail_count" -gt 0 ] && overall="fail"

RESULT=$(jq -n \
    --arg level "$diag_level" \
    --arg overall "$overall" \
    --argjson tests "$tests_json" \
    --arg pass "$pass_count" \
    --arg fail "$fail_count" \
    --arg warn "$warn_count" \
    --arg raw "$diag_output" \
    '{
        diag_level: ($level | tonumber),
        overall: $overall,
        pass_count: ($pass | tonumber),
        fail_count: ($fail | tonumber),
        warn_count: ($warn | tonumber),
        tests: $tests,
        raw_output: $raw
    }')

finish_module "dcgm-diag" "$overall" "$RESULT" '{diag_level, overall, pass_count, fail_count, warn_count}'
