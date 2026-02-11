#!/usr/bin/env bash
# dcgm-diag.sh — DCGM diagnostics level 3 (fallback to 2, then 1)
SCRIPT_NAME="dcgm-diag"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== DCGM Diagnostics ==="

if ! has_cmd dcgmi; then
    log_warn "dcgmi not found — skipping"
    echo '{"note":"dcgmi not available"}' | emit_json "dcgm-diag" "skipped"
    exit 0
fi

# Ensure nv-hostengine is running
if ! pgrep -x nv-hostengine &>/dev/null; then
    log_info "Starting nv-hostengine..."
    nv-hostengine 2>/dev/null || true
    sleep 2
fi

# Try levels 3 → 2 → 1
diag_output=""
diag_level=0
for level in 3 2 1; do
    log_info "Attempting DCGM diag level $level..."
    diag_output=$(run_with_timeout 1800 "dcgm-diag-r${level}" dcgmi diag -r "$level" 2>&1) && {
        diag_level=$level
        log_ok "DCGM diag level $level completed"
        break
    } || {
        log_warn "DCGM diag level $level failed, trying lower..."
    }
done

if [ "$diag_level" -eq 0 ]; then
    log_error "All DCGM diag levels failed"
    echo "{\"error\":\"all levels failed\",\"output\":$(json_str "$diag_output")}" | emit_json "dcgm-diag" "error"
    exit 1
fi

# Parse results — look for PASS/FAIL/WARN/SKIP per test
tests_json=$(echo "$diag_output" | awk '
BEGIN { print "["; first=1 }
/PASS|FAIL|WARN|SKIP/ {
    # Lines like: "  Diagnostic                  : PASS"
    if (match($0, /^[[:space:]]*([A-Za-z ]+[A-Za-z])[[:space:]]*:[[:space:]]*(PASS|FAIL|WARN|SKIP)/, m)) {
        name = m[1]; gsub(/^[ \t]+|[ \t]+$/, "", name)
        result = m[2]
        if (!first) printf ","
        first = 0
        printf "{\"test\":\"%s\",\"result\":\"%s\"}", name, result
    }
}
END { print "]" }
' 2>/dev/null || echo "[]")

# Count results
pass_count=$(echo "$tests_json" | jq '[.[] | select(.result=="PASS")] | length' 2>/dev/null || echo 0)
fail_count=$(echo "$tests_json" | jq '[.[] | select(.result=="FAIL")] | length' 2>/dev/null || echo 0)
warn_count=$(echo "$tests_json" | jq '[.[] | select(.result=="WARN")] | length' 2>/dev/null || echo 0)

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

echo "$RESULT" | emit_json "dcgm-diag" "$overall"
log_ok "DCGM diag: $overall (P:$pass_count F:$fail_count W:$warn_count)"
echo "$RESULT" | jq '{diag_level, overall, pass_count, fail_count, warn_count, tests}'
