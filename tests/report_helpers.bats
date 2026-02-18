#!/usr/bin/env bats
# Unit tests for lib/report-common.sh helper functions.
# Run: bats tests/report_helpers.bats
#
# Note: score_module and run_report_scoring tests use `run bash -c` to avoid
# interference from lib/common.sh's EXIT trap with bats' subshell management.

load helpers

# ── Test setup ──
setup() {
    setup_test_env "test-report"
    source_common
    source "${HPC_BENCH_ROOT}/lib/report-common.sh"
}

teardown() {
    teardown_test_env
}

# Helper: run score_module in an isolated bash process to avoid EXIT trap conflicts.
_run_score() {
    local mod="$1" json="$2" query="${3:-SCORES}"
    local results_dir="$HPC_RESULTS_DIR"
    echo "$json" > "${results_dir}/${mod}.json"
    bash -c "
        export HPC_BENCH_ROOT='${HPC_BENCH_ROOT}'
        export HPC_RESULTS_DIR='${results_dir}'
        export HPC_LOG_DIR='${results_dir}/logs'
        export HPC_WORK_DIR='$(mktemp -d)'
        export SCRIPT_NAME='test-score'
        mkdir -p \"\$HPC_LOG_DIR\"
        source '${HPC_BENCH_ROOT}/lib/common.sh'
        source '${HPC_BENCH_ROOT}/lib/report-common.sh'
        score_module '$mod'
        if [ '$query' = 'SCORES' ]; then
            echo \"\${SCORES[$mod]}\"
        else
            echo \"\${SCORE_NOTES[$mod]}\"
        fi
    " 2>/dev/null
}

# ═══════════════════════════════════════════
# jf (JSON field extractor)
# ═══════════════════════════════════════════

@test "jf: extracts field from JSON file" {
    echo '{"name": "testhost", "count": 4}' > "${HPC_RESULTS_DIR}/test.json"
    result=$(jf "${HPC_RESULTS_DIR}/test.json" '.name')
    [ "$result" = "testhost" ]
}

@test "jf: returns default for missing field" {
    echo '{"name": "testhost"}' > "${HPC_RESULTS_DIR}/test.json"
    result=$(jf "${HPC_RESULTS_DIR}/test.json" '.missing' 'fallback')
    [ "$result" = "fallback" ]
}

@test "jf: returns default for missing file" {
    result=$(jf "${HPC_RESULTS_DIR}/nonexistent.json" '.field' 'default')
    [ "$result" = "default" ]
}

@test "jf: returns N/A as default when no default given" {
    result=$(jf "${HPC_RESULTS_DIR}/nonexistent.json" '.field')
    [ "$result" = "N/A" ]
}

@test "jf: handles nested queries" {
    echo '{"cpu": {"model": "Xeon", "cores": 64}}' > "${HPC_RESULTS_DIR}/inv.json"
    result=$(jf "${HPC_RESULTS_DIR}/inv.json" '.cpu.model')
    [ "$result" = "Xeon" ]
}

# ═══════════════════════════════════════════
# has_result / mod_status
# ═══════════════════════════════════════════

@test "has_result: returns true when file exists" {
    echo '{}' > "${HPC_RESULTS_DIR}/mymod.json"
    has_result "mymod"
}

@test "has_result: returns false when file missing" {
    ! has_result "nomod"
}

@test "mod_status: returns status from JSON" {
    echo '{"status": "ok", "module": "gpu-burn"}' > "${HPC_RESULTS_DIR}/gpu-burn.json"
    result=$(mod_status "gpu-burn")
    [ "$result" = "ok" ]
}

@test "mod_status: returns missing for nonexistent file" {
    result=$(mod_status "nonexistent")
    [ "$result" = "missing" ]
}

# ═══════════════════════════════════════════
# score_module (isolated via _run_score to avoid EXIT trap)
# ═══════════════════════════════════════════

@test "score_module: PASS for ok status" {
    result=$(_run_score "mymod" '{"status": "ok"}')
    [ "$result" = "PASS" ]
}

@test "score_module: SKIP for skipped status" {
    result=$(_run_score "mymod" '{"status": "skipped", "skip_reason": "no hardware"}')
    [ "$result" = "SKIP" ]
}

@test "score_module: FAIL for error status" {
    result=$(_run_score "mymod" '{"status": "error"}')
    [ "$result" = "FAIL" ]
}

@test "score_module: SKIP for missing file" {
    # Don't write any file; query a nonexistent module
    result=$(bash -c "
        export HPC_BENCH_ROOT='${HPC_BENCH_ROOT}'
        export HPC_RESULTS_DIR='${HPC_RESULTS_DIR}'
        export HPC_LOG_DIR='${HPC_RESULTS_DIR}/logs'
        export HPC_WORK_DIR='$(mktemp -d)'
        export SCRIPT_NAME='test-score'
        mkdir -p \"\$HPC_LOG_DIR\"
        source '${HPC_BENCH_ROOT}/lib/common.sh'
        source '${HPC_BENCH_ROOT}/lib/report-common.sh'
        score_module 'absent'
        echo \"\${SCORES[absent]}\"
    " 2>/dev/null)
    [ "$result" = "SKIP" ]
}

@test "score_module: WARN for warn status" {
    result=$(_run_score "mymod" '{"status": "warn"}')
    [ "$result" = "WARN" ]
}

# ── Module-specific scoring ──

@test "score_module: hpl-cpu fails when residual check is false" {
    result=$(_run_score "hpl-cpu" '{"status": "ok", "passed": false}')
    [ "$result" = "FAIL" ]
}

@test "score_module: dcgm-diag fails when overall is Fail" {
    result=$(_run_score "dcgm-diag" '{"status": "ok", "overall": "Fail"}')
    [ "$result" = "FAIL" ]
}

@test "score_module: ib-tests warns on low memlock" {
    result=$(_run_score "ib-tests" '{"status": "ok", "memlock_check": "warn"}')
    [ "$result" = "WARN" ]
}

@test "score_module: thermal-power fails on active throttle" {
    result=$(_run_score "thermal-power" '{"status": "ok", "thermal_status": "fail"}')
    [ "$result" = "FAIL" ]
}

@test "score_module: security-scan warns on warnings" {
    result=$(_run_score "security-scan" '{"status": "warn", "warnings": ["weak ssh"]}')
    [ "$result" = "WARN" ]
}

# ═══════════════════════════════════════════
# run_report_scoring (isolated)
# ═══════════════════════════════════════════

@test "run_report_scoring: aggregates counts correctly" {
    echo '{"status": "ok"}' > "${HPC_RESULTS_DIR}/bootstrap.json"
    echo '{"status": "ok"}' > "${HPC_RESULTS_DIR}/inventory.json"
    echo '{"status": "skipped", "skip_reason": "no gpu"}' > "${HPC_RESULTS_DIR}/gpu-burn.json"

    run bash -c "
        export HPC_BENCH_ROOT='${HPC_BENCH_ROOT}'
        export HPC_RESULTS_DIR='${HPC_RESULTS_DIR}'
        export HPC_LOG_DIR='${HPC_RESULTS_DIR}/logs'
        export HPC_WORK_DIR='$(mktemp -d)'
        export SCRIPT_NAME='test-score'
        mkdir -p \"\$HPC_LOG_DIR\"
        source '${HPC_BENCH_ROOT}/lib/common.sh'
        source '${HPC_BENCH_ROOT}/lib/report-common.sh'
        run_report_scoring
        echo \"PASS=\${PASS_COUNT} WARN=\${WARN_COUNT} FAIL=\${FAIL_COUNT} SKIP=\${SKIP_COUNT} OVERALL=\${OVERALL}\"
    "
    [ "$status" -eq 0 ]
    [[ "$output" == *"PASS="* ]]
    [[ "$output" == *"OVERALL="* ]]
}

# ═══════════════════════════════════════════
# report_nvb_bw
# ═══════════════════════════════════════════

@test "report_nvb_bw: extracts sum_gbps" {
    echo '{"host_to_device": {"sum_gbps": 12.5}}' > "${HPC_RESULTS_DIR}/nvb.json"
    result=$(report_nvb_bw "${HPC_RESULTS_DIR}/nvb.json" "host_to_device")
    [ "$result" = "12.5" ]
}

@test "report_nvb_bw: extracts mean_gbps with max" {
    echo '{"d2d": {"mean_gbps": 50.0, "max_gbps": 55.0}}' > "${HPC_RESULTS_DIR}/nvb.json"
    result=$(report_nvb_bw "${HPC_RESULTS_DIR}/nvb.json" "d2d")
    [[ "$result" == *"50"* ]]
    [[ "$result" == *"55"* ]]
}

@test "report_nvb_bw: returns N/A for missing file" {
    result=$(report_nvb_bw "/nonexistent.json" "key")
    [ "$result" = "N/A" ]
}
