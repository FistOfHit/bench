#!/usr/bin/env bash
# report-common.sh — Report helpers and scorecard logic for report.sh
# Source after common.sh. Expects: HPC_RESULTS_DIR, jq.

jf() {
    local file="$1" query="$2" default="${3:-N/A}"
    if [ -f "$file" ]; then
        local val
        val=$(jq -r "$query // empty" "$file" 2>/dev/null)
        echo "${val:-$default}"
    else
        echo "$default"
    fi
}

has_result() { [ -f "${HPC_RESULTS_DIR}/${1}.json" ]; }
mod_status() { jf "${HPC_RESULTS_DIR}/${1}.json" '.status' 'missing'; }

ALL_MODULES=(
    bootstrap inventory gpu-inventory topology network-inventory bmc-inventory software-audit
    dcgm-diag gpu-burn nccl-tests nvbandwidth stream-bench storage-bench hpl-cpu hpl-mxp ib-tests
    network-diag filesystem-diag thermal-power security-scan
)

declare -A SCORES
declare -A SCORE_NOTES

score_module() {
    local mod="$1" score="PASS" note=""
    local status
    status=$(mod_status "$mod")

    case "$status" in
        ok|pass) score="PASS" ;;
        skipped) score="SKIP"; note="Not applicable or missing hardware" ;;
        error|fail) score="FAIL"; note="Module errored" ;;
        warn) score="WARN" ;;
        missing) score="SKIP"; note="Not executed" ;;
        *) score="UNKNOWN"; note="Unrecognized status: $status" ;;
    esac

    case "$mod" in
        hpl-cpu)
            if [ "$score" = "PASS" ]; then
                local passed
                passed=$(jf "${HPC_RESULTS_DIR}/hpl-cpu.json" '.passed' 'false')
                if [ "$passed" != "true" ]; then
                    score="FAIL"; note="HPL residual check failed"
                fi
            fi
            ;;
        ib-tests)
            if [ "$score" = "PASS" ]; then
                local memlock
                memlock=$(jf "${HPC_RESULTS_DIR}/ib-tests.json" '.memlock_check' 'ok')
                if [ "$memlock" = "warn" ]; then
                    score="WARN"; note="memlock ulimit too low for optimal RDMA"
                fi
            fi
            ;;
        dcgm-diag)
            if [ "$score" = "PASS" ]; then
                local overall
                overall=$(jf "${HPC_RESULTS_DIR}/dcgm-diag.json" '.overall' 'N/A')
                if [ "$overall" = "Fail" ] || [ "$overall" = "FAIL" ]; then
                    score="FAIL"; note="DCGM diagnostic reported failures"
                elif [ "$overall" = "Warn" ] || [ "$overall" = "WARN" ]; then
                    score="WARN"; note="DCGM diagnostic reported warnings"
                fi
            fi
            ;;
        security-scan)
            if [ "$score" = "PASS" ]; then
                local sec_status warn_count
                sec_status=$(jf "${HPC_RESULTS_DIR}/security-scan.json" '.status' 'pass')
                if [ "$sec_status" = "warn" ]; then
                    warn_count=$(jf "${HPC_RESULTS_DIR}/security-scan.json" '.warnings | length' '0')
                    score="WARN"; note="${warn_count} security warnings"
                elif [ "$sec_status" = "fail" ]; then
                    score="FAIL"; note="Security scan failed"
                fi
            fi
            ;;
        thermal-power)
            if [ "$score" = "PASS" ]; then
                local thermal_st hot_gpus
                thermal_st=$(jf "${HPC_RESULTS_DIR}/thermal-power.json" '.thermal_status' 'ok')
                hot_gpus=$(jf "${HPC_RESULTS_DIR}/thermal-power.json" '.hot_gpus_above_85c' '0')
                if [ "$thermal_st" = "fail" ]; then
                    score="FAIL"; note="Active thermal throttling detected"
                elif [ "$thermal_st" = "warn" ] || [ "${hot_gpus:-0}" -gt 0 ] 2>/dev/null; then
                    score="WARN"; note="${hot_gpus} GPU(s) above 85°C"
                fi
            fi
            ;;
    esac

    SCORES[$mod]="$score"
    SCORE_NOTES[$mod]="$note"
}

run_report_scoring() {
    local mod
    for mod in "${ALL_MODULES[@]}"; do
        score_module "$mod"
    done

    PASS_COUNT=0
    WARN_COUNT=0
    FAIL_COUNT=0
    SKIP_COUNT=0
    for mod in "${ALL_MODULES[@]}"; do
        case "${SCORES[$mod]}" in
            PASS) PASS_COUNT=$((PASS_COUNT + 1)) ;;
            WARN) WARN_COUNT=$((WARN_COUNT + 1)) ;;
            FAIL) FAIL_COUNT=$((FAIL_COUNT + 1)) ;;
            SKIP) SKIP_COUNT=$((SKIP_COUNT + 1)) ;;
        esac
    done

    OVERALL="PASS"
    if [ "$WARN_COUNT" -gt 0 ]; then OVERALL="WARN"; fi
    if [ "$FAIL_COUNT" -gt 0 ]; then OVERALL="FAIL"; fi
}
