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

# Derive ALL_MODULES from the manifest (single source of truth).
# Sorted by phase + order so scorecard/report renders in execution order.
# Excludes the "report" module itself (it is the generator, not a scored module).
_modules_manifest="${HPC_BENCH_ROOT}/specs/modules.json"
if [ -f "$_modules_manifest" ]; then
    mapfile -t ALL_MODULES < <(
        jq -r '.modules | sort_by(.phase, .order) | .[].name' "$_modules_manifest" 2>/dev/null \
            | grep -v '^report$'
    )
else
    # Fallback: hardcoded list (kept in sync manually if manifest is missing)
    ALL_MODULES=(
        bootstrap runtime-sanity
        inventory gpu-inventory topology network-inventory bmc-inventory software-audit
        dcgm-diag gpu-burn nccl-tests nvbandwidth stream-bench storage-bench hpl-cpu hpl-mxp ib-tests
        network-diag filesystem-diag thermal-power security-scan
    )
fi

declare -A SCORES
declare -A SCORE_NOTES

score_module() {
    local mod="$1" score="PASS" note=""
    local status
    status=$(mod_status "$mod")

    case "$status" in
        ok|pass) score="PASS" ;;
        skipped)
            score="SKIP"
            note=$(jq -r 'if .skip_reason then .skip_reason elif .note then .note else "Not applicable or missing hardware" end' "${HPC_RESULTS_DIR}/${mod}.json" 2>/dev/null) || note="Not applicable or missing hardware"
            ;;
        error|fail) score="FAIL"; note="Module errored" ;;
        warn) score="WARN" ;;
        missing) score="SKIP"; note="Not executed" ;;
        *) score="UNKNOWN"; note="Unrecognized status: $status" ;;
    esac

    case "$mod" in
        runtime-sanity)
            if [ "$score" = "PASS" ]; then
                local rt_note
                rt_note=$(jf "${HPC_RESULTS_DIR}/runtime-sanity.json" '.note' '')
                if [ "$status" = "warn" ]; then
                    score="WARN"; note="${rt_note:-NVIDIA container runtime missing}"
                fi
            fi
            ;;
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
                sec_status=$(jf "${HPC_RESULTS_DIR}/security-scan.json" '.status' 'ok')
                if [ "$sec_status" = "warn" ]; then
                    warn_count=$(jf "${HPC_RESULTS_DIR}/security-scan.json" '.warnings | length' '0')
                    score="WARN"; note="${warn_count} security warnings"
                elif [ "$sec_status" = "error" ] || [ "$sec_status" = "fail" ]; then
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
                    score="WARN"; note="${hot_gpus} GPU(s) above ${GPU_THERMAL_WARN_C:-85}°C"
                fi
            fi
            ;;
    esac

    SCORES[$mod]="$score"
    SCORE_NOTES[$mod]="$note"
}

# NVBandwidth result: extract sum_gbps or mean_gbps (for report table)
report_nvb_bw() {
    local nvb_file="$1" key="$2"
    if [ -f "$nvb_file" ]; then
        jq -r ".$key | if .sum_gbps then \"\(.sum_gbps)\" elif .mean_gbps then \"\(.mean_gbps) (max: \(.max_gbps))\" else \"N/A\" end" "$nvb_file" 2>/dev/null || echo "N/A"
    else
        echo "N/A"
    fi
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

# Build verdict.issues array for report.json (machine-readable for CI/tooling).
# Call after run_report_scoring. Outputs compact JSON array to stdout.
build_verdict_issues_json() {
    local mod note sev
    local out="[]"
    for mod in "${ALL_MODULES[@]}"; do
        case "${SCORES[$mod]}" in
            PASS) continue ;;
            FAIL) sev="critical" ;;
            WARN) sev="warning" ;;
            SKIP|UNKNOWN|*) sev="info" ;;
        esac
        note="${SCORE_NOTES[$mod]:-}"
        out=$(echo "$out" | jq --arg m "$mod" --arg i "$note" --arg s "$sev" '. + [{module:$m, issue:$i, severity:$s}]')
    done
    echo "$out"
}
