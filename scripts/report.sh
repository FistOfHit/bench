#!/usr/bin/env bash
# report.sh — Generate comprehensive markdown report from benchmark results
# Run standalone: HPC_RESULTS_DIR=/path/to/results ./report.sh
SCRIPT_NAME="report"
source "$(dirname "$0")/../lib/common.sh"
source "$(dirname "$0")/../lib/report-common.sh"

log_info "=== Report Generation ==="

REPORT_FILE="${HPC_RESULTS_DIR}/report.md"
HPC_BENCH_VERSION="${HPC_BENCH_VERSION:-$(tr -d '[:space:]' < "${HPC_BENCH_ROOT}/VERSION" 2>/dev/null || echo "unknown")}"

HOSTNAME=$(jf "${HPC_RESULTS_DIR}/bootstrap.json" '.hostname' "$(hostname)")
RUN_DATE=$(jf "${HPC_RESULTS_DIR}/run-all.json" '.start_time' "$(date -u +%Y-%m-%dT%H:%M:%SZ)")
DURATION=$(jf "${HPC_RESULTS_DIR}/run-all.json" '.duration' 'standalone run')

run_report_scoring

GPU_MODEL="none"
GPU_COUNT=0
GPU_ARCH="N/A"
GPU_MEM_GB="N/A"
GPU_TDP_W="N/A"
if has_result "gpu-inventory"; then
    GPU_MODEL=$(jf "${HPC_RESULTS_DIR}/gpu-inventory.json" '.gpu_model // .gpus[0].name' 'none')
    GPU_COUNT=$(jf "${HPC_RESULTS_DIR}/gpu-inventory.json" '.gpu_count // (.gpus | length)' '0')
    GPU_ARCH=$(jf "${HPC_RESULTS_DIR}/gpu-inventory.json" '.gpus[0].compute_capability // "N/A"' 'N/A')
    _gpu_mem_mb=$(jf "${HPC_RESULTS_DIR}/gpu-inventory.json" '.gpus[0].memory_total_mb // empty' '')
    if [ -n "$_gpu_mem_mb" ] && [ "$_gpu_mem_mb" != "N/A" ]; then
        GPU_MEM_GB=$(echo "scale=1; $_gpu_mem_mb / 1024" | bc 2>/dev/null || echo "N/A")
    fi
    GPU_TDP_W=$(jf "${HPC_RESULTS_DIR}/gpu-inventory.json" '.gpus[0].power_limit_w // "N/A"' 'N/A')
fi
# Single source: fallback to bootstrap if gpu-inventory missing (e.g. smoke run)
if [ "${GPU_COUNT:-0}" = "0" ] && has_result "bootstrap"; then
    GPU_COUNT=$(jf "${HPC_RESULTS_DIR}/bootstrap.json" '.gpu_count' '0')
    [ "$GPU_MODEL" = "none" ] && GPU_MODEL=$(jf "${HPC_RESULTS_DIR}/bootstrap.json" '.gpu_model' 'none')
fi

# ── Generate Report ──
{
# First: device result (clear at a glance)
case "$OVERALL" in
    PASS)  RESULT_LABEL="PASSED" ;;
    WARN)  RESULT_LABEL="CONDITIONAL PASS (warnings)" ;;
    FAIL)  RESULT_LABEL="FAILED" ;;
    *)     RESULT_LABEL="$OVERALL" ;;
esac
echo ""
echo "## Device result: **${RESULT_LABEL}**"
echo ""
echo "---"
echo ""

cat <<HEADER
# HPC Benchmarking Report

**Host:** ${HOSTNAME}
**Date:** ${RUN_DATE}
**Duration:** ${DURATION}
**Suite Version:** ${HPC_BENCH_VERSION}
**Overall Result:** **${OVERALL}** (${PASS_COUNT} passed, ${WARN_COUNT} warnings, ${FAIL_COUNT} failed, ${SKIP_COUNT} skipped)

---

## Health Scorecard

| Module | Category | Result | Notes |
|--------|----------|--------|-------|
HEADER

# Emit scorecard rows
for mod in "${ALL_MODULES[@]}"; do
    local_score="${SCORES[$mod]}"
    local_note="${SCORE_NOTES[$mod]}"
    # Determine category
    case "$mod" in
        bootstrap)
            cat="Setup" ;;
        runtime-sanity)
            cat="Runtime" ;;
        inventory|gpu-inventory|topology|network-inventory|bmc-inventory|software-audit)
            cat="Discovery" ;;
        dcgm-diag|gpu-burn|nccl-tests|nvbandwidth|stream-bench|storage-bench|hpl-cpu|hpl-mxp|ib-tests)
            cat="Benchmark" ;;
        network-diag|filesystem-diag|thermal-power|security-scan)
            cat="Diagnostic" ;;
        *) cat="Other" ;;
    esac
    # Emoji for result
    case "$local_score" in
        PASS) icon="✅" ;;
        WARN) icon="⚠️" ;;
        FAIL) icon="❌" ;;
        SKIP) icon="⏭️" ;;
        *) icon="❓" ;;
    esac
    echo "| ${mod} | ${cat} | ${icon} ${local_score} | ${local_note} |"
done

cat <<SUMMARY_HEADER

**Summary:** ${PASS_COUNT} passed, ${WARN_COUNT} warnings, ${FAIL_COUNT} failed, ${SKIP_COUNT} skipped

---

## Executive Summary

SUMMARY_HEADER

# Executive summary text
if [ "$FAIL_COUNT" -eq 0 ] && [ "$WARN_COUNT" -eq 0 ]; then
    echo "All executed benchmarks and diagnostics **passed** without issues. The system appears healthy and ready for production workloads."
elif [ "$FAIL_COUNT" -eq 0 ]; then
    echo "All benchmarks completed successfully with **${WARN_COUNT} warning(s)**. Review the warnings below before accepting the system for production use."
else
    echo "**${FAIL_COUNT} benchmark(s) failed** and **${WARN_COUNT} warning(s)** were raised. The issues below must be addressed before the system can be accepted for production use."
fi

echo ""
echo "---"
echo ""

# ── Hardware Overview ──
cat <<'HW_HDR'
## Hardware Overview

HW_HDR

if has_result "inventory"; then
    echo "### System"
    echo ""
    echo "| Property | Value |"
    echo "|----------|-------|"
    INV="${HPC_RESULTS_DIR}/inventory.json"
    echo "| Hostname | $(jf "$INV" '.os.hostname') |"
    echo "| CPU | $(jf "$INV" '.cpu.model') |"
    echo "| Sockets | $(jf "$INV" '.cpu.sockets') |"
    echo "| Cores (total) | $(jf "$INV" '.cpu.total_cores') |"
    echo "| Threads | $(jf "$INV" '.cpu.threads') |"
    echo "| NUMA Nodes | $(jf "$INV" '.cpu.numa_nodes') |"
    echo "| AVX-512 | $(jf "$INV" '.cpu.has_avx512') |"
    echo "| Memory | $(jf "$INV" '.ram.total_gb') GB |"
    echo "| DIMMs | $(jf "$INV" '.ram.dimms | length') × $(jf "$INV" '.ram.dimms[0].size') $(jf "$INV" '.ram.dimms[0].type') $(jf "$INV" '.ram.dimms[0].speed') |"
    echo "| Storage | $(jf "$INV" '.storage.devices[0].model') ($(jf "$INV" '.storage.devices[0].size')) |"
    echo "| OS | $(jf "$INV" '.os.os') |"
    echo "| Kernel | $(jf "$INV" '.os.kernel') |"
    echo ""
fi

if has_result "gpu-inventory" && [ "$GPU_MODEL" != "none" ]; then
    echo "### GPUs"
    echo ""
    echo "- **Model:** ${GPU_MODEL}"
    echo "- **Count:** ${GPU_COUNT}"
    echo "- **Compute capability:** ${GPU_ARCH}"
    echo "- **Memory per GPU:** ${GPU_MEM_GB} GB"
    echo "- **Power limit:** ${GPU_TDP_W} W"
    echo ""
fi

# Primary storage (what was benchmarked) — one line for readers
PRIMARY_STORAGE=""
if has_result "storage-bench"; then
    PRIMARY_STORAGE=$(jf "${HPC_RESULTS_DIR}/storage-bench.json" '.test_config.device' '')
    [ -n "$PRIMARY_STORAGE" ] && PRIMARY_STORAGE="**Primary storage (benchmarked):** $PRIMARY_STORAGE"
fi
if [ -z "$PRIMARY_STORAGE" ] && has_result "inventory"; then
    PRIMARY_STORAGE=$(jf "${HPC_RESULTS_DIR}/inventory.json" '.storage.devices[0].model' '')
    [ -n "$PRIMARY_STORAGE" ] && PRIMARY_STORAGE="**Primary storage:** $PRIMARY_STORAGE (from inventory)"
fi
if [ -n "$PRIMARY_STORAGE" ]; then
    echo "$PRIMARY_STORAGE"
    echo ""
fi

if has_result "network-inventory"; then
    echo "### Network"
    echo ""
    jq -r '
        if .nics then
            .nics[] | "- **\(.name // "unknown"):** state=\(.state // "N/A"), speed=\(.speed // "N/A"), MTU=\(.mtu // "N/A")"
        else empty end
    ' "${HPC_RESULTS_DIR}/network-inventory.json" 2>/dev/null || echo "- (details unavailable)"
    if jq -e '.infiniband | length > 0' "${HPC_RESULTS_DIR}/network-inventory.json" &>/dev/null; then
        jq -r '.infiniband[]? | "- **IB \(.name // "unknown"):** \(.rate // "N/A") Gb/s, state=\(.state // "N/A")"' \
            "${HPC_RESULTS_DIR}/network-inventory.json" 2>/dev/null || true
    fi
    echo ""
fi

if has_result "topology"; then
    echo "### NUMA Topology"
    echo ""
    numa_nodes=$(jf "${HPC_RESULTS_DIR}/topology.json" '.numa_node_count // (.numa_nodes | length)' 'N/A')
    echo "- **NUMA nodes:** ${numa_nodes}"
    echo ""
fi

echo "---"
echo ""

# ── Benchmark Results ──
cat <<'BENCH_HDR'
## Benchmark Results

BENCH_HDR

# STREAM
if has_result "stream-bench"; then
    echo "### STREAM Memory Bandwidth"
    echo ""
    triad=$(jf "${HPC_RESULTS_DIR}/stream-bench.json" '.triad_mbps' '0')
    copy=$(jf "${HPC_RESULTS_DIR}/stream-bench.json" '.copy_mbps' '0')
    scale=$(jf "${HPC_RESULTS_DIR}/stream-bench.json" '.scale_mbps' '0')
    add=$(jf "${HPC_RESULTS_DIR}/stream-bench.json" '.add_mbps' '0')
    threads=$(jf "${HPC_RESULTS_DIR}/stream-bench.json" '.threads' 'N/A')
    echo "| Operation | Bandwidth (MB/s) |"
    echo "|-----------|-----------------|"
    echo "| Copy | ${copy} |"
    echo "| Scale | ${scale} |"
    echo "| Add | ${add} |"
    echo "| **Triad** | **${triad}** |"
    echo ""
    echo "Threads: ${threads}"
    echo ""
fi

# HPL CPU
if has_result "hpl-cpu" && [ "$(mod_status hpl-cpu)" = "ok" ]; then
    echo "### HPL (CPU Linpack)"
    echo ""
    gflops=$(jf "${HPC_RESULTS_DIR}/hpl-cpu.json" '.gflops' '0')
    hpl_n=$(jf "${HPC_RESULTS_DIR}/hpl-cpu.json" '.problem_size_N' 'N/A')
    hpl_passed=$(jf "${HPC_RESULTS_DIR}/hpl-cpu.json" '.passed' 'N/A')
    hpl_method=$(jf "${HPC_RESULTS_DIR}/hpl-cpu.json" '.method' 'N/A')
    hpl_mem=$(jf "${HPC_RESULTS_DIR}/hpl-cpu.json" '.memory_gb' 'N/A')
    echo "- **Performance:** ${gflops} GFLOPS"
    echo "- **Problem size (N):** ${hpl_n}"
    echo "- **Passed residual check:** ${hpl_passed}"
    echo "- **Method:** ${hpl_method}"
    echo "- **System RAM:** ${hpl_mem} GB"
    echo ""
fi

# HPL MxP (GPU)
if has_result "hpl-mxp" && [ "$(mod_status hpl-mxp)" = "ok" ]; then
    echo "### HPL-MxP (GPU Linpack)"
    echo ""
    gflops=$(jf "${HPC_RESULTS_DIR}/hpl-mxp.json" '.gflops // .tflops' '0')
    echo "- **Performance:** ${gflops} GFLOPS/TFLOPS"
    echo ""
fi

# GPU Burn
if has_result "gpu-burn" && [ "$(mod_status gpu-burn)" != "skipped" ]; then
    echo "### GPU Burn (Stress Test)"
    echo ""
    BURN="${HPC_RESULTS_DIR}/gpu-burn.json"
    burn_status=$(jf "$BURN" '.status' 'N/A')
    burn_duration=$(jf "$BURN" '.duration_seconds' 'N/A')
    burn_errors=$(jf "$BURN" '.errors_detected' '0')
    echo "- **Status:** ${burn_status}"
    echo "- **Duration:** ${burn_duration}s"
    echo "- **Errors detected:** ${burn_errors}"
    # Show per-GPU performance if available
    jq -r '.gpu_performance[]? | "- GPU \(.gpu): \(.gflops // "N/A") GFLOPS (\(.status // "N/A"))"' \
        "$BURN" 2>/dev/null || true
    # Show max temps
    jq -r '.max_temps[]? | "- GPU \(.gpu): max \(.max_temp_c)°C"' \
        "$BURN" 2>/dev/null || true
    echo ""
fi

# DCGM Diagnostics
if has_result "dcgm-diag" && [ "$(mod_status dcgm-diag)" = "ok" ]; then
    echo "### DCGM Diagnostics"
    echo ""
    dcgm_overall=$(jf "${HPC_RESULTS_DIR}/dcgm-diag.json" '.overall' 'N/A')
    echo "- **Overall:** ${dcgm_overall}"
    jq -r '.tests[]? | "- \(.name // .test): \(.result // "N/A")"' \
        "${HPC_RESULTS_DIR}/dcgm-diag.json" 2>/dev/null || true
    echo ""
fi

# NCCL Tests
if has_result "nccl-tests" && [ "$(mod_status nccl-tests)" = "ok" ]; then
    echo "### NCCL Tests (GPU-GPU Communication)"
    echo ""
    nccl_gpu_count=$(jf "${HPC_RESULTS_DIR}/nccl-tests.json" '.gpu_count' 'N/A')
    allreduce_bw=$(jf "${HPC_RESULTS_DIR}/nccl-tests.json" '.peak_allreduce_busbw_gbps' 'N/A')
    nccl_errs=$(jf "${HPC_RESULTS_DIR}/nccl-tests.json" '.runtime_error_count // 0' '0')
    echo "| Metric | Value |"
    echo "|--------|-------|"
    echo "| GPUs | ${nccl_gpu_count} |"
    echo "| **AllReduce Peak Bus BW** | **${allreduce_bw} GB/s** |"
    echo "| Runtime errors | ${nccl_errs} |"
    echo ""
    # Per-test summary
    jq -r '.tests[]? | "- **\(.test):** busbw=\(.summary.busbw_gbps // "N/A") GB/s, algbw=\(.summary.algbw_gbps // "N/A") GB/s\((if .error then ", error=" + .error else "" end))"' \
        "${HPC_RESULTS_DIR}/nccl-tests.json" 2>/dev/null || true
    echo ""
fi

# NVBandwidth
if has_result "nvbandwidth" && [ "$(mod_status nvbandwidth)" = "ok" ]; then
    echo "### NVBandwidth (GPU Memory/PCIe/NVLink)"
    echo ""
    NVB="${HPC_RESULTS_DIR}/nvbandwidth.json"
    echo "| Test | Bandwidth (GB/s) |"
    echo "|------|-----------------|"
    echo "| Host → Device | $(report_nvb_bw "$NVB" host_to_device) |"
    echo "| Device → Host | $(report_nvb_bw "$NVB" device_to_host) |"
    echo "| Device → Device (read) | $(report_nvb_bw "$NVB" device_to_device_read) |"
    echo "| Device → Device (write) | $(report_nvb_bw "$NVB" device_to_device_write) |"
    echo "| Device ↔ Device (bidir) | $(report_nvb_bw "$NVB" device_to_device_bidirectional) |"
    echo "| GPU P2P status | $(jf "$NVB" '.p2p_status' 'unknown') |"
    echo ""
fi

# Storage
if has_result "storage-bench" && [ "$(mod_status storage-bench)" != "skipped" ]; then
    echo "### Storage (fio)"
    echo ""
    STG="${HPC_RESULTS_DIR}/storage-bench.json"
    stg_device=$(jf "$STG" '.test_config.device' 'N/A')
    stg_size=$(jf "$STG" '.test_config.size' 'N/A')
    stg_rota=$(jf "$STG" '.test_config.rotational' 'N/A')
    echo "**Device:** ${stg_device} (rotational: ${stg_rota}), test size: ${stg_size}"
    echo ""
    echo "| Test | Read BW (MB/s) | Write BW (MB/s) | Read IOPS | Write IOPS | Read Lat (μs) | Write Lat (μs) |"
    echo "|------|---------------|----------------|-----------|------------|--------------|---------------|"
    for test_key in sequential_read_1M sequential_write_1M random_4k_read random_4k_write mixed_randrw_70_30 sequential_read_128k_qd128; do
        label=$(echo "$test_key" | tr '_' ' ')
        has_error=$(jq -r ".${test_key}.error // empty" "$STG" 2>/dev/null)
        is_skipped=$(jq -r ".${test_key}.quick_mode_skip // empty" "$STG" 2>/dev/null)
        if [ -n "$has_error" ]; then
            echo "| ${label} | ⚠ ${has_error} | | | | | |"
        elif [ "$is_skipped" = "true" ]; then
            echo "| ${label} | *(skipped — quick mode)* | | | | | |"
        else
            r_bw=$(jf "$STG" ".${test_key}.read_bw_mbps" '-')
            w_bw=$(jf "$STG" ".${test_key}.write_bw_mbps" '-')
            r_iops=$(jf "$STG" ".${test_key}.read_iops" '-')
            w_iops=$(jf "$STG" ".${test_key}.write_iops" '-')
            r_lat=$(jf "$STG" ".${test_key}.read_lat_usec" '-')
            w_lat=$(jf "$STG" ".${test_key}.write_lat_usec" '-')
            echo "| ${label} | ${r_bw} | ${w_bw} | ${r_iops} | ${w_iops} | ${r_lat} | ${w_lat} |"
        fi
    done
    echo ""
fi

# InfiniBand
if has_result "ib-tests" && [ "$(mod_status ib-tests)" != "skipped" ]; then
    echo "### InfiniBand"
    echo ""
    ib_dev=$(jf "${HPC_RESULTS_DIR}/ib-tests.json" '.device' 'N/A')
    ib_rate=$(jf "${HPC_RESULTS_DIR}/ib-tests.json" '.link_rate_gbps' 'N/A')
    ib_write=$(jf "${HPC_RESULTS_DIR}/ib-tests.json" '.ib_write_bw.peak_gbps' 'N/A')
    ib_read=$(jf "${HPC_RESULTS_DIR}/ib-tests.json" '.ib_read_bw.peak_gbps' 'N/A')
    ib_lat=$(jf "${HPC_RESULTS_DIR}/ib-tests.json" '.ib_send_lat.avg_lat_usec' 'N/A')
    ib_theo=$(jf "${HPC_RESULTS_DIR}/ib-tests.json" '.theoretical_rate_gbps' 'N/A')
    memlock=$(jf "${HPC_RESULTS_DIR}/ib-tests.json" '.memlock_check' 'N/A')
    echo "| Metric | Value |"
    echo "|--------|-------|"
    echo "| Device | ${ib_dev} |"
    echo "| Link Rate | ${ib_rate} Gb/s |"
    echo "| Theoretical | ${ib_theo} Gb/s |"
    echo "| Write BW (peak) | ${ib_write} Gb/s |"
    echo "| Read BW (peak) | ${ib_read} Gb/s |"
    echo "| Send Latency (avg) | ${ib_lat} μs |"
    echo "| memlock check | ${memlock} |"
    echo ""
    echo "> ⚠️ **Note:** These are **loopback** (single-node) tests. They validate NIC/driver"
    echo "> health only. They do **NOT** test switch fabric, cabling, or multi-node connectivity."
    echo "> Run inter-node perftest for full fabric validation."
    echo ""
fi

echo "---"
echo ""

# ── Issues Found ──
echo "## Issues Found"
echo ""

ISSUES_FOUND=false

for mod in "${ALL_MODULES[@]}"; do
    if [ "${SCORES[$mod]}" = "FAIL" ] || [ "${SCORES[$mod]}" = "WARN" ]; then
        ISSUES_FOUND=true
        icon="⚠️"
        [ "${SCORES[$mod]}" = "FAIL" ] && icon="❌"
        echo "- ${icon} **${mod}:** ${SCORE_NOTES[$mod]}"
    fi
done

# Check for memlock warning specifically
if has_result "ib-tests"; then
    memlock=$(jf "${HPC_RESULTS_DIR}/ib-tests.json" '.memlock_check' 'ok')
    if [ "$memlock" = "warn" ]; then
        memlock_val=$(jf "${HPC_RESULTS_DIR}/ib-tests.json" '.memlock_ulimit_kb' 'N/A')
        echo "- ⚠️ **memlock ulimit:** Currently ${memlock_val} KB — should be 'unlimited' for RDMA. Set in /etc/security/limits.conf"
        ISSUES_FOUND=true
    fi
fi

# Security-scan: remediation hint for SSH PasswordAuthentication (show when warning present, even if overall pass)
if has_result "security-scan"; then
    if jq -e '.warnings[]? | select(. | test("PasswordAuth|Password"; "i"))' "${HPC_RESULTS_DIR}/security-scan.json" &>/dev/null; then
        echo "- ⚠️ **SSH PasswordAuthentication:** Set \`PasswordAuthentication no\` in sshd_config and restart sshd (e.g. \`systemctl restart sshd\`)."
        ISSUES_FOUND=true
    fi
fi

if [ "$ISSUES_FOUND" = false ]; then
    echo "No issues found. ✅"
fi

echo ""
echo "---"
echo ""

# ── Recommendations ──
echo "## Recommendations"
echo ""

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "### Critical (must fix before acceptance)"
    echo ""
    for mod in "${ALL_MODULES[@]}"; do
        if [ "${SCORES[$mod]}" = "FAIL" ]; then
            echo "1. **${mod}:** ${SCORE_NOTES[$mod]} — investigate and resolve before production use."
        fi
    done
    echo ""
fi

if [ "$WARN_COUNT" -gt 0 ]; then
    echo "### Warnings (should address)"
    echo ""
    for mod in "${ALL_MODULES[@]}"; do
        if [ "${SCORES[$mod]}" = "WARN" ]; then
            echo "1. **${mod}:** ${SCORE_NOTES[$mod]}"
        fi
    done
    echo ""
fi

echo "### General"
echo ""
echo "1. Run inter-node InfiniBand tests (ib_write_bw between hosts) to validate full fabric."
echo "2. Run multi-node NCCL tests to validate GPU-GPU communication across nodes."
echo "3. Verify thermal behavior under sustained load (24h+ burn-in recommended)."
echo "4. Confirm firmware versions match vendor-recommended levels."
echo "5. Validate storage performance under realistic workload patterns."

if has_result "ib-tests"; then
    memlock=$(jf "${HPC_RESULTS_DIR}/ib-tests.json" '.memlock_check' 'ok')
    if [ "$memlock" = "warn" ]; then
        echo "6. **Set memlock ulimit to unlimited** — add to /etc/security/limits.conf:"
        echo '   ```'
        echo '   * soft memlock unlimited'
        echo '   * hard memlock unlimited'
        echo '   ```'
    fi
fi

echo ""
echo "---"
echo ""
echo "*Report generated by HPC Bench Suite v${HPC_BENCH_VERSION} on $(date -u +%Y-%m-%dT%H:%M:%SZ)*"

} > "$REPORT_FILE"

log_ok "Report written to $REPORT_FILE"
jq -n \
    --arg report_file "$REPORT_FILE" \
    --arg overall "$OVERALL" \
    --argjson pass "$PASS_COUNT" \
    --argjson warn "$WARN_COUNT" \
    --argjson fail "$FAIL_COUNT" \
    --argjson skip "$SKIP_COUNT" \
    '{report_file: $report_file, overall: $overall, pass: $pass, warn: $warn, fail: $fail, skip: $skip}' \
    | emit_json "report" "ok"
