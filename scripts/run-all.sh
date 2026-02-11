#!/usr/bin/env bash
# run-all.sh — Master orchestrator for HPC bench suite
# Runs all phases in order, handles timeouts, shows progress
SCRIPT_NAME="run-all"
source "$(dirname "$0")/../lib/common.sh"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAX_MODULE_TIME=${MAX_MODULE_TIME:-1800}  # 30 min per module

# Read version from single source of truth
HPC_BENCH_VERSION=$(cat "${HPC_BENCH_ROOT}/VERSION" 2>/dev/null | tr -d '[:space:]' || echo "unknown")
export HPC_BENCH_VERSION

# ── Exclusive lock — prevent concurrent runs clobbering results ──
LOCKFILE="${HPC_RESULTS_DIR}/.hpc-bench.lock"
mkdir -p "$(dirname "$LOCKFILE")" 2>/dev/null || true
exec 9>"$LOCKFILE"
if ! flock -n 9; then
    echo "ERROR: Another instance of hpc-bench is already running (lockfile: $LOCKFILE)"
    echo "If this is stale, remove $LOCKFILE and retry."
    exit 2
fi
# Lock acquired — fd 9 held until process exits

log_info "================================================================"
log_info "  HPC Bench Suite v${HPC_BENCH_VERSION} — Full Run"
log_info "  Host: $(hostname)"
log_info "  Date: $(date -u)"
log_info "  Results: ${HPC_RESULTS_DIR}"
log_info "================================================================"

START_TIME=$(date +%s)

# Track results
declare -A MODULE_STATUS
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0

run_module() {
    local script="$1"
    local name=$(basename "$script" .sh)
    TOTAL=$((TOTAL + 1))

    echo ""
    log_info "━━━ [$TOTAL] Running: $name ━━━"
    local mod_start=$(date +%s)

    timeout "$MAX_MODULE_TIME" bash "$script" 2>&1 | tee -a "${HPC_LOG_DIR}/${name}-stdout.log"
    local rc=${PIPESTATUS[0]:-$?}

    local mod_end=$(date +%s)
    local mod_duration=$((mod_end - mod_start))

    if [ $rc -eq 0 ]; then
        # Check if module emitted "skipped" status
        if [ -f "${HPC_RESULTS_DIR}/${name}.json" ] && jq -e '.status == "skipped"' "${HPC_RESULTS_DIR}/${name}.json" &>/dev/null; then
            MODULE_STATUS[$name]="SKIPPED (${mod_duration}s)"
            SKIPPED=$((SKIPPED + 1))
            log_warn "[$name] SKIPPED (${mod_duration}s)"
        else
            MODULE_STATUS[$name]="OK (${mod_duration}s)"
            PASSED=$((PASSED + 1))
            log_ok "[$name] OK (${mod_duration}s)"
        fi
    elif [ $rc -eq 137 ]; then
        MODULE_STATUS[$name]="TIMEOUT (${MAX_MODULE_TIME}s limit)"
        FAILED=$((FAILED + 1))
        log_error "[$name] TIMEOUT after ${MAX_MODULE_TIME}s"
    else
        MODULE_STATUS[$name]="FAILED rc=$rc (${mod_duration}s)"
        FAILED=$((FAILED + 1))
        log_error "[$name] FAILED with rc=$rc (${mod_duration}s)"
    fi
}

# ═══════════════════════════════════════════
# PHASE 0: Bootstrap
# ═══════════════════════════════════════════
log_info "╔══════════════════════════════════════╗"
log_info "║  PHASE 0: Bootstrap                  ║"
log_info "╚══════════════════════════════════════╝"
run_module "${SCRIPT_DIR}/bootstrap.sh"

# ═══════════════════════════════════════════
# PHASE 1: Discovery & Inventory
# ═══════════════════════════════════════════
log_info "╔══════════════════════════════════════╗"
log_info "║  PHASE 1: Discovery & Inventory      ║"
log_info "╚══════════════════════════════════════╝"
for script in inventory.sh gpu-inventory.sh topology.sh network-inventory.sh bmc-inventory.sh software-audit.sh; do
    run_module "${SCRIPT_DIR}/${script}"
done

# ═══════════════════════════════════════════
# PHASE 2: Benchmarks
# ═══════════════════════════════════════════
log_info "╔══════════════════════════════════════╗"
log_info "║  PHASE 2: Benchmarks                 ║"
log_info "╚══════════════════════════════════════╝"
for script in dcgm-diag.sh gpu-burn.sh nccl-tests.sh nvbandwidth.sh stream-bench.sh storage-bench.sh hpl-cpu.sh hpl-mxp.sh ib-tests.sh; do
    run_module "${SCRIPT_DIR}/${script}"
done

# ═══════════════════════════════════════════
# PHASE 3: Diagnostics
# ═══════════════════════════════════════════
log_info "╔══════════════════════════════════════╗"
log_info "║  PHASE 3: Diagnostics                ║"
log_info "╚══════════════════════════════════════╝"
for script in network-diag.sh filesystem-diag.sh thermal-power.sh security-scan.sh; do
    run_module "${SCRIPT_DIR}/${script}"
done

# ═══════════════════════════════════════════
# PHASE 4: Report Generation
# ═══════════════════════════════════════════
log_info "╔══════════════════════════════════════╗"
log_info "║  PHASE 4: Report                     ║"
log_info "╚══════════════════════════════════════╝"
run_module "${SCRIPT_DIR}/report.sh"

# ═══════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((TOTAL_DURATION / 60))
DURATION_SEC=$((TOTAL_DURATION % 60))

echo ""
log_info "================================================================"
log_info "  HPC Bench Suite — COMPLETE"
log_info "  Duration: ${DURATION_MIN}m ${DURATION_SEC}s"
log_info "  Modules: $TOTAL total, $PASSED passed, $FAILED failed, $SKIPPED skipped"
log_info "  Results: ${HPC_RESULTS_DIR}"
log_info "  Report:  ${HPC_RESULTS_DIR}/report.md"
log_info "================================================================"

# Module status table
echo ""
echo "Module Results:"
echo "─────────────────────────────────────────"
for name in "${!MODULE_STATUS[@]}"; do
    printf "  %-25s %s\n" "$name" "${MODULE_STATUS[$name]}"
done | sort
echo "─────────────────────────────────────────"

# Emit orchestrator JSON (portable ISO timestamps: GNU date -d not on macOS)
_iso_utc() {
    local ts="$1"
    if date -u -d "@${ts}" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null; then
        return
    fi
    perl -e 'use POSIX qw(strftime); print strftime("%Y-%m-%dT%H:%M:%SZ", gmtime(shift))' "$ts" 2>/dev/null || date -u +%Y-%m-%dT%H:%M:%SZ
}
_start_iso=$(_iso_utc "$START_TIME")
_end_iso=$(_iso_utc "$END_TIME")
SUMMARY_JSON=$(jq -n \
    --arg host "$(hostname)" \
    --arg start_time "$_start_iso" \
    --arg end_time "$_end_iso" \
    --arg dur "${DURATION_MIN}m${DURATION_SEC}s" \
    --argjson dur_s "$TOTAL_DURATION" \
    --argjson total "$TOTAL" \
    --argjson passed "$PASSED" \
    --argjson failed "$FAILED" \
    --argjson skipped "$SKIPPED" \
    '{
        hostname: $host,
        start_time: $start_time,
        end_time: $end_time,
        duration: $dur,
        duration_seconds: $dur_s,
        modules_total: $total,
        modules_passed: $passed,
        modules_failed: $failed,
        modules_skipped: $skipped
    }')

echo "$SUMMARY_JSON" | emit_json "run-all" "$([ "$FAILED" -gt 0 ] && echo "warn" || echo "ok")"

# ═══════════════════════════════════════════
# Results Archive
# ═══════════════════════════════════════════
ARCHIVE_NAME="hpc-bench-$(hostname)-$(date +%Y%m%dT%H%M%S).tar.gz"
ARCHIVE_PATH="${HPC_RESULTS_DIR}/${ARCHIVE_NAME}"

log_info "Bundling results archive..."
# Archive all JSON, report, and logs into a single portable file
_archive_err=$(mktemp)
if tar czf "$ARCHIVE_PATH" \
    -C "$(dirname "$HPC_RESULTS_DIR")" \
    "$(basename "$HPC_RESULTS_DIR")" \
    2>"$_archive_err"; then
    log_ok "Results archive: $ARCHIVE_PATH"
    log_info "  Transfer with: scp ${ARCHIVE_PATH} user@host:/path/"
else
    log_warn "Failed to create results archive (non-fatal): $(cat "$_archive_err" 2>/dev/null | head -3)"
fi
rm -f "$_archive_err"

# ═══════════════════════════════════════════
# Acceptance Gate
# ═══════════════════════════════════════════
# Check for warnings from the report module (which scores each module's internal status)
REPORT_WARN_COUNT=0
if [ -f "${HPC_RESULTS_DIR}/report.json" ]; then
    REPORT_WARN_COUNT=$(jq -r '.warn // 0' "${HPC_RESULTS_DIR}/report.json" 2>/dev/null || echo 0)
fi

if [ "$FAILED" -gt 0 ]; then
    log_error "ACCEPTANCE: FAIL — $FAILED module(s) failed"
    log_error "Review ${HPC_RESULTS_DIR}/report.md for details"
    exit 1
else
    if [ "${REPORT_WARN_COUNT:-0}" -gt 0 ] 2>/dev/null; then
        log_warn "ACCEPTANCE: CONDITIONAL — ${REPORT_WARN_COUNT} warning(s), review before sign-off"
    else
        log_ok "ACCEPTANCE: PASS — all modules passed"
    fi
    exit 0
fi
