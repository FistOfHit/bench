#!/usr/bin/env bash
# run-all.sh — Master orchestrator for HPC bench suite
# Runs all phases in order, handles timeouts, shows progress
# Usage: run-all.sh [--quick] [--smoke] [--ci] [--install-nvidia] [--install-nvidia-container-toolkit] [--auto-install-runtime] [--fail-fast-runtime]
#   --quick                       short benchmarks to verify suite end-to-end
#   --smoke                       bootstrap + inventory + report only
#   --ci                          CI-friendly mode (quick + quieter logs + deterministic defaults)
#   --install-nvidia              pass to bootstrap: install NVIDIA driver+CUDA when GPU present (Ubuntu; reboot after)
#   --install-nvidia-container-toolkit  pass to bootstrap: install Docker + NVIDIA container runtime
#   --auto-install-runtime        auto-install NVIDIA container runtime during runtime-sanity
#   --fail-fast-runtime           stop immediately if runtime-sanity fails
SCRIPT_NAME="run-all"
source "$(dirname "$0")/../lib/common.sh"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODULES_MANIFEST="${HPC_BENCH_ROOT}/specs/modules.json"

if ! has_cmd jq; then
    echo "FATAL: jq is required to run hpc-bench." >&2
    exit 1
fi
if [ ! -f "$MODULES_MANIFEST" ]; then
    echo "FATAL: module manifest missing: $MODULES_MANIFEST" >&2
    exit 1
fi

# Parse flags
HPC_SMOKE=0
BOOTSTRAP_EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --quick)
            export HPC_QUICK=1
            shift
            ;;
        --smoke)
            HPC_SMOKE=1
            export HPC_QUICK=1
            shift
            ;;
        --install-nvidia)
            BOOTSTRAP_EXTRA_ARGS+=(--install-nvidia)
            shift
            ;;
        --install-nvidia-container-toolkit)
            BOOTSTRAP_EXTRA_ARGS+=(--install-nvidia-container-toolkit)
            shift
            ;;
        --auto-install-runtime)
            export HPC_AUTO_INSTALL_CONTAINER_RUNTIME=1
            shift
            ;;
        --ci)
            export HPC_CI=1
            export HPC_QUICK=1
            shift
            ;;
        --fail-fast-runtime)
            export HPC_FAIL_FAST_RUNTIME=1
            shift
            ;;
        *)
            log_warn "Unknown option: $1 (use --quick, --smoke, --ci, --install-nvidia, --auto-install-runtime, --fail-fast-runtime)"
            shift
            ;;
    esac
done

# Quick mode: short runs per module so suite finishes fast (smoke test / VM validation)
# Defaults are in conf/defaults.sh.
if [ "${HPC_QUICK:-0}" = "1" ]; then
    MAX_MODULE_TIME=${MAX_MODULE_TIME:-${MAX_MODULE_TIME_QUICK:-600}}
else
    MAX_MODULE_TIME=${MAX_MODULE_TIME:-${MAX_MODULE_TIME_FULL:-1800}}
fi
if [ "${HPC_CI:-0}" = "1" ]; then
    MAX_MODULE_TIME=${MAX_MODULE_TIME:-${MAX_MODULE_TIME_QUICK:-600}}
fi

# Read version from single source of truth
HPC_BENCH_VERSION=$(tr -d '[:space:]' < "${HPC_BENCH_ROOT}/VERSION" 2>/dev/null || echo "unknown")
export HPC_BENCH_VERSION

# ── Exclusive lock — prevent concurrent runs clobbering results ──
LOCKFILE="${HPC_RESULTS_DIR}/.hpc-bench.lock"
LOCKDIR="${LOCKFILE}.d"
mkdir -p "$(dirname "$LOCKFILE")" 2>/dev/null || true
if has_cmd flock; then
    exec 9>"$LOCKFILE"
    if ! flock -n 9; then
        echo "ERROR: Another instance of hpc-bench is already running (lockfile: $LOCKFILE)"
        echo "If this is stale, remove $LOCKFILE and retry."
        exit 2
    fi
    # Lock acquired — fd 9 held until process exits
else
    # Fallback for minimal environments without flock(1): atomic lock directory.
    if mkdir "$LOCKDIR" 2>/dev/null; then
        echo "$$" > "${LOCKDIR}/pid" 2>/dev/null || true
        trap 'rm -rf "$LOCKDIR"' EXIT
    else
        echo "ERROR: Another instance of hpc-bench is already running (lockdir: $LOCKDIR)"
        echo "If this is stale, remove $LOCKDIR and retry."
        exit 2
    fi
fi

log_info "================================================================"
if [ "${HPC_SMOKE:-0}" = "1" ]; then
    log_info "  HPC Bench Suite v${HPC_BENCH_VERSION} — Smoke (bootstrap + inventory + report)"
elif [ "${HPC_QUICK:-0}" = "1" ]; then
    log_info "  HPC Bench Suite v${HPC_BENCH_VERSION} — Quick Run (short benchmarks)"
else
    log_info "  HPC Bench Suite v${HPC_BENCH_VERSION} — Full Run"
fi
log_info "  Host: $(hostname)"
log_info "  Date: $(date -u)"
log_info "  Results: ${HPC_RESULTS_DIR}"
if [ "${HPC_AUTO_INSTALL_CONTAINER_RUNTIME:-0}" = "1" ]; then
    log_info "  Runtime sanity auto-install: enabled"
fi
if [ "${HPC_FAIL_FAST_RUNTIME:-0}" = "1" ]; then
    log_info "  Runtime fail-fast: enabled"
fi
if [ "${HPC_CI:-0}" = "1" ]; then
    log_info "  CI mode: enabled (compact module stdout, quick-mode defaults)"
fi
if [ "${#BOOTSTRAP_EXTRA_ARGS[@]}" -gt 0 ]; then
    log_info "  Bootstrap flags: ${BOOTSTRAP_EXTRA_ARGS[*]}"
fi
log_info "================================================================"

START_TIME=$(date +%s)

# Track results
declare -A MODULE_STATUS
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0

# Get short skip/note reason from module JSON for display
get_skip_reason() {
    local name="$1"
    local j="${HPC_RESULTS_DIR}/${name}.json"
    if [ -f "$j" ]; then
        jq -r 'if .skip_reason then .skip_reason elif .note then .note else empty end' "$j" 2>/dev/null | head -c 60
    fi
}

# Manifest helpers
manifest_phase_scripts() {
    local phase="$1"
    local is_root="$2"
    jq -r --argjson phase "$phase" --argjson is_root "$is_root" '
        .modules
        | map(select(.phase == $phase))
        | map(select((.requires_root // false) | not or $is_root))
        | sort_by(.order)
        | .[].script
    ' "$MODULES_MANIFEST" 2>/dev/null
}

# One-line summary for a phase (passed/failed/skipped counts). Call after the phase completes.
phase_summary_line() {
    local phase="$1" label="$2"
    local -a names=()
    mapfile -t names < <(jq -r --argjson p "$phase" '.modules | map(select(.phase == $p)) | .[].name' "$MODULES_MANIFEST" 2>/dev/null)
    local p=0 f=0 s=0 n st
    for n in "${names[@]}"; do
        [ -z "$n" ] && continue
        st="${MODULE_STATUS[$n]:-}"
        if [[ "$st" == OK* ]]; then ((p++)) || true
        elif [[ "$st" == SKIPPED* ]]; then ((s++)) || true
        else ((f++)) || true
        fi
    done
    local total=$((p + f + s))
    [ "$total" -eq 0 ] && return
    log_info "  Phase $phase $label: $p passed, $f failed, $s skipped"
}

manifest_required_cmds() {
    local name="$1"
    jq -r --arg n "$name" '
        (.modules[]? | select(.name == $n) | .required_cmds[]?) // empty
    ' "$MODULES_MANIFEST" 2>/dev/null
}

module_required_cmds() {
    local name="$1"
    local -a cmds=()
    mapfile -t cmds < <(manifest_required_cmds "$name")
    if [ "${#cmds[@]}" -eq 0 ]; then
        printf '%s\n' jq
        return 0
    fi
    printf '%s\n' "${cmds[@]}"
}

module_missing_cmds() {
    local name="$1"
    local missing=()
    local req
    while IFS= read -r req; do
        [ -z "$req" ] && continue
        if ! has_cmd "$req"; then
            missing+=("$req")
        fi
    done < <(module_required_cmds "$name")
    if [ "${#missing[@]}" -gt 0 ]; then
        local IFS=","
        echo "${missing[*]}"
    fi
}

skip_module_due_prereq() {
    local name="$1" reason="$2" mod_duration="$3"
    MODULE_STATUS[$name]="SKIPPED (${mod_duration}s): $reason"
    SKIPPED=$((SKIPPED + 1))
    log_warn "[$name] SKIPPED (${mod_duration}s): $reason"
    jq -n --arg r "$reason" '{skip_reason: $r}' | emit_json "$name" "skipped" 2>/dev/null || true
}

_ci_tail_snippet() {
    local name="$1"
    if [ "${HPC_CI:-0}" = "1" ]; then
        log_error "[$name] tail log snippet:"
        tail -n 40 "${HPC_LOG_DIR}/${name}-stdout.log" 2>/dev/null || true
    fi
}

# Record result for a module execution (shared by sequential + parallel phase 1).
record_module_result() {
    local name="$1" rc="$2" mod_duration="$3"

    if [ "$rc" -eq 0 ]; then
        # Module can exit 0 but still emit "skipped".
        if [ -f "${HPC_RESULTS_DIR}/${name}.json" ] && jq -e '.status == "skipped"' "${HPC_RESULTS_DIR}/${name}.json" &>/dev/null; then
            local reason
            reason="$(get_skip_reason "$name")"
            if [ -n "$reason" ]; then
                MODULE_STATUS[$name]="SKIPPED (${mod_duration}s): $reason"
            else
                MODULE_STATUS[$name]="SKIPPED (${mod_duration}s)"
            fi
            SKIPPED=$((SKIPPED + 1))
            log_warn "[$name] SKIPPED (${mod_duration}s)"
        else
            MODULE_STATUS[$name]="OK (${mod_duration}s)"
            PASSED=$((PASSED + 1))
            log_ok "[$name] OK (${mod_duration}s)"
        fi
        return 0
    fi

    if [ "$rc" -eq 137 ]; then
        MODULE_STATUS[$name]="TIMEOUT (${MAX_MODULE_TIME}s limit)"
        FAILED=$((FAILED + 1))
        log_error "[$name] TIMEOUT after ${MAX_MODULE_TIME}s"
        _ci_tail_snippet "$name"
        return 0
    fi

    MODULE_STATUS[$name]="FAILED rc=$rc (${mod_duration}s)"
    FAILED=$((FAILED + 1))
    log_error "[$name] FAILED with rc=$rc (${mod_duration}s)"
    _ci_tail_snippet "$name"
    return 0
}

run_module_command() {
    local script="$1" name="$2"
    shift 2

    if [ "${HPC_CI:-0}" = "1" ]; then
        timeout "$MAX_MODULE_TIME" bash "$script" "$@" >> "${HPC_LOG_DIR}/${name}-stdout.log" 2>&1
        return $?
    fi

    timeout "$MAX_MODULE_TIME" bash "$script" "$@" 2>&1 | tee -a "${HPC_LOG_DIR}/${name}-stdout.log"
    return ${PIPESTATUS[0]:-$?}
}

run_module() {
    local script="$1"
    local name
    name="$(basename "$script" .sh)"
    TOTAL=$((TOTAL + 1))

    echo ""
    log_info "━━━ [$TOTAL] Running: $name ━━━"
    local mod_start mod_end mod_duration missing
    mod_start=$(date +%s)
    missing=$(module_missing_cmds "$name")
    if [ -n "$missing" ]; then
        mod_end=$(date +%s)
        mod_duration=$((mod_end - mod_start))
        skip_module_due_prereq "$name" "missing commands: $missing" "$mod_duration"
        return 0
    fi

    # Disable errexit/pipefail so a module failure doesn't kill the orchestrator
    set +eo pipefail
    if [ "$name" = "bootstrap" ] && [ "${#BOOTSTRAP_EXTRA_ARGS[@]}" -gt 0 ]; then
        run_module_command "$script" "$name" "${BOOTSTRAP_EXTRA_ARGS[@]}"
    else
        run_module_command "$script" "$name"
    fi
    local rc=$?
    set -eo pipefail

    mod_end=$(date +%s)
    mod_duration=$((mod_end - mod_start))
    record_module_result "$name" "$rc" "$mod_duration"
}

# ── User-level mode: when not root, skip root-only modules ──
HPC_IS_ROOT=0
[ "$(id -u)" -eq 0 ] && HPC_IS_ROOT=1
if [ "$HPC_IS_ROOT" -eq 0 ]; then
    log_warn "Running as non-root: results in ${HPC_RESULTS_DIR}; bootstrap and bmc-inventory skipped."
fi

# ═══════════════════════════════════════════
# PHASE 0: Bootstrap (manifest phase=0, requires root)
# ═══════════════════════════════════════════
log_info "╔══════════════════════════════════════╗"
log_info "║  PHASE 0: Bootstrap                  ║"
log_info "╚══════════════════════════════════════╝"
if [ "$HPC_IS_ROOT" -eq 1 ]; then
    run_module "${SCRIPT_DIR}/bootstrap.sh"
    # Bootstrap may exit after installing driver — requires reboot before GPU stack works
    if [ -f "${HPC_RESULTS_DIR}/bootstrap.json" ] && jq -e '.reboot_required == true' "${HPC_RESULTS_DIR}/bootstrap.json" >/dev/null 2>&1; then
        log_info "================================================================"
        log_info "  NVIDIA driver was installed. Reboot required before GPU benchmarks."
        _rerun="sudo bash scripts/run-all.sh"
        [ "${#BOOTSTRAP_EXTRA_ARGS[@]}" -gt 0 ] && _rerun="$_rerun ${BOOTSTRAP_EXTRA_ARGS[*]}"
        log_info "  After reboot, run: $_rerun"
        log_info "================================================================"
        exit 0
    fi
else
    TOTAL=$((TOTAL + 1))
    MODULE_STATUS[bootstrap]="SKIPPED (0s): requires root"
    SKIPPED=$((SKIPPED + 1))
    log_warn "[bootstrap] SKIPPED (requires root)"
    # Emit minimal bootstrap.json so report has hostname/results_dir (optional)
    jq -n --arg h "$(hostname)" --arg r "${HPC_RESULTS_DIR}" '{hostname: $h, results_dir: $r, status: "skipped", note: "requires root"}' | emit_json "bootstrap" "skipped" 2>/dev/null || true
fi
phase_summary_line 0 "Bootstrap"

# ═══════════════════════════════════════════
# PHASE 1: Runtime Sanity (manifest phase=1, early fail-fast/auto-install checks)
# ═══════════════════════════════════════════
log_info "╔══════════════════════════════════════╗"
log_info "║  PHASE 1: Runtime Sanity             ║"
log_info "╚══════════════════════════════════════╝"
run_module "${SCRIPT_DIR}/runtime-sanity.sh"
if [ "${HPC_FAIL_FAST_RUNTIME:-0}" = "1" ] && \
   [ -f "${HPC_RESULTS_DIR}/runtime-sanity.json" ] && \
   jq -e '.status == "error"' "${HPC_RESULTS_DIR}/runtime-sanity.json" >/dev/null 2>&1; then
    log_error "Runtime fail-fast triggered by runtime-sanity check."
    log_error "Resolve runtime prerequisites or rerun without --fail-fast-runtime."
    exit 1
fi
phase_summary_line 1 "Runtime Sanity"

# ═══════════════════════════════════════════
# PHASE 2: Discovery & Inventory (manifest phase=2, parallel)
# ═══════════════════════════════════════════

# ── GPU driver warmup ──
# After bootstrap starts nv-hostengine (DCGM daemon), it needs time to
# initialise.  If Phase 1 modules all hit nvidia-smi in parallel while the
# daemon is still probing the GPU, they serialise on the driver lock and
# individual modules can appear to take 2+ minutes instead of <1s.
# A single blocking nvidia-smi call here ensures the driver (and DCGM) are
# ready before we fan out.
if has_cmd nvidia-smi; then
    log_info "Warming up GPU driver before parallel inventory..."
    nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1 || true
fi

log_info "╔══════════════════════════════════════╗"
log_info "║  PHASE 2: Discovery & Inventory      ║"
log_info "╚══════════════════════════════════════╝"
mapfile -t PHASE2_SCRIPTS < <(manifest_phase_scripts 2 "$HPC_IS_ROOT")
_phase2_pids=()
_phase2_names=()
for script in "${PHASE2_SCRIPTS[@]}"; do
    name=$(basename "$script" .sh)
    TOTAL=$((TOTAL + 1))
    echo ""
    log_info "━━━ [$TOTAL] Running: $name ━━━"
    missing=$(module_missing_cmds "$name")
    if [ -n "$missing" ]; then
        MODULE_STATUS[$name]="SKIPPED (0s): missing commands: $missing"
        SKIPPED=$((SKIPPED + 1))
        log_warn "[$name] SKIPPED (0s): missing commands: $missing"
        jq -n --arg r "missing commands: $missing" '{skip_reason: $r}' | emit_json "$name" "skipped" 2>/dev/null || true
        continue
    fi
    (
        set +eo pipefail
        mod_start=$(date +%s)
        run_module_command "${SCRIPT_DIR}/${script}" "$name"
        rc=$?
        mod_end=$(date +%s)
        echo "$rc $((mod_end - mod_start))" > "${HPC_RESULTS_DIR}/.phase2_${name}.meta"
    ) &
    _phase2_pids+=($!)
    _phase2_names+=("$name")
done
for i in "${!_phase2_pids[@]}"; do
    wait "${_phase2_pids[$i]}" 2>/dev/null || true
    name="${_phase2_names[$i]}"
    rc=0
    mod_duration=0
    if [ -f "${HPC_RESULTS_DIR}/.phase2_${name}.meta" ]; then
        read -r rc mod_duration < "${HPC_RESULTS_DIR}/.phase2_${name}.meta"
        rm -f "${HPC_RESULTS_DIR}/.phase2_${name}.meta"
    fi
    record_module_result "$name" "$rc" "$mod_duration"
done
phase_summary_line 2 "Discovery"

# ═══════════════════════════════════════════
# PHASE 3: Benchmarks (manifest phase=3, skipped in smoke mode)
# ═══════════════════════════════════════════
if [ "${HPC_SMOKE:-0}" != "1" ]; then
    log_info "╔══════════════════════════════════════╗"
    log_info "║  PHASE 3: Benchmarks                 ║"
    log_info "╚══════════════════════════════════════╝"
    mapfile -t PHASE3_SCRIPTS < <(manifest_phase_scripts 3 "$HPC_IS_ROOT")
    for script in "${PHASE3_SCRIPTS[@]}"; do
        run_module "${SCRIPT_DIR}/${script}"
    done
    phase_summary_line 3 "Benchmarks"
fi

# ═══════════════════════════════════════════
# PHASE 4: Diagnostics (manifest phase=4, skipped in smoke mode)
# ═══════════════════════════════════════════
if [ "${HPC_SMOKE:-0}" != "1" ]; then
    log_info "╔══════════════════════════════════════╗"
    log_info "║  PHASE 4: Diagnostics                ║"
    log_info "╚══════════════════════════════════════╝"
    mapfile -t PHASE4_SCRIPTS < <(manifest_phase_scripts 4 "$HPC_IS_ROOT")
    for script in "${PHASE4_SCRIPTS[@]}"; do
        run_module "${SCRIPT_DIR}/${script}"
    done
    phase_summary_line 4 "Diagnostics"
fi

# ═══════════════════════════════════════════
# PHASE 5: Report Generation (manifest phase=5)
# ═══════════════════════════════════════════
log_info "╔══════════════════════════════════════╗"
log_info "║  PHASE 5: Report                     ║"
log_info "╚══════════════════════════════════════╝"
mapfile -t PHASE5_SCRIPTS < <(manifest_phase_scripts 5 "$HPC_IS_ROOT")
for script in "${PHASE5_SCRIPTS[@]}"; do
    run_module "${SCRIPT_DIR}/${script}"
done
phase_summary_line 5 "Report"

# ═══════════════════════════════════════════
# Summary — Progressive output: device result → checklist → details
# ═══════════════════════════════════════════
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((TOTAL_DURATION / 60))
DURATION_SEC=$((TOTAL_DURATION % 60))

# 1) Device result (first and foremost)
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
if [ "$FAILED" -gt 0 ]; then
    echo "║  DEVICE: FAILED  —  $FAILED module(s) failed  (review report)       ║"
elif [ "$SKIPPED" -eq "$TOTAL" ] && [ "$PASSED" -eq 0 ]; then
    echo "║  DEVICE: INCONCLUSIVE  —  all modules skipped (e.g. no GPU/IB)       ║"
else
    echo "║  DEVICE: PASSED  —  all ran modules passed                           ║"
fi
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# 2) Checklist (compact)
echo "  CHECKLIST  ($PASSED passed, $FAILED failed, $SKIPPED skipped)"
echo "  ─────────────────────────────────────────────────────────────"
for name in $(printf '%s\n' "${!MODULE_STATUS[@]}" | sort); do
    status="${MODULE_STATUS[$name]}"
    if [[ "$status" == OK* ]]; then
        symbol=$(status_display_string PASS)
    elif [[ "$status" == SKIPPED* ]]; then
        symbol=$(status_display_string SKIP)
    else
        symbol=$(status_display_string FAIL)
    fi
    printf "    %s  %-28s  %s\n" "$symbol" "$name" "$status"
done
echo "  ─────────────────────────────────────────────────────────────"
echo ""

# 3) Detail: passed vs failed/skipped
echo "  PASSED:"
for name in $(printf '%s\n' "${!MODULE_STATUS[@]}" | sort); do
    status="${MODULE_STATUS[$name]}"
    [[ "$status" == OK* ]] && printf "    • %s — %s\n" "$name" "$status"
done
echo ""
echo "  FAILED / SKIPPED:"
for name in $(printf '%s\n' "${!MODULE_STATUS[@]}" | sort); do
    status="${MODULE_STATUS[$name]}"
    [[ "$status" != OK* ]] && printf "    • %s — %s\n" "$name" "$status"
done
echo ""

log_info "================================================================"
log_info "  HPC Bench Suite — COMPLETE"
log_info "  Duration: ${DURATION_MIN}m ${DURATION_SEC}s"
log_info "  Results: ${HPC_RESULTS_DIR}"
log_info "  Report:  ${HPC_RESULTS_DIR}/report.md"
log_info "================================================================"

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
# Copy results to a temp snapshot so tar does not see "file changed as we read it"
_archive_err=$(mktemp)
_archive_snap=$(mktemp -d)
if cp -a "$HPC_RESULTS_DIR" "$_archive_snap/"; then
    if tar czf "$ARCHIVE_PATH" -C "$_archive_snap" "$(basename "$HPC_RESULTS_DIR")" 2>"$_archive_err"; then
        log_ok "Results archive: $ARCHIVE_PATH"
        log_info "  Transfer with: scp ${ARCHIVE_PATH} user@host:/path/"
    else
        log_warn "Failed to create results archive (non-fatal): $(cat "$_archive_err" 2>/dev/null | head -3)"
    fi
else
    log_warn "Failed to snapshot results for archive (non-fatal)"
fi
rm -rf "$_archive_snap" "$_archive_err"

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
