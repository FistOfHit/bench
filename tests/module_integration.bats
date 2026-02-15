#!/usr/bin/env bats
# Integration tests for HPC bench module scripts.
# Validates syntax, manifest consistency, early-exit guards, and report generation.
# Run: bats tests/module_integration.bats

# Portable fail helper (not built-in in all bats versions).
_fail() { echo "$1" >&2; return 1; }

# ── Test setup ──
setup() {
    export HPC_BENCH_ROOT="${BATS_TEST_DIRNAME}/.."
    export HPC_RESULTS_DIR="$(mktemp -d)"
    export HPC_LOG_DIR="${HPC_RESULTS_DIR}/logs"
    export HPC_WORK_DIR="$(mktemp -d)"
    mkdir -p "$HPC_LOG_DIR"
    SCRIPTS_DIR="${HPC_BENCH_ROOT}/scripts"
    MANIFEST="${HPC_BENCH_ROOT}/specs/modules.json"
}

teardown() {
    rm -rf "$HPC_RESULTS_DIR" "$HPC_WORK_DIR"
}

# ═══════════════════════════════════════════
# Syntax validation (bash -n)
# ═══════════════════════════════════════════

@test "syntax: all scripts in scripts/ pass bash -n" {
    local failed=""
    for f in "${SCRIPTS_DIR}"/*.sh; do
        if ! bash -n "$f" 2>/dev/null; then
            failed="${failed} $(basename "$f")"
        fi
    done
    [ -z "$failed" ] || _fail "Syntax errors in:${failed}"
}

@test "syntax: all libraries in lib/ pass bash -n" {
    local failed=""
    for f in "${HPC_BENCH_ROOT}/lib"/*.sh; do
        if ! bash -n "$f" 2>/dev/null; then
            failed="${failed} $(basename "$f")"
        fi
    done
    [ -z "$failed" ] || _fail "Syntax errors in:${failed}"
}

# ═══════════════════════════════════════════
# Manifest consistency
# ═══════════════════════════════════════════

@test "manifest: modules.json is valid JSON" {
    jq . "$MANIFEST" >/dev/null 2>&1
}

@test "manifest: every listed script file exists" {
    local missing=""
    while IFS= read -r script; do
        if [ ! -f "${SCRIPTS_DIR}/${script}" ]; then
            missing="${missing} ${script}"
        fi
    done < <(jq -r '.modules[].script' "$MANIFEST")
    [ -z "$missing" ] || _fail "Scripts listed in manifest but missing on disk:${missing}"
}

@test "manifest: every module has required fields (name, script, phase, order)" {
    local bad=""
    local count
    count=$(jq '.modules | length' "$MANIFEST")
    for ((i=0; i<count; i++)); do
        local name script phase order
        name=$(jq -r ".modules[$i].name" "$MANIFEST")
        script=$(jq -r ".modules[$i].script" "$MANIFEST")
        phase=$(jq -r ".modules[$i].phase" "$MANIFEST")
        order=$(jq -r ".modules[$i].order" "$MANIFEST")
        if [ "$name" = "null" ] || [ "$script" = "null" ] || [ "$phase" = "null" ] || [ "$order" = "null" ]; then
            bad="${bad} index=$i(${name:-?})"
        fi
    done
    [ -z "$bad" ] || _fail "Modules missing required fields:${bad}"
}

@test "manifest: no duplicate module names" {
    local dupes
    dupes=$(jq -r '.modules[].name' "$MANIFEST" | sort | uniq -d)
    [ -z "$dupes" ] || _fail "Duplicate module names: $dupes"
}

@test "manifest: no duplicate order values within the same phase" {
    local dupes
    dupes=$(jq -r '.modules[] | "\(.phase):\(.order)"' "$MANIFEST" | sort | uniq -d)
    [ -z "$dupes" ] || _fail "Duplicate phase:order pairs: $dupes"
}

@test "manifest: every script with SCRIPT_NAME in scripts/ is in the manifest" {
    local unlisted=""
    local manifest_scripts
    manifest_scripts=$(jq -r '.modules[].script' "$MANIFEST")
    for f in "${SCRIPTS_DIR}"/*.sh; do
        local base
        base=$(basename "$f")
        # Skip non-module scripts (orchestrator and CI utility)
        [ "$base" = "ci-static-checks.sh" ] && continue
        [ "$base" = "run-all.sh" ] && continue
        if grep -q '^SCRIPT_NAME=' "$f" 2>/dev/null; then
            if ! echo "$manifest_scripts" | grep -qx "$base"; then
                unlisted="${unlisted} ${base}"
            fi
        fi
    done
    [ -z "$unlisted" ] || _fail "Scripts with SCRIPT_NAME not in manifest:${unlisted}"
}

# ═══════════════════════════════════════════
# Module source-gate tests
# Each module should either:
#   - Source common.sh successfully, then skip/exit cleanly when prerequisites are missing
#   - NOT crash with a bash error (set -e / unbound variable / syntax)
# We run each in an isolated subshell with restricted PATH to simulate missing tools.
# ═══════════════════════════════════════════

# Helper: run a module script in isolation expecting a clean skip.
# Returns 0 if the module exited 0 (skip) or wrote a JSON file.
# Returns 1 if the module crashed (non-zero exit without writing JSON).
_run_module_isolated() {
    local script="$1"
    local name
    name=$(basename "$script" .sh)
    local tmp_results tmp_logs tmp_work
    tmp_results=$(mktemp -d)
    tmp_logs="${tmp_results}/logs"
    tmp_work=$(mktemp -d)
    mkdir -p "$tmp_logs"

    # Run in a subshell with our controlled environment.
    # We don't restrict PATH — the module should handle missing tools via has_cmd/require_gpu/skip_module.
    local rc=0
    bash -c "
        export HPC_BENCH_ROOT='${HPC_BENCH_ROOT}'
        export HPC_RESULTS_DIR='${tmp_results}'
        export HPC_LOG_DIR='${tmp_logs}'
        export HPC_WORK_DIR='${tmp_work}'
        export HPC_QUICK=1
        bash '${script}' 2>/dev/null
    " >/dev/null 2>&1 || rc=$?

    # Check: did it produce a JSON result file?
    local json_file="${tmp_results}/${name}.json"
    local has_json=false
    [ -f "$json_file" ] && has_json=true

    rm -rf "$tmp_results" "$tmp_work"

    # A module that exits 0 is fine (clean skip or success).
    # A module that exits non-zero but wrote JSON is also fine (error record).
    # A module that exits non-zero WITHOUT JSON crashed — that's a problem.
    if [ "$rc" -eq 0 ]; then
        return 0
    elif [ "$has_json" = true ]; then
        return 0
    else
        return 1
    fi
}

# Helper: build a minimal PATH that includes common tools but excludes specific ones.
# Usage: _path_hiding "fio" "dcgmi"  → prints a PATH with a curated bin dir
# Creates a temp dir with symlinks to essential tools (jq, awk, grep, etc.)
# but deliberately omits the listed commands, then sets PATH to only that dir.
_path_hiding() {
    local hide_dir
    hide_dir=$(mktemp -d)
    local -a blocked=("$@")

    # Essential tools that common.sh and modules need to function
    local -a essentials=(
        bash sh jq awk grep sed cat tr head tail wc cut sort uniq tee
        date hostname id mkdir rm cp mv mktemp timeout kill sleep touch
        printf dirname basename uname pgrep ldconfig find bc lsblk df
        python3 nvidia-smi git make gcc g++ curl wget tar gzip
    )

    for tool in "${essentials[@]}"; do
        # Skip if this tool is one we're blocking
        local is_blocked=false
        for b in "${blocked[@]}"; do
            [ "$tool" = "$b" ] && is_blocked=true && break
        done
        $is_blocked && continue

        # Symlink if it exists on the system
        local real_path
        real_path=$(command -v "$tool" 2>/dev/null) || continue
        ln -sf "$real_path" "${hide_dir}/${tool}" 2>/dev/null || true
    done

    echo "${hide_dir}"
}

# Test modules that should skip cleanly when their prerequisite tools are missing.

@test "source-gate: storage-bench skips when fio is missing" {
    local safe_bin
    safe_bin=$(_path_hiding fio)
    run bash -c "
        export HPC_BENCH_ROOT='${HPC_BENCH_ROOT}'
        export HPC_RESULTS_DIR='${HPC_RESULTS_DIR}'
        export HPC_LOG_DIR='${HPC_LOG_DIR}'
        export HPC_WORK_DIR='${HPC_WORK_DIR}'
        export PATH='${safe_bin}'
        bash '${SCRIPTS_DIR}/storage-bench.sh'
    " 2>/dev/null
    # Should exit 0 (clean skip) and write a JSON file
    [ "$status" -eq 0 ]
    [ -f "${HPC_RESULTS_DIR}/storage-bench.json" ]
    local st
    st=$(jq -r '.status' "${HPC_RESULTS_DIR}/storage-bench.json" 2>/dev/null)
    [ "$st" = "skipped" ]
}

@test "source-gate: dcgm-diag skips when dcgmi is missing" {
    local safe_bin
    safe_bin=$(_path_hiding dcgmi)
    run bash -c "
        export HPC_BENCH_ROOT='${HPC_BENCH_ROOT}'
        export HPC_RESULTS_DIR='${HPC_RESULTS_DIR}'
        export HPC_LOG_DIR='${HPC_LOG_DIR}'
        export HPC_WORK_DIR='${HPC_WORK_DIR}'
        export PATH='${safe_bin}'
        bash '${SCRIPTS_DIR}/dcgm-diag.sh'
    " 2>/dev/null
    [ "$status" -eq 0 ]
    [ -f "${HPC_RESULTS_DIR}/dcgm-diag.json" ]
    local st
    st=$(jq -r '.status' "${HPC_RESULTS_DIR}/dcgm-diag.json" 2>/dev/null)
    [ "$st" = "skipped" ]
}

@test "source-gate: ib-tests skips when no IB hardware present" {
    run bash -c "
        export HPC_BENCH_ROOT='${HPC_BENCH_ROOT}'
        export HPC_RESULTS_DIR='${HPC_RESULTS_DIR}'
        export HPC_LOG_DIR='${HPC_LOG_DIR}'
        export HPC_WORK_DIR='${HPC_WORK_DIR}'
        export HPC_QUICK=1
        bash '${SCRIPTS_DIR}/ib-tests.sh'
    " 2>/dev/null
    # ib-tests should either skip (exit 0) or produce a JSON file
    [ -f "${HPC_RESULTS_DIR}/ib-tests.json" ] || [ "$status" -eq 0 ]
}

# ═══════════════════════════════════════════
# Report generation (synthetic data)
# ═══════════════════════════════════════════

@test "report: generates valid markdown from synthetic results" {
    # Plant minimal synthetic module results
    echo '{"status":"ok","hostname":"test-host","gpu_count":2,"gpu_model":"Test GPU"}' \
        > "${HPC_RESULTS_DIR}/bootstrap.json"
    echo '{"status":"ok","has_gpu_driver":true,"gpu_count":2}' \
        > "${HPC_RESULTS_DIR}/runtime-sanity.json"
    echo '{"status":"ok","cpu":{"model":"Test CPU","cores":16},"ram":{"total_gb":64},"os":{"hostname":"test-host","os":"Ubuntu 24.04","kernel":"6.8.0"}}' \
        > "${HPC_RESULTS_DIR}/inventory.json"
    echo '{"status":"ok","gpu_model":"Test GPU","gpu_count":2,"gpus":[{"name":"Test GPU","memory_total_mb":81920,"compute_capability":"8.0"}]}' \
        > "${HPC_RESULTS_DIR}/gpu-inventory.json"
    echo '{"status":"skipped","skip_reason":"no hardware"}' \
        > "${HPC_RESULTS_DIR}/nccl-tests.json"
    echo '{"status":"ok","errors_detected":0,"duration_seconds":10}' \
        > "${HPC_RESULTS_DIR}/gpu-burn.json"
    echo '{"status":"ok","sequential_read_1M":{"read_bw_mbps":1000}}' \
        > "${HPC_RESULTS_DIR}/storage-bench.json"

    # Run report.sh
    run bash -c "
        export HPC_BENCH_ROOT='${HPC_BENCH_ROOT}'
        export HPC_RESULTS_DIR='${HPC_RESULTS_DIR}'
        export HPC_LOG_DIR='${HPC_LOG_DIR}'
        export HPC_WORK_DIR='${HPC_WORK_DIR}'
        export SCRIPT_NAME='report'
        bash '${SCRIPTS_DIR}/report.sh'
    " 2>/dev/null
    [ "$status" -eq 0 ]

    # Verify report file was created and has expected content
    [ -f "${HPC_RESULTS_DIR}/report.md" ]
    grep -q "HPC Benchmarking Report" "${HPC_RESULTS_DIR}/report.md"
    grep -q "Health Scorecard" "${HPC_RESULTS_DIR}/report.md"
    grep -q "test-host" "${HPC_RESULTS_DIR}/report.md"

    # Verify report.json was created with valid scoring
    [ -f "${HPC_RESULTS_DIR}/report.json" ]
    local overall
    overall=$(jq -r '.overall' "${HPC_RESULTS_DIR}/report.json" 2>/dev/null)
    [[ "$overall" =~ ^(PASS|WARN|FAIL)$ ]]
}

@test "report: handles all-skipped scenario without crashing" {
    # Only skipped modules — report should still generate
    for mod in bootstrap runtime-sanity inventory gpu-inventory; do
        echo '{"status":"skipped","skip_reason":"test"}' \
            > "${HPC_RESULTS_DIR}/${mod}.json"
    done

    run bash -c "
        export HPC_BENCH_ROOT='${HPC_BENCH_ROOT}'
        export HPC_RESULTS_DIR='${HPC_RESULTS_DIR}'
        export HPC_LOG_DIR='${HPC_LOG_DIR}'
        export HPC_WORK_DIR='${HPC_WORK_DIR}'
        export SCRIPT_NAME='report'
        bash '${SCRIPTS_DIR}/report.sh'
    " 2>/dev/null
    [ "$status" -eq 0 ]
    [ -f "${HPC_RESULTS_DIR}/report.md" ]
}

@test "report: handles failure scenario and shows issues" {
    echo '{"status":"ok"}' > "${HPC_RESULTS_DIR}/bootstrap.json"
    echo '{"status":"error","error":"DCGM failed"}' > "${HPC_RESULTS_DIR}/dcgm-diag.json"
    echo '{"status":"ok","thermal_status":"fail"}' > "${HPC_RESULTS_DIR}/thermal-power.json"

    run bash -c "
        export HPC_BENCH_ROOT='${HPC_BENCH_ROOT}'
        export HPC_RESULTS_DIR='${HPC_RESULTS_DIR}'
        export HPC_LOG_DIR='${HPC_LOG_DIR}'
        export HPC_WORK_DIR='${HPC_WORK_DIR}'
        export SCRIPT_NAME='report'
        bash '${SCRIPTS_DIR}/report.sh'
    " 2>/dev/null
    [ "$status" -eq 0 ]
    [ -f "${HPC_RESULTS_DIR}/report.md" ]

    # Verify the report shows failures
    grep -q "FAIL" "${HPC_RESULTS_DIR}/report.md"
    grep -q "Issues Found" "${HPC_RESULTS_DIR}/report.md"

    # Verify overall result is FAIL
    local overall
    overall=$(jq -r '.overall' "${HPC_RESULTS_DIR}/report.json" 2>/dev/null)
    [ "$overall" = "FAIL" ]
}

# ═══════════════════════════════════════════
# Common library sanity
# ═══════════════════════════════════════════

@test "common.sh: exports expected path variables" {
    run bash -c "
        export HPC_BENCH_ROOT='${HPC_BENCH_ROOT}'
        export HPC_RESULTS_DIR='${HPC_RESULTS_DIR}'
        export HPC_LOG_DIR='${HPC_LOG_DIR}'
        export HPC_WORK_DIR='${HPC_WORK_DIR}'
        export SCRIPT_NAME='test'
        source '${HPC_BENCH_ROOT}/lib/common.sh'
        [ -n \"\$HPC_BENCH_ROOT\" ] && [ -n \"\$HPC_RESULTS_DIR\" ] && [ -n \"\$HPC_LOG_DIR\" ] && [ -n \"\$HPC_WORK_DIR\" ]
    "
    [ "$status" -eq 0 ]
}

@test "common.sh: creates required directories on source" {
    local tmp_results tmp_logs tmp_work
    tmp_results=$(mktemp -d)
    tmp_logs="${tmp_results}/logs"
    tmp_work=$(mktemp -d)/sub

    run bash -c "
        export HPC_BENCH_ROOT='${HPC_BENCH_ROOT}'
        export HPC_RESULTS_DIR='${tmp_results}'
        export HPC_LOG_DIR='${tmp_logs}'
        export HPC_WORK_DIR='${tmp_work}'
        export SCRIPT_NAME='test'
        source '${HPC_BENCH_ROOT}/lib/common.sh'
        [ -d '${tmp_results}' ] && [ -d '${tmp_logs}' ] && [ -d '${tmp_work}' ]
    "
    rm -rf "$tmp_results" "$(dirname "$tmp_work")"
    [ "$status" -eq 0 ]
}

@test "common.sh: rejects protected HPC_RESULTS_DIR paths" {
    run bash -c "
        export HPC_BENCH_ROOT='${HPC_BENCH_ROOT}'
        export HPC_RESULTS_DIR='/etc'
        export SCRIPT_NAME='test'
        source '${HPC_BENCH_ROOT}/lib/common.sh'
    "
    [ "$status" -ne 0 ]
}

@test "defaults.sh: is valid bash and all defaults are set" {
    run bash -c "
        source '${HPC_BENCH_ROOT}/conf/defaults.sh'
        [ -n \"\$MAX_MODULE_TIME_QUICK\" ] && [ -n \"\$GPU_BURN_DURATION_QUICK\" ] && [ -n \"\$GPU_THERMAL_WARN_C\" ]
    "
    [ "$status" -eq 0 ]
}
