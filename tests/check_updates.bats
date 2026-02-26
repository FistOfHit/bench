#!/usr/bin/env bats
# Tests for specs/dependencies.json manifest and scripts/check-updates.sh.
# Run: bats tests/check_updates.bats

load helpers

# ── Test setup ──
setup() {
    setup_test_env "test-check-updates"
    SCRIPT="${HPC_BENCH_ROOT}/scripts/check-updates.sh"
    MANIFEST="${HPC_BENCH_ROOT}/specs/dependencies.json"
    DEFAULTS="${HPC_BENCH_ROOT}/conf/defaults.sh"
    PRECOMMIT_CFG="${HPC_BENCH_ROOT}/.pre-commit-config.yaml"
}

teardown() {
    teardown_test_env
}

# ═══════════════════════════════════════════
# Manifest schema validation
# ═══════════════════════════════════════════

@test "deps-manifest: dependencies.json is valid JSON" {
    jq . "$MANIFEST" >/dev/null 2>&1
}

@test "deps-manifest: has schema_version field" {
    local ver
    ver=$(jq -r '.schema_version' "$MANIFEST")
    [ "$ver" != "null" ] && [ -n "$ver" ]
}

@test "deps-manifest: every dependency has required fields" {
    local bad=""
    local count
    count=$(jq '.dependencies | length' "$MANIFEST")
    for ((i = 0; i < count; i++)); do
        local name category version method
        name=$(jq -r ".dependencies[$i].name" "$MANIFEST")
        category=$(jq -r ".dependencies[$i].category" "$MANIFEST")
        version=$(jq -r ".dependencies[$i].current_version" "$MANIFEST")
        method=$(jq -r ".dependencies[$i].source.check_method" "$MANIFEST")
        if [ "$name" = "null" ] || [ "$category" = "null" ] || \
            [ "$version" = "null" ] || [ "$method" = "null" ]; then
            bad="${bad} index=${i}(${name:-?})"
        fi
    done
    [ -z "$bad" ] || _fail "Dependencies missing required fields:${bad}"
}

@test "deps-manifest: no duplicate dependency names" {
    local dupes
    dupes=$(jq -r '.dependencies[].name' "$MANIFEST" | sort | uniq -d)
    [ -z "$dupes" ] || _fail "Duplicate dependency names: $dupes"
}

@test "deps-manifest: all check_method values are recognized" {
    local known="nvcr_registry dockerhub github_releases github_tags github_commits nvidia_apt_repo"
    local bad=""
    local count
    count=$(jq '.dependencies | length' "$MANIFEST")
    for ((i = 0; i < count; i++)); do
        local method
        method=$(jq -r ".dependencies[$i].source.check_method" "$MANIFEST")
        local found=false
        for k in $known; do
            [ "$method" = "$k" ] && found=true && break
        done
        if [ "$found" = false ]; then
            bad="${bad} ${method}"
        fi
    done
    [ -z "$bad" ] || _fail "Unknown check methods:${bad}"
}

@test "deps-manifest: container_image deps have update_targets" {
    local bad=""
    local count
    count=$(jq '.dependencies | length' "$MANIFEST")
    for ((i = 0; i < count; i++)); do
        local cat targets
        cat=$(jq -r ".dependencies[$i].category" "$MANIFEST")
        if [ "$cat" = "container_image" ]; then
            targets=$(jq ".dependencies[$i].update_targets | length" "$MANIFEST")
            if [ "$targets" -eq 0 ]; then
                local name
                name=$(jq -r ".dependencies[$i].name" "$MANIFEST")
                bad="${bad} ${name}"
            fi
        fi
    done
    [ -z "$bad" ] || _fail "Container image deps without update_targets:${bad}"
}

@test "deps-manifest: upstream_source deps have empty update_targets" {
    local bad=""
    local count
    count=$(jq '.dependencies | length' "$MANIFEST")
    for ((i = 0; i < count; i++)); do
        local cat targets
        cat=$(jq -r ".dependencies[$i].category" "$MANIFEST")
        if [ "$cat" = "upstream_source" ]; then
            targets=$(jq ".dependencies[$i].update_targets | length" "$MANIFEST")
            if [ "$targets" -ne 0 ]; then
                local name
                name=$(jq -r ".dependencies[$i].name" "$MANIFEST")
                bad="${bad} ${name}"
            fi
        fi
    done
    [ -z "$bad" ] || _fail "Upstream source deps should have empty update_targets:${bad}"
}

# ═══════════════════════════════════════════
# Cross-check: manifest vs actual files
# ═══════════════════════════════════════════

@test "deps-crosscheck: HPL_IMAGE version in defaults.sh matches manifest" {
    local manifest_ver
    manifest_ver=$(jq -r '.dependencies[] | select(.name=="hpc-benchmarks") | .current_version' "$MANIFEST")
    [ -n "$manifest_ver" ]
    grep -q "hpc-benchmarks:${manifest_ver}" "$DEFAULTS"
}

@test "deps-crosscheck: HPL_IMAGE_ALT version in defaults.sh matches manifest" {
    local manifest_alt
    manifest_alt=$(jq -r '.dependencies[] | select(.name=="hpc-benchmarks") | .current_alt_version' "$MANIFEST")
    [ -n "$manifest_alt" ]
    grep -q "hpc-benchmarks:${manifest_alt}" "$DEFAULTS"
}

@test "deps-crosscheck: pre-commit-hooks rev matches manifest" {
    local manifest_ver
    manifest_ver=$(jq -r '.dependencies[] | select(.name=="pre-commit-hooks") | .current_version' "$MANIFEST")
    [ -n "$manifest_ver" ]
    grep -A1 'pre-commit/pre-commit-hooks' "$PRECOMMIT_CFG" | grep -q "rev: ${manifest_ver}"
}

@test "deps-crosscheck: pre-commit-shfmt rev matches manifest" {
    local manifest_ver
    manifest_ver=$(jq -r '.dependencies[] | select(.name=="pre-commit-shfmt") | .current_version' "$MANIFEST")
    [ -n "$manifest_ver" ]
    grep -A1 'scop/pre-commit-shfmt' "$PRECOMMIT_CFG" | grep -q "rev: ${manifest_ver}"
}

@test "deps-crosscheck: shellcheck-py rev matches manifest" {
    local manifest_ver
    manifest_ver=$(jq -r '.dependencies[] | select(.name=="shellcheck-py") | .current_version' "$MANIFEST")
    [ -n "$manifest_ver" ]
    grep -A1 'shellcheck-py/shellcheck-py' "$PRECOMMIT_CFG" | grep -q "rev: ${manifest_ver}"
}

# ═══════════════════════════════════════════
# Script validation
# ═══════════════════════════════════════════

@test "check-updates: script passes bash -n" {
    bash -n "$SCRIPT"
}

@test "check-updates: --help exits 0 and prints usage" {
    run bash "$SCRIPT" --help
    [ "$status" -eq 0 ]
    [[ "$output" == *"Usage:"* ]]
    [[ "$output" == *"--json"* ]]
    [[ "$output" == *"--apply"* ]]
}

@test "check-updates: unknown flag exits non-zero" {
    run bash "$SCRIPT" --bogus-flag
    [ "$status" -ne 0 ]
}

@test "check-updates: --category with missing value exits non-zero" {
    run bash "$SCRIPT" --category
    [ "$status" -ne 0 ]
}

# ═══════════════════════════════════════════
# NVIDIA package manifest validation
# ═══════════════════════════════════════════

@test "deps-manifest: nvidia_package deps with update_targets have target files" {
    local bad=""
    local count
    count=$(jq '.dependencies | length' "$MANIFEST")
    for ((i = 0; i < count; i++)); do
        local cat targets_len
        cat=$(jq -r ".dependencies[$i].category" "$MANIFEST")
        if [ "$cat" = "nvidia_package" ]; then
            targets_len=$(jq ".dependencies[$i].update_targets | length" "$MANIFEST")
            for ((j = 0; j < targets_len; j++)); do
                local tgt
                tgt=$(jq -r ".dependencies[$i].update_targets[$j]" "$MANIFEST")
                if [ ! -f "${HPC_BENCH_ROOT}/${tgt}" ]; then
                    local name
                    name=$(jq -r ".dependencies[$i].name" "$MANIFEST")
                    bad="${bad} ${name}:${tgt}"
                fi
            done
        fi
    done
    [ -z "$bad" ] || _fail "Update target files not found:${bad}"
}

@test "deps-manifest: nvidia_apt_repo deps have required source fields" {
    local bad=""
    local count
    count=$(jq '.dependencies | length' "$MANIFEST")
    for ((i = 0; i < count; i++)); do
        local method
        method=$(jq -r ".dependencies[$i].source.check_method" "$MANIFEST")
        if [ "$method" = "nvidia_apt_repo" ]; then
            local pattern extract
            pattern=$(jq -r ".dependencies[$i].source.package_pattern" "$MANIFEST")
            extract=$(jq -r ".dependencies[$i].source.version_extract" "$MANIFEST")
            if [ "$pattern" = "null" ] || [ "$extract" = "null" ]; then
                local name
                name=$(jq -r ".dependencies[$i].name" "$MANIFEST")
                bad="${bad} ${name}"
            fi
        fi
    done
    [ -z "$bad" ] || _fail "nvidia_apt_repo deps missing package_pattern/version_extract:${bad}"
}

@test "deps-crosscheck: nvidia-driver fallback in bootstrap.sh matches manifest" {
    local manifest_ver
    manifest_ver=$(jq -r '.dependencies[] | select(.name=="nvidia-driver") | .current_version' "$MANIFEST")
    [ -n "$manifest_ver" ]
    grep -q "nvidia-driver-${manifest_ver}-server" "${HPC_BENCH_ROOT}/scripts/bootstrap.sh"
}

@test "deps-crosscheck: CUDA fallback in bootstrap.sh matches manifest" {
    local manifest_ver
    manifest_ver=$(jq -r '.dependencies[] | select(.name=="cuda-toolkit") | .current_version' "$MANIFEST")
    [ -n "$manifest_ver" ]
    grep -q "_cuda_runtime:-${manifest_ver}" "${HPC_BENCH_ROOT}/scripts/bootstrap.sh"
}

@test "deps-manifest: total dependency count is 14" {
    local count
    count=$(jq '.dependencies | length' "$MANIFEST")
    [ "$count" -eq 14 ] || _fail "Expected 14 dependencies, got ${count}"
}

# ═══════════════════════════════════════════
# Phase 4: Constraints, dry-run, history
# ═══════════════════════════════════════════

@test "deps-manifest: constraints reference valid dependency names" {
    local all_names
    all_names=$(jq -r '[.dependencies[].name] | join(" ")' "$MANIFEST")
    local bad=""
    local count
    count=$(jq '.dependencies | length' "$MANIFEST")
    for ((i = 0; i < count; i++)); do
        local n_constraints
        n_constraints=$(jq ".dependencies[$i].constraints // [] | length" "$MANIFEST")
        for ((ci = 0; ci < n_constraints; ci++)); do
            local req
            req=$(jq -r ".dependencies[$i].constraints[$ci].requires" "$MANIFEST")
            local found=false
            for n in $all_names; do
                [ "$n" = "$req" ] && found=true && break
            done
            if [ "$found" = false ]; then
                local dname
                dname=$(jq -r ".dependencies[$i].name" "$MANIFEST")
                bad="${bad} ${dname}→${req}"
            fi
        done
    done
    [ -z "$bad" ] || _fail "Constraints reference unknown deps:${bad}"
}

@test "deps-manifest: constraints have required fields" {
    local bad=""
    local count
    count=$(jq '.dependencies | length' "$MANIFEST")
    for ((i = 0; i < count; i++)); do
        local n_constraints
        n_constraints=$(jq ".dependencies[$i].constraints // [] | length" "$MANIFEST")
        for ((ci = 0; ci < n_constraints; ci++)); do
            local req min_ver desc
            req=$(jq -r ".dependencies[$i].constraints[$ci].requires" "$MANIFEST")
            min_ver=$(jq -r ".dependencies[$i].constraints[$ci].minimum_version" "$MANIFEST")
            desc=$(jq -r ".dependencies[$i].constraints[$ci].description" "$MANIFEST")
            if [ "$req" = "null" ] || [ "$min_ver" = "null" ] || [ "$desc" = "null" ]; then
                local dname
                dname=$(jq -r ".dependencies[$i].name" "$MANIFEST")
                bad="${bad} ${dname}[$ci]"
            fi
        done
    done
    [ -z "$bad" ] || _fail "Constraints missing required fields:${bad}"
}

@test "check-updates: --dry-run without --apply exits non-zero" {
    run bash "$SCRIPT" --dry-run
    [ "$status" -ne 0 ]
    [[ "$output" == *"--dry-run requires --apply"* ]]
}

@test "check-updates: --help shows dry-run option" {
    run bash "$SCRIPT" --help
    [ "$status" -eq 0 ]
    [[ "$output" == *"--dry-run"* ]]
}

@test "update-history: initial file is valid JSON array" {
    local history_file="${HPC_BENCH_ROOT}/specs/update-history.json"
    [ -f "$history_file" ]
    local val
    val=$(jq 'type' "$history_file")
    [ "$val" = '"array"' ]
}

@test "deps-manifest: cuda-toolkit has constraints for nvidia-driver" {
    local has_constraint
    has_constraint=$(jq '.dependencies[] | select(.name=="cuda-toolkit") | .constraints[]? | select(.requires=="nvidia-driver") | .requires' "$MANIFEST")
    [ -n "$has_constraint" ] || _fail "cuda-toolkit should have a constraint on nvidia-driver"
}

@test "deps-manifest: cuda-toolkit constraint has CUDA 13 driver minimum" {
    local min_13
    min_13=$(jq -r '.dependencies[] | select(.name=="cuda-toolkit") | .constraints[] | select(.requires=="nvidia-driver") | .minimum_version["13"] // empty' "$MANIFEST")
    [ -n "$min_13" ] || _fail "cuda-toolkit should have minimum driver for CUDA 13"
    [ "$min_13" -ge 500 ] || _fail "CUDA 13 minimum driver ${min_13} seems too low"
}
