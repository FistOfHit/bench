#!/usr/bin/env bash
# check-updates.sh — Check tracked dependencies for available updates
# Usage: bash scripts/check-updates.sh [--json] [--apply] [--dry-run] [--category CAT] [--help]
#
# Reads specs/dependencies.json, queries upstream sources, and reports
# which dependencies have newer versions available.
#
# Not a benchmark module — does not source lib/common.sh or emit module JSON.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${ROOT_DIR}/specs/dependencies.json"
VERSION_FILE="${ROOT_DIR}/VERSION"
HISTORY_FILE="${ROOT_DIR}/specs/update-history.json"

# ── Defaults ──
OUTPUT_JSON=false
APPLY=false
DRY_RUN=false
FILTER_CATEGORY=""
CURL_TIMEOUT=10

# ── Colors (disabled when not a terminal or when NO_COLOR is set) ──
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    C_GREEN='\033[0;32m'
    C_YELLOW='\033[0;33m'
    C_RED='\033[0;31m'
    C_CYAN='\033[0;36m'
    C_RESET='\033[0m'
else
    C_GREEN="" C_YELLOW="" C_RED="" C_CYAN="" C_RESET=""
fi

# ── Logging ──
log() { echo -e "[check-updates] $*"; }
log_ok() { echo -e "[check-updates] ${C_GREEN}✓${C_RESET} $*"; }
log_update() { echo -e "[check-updates] ${C_YELLOW}↑${C_RESET} $*"; }
log_fail() { echo -e "[check-updates] ${C_RED}✗${C_RESET} $*"; }
log_info() { echo -e "[check-updates] ${C_CYAN}ℹ${C_RESET} $*"; }

# ── Usage ──
usage() {
    cat <<'EOF'
Usage: bash scripts/check-updates.sh [OPTIONS]

Check tracked dependencies for available upstream updates.

Options:
  --json            Output machine-readable JSON report to stdout
  --apply           Apply available updates to source files
  --dry-run         Preview what --apply would change (use with --apply)
  --category CAT    Check only one category:
                      container_image, upstream_source, pre_commit_hook,
                      nvidia_package
  --help            Show this help

Environment:
  GITHUB_TOKEN      GitHub API token for higher rate limits (5000/hr vs 60/hr)
  NGC_API_KEY       NGC API key (currently unused; reserved for future auth)
  NVIDIA_REPO_OS    Ubuntu version for NVIDIA apt repo (default: auto-detect or 2204)
  NO_COLOR          Disable colored output

Examples:
  bash scripts/check-updates.sh                     # Human-readable report
  bash scripts/check-updates.sh --json              # JSON report
  bash scripts/check-updates.sh --apply             # Check and apply updates
  bash scripts/check-updates.sh --category pre_commit_hook  # Check one category
  bash scripts/check-updates.sh --apply --dry-run           # Preview changes
EOF
}

# ── Argument parsing ──
parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --json) OUTPUT_JSON=true ;;
            --apply) APPLY=true ;;
            --dry-run) DRY_RUN=true ;;
            --category)
                shift
                FILTER_CATEGORY="${1:-}"
                if [ -z "$FILTER_CATEGORY" ]; then
                    echo "Error: --category requires a value" >&2
                    exit 1
                fi
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1" >&2
                usage >&2
                exit 1
                ;;
        esac
        shift
    done
}

# ── Prerequisites ──
require_tools() {
    local missing=""
    for tool in jq curl; do
        if ! command -v "$tool" &> /dev/null; then
            missing="${missing} ${tool}"
        fi
    done
    if [ -n "$missing" ]; then
        echo "Error: required tools not found:${missing}" >&2
        exit 1
    fi
}

# ── GitHub API helper ──
github_api() {
    local endpoint="$1"
    local -a curl_args=(
        -s --connect-timeout "$CURL_TIMEOUT" --max-time 30
        -H "Accept: application/vnd.github+json"
    )
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        curl_args+=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
    fi
    curl "${curl_args[@]}" "https://api.github.com${endpoint}" 2>/dev/null
}

# ── Generic curl helper ──
api_get() {
    local url="$1"
    shift
    curl -s --connect-timeout "$CURL_TIMEOUT" --max-time 30 "$@" "$url" 2>/dev/null
}

# ═══════════════════════════════════════════════════════════════════
# Check methods — each returns JSON: {"latest":"...", "extra":"..."}
# On failure returns: {"error":"..."}
# ═══════════════════════════════════════════════════════════════════

check_nvcr_registry() {
    local image="$1"
    local tag_filter="${2:-}"

    local token
    token=$(api_get "https://nvcr.io/proxy_auth?scope=repository:${image}:pull" \
        | jq -r '.token // empty')
    if [ -z "$token" ]; then
        echo '{"error":"failed to obtain registry token"}'
        return 0
    fi

    local tags_json
    tags_json=$(api_get "https://nvcr.io/v2/${image}/tags/list" \
        -H "Authorization: Bearer ${token}")
    if [ -z "$tags_json" ] || ! echo "$tags_json" | jq -e '.tags' &> /dev/null; then
        echo '{"error":"failed to fetch tags from registry"}'
        return 0
    fi

    local latest
    if [ -n "$tag_filter" ]; then
        latest=$(echo "$tags_json" | jq -r --arg f "$tag_filter" \
            '[.tags[] | select(test($f))] | sort | last // empty')
    else
        latest=$(echo "$tags_json" | jq -r '.tags | sort | last // empty')
    fi

    if [ -z "$latest" ]; then
        echo '{"error":"no tags matched filter"}'
        return 0
    fi
    jq -n --arg l "$latest" '{"latest":$l}'
}

check_dockerhub() {
    local repo="$1"
    local tag_filter="${2:-}"

    local resp
    resp=$(api_get "https://hub.docker.com/v2/repositories/${repo}/tags/?page_size=100&ordering=-last_updated")
    if [ -z "$resp" ] || ! echo "$resp" | jq -e '.results' &> /dev/null; then
        echo '{"error":"failed to fetch tags from Docker Hub"}'
        return 0
    fi

    local latest
    if [ -n "$tag_filter" ]; then
        latest=$(echo "$resp" | jq -r --arg f "$tag_filter" \
            '[.results[].name | select(test($f))] | sort | last // empty')
    else
        latest=$(echo "$resp" | jq -r '.results[0].name // empty')
    fi

    if [ -z "$latest" ]; then
        echo '{"error":"no tags matched filter"}'
        return 0
    fi
    jq -n --arg l "$latest" '{"latest":$l}'
}

check_github_releases() {
    local repo="$1"

    local resp
    resp=$(github_api "/repos/${repo}/releases/latest")

    local tag
    tag=$(echo "$resp" | jq -r '.tag_name // empty')
    if [ -z "$tag" ]; then
        local msg
        msg=$(echo "$resp" | jq -r '.message // "no releases found"')
        echo "{\"error\":$(echo "$msg" | jq -Rs .)}"
        return 0
    fi

    local date
    date=$(echo "$resp" | jq -r '.published_at // empty')
    jq -n --arg l "$tag" --arg d "$date" '{"latest":$l,"published":$d}'
}

check_github_tags() {
    local repo="$1"

    local resp
    resp=$(github_api "/repos/${repo}/tags?per_page=1")

    local tag
    tag=$(echo "$resp" | jq -r '.[0].name // empty' 2>/dev/null)
    if [ -z "$tag" ]; then
        echo '{"error":"no tags found"}'
        return 0
    fi
    jq -n --arg l "$tag" '{"latest":$l}'
}

check_github_commits() {
    local repo="$1"

    local resp
    resp=$(github_api "/repos/${repo}/commits?per_page=1")

    local sha date
    sha=$(echo "$resp" | jq -r '.[0].sha // empty' 2>/dev/null)
    if [ -z "$sha" ]; then
        echo '{"error":"failed to fetch commits"}'
        return 0
    fi
    sha="${sha:0:8}"
    date=$(echo "$resp" | jq -r '.[0].commit.committer.date // empty' 2>/dev/null)
    jq -n --arg s "$sha" --arg d "$date" '{"latest":$s,"commit_date":$d}'
}

# ═══════════════════════════════════════════════════════════════════
# NVIDIA apt repo — download once, cache for the run
# ═══════════════════════════════════════════════════════════════════

_NVIDIA_PACKAGES_CACHE=""

_nvidia_repo_os() {
    if [ -n "${NVIDIA_REPO_OS:-}" ]; then
        echo "$NVIDIA_REPO_OS"
        return
    fi
    if [ -f /etc/os-release ]; then
        # shellcheck disable=SC1091
        local ver_id
        ver_id=$(. /etc/os-release && echo "${VERSION_ID:-}")
        if [ -n "$ver_id" ]; then
            echo "$ver_id" | tr -d '.'
            return
        fi
    fi
    echo "2204"
}

_ensure_nvidia_packages_cache() {
    if [ -n "$_NVIDIA_PACKAGES_CACHE" ] && [ -f "$_NVIDIA_PACKAGES_CACHE" ]; then
        return 0
    fi
    local os_ver
    os_ver=$(_nvidia_repo_os)
    local url="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${os_ver}/x86_64/Packages"
    _NVIDIA_PACKAGES_CACHE=$(mktemp /tmp/hpc-bench-nvidia-packages.XXXXXX)
    if ! api_get "$url" -o "$_NVIDIA_PACKAGES_CACHE" --max-time 60; then
        rm -f "$_NVIDIA_PACKAGES_CACHE"
        _NVIDIA_PACKAGES_CACHE=""
        return 1
    fi
    if [ ! -s "$_NVIDIA_PACKAGES_CACHE" ]; then
        rm -f "$_NVIDIA_PACKAGES_CACHE"
        _NVIDIA_PACKAGES_CACHE=""
        return 1
    fi
    return 0
}

_cleanup_nvidia_cache() {
    [ -n "$_NVIDIA_PACKAGES_CACHE" ] && rm -f "$_NVIDIA_PACKAGES_CACHE" || true
}
trap '_cleanup_nvidia_cache' EXIT

check_nvidia_apt_repo() {
    local dep_json="$1"

    if ! _ensure_nvidia_packages_cache; then
        echo '{"error":"failed to download NVIDIA apt repo index"}'
        return 0
    fi

    local pattern version_extract
    pattern=$(echo "$dep_json" | jq -r '.source.package_pattern')
    version_extract=$(echo "$dep_json" | jq -r '.source.version_extract')

    case "$version_extract" in
        highest_match)
            local latest
            latest=$(grep -oP "$pattern" "$_NVIDIA_PACKAGES_CACHE" \
                | grep -oP '\d+$' | sort -nr | head -1)
            if [ -z "$latest" ]; then
                echo '{"error":"no matching packages in repo"}'
                return 0
            fi
            jq -n --arg l "$latest" '{"latest":$l}'
            ;;
        cuda_major_minor)
            local latest_pkg
            latest_pkg=$(grep -oP "$pattern" "$_NVIDIA_PACKAGES_CACHE" \
                | sed 's/Package: cuda-toolkit-//' \
                | sort -t- -k1,1nr -k2,2nr | head -1)
            if [ -z "$latest_pkg" ]; then
                echo '{"error":"no CUDA toolkit packages in repo"}'
                return 0
            fi
            local latest
            latest=$(echo "$latest_pkg" | tr '-' '.')
            jq -n --arg l "$latest" '{"latest":$l}'
            ;;
        package_version)
            local latest
            latest=$(grep -A1 "$pattern" "$_NVIDIA_PACKAGES_CACHE" \
                | grep '^Version:' | sed 's/^Version: //' \
                | sed 's/^[0-9]*://' | sort -Vr | head -1)
            if [ -z "$latest" ]; then
                echo '{"error":"package not found in repo"}'
                return 0
            fi
            jq -n --arg l "$latest" '{"latest":$l}'
            ;;
        nccl_version)
            local raw
            raw=$(grep -A1 "$pattern" "$_NVIDIA_PACKAGES_CACHE" \
                | grep '^Version:' | sed 's/^Version: //' | sort -Vr | head -1)
            if [ -z "$raw" ]; then
                echo '{"error":"libnccl2 not found in repo"}'
                return 0
            fi
            local latest
            latest=$(echo "$raw" | grep -oP '^[0-9]+\.[0-9]+\.[0-9]+')
            jq -n --arg l "$latest" --arg r "$raw" '{"latest":$l,"full_version":$r}'
            ;;
        *)
            echo "{\"error\":\"unknown version_extract: ${version_extract}\"}"
            ;;
    esac
}

# ═══════════════════════════════════════════════════════════════════
# Dispatch: run the right check method for a dependency
# ═══════════════════════════════════════════════════════════════════

run_check() {
    local dep_json="$1"

    local method image repo tag_filter
    method=$(echo "$dep_json" | jq -r '.source.check_method')

    case "$method" in
        nvcr_registry)
            image=$(echo "$dep_json" | jq -r '.source.image')
            tag_filter=$(echo "$dep_json" | jq -r '.source.tag_filter // empty')
            check_nvcr_registry "$image" "$tag_filter"
            ;;
        dockerhub)
            repo=$(echo "$dep_json" | jq -r '.source.repo')
            tag_filter=$(echo "$dep_json" | jq -r '.source.tag_filter // empty')
            check_dockerhub "$repo" "$tag_filter"
            ;;
        github_releases)
            repo=$(echo "$dep_json" | jq -r '.source.repo')
            check_github_releases "$repo"
            ;;
        github_tags)
            repo=$(echo "$dep_json" | jq -r '.source.repo')
            check_github_tags "$repo"
            ;;
        github_commits)
            repo=$(echo "$dep_json" | jq -r '.source.repo')
            check_github_commits "$repo"
            ;;
        nvidia_apt_repo)
            check_nvidia_apt_repo "$dep_json"
            ;;
        *)
            echo "{\"error\":\"unknown check method: ${method}\"}"
            ;;
    esac
}

# ═══════════════════════════════════════════════════════════════════
# Version comparison
# ═══════════════════════════════════════════════════════════════════

has_update() {
    local current="$1" latest="$2" category="$3"

    # upstream_source deps track HEAD — always "informational", never "update"
    if [ "$category" = "upstream_source" ]; then
        echo "false"
        return 0
    fi
    # Simple string comparison: different = update available
    if [ "$current" != "$latest" ] && [ -n "$latest" ]; then
        echo "true"
    else
        echo "false"
    fi
}

# ═══════════════════════════════════════════════════════════════════
# Version utilities
# ═══════════════════════════════════════════════════════════════════

version_gte() {
    local a="$1" b="$2"
    [ "$(printf '%s\n%s' "$a" "$b" | sort -V | head -1)" = "$b" ]
}

version_major() {
    echo "${1%%.*}" | tr -cd '0-9'
}

# ═══════════════════════════════════════════════════════════════════
# Constraint checking
# ═══════════════════════════════════════════════════════════════════

check_constraints() {
    local dep_json="$1" latest_version="$2"

    local constraints
    constraints=$(echo "$dep_json" | jq -c '.constraints // []')
    local n_constraints
    n_constraints=$(echo "$constraints" | jq 'length')

    if [ "$n_constraints" -eq 0 ]; then
        echo '[]'
        return 0
    fi

    local warnings="[]"
    local ci
    for ((ci = 0; ci < n_constraints; ci++)); do
        local constraint
        constraint=$(echo "$constraints" | jq -c ".[$ci]")
        local req_dep req_desc
        req_dep=$(echo "$constraint" | jq -r '.requires')
        req_desc=$(echo "$constraint" | jq -r '.description // ""')

        local current_req_ver
        current_req_ver=$(jq -r --arg n "$req_dep" \
            '.dependencies[] | select(.name==$n) | .current_version' "$MANIFEST")

        local latest_major
        latest_major=$(version_major "$latest_version")

        local min_required
        min_required=$(echo "$constraint" | jq -r --arg m "$latest_major" \
            '.minimum_version[$m] // empty')

        if [ -z "$min_required" ]; then
            warnings=$(echo "$warnings" | jq \
                --arg dep "$req_dep" --arg desc "$req_desc" \
                --arg major "$latest_major" \
                '. + [{"requires":$dep,"status":"unknown","message":"no constraint data for major version "+$major,"description":$desc}]')
            continue
        fi

        local status="ok"
        if version_gte "$current_req_ver" "$min_required"; then
            status="ok"
        else
            status="incompatible"
        fi

        warnings=$(echo "$warnings" | jq \
            --arg dep "$req_dep" --arg min "$min_required" \
            --arg cur "$current_req_ver" --arg st "$status" --arg desc "$req_desc" \
            '. + [{"requires":$dep,"minimum":$min,"current":$cur,"status":$st,"description":$desc}]')
    done

    echo "$warnings"
}

# ═══════════════════════════════════════════════════════════════════
# Post-apply validation
# ═══════════════════════════════════════════════════════════════════

validate_modified_files() {
    local -a files=("$@")
    local all_ok=true

    for file in "${files[@]}"; do
        [ -z "$file" ] && continue
        local full_path="${ROOT_DIR}/${file}"
        [ ! -f "$full_path" ] && continue

        case "$file" in
            *.sh)
                if ! bash -n "$full_path" 2>/dev/null; then
                    log_fail "Post-apply validation failed: ${file} has bash syntax errors"
                    git -C "$ROOT_DIR" checkout -- "$file" 2>/dev/null || true
                    all_ok=false
                fi
                ;;
            *.json)
                if ! jq . "$full_path" > /dev/null 2>&1; then
                    log_fail "Post-apply validation failed: ${file} has invalid JSON"
                    git -C "$ROOT_DIR" checkout -- "$file" 2>/dev/null || true
                    all_ok=false
                fi
                ;;
            *.yaml | *.yml)
                if command -v python3 &> /dev/null; then
                    if ! python3 -c "import yaml; yaml.safe_load(open('${full_path}'))" 2>/dev/null; then
                        log_fail "Post-apply validation failed: ${file} has invalid YAML"
                        git -C "$ROOT_DIR" checkout -- "$file" 2>/dev/null || true
                        all_ok=false
                    fi
                fi
                ;;
        esac
    done

    if [ "$all_ok" = true ]; then
        return 0
    else
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════════════
# Update history
# ═══════════════════════════════════════════════════════════════════

append_update_history() {
    local updates_json="$1"
    local check_date="$2"

    if [ "$(echo "$updates_json" | jq 'length')" -eq 0 ]; then
        return 0
    fi

    local history="[]"
    if [ -f "$HISTORY_FILE" ]; then
        history=$(jq '.' "$HISTORY_FILE" 2>/dev/null || echo '[]')
    fi

    local tmp
    tmp=$(mktemp)
    jq --arg d "$check_date" --argjson u "$updates_json" \
        '. + [{"date": $d, "updates": $u}]' <<< "$history" > "$tmp" \
        && mv "$tmp" "$HISTORY_FILE"
}

# ═══════════════════════════════════════════════════════════════════
# Apply logic
# ═══════════════════════════════════════════════════════════════════

apply_container_image() {
    local name="$1" current="$2" latest="$3"

    if [ "$name" = "hpc-benchmarks" ]; then
        if [ "$DRY_RUN" = true ]; then
            log_info "  conf/defaults.sh: HPL_IMAGE ${current} → ${latest}, HPL_IMAGE_ALT → ${current}"
            return 0
        fi
        local file="${ROOT_DIR}/conf/defaults.sh"
        local old_alt
        old_alt=$(jq -r '.dependencies[] | select(.name=="hpc-benchmarks") | .current_alt_version' "$MANIFEST")
        sed -i "s|hpc-benchmarks:${current}}|hpc-benchmarks:${latest}}|" "$file"
        sed -i "s|hpc-benchmarks:${old_alt}}|hpc-benchmarks:${current}}|" "$file"
        MODIFIED_FILES+=("conf/defaults.sh")
        local tmp
        tmp=$(mktemp)
        jq --arg v "$latest" --arg a "$current" \
            '(.dependencies[] | select(.name=="hpc-benchmarks")).current_version = $v
           | (.dependencies[] | select(.name=="hpc-benchmarks")).current_alt_version = $a' \
            "$MANIFEST" > "$tmp" && mv "$tmp" "$MANIFEST"
        MODIFIED_FILES+=("specs/dependencies.json")
        log_info "Updated conf/defaults.sh: HPL_IMAGE → ${latest}, HPL_IMAGE_ALT → ${current}"
    fi

    if [ "$name" = "intel-hpckit" ]; then
        if [ "$DRY_RUN" = true ]; then
            log_info "  scripts/hpl-cpu.sh: intel/hpckit ${current} → ${latest}"
            return 0
        fi
        local file="${ROOT_DIR}/scripts/hpl-cpu.sh"
        sed -i "s|intel/hpckit:[^\"]*|intel/hpckit:${latest}|" "$file"
        MODIFIED_FILES+=("scripts/hpl-cpu.sh")
        local tmp
        tmp=$(mktemp)
        jq --arg v "$latest" \
            '(.dependencies[] | select(.name=="intel-hpckit")).current_version = $v' \
            "$MANIFEST" > "$tmp" && mv "$tmp" "$MANIFEST"
        MODIFIED_FILES+=("specs/dependencies.json")
        log_info "Updated scripts/hpl-cpu.sh: intel/hpckit → ${latest}"
    fi
}

apply_precommit_hooks() {
    if [ "$DRY_RUN" = true ]; then
        log_info "  .pre-commit-config.yaml: would run pre-commit autoupdate"
        return 0
    fi
    if ! command -v pre-commit &> /dev/null; then
        log_fail "pre-commit not installed — cannot apply pre-commit hook updates"
        return 1
    fi

    log_info "Running pre-commit autoupdate..."
    (cd "$ROOT_DIR" && pre-commit autoupdate)
    MODIFIED_FILES+=(".pre-commit-config.yaml")

    local config="${ROOT_DIR}/.pre-commit-config.yaml"
    local tmp
    tmp=$(mktemp)
    local hooks_rev shfmt_rev shellcheck_rev
    hooks_rev=$(grep -A1 'pre-commit/pre-commit-hooks' "$config" | grep 'rev:' | awk '{print $2}')
    shfmt_rev=$(grep -A1 'scop/pre-commit-shfmt' "$config" | grep 'rev:' | awk '{print $2}')
    shellcheck_rev=$(grep -A1 'shellcheck-py/shellcheck-py' "$config" | grep 'rev:' | awk '{print $2}')

    jq --arg h "${hooks_rev:-}" --arg s "${shfmt_rev:-}" --arg sc "${shellcheck_rev:-}" '
        (if $h != "" then (.dependencies[] | select(.name=="pre-commit-hooks")).current_version = $h else . end)
      | (if $s != "" then (.dependencies[] | select(.name=="pre-commit-shfmt")).current_version = $s else . end)
      | (if $sc != "" then (.dependencies[] | select(.name=="shellcheck-py")).current_version = $sc else . end)
    ' "$MANIFEST" > "$tmp" && mv "$tmp" "$MANIFEST"
    MODIFIED_FILES+=("specs/dependencies.json")

    log_info "Updated .pre-commit-config.yaml and synced manifest"
}

apply_nvidia_package() {
    local name="$1" current="$2" latest="$3"
    local file="${ROOT_DIR}/scripts/bootstrap.sh"

    if [ "$name" = "nvidia-driver" ]; then
        if [ "$DRY_RUN" = true ]; then
            log_info "  scripts/bootstrap.sh: driver fallback ${current} → ${latest}"
            return 0
        fi
        sed -i "s|nvidia-driver-${current}-server|nvidia-driver-${latest}-server|" "$file"
        MODIFIED_FILES+=("scripts/bootstrap.sh")
        local tmp
        tmp=$(mktemp)
        jq --arg v "$latest" \
            '(.dependencies[] | select(.name=="nvidia-driver")).current_version = $v' \
            "$MANIFEST" > "$tmp" && mv "$tmp" "$MANIFEST"
        MODIFIED_FILES+=("specs/dependencies.json")
        log_info "Updated bootstrap.sh: driver fallback → nvidia-driver-${latest}-server"
    elif [ "$name" = "cuda-toolkit" ]; then
        if [ "$DRY_RUN" = true ]; then
            log_info "  scripts/bootstrap.sh: CUDA fallback ${current} → ${latest}"
            return 0
        fi
        sed -i "s|_cuda_runtime:-${current}|_cuda_runtime:-${latest}|g" "$file"
        MODIFIED_FILES+=("scripts/bootstrap.sh")
        local tmp
        tmp=$(mktemp)
        jq --arg v "$latest" \
            '(.dependencies[] | select(.name=="cuda-toolkit")).current_version = $v' \
            "$MANIFEST" > "$tmp" && mv "$tmp" "$MANIFEST"
        MODIFIED_FILES+=("specs/dependencies.json")
        log_info "Updated bootstrap.sh: CUDA fallback → ${latest}"
    else
        if [ "$DRY_RUN" = true ]; then
            log_info "  manifest: ${name} → ${latest} (report-only)"
            return 0
        fi
        local tmp
        tmp=$(mktemp)
        jq --arg n "$name" --arg v "$latest" \
            '(.dependencies[] | select(.name==$n)).current_version = $v' \
            "$MANIFEST" > "$tmp" && mv "$tmp" "$MANIFEST"
        MODIFIED_FILES+=("specs/dependencies.json")
        log_info "Updated manifest: ${name} → ${latest} (report-only, no file changes)"
    fi
}

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

main() {
    parse_args "$@"
    require_tools

    if [ ! -f "$MANIFEST" ]; then
        echo "Error: manifest not found: ${MANIFEST}" >&2
        exit 1
    fi

    if [ "$DRY_RUN" = true ] && [ "$APPLY" = false ]; then
        echo "Error: --dry-run requires --apply" >&2
        exit 1
    fi

    local suite_version="unknown"
    [ -f "$VERSION_FILE" ] && suite_version=$(cat "$VERSION_FILE")
    local check_date
    check_date=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    local dep_count
    dep_count=$(jq '.dependencies | length' "$MANIFEST")

    if [ "$OUTPUT_JSON" = false ]; then
        log "HPC Bench Suite v${suite_version} — Dependency Update Check"
        log "Date: ${check_date}"
        log "Manifest: ${MANIFEST} (${dep_count} dependencies)"
        log "─────────────────────────────────────────────────────"
    fi

    # Collect results as a JSON array
    local results="[]"
    local count_update=0 count_ok=0 count_failed=0 count_constraint_warnings=0
    local apply_containers="" apply_precommit=false apply_nvidia=""
    declare -a MODIFIED_FILES=()

    local i
    for ((i = 0; i < dep_count; i++)); do
        local dep
        dep=$(jq -c ".dependencies[$i]" "$MANIFEST")

        local name category current
        name=$(echo "$dep" | jq -r '.name')
        category=$(echo "$dep" | jq -r '.category')
        current=$(echo "$dep" | jq -r '.current_version')

        # Category filter
        if [ -n "$FILTER_CATEGORY" ] && [ "$category" != "$FILTER_CATEGORY" ]; then
            continue
        fi

        # Run the check
        local check_result
        check_result=$(run_check "$dep")

        local error latest
        error=$(echo "$check_result" | jq -r '.error // empty')

        if [ -n "$error" ]; then
            count_failed=$((count_failed + 1))
            if [ "$OUTPUT_JSON" = false ]; then
                log_fail "$(printf '%-24s check failed: %s' "$name" "$error")"
            fi
            results=$(echo "$results" | jq --arg n "$name" --arg c "$category" \
                --arg cv "$current" --arg e "$error" \
                '. + [{"name":$n,"category":$c,"current_version":$cv,"status":"check_failed","error":$e}]')
            continue
        fi

        latest=$(echo "$check_result" | jq -r '.latest // empty')
        local update_available
        update_available=$(has_update "$current" "$latest" "$category")

        local targets
        targets=$(echo "$dep" | jq -c '.update_targets // []')

        # Check constraints for this dependency
        local cw_json="[]"
        if [ "$update_available" = "true" ]; then
            cw_json=$(check_constraints "$dep" "$latest")
        fi

        if [ "$update_available" = "true" ]; then
            count_update=$((count_update + 1))
            if [ "$OUTPUT_JSON" = false ]; then
                log_update "$(printf '%-24s %s → %s' "$name" "$current" "$latest")"
                local n_cw
                n_cw=$(echo "$cw_json" | jq 'length')
                local cwi
                for ((cwi = 0; cwi < n_cw; cwi++)); do
                    local cw_req cw_min cw_cur cw_status
                    cw_req=$(echo "$cw_json" | jq -r ".[$cwi].requires")
                    cw_min=$(echo "$cw_json" | jq -r ".[$cwi].minimum")
                    cw_cur=$(echo "$cw_json" | jq -r ".[$cwi].current")
                    cw_status=$(echo "$cw_json" | jq -r ".[$cwi].status")
                    if [ "$cw_status" = "incompatible" ]; then
                        log_fail "  ⚠ requires ${cw_req} ≥ ${cw_min} (current: ${cw_cur} — INCOMPATIBLE)"
                        count_constraint_warnings=$((count_constraint_warnings + 1))
                    elif [ "$cw_status" = "ok" ]; then
                        log_ok "  ⚠ requires ${cw_req} ≥ ${cw_min} (current: ${cw_cur} — OK)"
                    fi
                done
            fi
            # Queue for --apply
            if [ "$category" = "container_image" ]; then
                apply_containers="${apply_containers} ${name}:${current}:${latest}"
            elif [ "$category" = "pre_commit_hook" ]; then
                apply_precommit=true
            elif [ "$category" = "nvidia_package" ]; then
                apply_nvidia="${apply_nvidia} ${name}:${current}:${latest}"
            fi
        else
            count_ok=$((count_ok + 1))
            if [ "$OUTPUT_JSON" = false ]; then
                local extra=""
                if [ "$category" = "upstream_source" ]; then
                    local commit_date published
                    commit_date=$(echo "$check_result" | jq -r '.commit_date // empty')
                    published=$(echo "$check_result" | jq -r '.published // empty')
                    if [ -n "$commit_date" ]; then
                        extra=" (latest commit: ${latest}, ${commit_date%%T*})"
                    elif [ -n "$published" ]; then
                        extra=" (latest release: ${latest}, ${published%%T*})"
                    elif [ -n "$latest" ]; then
                        extra=" (latest tag: ${latest})"
                    fi
                fi
                log_ok "$(printf '%-24s %s%s' "$name" "$current" "$extra")"
            fi
        fi

        results=$(echo "$results" | jq \
            --arg n "$name" --arg c "$category" --arg cv "$current" \
            --arg lv "$latest" --argjson ua "$update_available" --argjson t "$targets" \
            --argjson cr "$check_result" --argjson cw "$cw_json" \
            '. + [{"name":$n,"category":$c,"current_version":$cv,"latest_version":$lv,
                   "update_available":$ua,"update_targets":$t,"check_detail":$cr,
                   "constraint_warnings":$cw}]')
    done

    local total=$((count_update + count_ok + count_failed))

    if [ "$OUTPUT_JSON" = false ]; then
        log "─────────────────────────────────────────────────────"
        log "Summary: ${count_update} updates available, ${count_ok} up-to-date, ${count_failed} check failed"
    fi

    # JSON output
    if [ "$OUTPUT_JSON" = true ]; then
        jq -n \
            --arg date "$check_date" --arg ver "$suite_version" \
            --argjson total "$total" --argjson updates "$count_update" \
            --argjson ok "$count_ok" --argjson failed "$count_failed" \
            --argjson cw_total "$count_constraint_warnings" \
            --argjson deps "$results" \
            '{
                check_date: $date,
                suite_version: $ver,
                summary: {total: $total, updates_available: $updates, up_to_date: $ok, check_failed: $failed, constraint_warnings: $cw_total},
                dependencies: $deps
            }'
    fi

    # Apply updates
    if [ "$APPLY" = true ] && [ "$count_update" -gt 0 ]; then
        if [ "$OUTPUT_JSON" = false ]; then
            echo ""
            if [ "$DRY_RUN" = true ]; then
                log "Dry run — showing what --apply would change:"
            else
                log "Applying updates..."
            fi
        fi

        local applied_updates="[]"

        for entry in $apply_containers; do
            local cname ccurrent clatest
            cname="${entry%%:*}"
            entry="${entry#*:}"
            ccurrent="${entry%%:*}"
            clatest="${entry#*:}"
            apply_container_image "$cname" "$ccurrent" "$clatest"
            applied_updates=$(echo "$applied_updates" | jq \
                --arg n "$cname" --arg f "$ccurrent" --arg t "$clatest" \
                '. + [{"name":$n,"from":$f,"to":$t}]')
        done

        for entry in $apply_nvidia; do
            local nname ncurrent nlatest
            nname="${entry%%:*}"
            entry="${entry#*:}"
            ncurrent="${entry%%:*}"
            nlatest="${entry#*:}"
            apply_nvidia_package "$nname" "$ncurrent" "$nlatest"
            applied_updates=$(echo "$applied_updates" | jq \
                --arg n "$nname" --arg f "$ncurrent" --arg t "$nlatest" \
                '. + [{"name":$n,"from":$f,"to":$t}]')
        done

        if [ "$apply_precommit" = true ]; then
            apply_precommit_hooks
            applied_updates=$(echo "$applied_updates" | jq \
                '. + [{"name":"pre-commit-hooks","from":"(multiple)","to":"latest"}]')
        fi

        if [ "$DRY_RUN" = false ]; then
            # Post-apply validation
            local unique_files
            unique_files=$(printf '%s\n' "${MODIFIED_FILES[@]}" | sort -u)
            local -a files_array
            mapfile -t files_array <<< "$unique_files"
            if validate_modified_files "${files_array[@]}"; then
                if [ "$OUTPUT_JSON" = false ]; then
                    log_ok "Post-apply validation passed"
                fi
            else
                if [ "$OUTPUT_JSON" = false ]; then
                    log_fail "Post-apply validation failed — reverted broken files"
                fi
                return 1
            fi

            # Update history
            append_update_history "$applied_updates" "$check_date"

            if [ "$OUTPUT_JSON" = false ]; then
                log "Done. Review changes with: git diff"
            fi
        else
            if [ "$OUTPUT_JSON" = false ]; then
                log_info "Dry run complete — no files were modified."
            fi
        fi
    elif [ "$APPLY" = true ] && [ "$count_update" -eq 0 ]; then
        if [ "$OUTPUT_JSON" = false ]; then
            log_info "Nothing to apply — all dependencies are up to date."
        fi
    fi

    return 0
}

main "$@"
