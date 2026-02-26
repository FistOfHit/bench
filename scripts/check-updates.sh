#!/usr/bin/env bash
# check-updates.sh — Check tracked dependencies for available updates
# Usage: bash scripts/check-updates.sh [--json] [--apply] [--category CAT] [--help]
#
# Reads specs/dependencies.json, queries upstream sources, and reports
# which dependencies have newer versions available.
#
# Not a benchmark module — does not source lib/common.sh or emit module JSON.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${ROOT_DIR}/specs/dependencies.json"
VERSION_FILE="${ROOT_DIR}/VERSION"

# ── Defaults ──
OUTPUT_JSON=false
APPLY=false
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
  --category CAT    Check only one category:
                      container_image, upstream_source, pre_commit_hook
  --help            Show this help

Environment:
  GITHUB_TOKEN      GitHub API token for higher rate limits (5000/hr vs 60/hr)
  NGC_API_KEY       NGC API key (currently unused; reserved for future auth)
  NO_COLOR          Disable colored output

Examples:
  bash scripts/check-updates.sh                     # Human-readable report
  bash scripts/check-updates.sh --json              # JSON report
  bash scripts/check-updates.sh --apply             # Check and apply updates
  bash scripts/check-updates.sh --category pre_commit_hook  # Check one category
EOF
}

# ── Argument parsing ──
parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --json) OUTPUT_JSON=true ;;
            --apply) APPLY=true ;;
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
# Apply logic
# ═══════════════════════════════════════════════════════════════════

apply_container_image() {
    local name="$1" current="$2" latest="$3"

    if [ "$name" = "hpc-benchmarks" ]; then
        local file="${ROOT_DIR}/conf/defaults.sh"
        local old_alt
        old_alt=$(jq -r '.dependencies[] | select(.name=="hpc-benchmarks") | .current_alt_version' "$MANIFEST")

        # Rotate: current primary → alt, latest → primary
        # Match version between colon and closing brace-quote: :VER}"
        sed -i "s|hpc-benchmarks:${current}}|hpc-benchmarks:${latest}}|" "$file"
        sed -i "s|hpc-benchmarks:${old_alt}}|hpc-benchmarks:${current}}|" "$file"

        # Update manifest
        local tmp
        tmp=$(mktemp)
        jq --arg v "$latest" --arg a "$current" \
            '(.dependencies[] | select(.name=="hpc-benchmarks")).current_version = $v
           | (.dependencies[] | select(.name=="hpc-benchmarks")).current_alt_version = $a' \
            "$MANIFEST" > "$tmp" && mv "$tmp" "$MANIFEST"

        log_info "Updated conf/defaults.sh: HPL_IMAGE → ${latest}, HPL_IMAGE_ALT → ${current}"
    fi

    if [ "$name" = "intel-hpckit" ]; then
        local file="${ROOT_DIR}/scripts/hpl-cpu.sh"
        sed -i "s|intel/hpckit:[^\"]*|intel/hpckit:${latest}|" "$file"

        local tmp
        tmp=$(mktemp)
        jq --arg v "$latest" \
            '(.dependencies[] | select(.name=="intel-hpckit")).current_version = $v' \
            "$MANIFEST" > "$tmp" && mv "$tmp" "$MANIFEST"

        log_info "Updated scripts/hpl-cpu.sh: intel/hpckit → ${latest}"
    fi
}

apply_precommit_hooks() {
    if ! command -v pre-commit &> /dev/null; then
        log_fail "pre-commit not installed — cannot apply pre-commit hook updates"
        return 1
    fi

    log_info "Running pre-commit autoupdate..."
    (cd "$ROOT_DIR" && pre-commit autoupdate)

    # Sync manifest with the updated .pre-commit-config.yaml
    local config="${ROOT_DIR}/.pre-commit-config.yaml"
    local tmp
    tmp=$(mktemp)

    # Extract actual revs from the config file
    local hooks_rev shfmt_rev shellcheck_rev
    hooks_rev=$(grep -A1 'pre-commit/pre-commit-hooks' "$config" | grep 'rev:' | awk '{print $2}')
    shfmt_rev=$(grep -A1 'scop/pre-commit-shfmt' "$config" | grep 'rev:' | awk '{print $2}')
    shellcheck_rev=$(grep -A1 'shellcheck-py/shellcheck-py' "$config" | grep 'rev:' | awk '{print $2}')

    jq --arg h "${hooks_rev:-}" --arg s "${shfmt_rev:-}" --arg sc "${shellcheck_rev:-}" '
        (if $h != "" then (.dependencies[] | select(.name=="pre-commit-hooks")).current_version = $h else . end)
      | (if $s != "" then (.dependencies[] | select(.name=="pre-commit-shfmt")).current_version = $s else . end)
      | (if $sc != "" then (.dependencies[] | select(.name=="shellcheck-py")).current_version = $sc else . end)
    ' "$MANIFEST" > "$tmp" && mv "$tmp" "$MANIFEST"

    log_info "Updated .pre-commit-config.yaml and synced manifest"
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
    local count_update=0 count_ok=0 count_failed=0
    local apply_containers="" apply_precommit=false

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

        if [ "$update_available" = "true" ]; then
            count_update=$((count_update + 1))
            if [ "$OUTPUT_JSON" = false ]; then
                log_update "$(printf '%-24s %s → %s' "$name" "$current" "$latest")"
            fi
            # Queue for --apply
            if [ "$category" = "container_image" ]; then
                apply_containers="${apply_containers} ${name}:${current}:${latest}"
            elif [ "$category" = "pre_commit_hook" ]; then
                apply_precommit=true
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
            --argjson cr "$check_result" \
            '. + [{"name":$n,"category":$c,"current_version":$cv,"latest_version":$lv,
                   "update_available":$ua,"update_targets":$t,"check_detail":$cr}]')
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
            --argjson deps "$results" \
            '{
                check_date: $date,
                suite_version: $ver,
                summary: {total: $total, updates_available: $updates, up_to_date: $ok, check_failed: $failed},
                dependencies: $deps
            }'
    fi

    # Apply updates
    if [ "$APPLY" = true ] && [ "$count_update" -gt 0 ]; then
        if [ "$OUTPUT_JSON" = false ]; then
            echo ""
            log "Applying updates..."
        fi

        for entry in $apply_containers; do
            local cname ccurrent clatest
            cname="${entry%%:*}"
            entry="${entry#*:}"
            ccurrent="${entry%%:*}"
            clatest="${entry#*:}"
            apply_container_image "$cname" "$ccurrent" "$clatest"
        done

        if [ "$apply_precommit" = true ]; then
            apply_precommit_hooks
        fi

        if [ "$OUTPUT_JSON" = false ]; then
            log "Done. Review changes with: git diff"
        fi
    elif [ "$APPLY" = true ] && [ "$count_update" -eq 0 ]; then
        if [ "$OUTPUT_JSON" = false ]; then
            log_info "Nothing to apply — all dependencies are up to date."
        fi
    fi

    return 0
}

main "$@"
