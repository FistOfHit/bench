#!/usr/bin/env bats
# Unit tests for lib/common.sh helper functions.
# Run: bats tests/common_helpers.bats
# Install bats: sudo apt-get install bats  (or: npm install -g bats)

# ── Test setup ──
setup() {
    export HPC_BENCH_ROOT="${BATS_TEST_DIRNAME}/.."
    export HPC_RESULTS_DIR="$(mktemp -d)"
    export HPC_LOG_DIR="${HPC_RESULTS_DIR}/logs"
    export HPC_WORK_DIR="$(mktemp -d)"
    export SCRIPT_NAME="test-module"
    mkdir -p "$HPC_LOG_DIR"

    # Source common.sh (sets up paths, loads defaults, creates dirs)
    source "${HPC_BENCH_ROOT}/lib/common.sh"

    # Override common.sh's EXIT trap and strict mode: bats manages its own
    # subshell lifecycle and set -e behavior; common.sh's settings interfere.
    trap - EXIT
    set +euo pipefail
}

teardown() {
    rm -rf "$HPC_RESULTS_DIR" "$HPC_WORK_DIR"
}

# ═══════════════════════════════════════════
# trim_ws
# ═══════════════════════════════════════════

@test "trim_ws: removes leading and trailing whitespace" {
    result=$(trim_ws "  hello world  ")
    [ "$result" = "hello world" ]
}

@test "trim_ws: handles tabs and mixed whitespace" {
    result=$(trim_ws "$(printf '\t  foo bar \t ')")
    [ "$result" = "foo bar" ]
}

@test "trim_ws: returns empty for whitespace-only input" {
    result=$(trim_ws "   ")
    [ -z "$result" ]
}

@test "trim_ws: handles empty string" {
    result=$(trim_ws "")
    [ -z "$result" ]
}

@test "trim_ws: preserves internal whitespace" {
    result=$(trim_ws "  a   b   c  ")
    [ "$result" = "a   b   c" ]
}

# ═══════════════════════════════════════════
# json_numeric_or_null
# ═══════════════════════════════════════════

@test "json_numeric_or_null: integer" {
    result=$(json_numeric_or_null "42")
    [ "$result" = "42" ]
}

@test "json_numeric_or_null: float" {
    result=$(json_numeric_or_null "3.14")
    [ "$result" = "3.14" ]
}

@test "json_numeric_or_null: negative number" {
    result=$(json_numeric_or_null "-7")
    [ "$result" = "-7" ]
}

@test "json_numeric_or_null: negative float" {
    result=$(json_numeric_or_null "-12.5")
    [ "$result" = "-12.5" ]
}

@test "json_numeric_or_null: empty string returns null" {
    result=$(json_numeric_or_null "")
    [ "$result" = "null" ]
}

@test "json_numeric_or_null: N/A returns null" {
    result=$(json_numeric_or_null "N/A")
    [ "$result" = "null" ]
}

@test "json_numeric_or_null: [Not Supported] returns null" {
    result=$(json_numeric_or_null "[Not Supported]")
    [ "$result" = "null" ]
}

@test "json_numeric_or_null: unknown returns null" {
    result=$(json_numeric_or_null "unknown")
    [ "$result" = "null" ]
}

@test "json_numeric_or_null: non-numeric text returns null" {
    result=$(json_numeric_or_null "abc")
    [ "$result" = "null" ]
}

@test "json_numeric_or_null: trims whitespace around number" {
    result=$(json_numeric_or_null "  100  ")
    [ "$result" = "100" ]
}

@test "json_numeric_or_null: number with units returns null" {
    result=$(json_numeric_or_null "100W")
    [ "$result" = "null" ]
}

# ═══════════════════════════════════════════
# int_or_default
# ═══════════════════════════════════════════

@test "int_or_default: valid integer" {
    result=$(int_or_default "42")
    [ "$result" = "42" ]
}

@test "int_or_default: empty returns default 0" {
    result=$(int_or_default "")
    [ "$result" = "0" ]
}

@test "int_or_default: non-integer returns custom default" {
    result=$(int_or_default "abc" "99")
    [ "$result" = "99" ]
}

@test "int_or_default: float returns default (not integer)" {
    result=$(int_or_default "3.14" "0")
    [ "$result" = "0" ]
}

@test "int_or_default: negative not treated as int (contains dash)" {
    result=$(int_or_default "-5" "0")
    [ "$result" = "0" ]
}

@test "int_or_default: trims whitespace" {
    result=$(int_or_default "  7  " "0")
    [ "$result" = "7" ]
}

# ═══════════════════════════════════════════
# count_grep_re
# ═══════════════════════════════════════════

@test "count_grep_re: counts matches in string argument" {
    result=$(count_grep_re 'FAULTY' "GPU 0: OK
GPU 1: FAULTY
GPU 2: FAULTY")
    [ "$result" = "2" ]
}

@test "count_grep_re: returns 0 for no matches" {
    result=$(count_grep_re 'FAULTY' "GPU 0: OK
GPU 1: OK")
    [ "$result" = "0" ]
}

@test "count_grep_re: works with regex" {
    result=$(count_grep_re 'GPU [0-9]+' "GPU 0: OK
GPU 1: OK
GPU 2: FAIL")
    [ "$result" = "3" ]
}

@test "count_grep_re: handles empty input" {
    result=$(count_grep_re 'FAULTY' "")
    [ "$result" = "0" ]
}

@test "count_grep_re: reads from stdin" {
    result=$(printf 'line1\nFAULTY\nline3\nFAULTY\n' | count_grep_re 'FAULTY')
    [ "$result" = "2" ]
}

# ═══════════════════════════════════════════
# json_compact_or
# ═══════════════════════════════════════════

@test "json_compact_or: compacts valid JSON" {
    result=$(json_compact_or '{ "a": 1, "b": 2 }')
    [ "$result" = '{"a":1,"b":2}' ]
}

@test "json_compact_or: returns fallback for invalid JSON" {
    result=$(json_compact_or 'not json')
    [ "$result" = '{}' ]
}

@test "json_compact_or: returns custom fallback" {
    local result
    result=$(json_compact_or '' '[]') || true
    [ "$result" = '[]' ]
}

@test "json_compact_or: handles empty string" {
    local result
    result=$(json_compact_or '' '{"default":true}') || true
    [ "$result" = '{"default":true}' ]
}

# ═══════════════════════════════════════════
# json_array_from_lines
# ═══════════════════════════════════════════

@test "json_array_from_lines: converts lines to JSON array" {
    result=$(printf 'alpha\nbeta\ngamma\n' | json_array_from_lines)
    [ "$result" = '["alpha","beta","gamma"]' ]
}

@test "json_array_from_lines: drops empty lines" {
    result=$(printf 'a\n\nb\n\n' | json_array_from_lines)
    [ "$result" = '["a","b"]' ]
}

@test "json_array_from_lines: returns fallback for empty input" {
    result=$(printf '' | json_array_from_lines '[]')
    [ "$result" = '[]' ]
}

# ═══════════════════════════════════════════
# sanitize_json_str
# ═══════════════════════════════════════════

@test "sanitize_json_str: escapes quotes" {
    result=$(sanitize_json_str 'say "hello"')
    [ "$result" = '"say \"hello\""' ]
}

@test "sanitize_json_str: handles simple string" {
    result=$(sanitize_json_str 'hello world')
    [ "$result" = '"hello world"' ]
}

@test "sanitize_json_str: handles empty string" {
    result=$(sanitize_json_str '')
    [ "$result" = '""' ]
}

@test "sanitize_json_str: strips control characters" {
    # Input has a tab character — should be stripped before JSON encoding
    input=$'hello\tworld'
    result=$(sanitize_json_str "$input")
    # The tab is stripped, so result should be "helloworld" as a JSON string
    [ "$result" = '"helloworld"' ]
}

@test "sanitize_json_str: output is valid JSON" {
    result=$(sanitize_json_str 'test with "quotes" and backslash\\')
    echo "$result" | jq . >/dev/null 2>&1
}

# ═══════════════════════════════════════════
# json_str (alias for sanitize_json_str)
# ═══════════════════════════════════════════

@test "json_str: behaves identically to sanitize_json_str" {
    a=$(sanitize_json_str 'say "hello"')
    b=$(json_str 'say "hello"')
    [ "$a" = "$b" ]
}

@test "json_str: strips control characters like sanitize_json_str" {
    input=$'tab\there'
    result=$(json_str "$input")
    [ "$result" = '"tabhere"' ]
}

@test "json_str: output has no trailing newline" {
    # Verify no trailing newline (was a bug in the old standalone json_str)
    raw=$(json_str "test")
    # If there's a trailing newline, wc -l will be > 0
    lines=$(printf '%s' "$raw" | wc -l | tr -d '[:space:]')
    [ "$lines" = "0" ]
}

# ═══════════════════════════════════════════
# has_cmd / cmd_path
# ═══════════════════════════════════════════

@test "has_cmd: returns 0 for existing command" {
    has_cmd bash
}

@test "has_cmd: returns 1 for nonexistent command" {
    ! has_cmd this_command_does_not_exist_xyz
}

@test "cmd_path: returns path for existing command" {
    result=$(cmd_path bash)
    [ -n "$result" ]
    [ -x "$result" ]
}

@test "cmd_path: returns empty for nonexistent command" {
    result=$(cmd_path this_command_does_not_exist_xyz)
    [ -z "$result" ]
}

# ═══════════════════════════════════════════
# emit_json / emit_json_safe
# ═══════════════════════════════════════════

@test "emit_json: writes valid JSON with module and status" {
    echo '{"foo":"bar"}' | emit_json "test-mod" "ok"
    [ -f "${HPC_RESULTS_DIR}/test-mod.json" ]
    result=$(jq -r '.module' "${HPC_RESULTS_DIR}/test-mod.json")
    [ "$result" = "test-mod" ]
    result=$(jq -r '.status' "${HPC_RESULTS_DIR}/test-mod.json")
    [ "$result" = "ok" ]
    result=$(jq -r '.foo' "${HPC_RESULTS_DIR}/test-mod.json")
    [ "$result" = "bar" ]
}

@test "emit_json: includes timestamp" {
    echo '{}' | emit_json "ts-test" "ok"
    result=$(jq -r '.timestamp' "${HPC_RESULTS_DIR}/ts-test.json")
    [[ "$result" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]]
}

@test "emit_json_safe: rejects invalid JSON" {
    run bash -c 'source "'"${HPC_BENCH_ROOT}/lib/common.sh"'" && echo "not json" | emit_json_safe "bad-mod" "ok"'
    [ "$status" -ne 0 ]
    # Should have written an error record
    result=$(jq -r '.status' "${HPC_RESULTS_DIR}/bad-mod.json" 2>/dev/null || echo "missing")
    [ "$result" = "error" ]
}

@test "emit_json_safe: accepts valid JSON" {
    echo '{"valid": true}' | emit_json_safe "good-mod" "ok"
    result=$(jq -r '.valid' "${HPC_RESULTS_DIR}/good-mod.json")
    [ "$result" = "true" ]
}

# ═══════════════════════════════════════════
# finish_module
# ═══════════════════════════════════════════

@test "finish_module: emits JSON and sets status" {
    json='{"score": 95}'
    finish_module "fm-test" "ok" "$json" '.'
    [ -f "${HPC_RESULTS_DIR}/fm-test.json" ]
    result=$(jq -r '.status' "${HPC_RESULTS_DIR}/fm-test.json")
    [ "$result" = "ok" ]
    result=$(jq -r '.score' "${HPC_RESULTS_DIR}/fm-test.json")
    [ "$result" = "95" ]
}

# ═══════════════════════════════════════════
# json_tmpfile
# ═══════════════════════════════════════════

@test "json_tmpfile: creates valid JSON temp file" {
    path=$(json_tmpfile "test" '{"key":"val"}')
    [ -f "$path" ]
    result=$(jq -r '.key' "$path")
    [ "$result" = "val" ]
}

@test "json_tmpfile: writes fallback on invalid JSON" {
    local path
    path=$(json_tmpfile "test" 'not json' '{"fallback":true}') || true
    [ -f "$path" ]
    result=$(jq -r '.fallback' "$path")
    [ "$result" = "true" ]
}

# ═══════════════════════════════════════════
# safe_json_pipe
# ═══════════════════════════════════════════

@test "safe_json_pipe: returns command output as JSON" {
    result=$(safe_json_pipe '{}' echo '{"data":1}')
    echo "$result" | jq . >/dev/null 2>&1
    [ "$(echo "$result" | jq -r '.data')" = "1" ]
}

@test "safe_json_pipe: returns fallback on command failure" {
    result=$(safe_json_pipe '{"fallback":true}' false)
    [ "$(echo "$result" | jq -r '.fallback')" = "true" ]
}

@test "safe_json_pipe: returns fallback on invalid output" {
    result=$(safe_json_pipe '{"fb":1}' echo "not json")
    [ "$(echo "$result" | jq -r '.fb')" = "1" ]
}

# ═══════════════════════════════════════════
# detect_compute_capability (requires nvidia-smi to be meaningful, test graceful fallback)
# ═══════════════════════════════════════════

@test "detect_compute_capability: returns empty when nvidia-smi unavailable" {
    # Override PATH to hide nvidia-smi
    PATH="/usr/bin:/bin" result=$(detect_compute_capability)
    [ -z "$result" ] || [[ "$result" =~ ^[0-9]+$ ]]
}

# ═══════════════════════════════════════════
# detect_pkg_manager
# ═══════════════════════════════════════════

@test "detect_pkg_manager: returns a known manager or unknown" {
    result=$(detect_pkg_manager)
    [[ "$result" =~ ^(apt|dnf|yum|unknown)$ ]]
}

# ═══════════════════════════════════════════
# is_root / try_sudo
# ═══════════════════════════════════════════

@test "is_root: returns correct status" {
    if [ "$(id -u)" -eq 0 ]; then
        is_root
    else
        ! is_root
    fi
}

@test "try_sudo: runs command" {
    result=$(try_sudo echo "hello")
    [ "$result" = "hello" ]
}
