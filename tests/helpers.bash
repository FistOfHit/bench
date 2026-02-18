# helpers.bash -- Shared helpers for BATS test files.
# Load from .bats files: load helpers

# Portable fail helper (not built-in in all bats versions).
_fail() { echo "$1" >&2; return 1; }

# Set up the common test environment variables and directories.
# Call from each .bats file's setup() function.
setup_test_env() {
    export HPC_BENCH_ROOT="${BATS_TEST_DIRNAME}/.."
    export HPC_RESULTS_DIR="$(mktemp -d)"
    export HPC_LOG_DIR="${HPC_RESULTS_DIR}/logs"
    export HPC_WORK_DIR="$(mktemp -d)"
    export SCRIPT_NAME="${1:-test-module}"
    mkdir -p "$HPC_LOG_DIR"
}

# Clean up temp directories. Call from each .bats file's teardown().
teardown_test_env() {
    rm -rf "$HPC_RESULTS_DIR" "$HPC_WORK_DIR"
}

# Source common.sh and disable its EXIT trap / strict mode for bats compat.
# Call after setup_test_env when tests need common.sh functions.
source_common() {
    source "${HPC_BENCH_ROOT}/lib/common.sh"
    # Override common.sh's EXIT trap and strict mode: bats manages its own
    # subshell lifecycle and set -e behavior; common.sh's settings interfere.
    trap - EXIT
    set +euo pipefail
}
