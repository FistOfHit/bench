# defaults.sh — Central configuration for HPC bench suite thresholds and tunables
# Sourced by lib/common.sh. Do not run standalone.
#
# Override any value by exporting the variable before running the suite, e.g.:
#   export GPU_THERMAL_WARN_C=90
#   sudo bash scripts/run-all.sh
#
# Or create a conf/local.sh file (gitignored) that exports overrides.
# ─────────────────────────────────────────────────────────────────────

# ── Orchestrator ──
: "${MAX_MODULE_TIME_QUICK:=600}"      # Per-module timeout in quick mode (seconds)
: "${MAX_MODULE_TIME_FULL:=1800}"      # Per-module timeout in full mode (seconds)

# ── GPU Burn ──
: "${GPU_BURN_DURATION_QUICK:=10}"     # Quick mode burn (seconds, ≥10 for GFLOPS)
: "${GPU_BURN_DURATION_VM:=60}"        # VM burn duration (seconds)
: "${GPU_BURN_DURATION_FULL:=300}"     # Bare-metal burn duration (seconds)

# ── NCCL Tests ──
: "${NCCL_TIMEOUT_QUICK:=45}"         # Timeout per test in quick mode
: "${NCCL_TIMEOUT_FULL:=600}"         # Timeout per test in full mode

# ── HPL-MxP ──
: "${HPL_MXP_TIMEOUT_QUICK:=120}"     # Quick mode timeout (seconds)
: "${HPL_MXP_GPU_MEM_FRACTION:=0.8}"  # Fraction of GPU memory to use for problem size

# ── Thermal / Power ──
: "${GPU_THERMAL_WARN_C:=85}"         # GPU temperature warning threshold (°C)
: "${GPU_THERMAL_CRIT_C:=95}"         # GPU temperature critical threshold (°C)

# ── Storage Bench ──
: "${STORAGE_BENCH_DURATION_QUICK:=5}" # fio runtime per profile in quick mode (seconds)
: "${STORAGE_BENCH_DURATION_FULL:=60}" # fio runtime per profile in full mode (seconds)

# ── STREAM Bench ──
: "${STREAM_ARRAY_SIZE_QUICK:=1000000}"   # Array elements in quick mode
: "${STREAM_ITERS_QUICK:=3}"              # Iterations in quick mode
: "${STREAM_ITERS_FULL:=20}"              # Iterations in full mode

# ── Container images ──
: "${HPL_IMAGE:=nvcr.io/nvidia/hpc-benchmarks:24.03}"
: "${HPL_IMAGE_ALT:=nvcr.io/nvidia/hpc-benchmarks:23.10}"

# ── Load local overrides if present (gitignored) ──
_local_conf="${HPC_BENCH_ROOT}/conf/local.sh"
if [ -f "$_local_conf" ]; then
    # shellcheck source=/dev/null
    source "$_local_conf"
fi
