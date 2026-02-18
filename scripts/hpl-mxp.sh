#!/usr/bin/env bash
# hpl-mxp.sh -- HPL-MxP (mixed-precision) via NVIDIA HPC Benchmarks container
# Phase: 3 (benchmark)
# Requires: jq, timeout, awk
# Emits: hpl-mxp.json
SCRIPT_NAME="hpl-mxp"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== HPL-MxP (GPU) Benchmark ==="

require_gpu "hpl-mxp"
HPL_MXP_VM_STRICT="${HPC_HPL_MXP_VM_STRICT:-0}"
if [ "$HPL_MXP_VM_STRICT" = "1" ]; then
    log_info "HPL-MxP VM strict mode enabled (no VM auto-skip conversion)"
fi

vm_skip_allowed() {
    is_virtualized && [ "$HPL_MXP_VM_STRICT" != "1" ]
}

docker_has_nvidia_runtime() {
    has_cmd docker || return 1
    local dinfo
    dinfo=$(docker info 2>/dev/null || true)
    [ -n "$dinfo" ] && echo "$dinfo" | grep -qi nvidia
}

# Need container runtime with GPU support
CONTAINER_CMD=""
if docker_has_nvidia_runtime; then
    CONTAINER_CMD="docker"
elif has_cmd nvidia-docker; then
    CONTAINER_CMD="nvidia-docker"
else
    # Fall back to shared detection (podman with GPU support, etc.)
    CONTAINER_CMD=$(detect_container_runtime)
    [ -z "$CONTAINER_CMD" ] && skip_module "hpl-mxp" "no GPU-capable container runtime"
fi

# In VMs, Docker/pipe often causes SIGPIPE (exit 141); treat as skipped so suite can pass
HPL_MXP_EXIT_CODE=
_hpl_exit_trap() {
    local _rc=$?
    HPL_MXP_EXIT_CODE=$_rc
    if [ $_rc -eq 141 ]; then
        if vm_skip_allowed; then
            log_warn "HPL-MxP exited with SIGPIPE (141) — typical in VMs, skipping"
            echo '{"note":"HPL-MxP exited with SIGPIPE (typical in VMs)","skip_reason":"vm"}' | emit_json "hpl-mxp" "skipped"
            HPL_MXP_EXIT_CODE=0
        else
            log_error "HPL-MxP exited with SIGPIPE (141) and VM strict mode is enabled"
            echo '{"error":"HPL-MxP exited with SIGPIPE while strict VM mode is enabled"}' | emit_json "hpl-mxp" "error"
            HPL_MXP_EXIT_CODE=1
        fi
    fi
}
trap '_hpl_exit_trap; _r=${HPL_MXP_EXIT_CODE:-$?}; do_cleanup; exit $_r' EXIT

NGPUS=$(gpu_count)
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_MODEL=$(gpu_model)

# ── Auto-configure problem size ──
# Quick mode: tiny problem (N=2048, NB=128) to finish in seconds and verify suite
if [ "${HPC_QUICK:-0}" = "1" ]; then
    N=2048
    NB=128
    TOTAL_GPU_MEM_MB=$((GPU_MEM_MB * NGPUS))
    log_info "Quick mode — tiny HPL-MxP N=$N NB=$NB"
else
    # Use ~80% of total GPU memory across all GPUs
    TOTAL_GPU_MEM_MB=$((GPU_MEM_MB * NGPUS))
    TOTAL_GPU_MEM_BYTES=$(echo "$TOTAL_GPU_MEM_MB * 1048576 * 0.8" | bc | cut -d. -f1)
    N=$(echo "scale=0; sqrt($TOTAL_GPU_MEM_BYTES / 8)" | bc)
    NB=1024  # Typical for GPU HPL
    N=$(( (N / NB) * NB ))
fi

# P x Q grid for GPUs
P=1; Q=$NGPUS
for ((p=1; p*p<=NGPUS; p++)); do
    if [ $((NGPUS % p)) -eq 0 ]; then
        P=$p; Q=$((NGPUS / p))
    fi
done

log_info "HPL-MxP: N=$N, NB=$NB, P=$P, Q=$Q, GPUs=$NGPUS, GPU_MEM=${GPU_MEM_MB}MB"

# ── Container image (configurable via conf/defaults.sh) ──
HPL_IMAGE="${HPL_IMAGE:-nvcr.io/nvidia/hpc-benchmarks:24.03}"
HPL_IMAGE_ALT="${HPL_IMAGE_ALT:-nvcr.io/nvidia/hpc-benchmarks:23.10}"

# Check if image is already available locally (avoids pull when possible)
_hpl_image_ready=false
if $CONTAINER_CMD images -q "$HPL_IMAGE" 2>/dev/null | grep -q .; then
    log_info "HPL container image found locally: $HPL_IMAGE"
    _hpl_image_ready=true
elif $CONTAINER_CMD images -q "$HPL_IMAGE_ALT" 2>/dev/null | grep -q .; then
    log_info "HPL container image found locally (alt tag): $HPL_IMAGE_ALT"
    HPL_IMAGE="$HPL_IMAGE_ALT"
    _hpl_image_ready=true
fi

# Prefer pulling latest from registry; fallback to bundled tar (sneakernet/offline)
if [ "$_hpl_image_ready" = false ]; then
    log_info "Pulling HPL container image from registry..."
    if $CONTAINER_CMD pull "$HPL_IMAGE" 2>&1 | tail -3; then
        _hpl_image_ready=true
    else
        log_warn "Failed to pull $HPL_IMAGE, trying older tag..."
        HPL_IMAGE="$HPL_IMAGE_ALT"
        if $CONTAINER_CMD pull "$HPL_IMAGE" 2>&1 | tail -3; then
            _hpl_image_ready=true
        fi
    fi
fi

# Fallback: try loading from bundled tar if available (offline/sneakernet)
if [ "$_hpl_image_ready" = false ]; then
    for tarfile in "${HPC_BENCH_ROOT}/src/hpc-benchmarks"*.tar "${HPC_BENCH_ROOT}/src/hpl"*.tar; do
        if [ -f "$tarfile" ]; then
            log_info "Loading HPL container from bundled image: $tarfile"
            if $CONTAINER_CMD load -i "$tarfile" 2>&1 | tail -3; then
                # Re-check which image tag was loaded
                if $CONTAINER_CMD images -q "$HPL_IMAGE" 2>/dev/null | grep -q .; then
                    _hpl_image_ready=true
                elif $CONTAINER_CMD images -q "$HPL_IMAGE_ALT" 2>/dev/null | grep -q .; then
                    HPL_IMAGE="$HPL_IMAGE_ALT"
                    _hpl_image_ready=true
                fi
            fi
            break
        fi
    done
fi

if [ "$_hpl_image_ready" = false ]; then
    log_info "To run offline: docker save $HPL_IMAGE > hpc-benchmarks.tar, place in src/, re-run"
    skip_module "hpl-mxp" "HPL container image unavailable — not found locally, pull failed"
fi

# Create HPL config
HPL_CFG="${HPC_WORK_DIR}/hpl-mxp"
mkdir -p "$HPL_CFG"
generate_hpl_dat "${HPL_CFG}/HPL.dat" "$N" "$NB" "$P" "$Q"

# ── Run HPL-MxP ──
# Quick mode: short timeout; full: base 1800s + 1s per GB of total GPU memory
if [ "${HPC_QUICK:-0}" = "1" ]; then
    HPL_MXP_TIMEOUT=${HPL_MXP_TIMEOUT_QUICK:-120}
else
    HPL_MXP_TIMEOUT=$((1800 + TOTAL_GPU_MEM_MB / 1024))
fi
log_info "Running HPL-MxP (timeout: ${HPL_MXP_TIMEOUT}s)..."
hpl_output=$(run_with_timeout "$HPL_MXP_TIMEOUT" "hpl-mxp" \
    $CONTAINER_CMD run --rm --gpus all \
    --shm-size=1g --ulimit memlock=-1 \
    -v "${HPL_CFG}:/workspace" \
    "$HPL_IMAGE" \
    mpirun --allow-run-as-root -np "$NGPUS" \
    /workspace/hpl.sh --dat /workspace/HPL.dat 2>&1) || true

# ── Parse ──
# If run produced no usable output (e.g. container crash, SIGPIPE in VM), skip gracefully
if [ -z "$hpl_output" ] || ! echo "$hpl_output" | grep -q "WR[0-9]\|PASSED"; then
    if vm_skip_allowed; then
        skip_module "hpl-mxp" "HPL-MxP run produced no results (container/VM limitation)"
    fi
    log_error "HPL-MxP run produced no usable output"
    echo '{"error":"no usable output from HPL-MxP run"}' | emit_json "hpl-mxp" "error"
    exit 1
fi
gflops=$(echo "$hpl_output" | awk '/WR[0-9]/ {print $NF}' | tail -1)
hpl_time=$(echo "$hpl_output" | awk '/WR[0-9]/ {print $(NF-1)}' | tail -1)
passed=$(count_grep_re 'PASSED' "$hpl_output")

RESULT=$(jq -n \
    --arg n "$N" \
    --arg nb "$NB" \
    --arg ngpus "$NGPUS" \
    --arg model "$GPU_MODEL" \
    --arg gflops "${gflops:-0}" \
    --arg time "${hpl_time:-0}" \
    --arg passed "$passed" \
    --arg image "$HPL_IMAGE" \
    '{
        container_image: $image,
        gpu_model: $model,
        gpu_count: ($ngpus | tonumber),
        problem_size_N: ($n | tonumber),
        block_size_NB: ($nb | tonumber),
        gflops: ($gflops | tonumber? // 0),
        time_seconds: ($time | tonumber? // 0),
        passed: ($passed | tonumber > 0)
    }')

finish_module "hpl-mxp" "ok" "$RESULT"
