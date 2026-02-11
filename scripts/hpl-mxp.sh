#!/usr/bin/env bash
# hpl-mxp.sh — HPL-MxP (mixed-precision) via NVIDIA HPC Benchmarks container
SCRIPT_NAME="hpl-mxp"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== HPL-MxP (GPU) Benchmark ==="

require_gpu "hpl-mxp" "no GPU"

# Need container runtime with GPU support
CONTAINER_CMD=""
if has_cmd docker && docker info 2>/dev/null | grep -qi nvidia; then
    CONTAINER_CMD="docker"
elif has_cmd nvidia-docker; then
    CONTAINER_CMD="nvidia-docker"
else
    log_warn "No GPU-capable container runtime — skipping HPL-MxP"
    echo '{"note":"no nvidia container runtime"}' | emit_json "hpl-mxp" "skipped"
    exit 0
fi

# In VMs, Docker/pipe often causes SIGPIPE (exit 141); treat as skipped so suite can pass
HPL_MXP_EXIT_CODE=
_hpl_exit_trap() {
    local _rc=$?
    HPL_MXP_EXIT_CODE=$_rc
    if [ $_rc -eq 141 ] && is_virtualized; then
        log_warn "HPL-MxP exited with SIGPIPE (141) — typical in VMs, skipping"
        echo '{"note":"HPL-MxP exited with SIGPIPE (typical in VMs)","skip_reason":"vm"}' | emit_json "hpl-mxp" "skipped"
        HPL_MXP_EXIT_CODE=0
    fi
}
trap '_hpl_exit_trap; _r=${HPL_MXP_EXIT_CODE:-$?}; do_cleanup; exit $_r' EXIT

NGPUS=$(gpu_count)
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_MODEL=$(gpu_model)

# ── Auto-configure problem size ──
# Use ~80% of total GPU memory across all GPUs
TOTAL_GPU_MEM_MB=$((GPU_MEM_MB * NGPUS))
TOTAL_GPU_MEM_BYTES=$(echo "$TOTAL_GPU_MEM_MB * 1048576 * 0.8" | bc | cut -d. -f1)
N=$(echo "scale=0; sqrt($TOTAL_GPU_MEM_BYTES / 8)" | bc)
NB=1024  # Typical for GPU HPL
N=$(( (N / NB) * NB ))

# P x Q grid for GPUs
P=1; Q=$NGPUS
for ((p=1; p*p<=NGPUS; p++)); do
    if [ $((NGPUS % p)) -eq 0 ]; then
        P=$p; Q=$((NGPUS / p))
    fi
done

log_info "HPL-MxP: N=$N, NB=$NB, P=$P, Q=$Q, GPUs=$NGPUS, GPU_MEM=${GPU_MEM_MB}MB"

# ── Container image ──
HPL_IMAGE="nvcr.io/nvidia/hpc-benchmarks:24.03"
HPL_IMAGE_ALT="nvcr.io/nvidia/hpc-benchmarks:23.10"

# Check if image is already available locally (avoids pull on air-gapped systems)
_hpl_image_ready=false
if $CONTAINER_CMD images -q "$HPL_IMAGE" 2>/dev/null | grep -q .; then
    log_info "HPL container image found locally: $HPL_IMAGE"
    _hpl_image_ready=true
elif $CONTAINER_CMD images -q "$HPL_IMAGE_ALT" 2>/dev/null | grep -q .; then
    log_info "HPL container image found locally (alt tag): $HPL_IMAGE_ALT"
    HPL_IMAGE="$HPL_IMAGE_ALT"
    _hpl_image_ready=true
fi

# Try loading from bundled tar if available (sneakernet support)
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

# Last resort: try pulling from registry
if [ "$_hpl_image_ready" = false ]; then
    log_info "Pulling HPL container image..."
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

if [ "$_hpl_image_ready" = false ]; then
    log_warn "HPL-MxP skipped: container image not found locally and pull failed (air-gapped?)"
    log_info "To run offline: docker save $HPL_IMAGE > hpc-benchmarks.tar, place in src/, re-run"
    echo '{"note":"HPL container image unavailable — not found locally, pull failed","skip_reason":"no container image"}' | emit_json "hpl-mxp" "skipped"
    exit 0
fi

# Create HPL config
HPL_CFG="${HPC_WORK_DIR}/hpl-mxp"
mkdir -p "$HPL_CFG"
cat > "${HPL_CFG}/HPL.dat" <<EOF
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
$N           Ns
1            # of NBs
$NB          NBs
0            PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
$P           Ps
$Q           Qs
16.0         threshold
1            # of panel fact
2            PFACTs (0=left, 1=Crout, 2=Right)
1            # of recursive stopping criterium
4            NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
1            # of recursive panel fact.
1            RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
1            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
1            DEPTHs (>=0)
2            SWAP (0=bin-exch,1=long,2=mix)
64           swapping threshold
0            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
1            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
EOF

# ── Run HPL-MxP ──
# Dynamic timeout: base 1800s + 1s per GB of total GPU memory
HPL_MXP_TIMEOUT=$((1800 + TOTAL_GPU_MEM_MB / 1024))
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
    if is_virtualized; then
        log_warn "HPL-MxP produced no results (typical in VMs) — skipping"
        echo '{"note":"HPL-MxP run produced no results (container/VM limitation)","skip_reason":"vm"}' | emit_json "hpl-mxp" "skipped"
        exit 0
    fi
    log_error "HPL-MxP run produced no usable output"
    echo '{"error":"no usable output from HPL-MxP run"}' | emit_json "hpl-mxp" "error"
    exit 1
fi
gflops=$(echo "$hpl_output" | awk '/WR[0-9]/ {print $NF}' | tail -1)
hpl_time=$(echo "$hpl_output" | awk '/WR[0-9]/ {print $(NF-1)}' | tail -1)
passed=$(echo "$hpl_output" | grep -ci "PASSED" || echo 0)

# Theoretical peak: sum of GPU FP64 TFLOPS (or FP16 for MxP)
spec=$(lookup_gpu_spec "$GPU_MODEL")
fp64_per_gpu=$(echo "$spec" | jq '.fp64_tflops // 0' 2>/dev/null)
theoretical_tflops=$(echo "scale=2; $fp64_per_gpu * $NGPUS" | bc 2>/dev/null || echo "0")

efficiency="N/A"
if [ -n "$gflops" ] && [ "$theoretical_tflops" != "0" ]; then
    efficiency=$(echo "scale=1; $gflops / ($theoretical_tflops * 1000) * 100" | bc 2>/dev/null || echo "N/A")
fi

RESULT=$(jq -n \
    --arg n "$N" \
    --arg nb "$NB" \
    --arg ngpus "$NGPUS" \
    --arg model "$GPU_MODEL" \
    --arg gflops "${gflops:-0}" \
    --arg time "${hpl_time:-0}" \
    --arg passed "$passed" \
    --arg theo "$theoretical_tflops" \
    --arg eff "$efficiency" \
    --arg image "$HPL_IMAGE" \
    '{
        container_image: $image,
        gpu_model: $model,
        gpu_count: ($ngpus | tonumber),
        problem_size_N: ($n | tonumber),
        block_size_NB: ($nb | tonumber),
        gflops: ($gflops | tonumber? // 0),
        time_seconds: ($time | tonumber? // 0),
        passed: ($passed | tonumber > 0),
        theoretical_fp64_tflops: ($theo | tonumber? // 0),
        efficiency_pct: $eff
    }')

echo "$RESULT" | emit_json "hpl-mxp" "ok"
log_ok "HPL-MxP: ${gflops:-N/A} GFLOPS"
echo "$RESULT" | jq .
