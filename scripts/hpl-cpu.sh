#!/usr/bin/env bash
# hpl-cpu.sh — HPL (High Performance Linpack) on CPUs
SCRIPT_NAME="hpl-cpu"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== HPL CPU Benchmark ==="

# ── Auto-tune N based on available RAM ──
total_mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
total_mem_gb=$(echo "scale=0; $total_mem_kb / 1048576" | bc)

# Quick mode: tiny problem to verify suite end-to-end
if [ "${HPC_QUICK:-0}" = "1" ]; then
    N=10000
    NB=128
    log_info "Quick mode — tiny HPL CPU N=$N NB=$NB"
else
    # Detect swap — be more conservative without it
    total_swap_kb=$(awk '/SwapTotal/ {print $2}' /proc/meminfo)
    if [ "${total_swap_kb:-0}" -le 0 ] 2>/dev/null; then
        # No swap: cap at 70% to prevent OOM
        mem_pct=0.70
        log_warn "No swap detected — capping HPL memory at 70% to prevent OOM"
    else
        # Swap available: use 75%
        mem_pct=0.75
        log_info "Swap detected (${total_swap_kb} KB) — capping HPL memory at 75%"
    fi
    mem_for_hpl=$(echo "scale=0; $total_mem_kb * 1024 * $mem_pct" | bc)  # bytes
    # N = sqrt(mem_bytes / 8) for double precision
    N=$(echo "scale=0; sqrt($mem_for_hpl / 8)" | bc)
    # Round down to nearest multiple of block size (NB)
    NB=256
    N=$(( (N / NB) * NB ))
fi

NPROCS=$(nproc)
# P x Q grid: try to make it square-ish
P=1; Q=$NPROCS
for ((p=1; p*p<=NPROCS; p++)); do
    if [ $((NPROCS % p)) -eq 0 ]; then
        P=$p; Q=$((NPROCS / p))
    fi
done

log_info "HPL params: N=$N, NB=$NB, P=$P, Q=$Q, procs=$NPROCS, RAM=${total_mem_gb}GB"

# Quick mode: 30s timeout so we skip fast if container not cached / not available; full: base 1800s + 2s per GB of RAM
if [ "${HPC_QUICK:-0}" = "1" ]; then
    HPL_TIMEOUT=30
    HPL_PULL_TIMEOUT=180
else
    HPL_TIMEOUT=$((1800 + total_mem_gb * 2))
    HPL_PULL_TIMEOUT=600
fi
log_info "HPL timeout: ${HPL_TIMEOUT}s (dynamic based on ${total_mem_gb}GB RAM), pull timeout: ${HPL_PULL_TIMEOUT}s"

# ── Try containerized HPL first ──
hpl_output=""
hpl_method="none"
hpl_skip_note="HPL not available"
HPL_CPU_IMAGE="intel/hpckit:latest"
CONTAINER_CMD=""
hpl_image_ready=true

if has_cmd docker && docker info &>/dev/null; then
    CONTAINER_CMD="docker"
elif has_cmd podman && podman info &>/dev/null; then
    CONTAINER_CMD="podman"
fi

if [ -n "$CONTAINER_CMD" ]; then
    log_info "Trying containerized HPL with $CONTAINER_CMD..."
    # Create HPL.dat
    HPL_DAT="${HPC_WORK_DIR}/HPL.dat"
    cat > "$HPL_DAT" <<HPLEOF
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
HPLEOF

    if ! $CONTAINER_CMD images -q "$HPL_CPU_IMAGE" 2>/dev/null | grep -q .; then
        log_info "HPL CPU image not cached; pulling $HPL_CPU_IMAGE..."
        if ! run_with_timeout "$HPL_PULL_TIMEOUT" "hpl-cpu-image-pull" "$CONTAINER_CMD" pull "$HPL_CPU_IMAGE" >/dev/null 2>&1; then
            hpl_skip_note="container runtime available but image pull failed ($HPL_CPU_IMAGE)"
            hpl_image_ready=false
        fi
    fi

    if [ "$hpl_method" = "none" ] && [ "$hpl_image_ready" = true ]; then
        hpl_output=$(run_with_timeout "$HPL_TIMEOUT" "hpl-container" \
            "$CONTAINER_CMD" run --rm -v "${HPC_WORK_DIR}:/work" \
            "$HPL_CPU_IMAGE" bash -c "cd /work && mpirun -np $NPROCS xhpl" 2>&1) && hpl_method="container" || {
                hpl_skip_note="container runtime available but HPL container run failed"
                if echo "$hpl_output" | grep -qi "xhpl: not found"; then
                    hpl_skip_note="container image does not include xhpl binary ($HPL_CPU_IMAGE)"
                fi
            }
    fi
else
    hpl_skip_note="no usable container runtime (docker/podman)"
fi

# ── Fallback: system xhpl if available ──
if [ "$hpl_method" = "none" ] && has_cmd xhpl; then
    if ! has_cmd mpirun && [ "$(id -u)" -eq 0 ] && has_cmd apt-get; then
        log_info "mpirun missing — attempting install: openmpi-bin"
        run_with_timeout 300 "install-openmpi" apt-get install -y openmpi-bin >/dev/null 2>&1 || true
    fi
    if has_cmd mpirun; then
        log_info "Using system xhpl..."
        cd "$HPC_WORK_DIR"
        hpl_output=$(run_with_timeout "$HPL_TIMEOUT" "hpl-system" mpirun --allow-run-as-root -np "$NPROCS" xhpl 2>&1) && hpl_method="system" || {
            hpl_skip_note="system xhpl detected but execution failed"
        }
    else
        hpl_skip_note="system xhpl found but mpirun is unavailable"
    fi
fi

# ── Fallback: hpcc package (CPU-only path) ──
if [ "$hpl_method" = "none" ]; then
    if ! has_cmd hpcc && [ "$(id -u)" -eq 0 ] && has_cmd apt-get; then
        log_info "hpcc missing — attempting install: hpcc"
        run_with_timeout 300 "install-hpcc" apt-get install -y hpcc >/dev/null 2>&1 || true
    fi
    if has_cmd hpcc; then
        hpcc_dir="${HPC_WORK_DIR}/hpcc-run"
        mkdir -p "$hpcc_dir"
        if [ -f /usr/share/doc/hpcc/examples/_hpccinf.txt ]; then
            cp /usr/share/doc/hpcc/examples/_hpccinf.txt "$hpcc_dir/hpccinf.txt"
        fi
        hpl_output=$(run_with_timeout "$HPL_TIMEOUT" "hpl-hpcc" bash -lc "cd \"$hpcc_dir\" && mpirun --allow-run-as-root -np 4 hpcc >/dev/null 2>&1 && cat hpccoutf.txt" 2>&1) && hpl_method="hpcc" || {
            hpl_skip_note="hpcc fallback execution failed"
        }
    fi
fi

# ── Fallback: skip ──
if [ "$hpl_method" = "none" ]; then
    if ! has_cmd xhpl && [ "$hpl_skip_note" = "no usable container runtime (docker/podman)" ]; then
        hpl_skip_note="no usable container runtime and xhpl binary not found"
    fi
    hpl_err_detail=$(printf '%s' "${hpl_output:-}" | awk 'NF{print; exit}' | tr -d '[:cntrl:]')
    # In CI / quick runs, HPL is best-effort: missing container image, blocked egress,
    # or unavailable packages shouldn't fail the entire suite.
    if [ "${HPC_CI:-0}" = "1" ] || [ "${HPC_QUICK:-0}" = "1" ]; then
        log_warn "HPL unavailable; skipping: $hpl_skip_note"
        jq -n --arg note "$hpl_skip_note" --arg detail "$hpl_err_detail" \
            '{note: "HPL unavailable", skip_reason: $note, detail: (if $detail != "" then $detail else null end)}' \
            | emit_json "hpl-cpu" "skipped"
        exit 0
    fi

    log_error "HPL unavailable: $hpl_skip_note"
    jq -n --arg error "$hpl_skip_note" --arg detail "$hpl_err_detail" \
        '{error: $error, detail: (if $detail != "" then $detail else null end)}' | emit_json "hpl-cpu" "error"
    exit 1
fi

# ── Parse results ──
# HPL output: WR11C2R4   N  NB  P  Q  Time  Gflops
gflops=$(echo "$hpl_output" | awk '/WR[0-9]/ {print $NF}' | tail -1)
hpl_time=$(echo "$hpl_output" | awk '/WR[0-9]/ {print $(NF-1)}' | tail -1)
residual=$(echo "$hpl_output" | grep -i "||Ax-b||" | tail -1 | awk '{print $NF}')
# `grep -c` prints a count even when it exits 1 (no matches). Avoid `|| echo 0`
# which would produce `0\n0` and break jq's `tonumber`.
passed=$(echo "$hpl_output" | grep -ci "PASSED" || true)

RESULT=$(jq -n \
    --arg method "$hpl_method" \
    --arg n "$N" \
    --arg nb "$NB" \
    --arg p "$P" \
    --arg q "$Q" \
    --arg nprocs "$NPROCS" \
    --arg gflops "${gflops:-0}" \
    --arg time "${hpl_time:-0}" \
    --arg residual "${residual:-N/A}" \
    --arg passed "$passed" \
    --arg mem "$total_mem_gb" \
    '{
        method: $method,
        problem_size_N: ($n | tonumber),
        block_size_NB: ($nb | tonumber),
        grid_P: ($p | tonumber),
        grid_Q: ($q | tonumber),
        num_processes: ($nprocs | tonumber),
        memory_gb: ($mem | tonumber),
        gflops: ($gflops | tonumber? // 0),
        time_seconds: ($time | tonumber? // 0),
        residual: $residual,
        passed: ($passed | tonumber > 0)
    }')

echo "$RESULT" | emit_json "hpl-cpu" "ok"
log_ok "HPL CPU: ${gflops:-N/A} GFLOPS"
echo "$RESULT" | jq .
