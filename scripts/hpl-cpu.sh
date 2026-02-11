#!/usr/bin/env bash
# hpl-cpu.sh — HPL (High Performance Linpack) on CPUs
SCRIPT_NAME="hpl-cpu"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== HPL CPU Benchmark ==="

# ── Auto-tune N based on available RAM ──
total_mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
total_mem_gb=$(echo "scale=0; $total_mem_kb / 1048576" | bc)

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

NPROCS=$(nproc)
# P x Q grid: try to make it square-ish
P=1; Q=$NPROCS
for ((p=1; p*p<=NPROCS; p++)); do
    if [ $((NPROCS % p)) -eq 0 ]; then
        P=$p; Q=$((NPROCS / p))
    fi
done

log_info "HPL params: N=$N, NB=$NB, P=$P, Q=$Q, procs=$NPROCS, RAM=${total_mem_gb}GB"

# Dynamic timeout: base 1800s (30 min) + 2s per GB of RAM
# A 2TB node gets ~5800s (~97 min), a 256GB node gets ~2300s (~38 min)
HPL_TIMEOUT=$((1800 + total_mem_gb * 2))
log_info "HPL timeout: ${HPL_TIMEOUT}s (dynamic based on ${total_mem_gb}GB RAM)"

# ── Try containerized HPL first ──
hpl_output=""
hpl_method="none"

if has_cmd docker && docker info &>/dev/null; then
    log_info "Trying containerized HPL..."
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

    hpl_output=$(run_with_timeout "$HPL_TIMEOUT" "hpl-container" \
        docker run --rm -v "${HPC_WORK_DIR}:/work" \
        intel/hpckit:latest bash -c "cd /work && mpirun -np $NPROCS xhpl" 2>&1) && hpl_method="container" || true
fi

# ── Fallback: system HPL ──
if [ "$hpl_method" = "none" ] && has_cmd xhpl; then
    log_info "Using system xhpl..."
    cd "$HPC_WORK_DIR"
    hpl_output=$(run_with_timeout "$HPL_TIMEOUT" "hpl-system" mpirun -np "$NPROCS" xhpl 2>&1) && hpl_method="system" || true
fi

# ── Fallback: skip ──
if [ "$hpl_method" = "none" ]; then
    log_warn "HPL not available (no container or xhpl binary)"
    echo '{"note":"HPL not available"}' | emit_json "hpl-cpu" "skipped"
    exit 0
fi

# ── Parse results ──
# HPL output: WR11C2R4   N  NB  P  Q  Time  Gflops
gflops=$(echo "$hpl_output" | awk '/WR[0-9]/ {print $NF}' | tail -1)
hpl_time=$(echo "$hpl_output" | awk '/WR[0-9]/ {print $(NF-1)}' | tail -1)
residual=$(echo "$hpl_output" | grep -i "||Ax-b||" | tail -1 | awk '{print $NF}')
passed=$(echo "$hpl_output" | grep -ci "PASSED" || echo 0)

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
