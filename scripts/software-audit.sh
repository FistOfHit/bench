#!/usr/bin/env bash
# software-audit.sh — CUDA, cuDNN, NCCL, MPI, container runtime, fabric manager, etc.
SCRIPT_NAME="software-audit"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== Software Audit ==="

# Helper: find version of a library
find_lib_version() {
    local name="$1"
    # Try ldconfig
    ldconfig -v 2>/dev/null | grep -i "$name" | head -1 | awk '{print $1}' | sed 's/.*\.so\.//'
}

# ── CUDA Toolkit ──
cuda_version="none"
cuda_path="none"
if has_cmd nvcc; then
    cuda_version=$(nvcc --version 2>/dev/null | awk '/release/ {print $NF}' | tr -d ',')
    cuda_path=$(which nvcc | sed 's|/bin/nvcc||')
elif [ -d /usr/local/cuda ]; then
    cuda_version=$(cat /usr/local/cuda/version.txt 2>/dev/null | awk '{print $NF}' || \
                   cat /usr/local/cuda/version.json 2>/dev/null | jq -r '.cuda.version' || echo "detected-noversion")
    cuda_path="/usr/local/cuda"
fi

# ── cuDNN ──
cudnn_version="none"
if [ -f /usr/include/cudnn_version.h ]; then
    major=$(grep '#define CUDNN_MAJOR' /usr/include/cudnn_version.h 2>/dev/null | awk '{print $3}')
    minor=$(grep '#define CUDNN_MINOR' /usr/include/cudnn_version.h 2>/dev/null | awk '{print $3}')
    patch=$(grep '#define CUDNN_PATCHLEVEL' /usr/include/cudnn_version.h 2>/dev/null | awk '{print $3}')
    cudnn_version="${major}.${minor}.${patch}"
elif [ -f /usr/include/cudnn.h ]; then
    major=$(grep '#define CUDNN_MAJOR' /usr/include/cudnn.h 2>/dev/null | awk '{print $3}')
    minor=$(grep '#define CUDNN_MINOR' /usr/include/cudnn.h 2>/dev/null | awk '{print $3}')
    cudnn_version="${major}.${minor}"
fi
# Try dpkg/rpm
if [ "$cudnn_version" = "none" ]; then
    cudnn_version=$(dpkg -l 2>/dev/null | grep -i cudnn | head -1 | awk '{print $3}' || \
                    rpm -qa 2>/dev/null | grep -i cudnn | head -1 | sed 's/.*cudnn-//' || echo "none")
fi

# ── NCCL ──
nccl_version="none"
if [ -f /usr/include/nccl.h ]; then
    major=$(grep '#define NCCL_MAJOR' /usr/include/nccl.h 2>/dev/null | awk '{print $3}')
    minor=$(grep '#define NCCL_MINOR' /usr/include/nccl.h 2>/dev/null | awk '{print $3}')
    patch=$(grep '#define NCCL_PATCH' /usr/include/nccl.h 2>/dev/null | awk '{print $3}')
    nccl_version="${major}.${minor}.${patch}"
fi
if [ "$nccl_version" = "none" ] || [ "$nccl_version" = ".." ]; then
    nccl_version=$(dpkg -l 2>/dev/null | grep libnccl | head -1 | awk '{print $3}' || \
                   rpm -qa 2>/dev/null | grep nccl | head -1 || echo "none")
fi

# ── MPI ──
mpi_json="[]"
mpi_arr=()
if has_cmd mpirun; then
    mpi_ver=$(mpirun --version 2>&1 | head -3 | tr '\n' ' ')
    mpi_arr+=("{\"name\":\"mpirun\",\"version\":\"$mpi_ver\"}")
fi
if has_cmd ompi_info; then
    ompi_ver=$(ompi_info --parsable 2>/dev/null | grep "ompi:version:full" | cut -d: -f4 | head -1)
    mpi_arr+=("{\"name\":\"OpenMPI\",\"version\":\"$ompi_ver\"}")
fi
if has_cmd mpiexec.hydra; then
    mvapich_ver=$(mpiexec.hydra --version 2>&1 | head -1)
    mpi_arr+=("{\"name\":\"MVAPICH/IntelMPI\",\"version\":\"$mvapich_ver\"}")
fi
if [ ${#mpi_arr[@]} -gt 0 ]; then
    mpi_json=$(printf '%s\n' "${mpi_arr[@]}" | jq -s '.' 2>/dev/null) || mpi_json="[]"
else
    mpi_json="[]"
fi

# ── Container Runtime ──
container_json="{}"
if has_cmd docker; then
    docker_ver=$(docker --version 2>/dev/null)
    nvidia_docker=$(docker info 2>/dev/null | grep -c -i "nvidia" || echo 0)
    container_json=$(jq -n --arg v "$docker_ver" --arg nv "$nvidia_docker" \
        '{runtime: "docker", version: $v, nvidia_runtime: ($nv | tonumber > 0)}')
elif has_cmd podman; then
    podman_ver=$(podman --version 2>/dev/null)
    container_json=$(jq -n --arg v "$podman_ver" '{runtime: "podman", version: $v}')
else
    container_json='{"runtime": "none"}'
fi

# ── Fabric Manager ──
fm_status="none"
if has_cmd systemctl; then
    if systemctl is-active nvidia-fabricmanager &>/dev/null; then
        fm_status="active"
    elif systemctl list-unit-files 2>/dev/null | grep -q nvidia-fabricmanager; then
        fm_status="installed-inactive"
    fi
fi

# ── GDRCopy ──
gdrcopy="none"
if has_cmd gdrcopy_sanity; then
    gdrcopy="installed"
elif ldconfig -v 2>/dev/null | grep -q gdrapi; then
    gdrcopy="library-found"
fi

# ── Perf tools ──
perf_tools=()
for tool in perf nsys ncu nvtop htop gpustat; do
    has_cmd "$tool" && perf_tools+=("$tool")
done

RESULT=$(jq -n \
    --arg cuda "$cuda_version" \
    --arg cuda_path "$cuda_path" \
    --arg cudnn "$cudnn_version" \
    --arg nccl "$nccl_version" \
    --argjson mpi "$mpi_json" \
    --argjson container "$container_json" \
    --arg fm "$fm_status" \
    --arg gdrcopy "$gdrcopy" \
    --argjson perf_tools "$(if [ ${#perf_tools[@]} -gt 0 ]; then printf '%s\n' "${perf_tools[@]}" | jq -R . | jq -s '.'; else echo '[]'; fi)" \
    '{
        cuda: {version: $cuda, path: $cuda_path},
        cudnn: $cudnn,
        nccl: $nccl,
        mpi: $mpi,
        container: $container,
        fabric_manager: $fm,
        gdrcopy: $gdrcopy,
        perf_tools: $perf_tools
    }')

echo "$RESULT" | emit_json "software-audit" "ok"
log_ok "Software audit complete"
echo "$RESULT" | jq .
