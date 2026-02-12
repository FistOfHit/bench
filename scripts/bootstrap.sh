#!/usr/bin/env bash
# bootstrap.sh — Entry point: detect hardware, install missing tools, set up environment
# Idempotent — safe to run multiple times
SCRIPT_NAME="bootstrap"

# ── Parse flags (before sourcing common.sh) ──
CHECK_ONLY=false
INSTALL_NVIDIA=false
for arg in "$@"; do
    case "$arg" in
        --check-only) CHECK_ONLY=true ;;
        --install-nvidia) INSTALL_NVIDIA=true ;;
    esac
done

# ── Ensure jq and curl so common.sh and connectivity check can run (root + package manager only) ──
if [ "$(id -u)" -eq 0 ]; then
    for _pkg in jq curl; do
        if ! command -v "$_pkg" &>/dev/null; then
            if command -v apt-get &>/dev/null; then
                apt-get update -qq 2>/dev/null; apt-get install -y "$_pkg" 2>/dev/null || true
            elif command -v dnf &>/dev/null; then
                dnf install -y "$_pkg" 2>/dev/null || true
            elif command -v yum &>/dev/null; then
                yum install -y "$_pkg" 2>/dev/null || true
            fi
        fi
    done
fi

source "$(dirname "$0")/../lib/common.sh"

# ── Detect virtualization early ──
VIRT_INFO=$(detect_virtualization)
VIRT_TYPE=$(echo "$VIRT_INFO" | jq -r '.type')

log_info "=== HPC Bench Bootstrap ==="
log_info "Host: $(hostname), Date: $(date -u)"
if [ "$CHECK_ONLY" = true ]; then
    log_info "Mode: CHECK-ONLY (dry run — no installs)"
fi

# ── Detect OS ──
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_ID="${ID:-unknown}"
    OS_VERSION="${VERSION_ID:-unknown}"
else
    OS_ID="unknown"
    OS_VERSION="unknown"
fi
log_info "OS: $OS_ID $OS_VERSION"

PKG_MGR=$(detect_pkg_manager)
log_info "Package manager: $PKG_MGR"

# ── Check root (skip for check-only) ──
if [ "$CHECK_ONLY" = false ] && [ "$(id -u)" -ne 0 ]; then
    log_error "Must run as root. Use sudo."
    exit 1
fi

# ── Core tools list ──
CORE_TOOLS=(jq bc curl dmidecode lshw pciutils util-linux numactl hwloc smartmontools ethtool sysstat)
case "$PKG_MGR" in
    apt)
        CORE_TOOLS+=(gnupg ipmitool fio net-tools build-essential linux-tools-common)
        KVER=$(uname -r)
        if [ "$CHECK_ONLY" = false ] && apt-cache show "linux-tools-${KVER}" &>/dev/null; then
            CORE_TOOLS+=("linux-tools-${KVER}")
        fi
        ;;
    dnf|yum)
        CORE_TOOLS+=(gnupg2 ipmitool fio net-tools gcc gcc-c++ make kernel-tools perf)
        ;;
esac

# ── Check-only mode: report and exit ──
if [ "$CHECK_ONLY" = true ]; then
    log_info "Checking dependencies..."
    PRESENT=()
    MISSING=()

    # Check core packages
    for tool in "${CORE_TOOLS[@]}"; do
        if dpkg -l "$tool" &>/dev/null 2>&1 || rpm -q "$tool" &>/dev/null 2>&1; then
            PRESENT+=("$tool")
        else
            MISSING+=("$tool")
        fi
    done

    # Check key binaries
    KEY_BINS=(python3 git cmake gcc docker nvidia-smi dcgmi ibstat ib_write_bw xhpl mpirun)
    for bin in "${KEY_BINS[@]}"; do
        if has_cmd "$bin"; then
            PRESENT+=("bin:$bin")
        else
            MISSING+=("bin:$bin")
        fi
    done

    # Report
    echo ""
    echo "═══════════════════════════════════════════"
    echo "  Dependency Check Report"
    echo "═══════════════════════════════════════════"
    echo ""
    echo "PRESENT (${#PRESENT[@]}):"
    for item in "${PRESENT[@]}"; do
        echo "  ✓ $item"
    done
    echo ""
    echo "MISSING (${#MISSING[@]}):"
    if [ ${#MISSING[@]} -eq 0 ]; then
        echo "  (none — all dependencies satisfied)"
    else
        for item in "${MISSING[@]}"; do
            echo "  ✗ $item"
        done
    fi
    echo ""
    echo "═══════════════════════════════════════════"

    # Emit JSON
    PRESENT_JSON=$(printf '%s\n' "${PRESENT[@]}" | jq -R . | jq -s .)
    MISSING_JSON=$(printf '%s\n' "${MISSING[@]}" | jq -R . | jq -s .)
    jq -n --argjson present "$PRESENT_JSON" --argjson missing "$MISSING_JSON" \
        '{check_only: true, present: $present, missing: $missing, present_count: ($present | length), missing_count: ($missing | length)}' \
        | emit_json "bootstrap" "$([ ${#MISSING[@]} -gt 0 ] && echo "warn" || echo "ok")"
    exit 0
fi

# ── Connectivity preflight ──
# Tries multiple hosts in case one is blocked by firewall/proxy. Under sudo, proxy env (HTTP_PROXY/HTTPS_PROXY) may not be set.
HAS_INTERNET=false
log_info "Checking internet/repository connectivity..."
for _url in https://google.com https://archive.ubuntu.com https://mirror.centos.org https://cloudflare.com https://1.1.1.1; do
    if curl -sI --connect-timeout 5 "$_url" &>/dev/null; then
        HAS_INTERNET=true
        log_info "Internet connectivity: OK (reachable: $_url)"
        break
    fi
done
if [ "$HAS_INTERNET" = false ]; then
    log_warn "No internet/repository access — will skip package installation"
    log_warn "Pre-install packages manually or use --check-only to see what's needed"
    # Show why (e.g. DNS vs timeout); use a short timeout so we don't hang
    _err=$(curl -sI --connect-timeout 3 https://archive.ubuntu.com 2>&1) || true
    if [ -n "$_err" ]; then
        log_info "Connectivity check hint: $(echo "$_err" | head -1)"
    fi
fi

if [ "$HAS_INTERNET" = true ]; then
    # ── Update package cache ──
    log_info "Updating package cache..."
    pkg_update 2>&1 | tail -5

    # ── EPEL for RHEL-family ──
    case "$PKG_MGR" in
        dnf|yum)
            if ! rpm -q epel-release &>/dev/null; then
                $PKG_MGR install -y epel-release 2>/dev/null || true
            fi
            ;;
    esac

    log_info "Installing core tools..."
    for tool in "${CORE_TOOLS[@]}"; do
        if ! dpkg -l "$tool" &>/dev/null && ! rpm -q "$tool" &>/dev/null; then
            pkg_install "$tool" 2>/dev/null || log_warn "Failed to install: $tool (non-fatal)"
        fi
    done

    # ── Python3 + pip ──
    if ! has_cmd python3; then
        log_info "Installing python3..."
        pkg_install python3 python3-pip 2>/dev/null || true
    fi
else
    log_info "Skipping package installation (no internet). Checking existing tools..."
fi

# ── NVIDIA GPU detection (PCI: no driver required) ──
# Prefer lspci; fallback to /sys (vendor 0x10de = NVIDIA) when lspci missing or PATH limited under sudo.
NVIDIA_GPU_PRESENT=false
if (command -v lspci &>/dev/null && lspci 2>/dev/null | grep -qi 'nvidia\|10de') || \
   grep -ql '0x10de' /sys/bus/pci/devices/*/vendor 2>/dev/null; then
    NVIDIA_GPU_PRESENT=true
fi

# ── NVIDIA tools check (driver working) ──
HAS_GPU=false
if has_cmd nvidia-smi; then
    if nvidia-smi &>/dev/null; then
        HAS_GPU=true
        GPU_MODEL=$(gpu_model)
        GPU_N=$(gpu_count)
        log_info "GPUs detected: ${GPU_N}x ${GPU_MODEL}"

        # Enable persistence mode
        nvidia-smi -pm 1 2>/dev/null || log_warn "Could not enable persistence mode"
    else
        log_warn "nvidia-smi found but not working — driver issue?"
    fi
else
    if [ "$NVIDIA_GPU_PRESENT" = true ]; then
        log_warn "nvidia-smi not found — NVIDIA GPU present but driver not installed"
    else
        log_warn "nvidia-smi not found — no NVIDIA GPU support"
    fi
fi

# ── --install-nvidia: install driver when GPU present but no working driver (Ubuntu, internet required) ──
if [ "$INSTALL_NVIDIA" = true ]; then
    if [ "$NVIDIA_GPU_PRESENT" = true ] && [ "$HAS_GPU" = false ] && [ "$HAS_INTERNET" = true ] && [ "$PKG_MGR" = "apt" ]; then
        log_info "Installing NVIDIA driver (--install-nvidia, Ubuntu)..."
        # Add NVIDIA CUDA repo for driver + toolkit
        if ! ls /etc/apt/sources.list.d/*cuda* 2>/dev/null | head -1 | grep -q .; then
            DCGM_KEY="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(echo "${OS_VERSION:-22.04}" | tr -d '.')/x86_64/3bf863cc.pub"
            curl -fsSL "$DCGM_KEY" 2>/dev/null | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg 2>/dev/null || true
            echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(echo "${OS_VERSION:-22.04}" | tr -d '.')/x86_64/ /" \
                > /etc/apt/sources.list.d/cuda-ubuntu.list 2>/dev/null || true
            apt-get update 2>/dev/null || true
        fi
        if apt-get install -y nvidia-driver-580-server 2>/dev/null; then
            log_ok "NVIDIA driver installed."
            echo ""
            echo "  Reboot required for the driver to load. After reboot, run:"
            echo "    sudo bash scripts/bootstrap.sh --install-nvidia"
            echo "  Then run the full suite as usual."
            echo ""
            exit 0
        else
            log_warn "NVIDIA driver install failed (try: apt-get install -y nvidia-driver-580-server)"
        fi
    else
        # --install-nvidia was requested but we didn't install the driver — explain why and optionally install CUDA toolkit
        if [ "$NVIDIA_GPU_PRESENT" = false ]; then
            log_warn "Skipping NVIDIA driver install: no NVIDIA GPU detected on this system (lspci)."
            log_info "Use --install-nvidia on a machine with an NVIDIA GPU to install the driver."
            if [ "$HAS_INTERNET" = true ] && [ "$PKG_MGR" = "apt" ] && ! command -v nvcc &>/dev/null; then
                log_info "Installing CUDA toolkit only (nvcc, libs) for development / future GPU..."
                if ! ls /etc/apt/sources.list.d/*cuda* 2>/dev/null | head -1 | grep -q .; then
                    DCGM_KEY="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(echo "${OS_VERSION:-22.04}" | tr -d '.')/x86_64/3bf863cc.pub"
                    curl -fsSL "$DCGM_KEY" 2>/dev/null | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg 2>/dev/null || true
                    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(echo "${OS_VERSION:-22.04}" | tr -d '.')/x86_64/ /" \
                        > /etc/apt/sources.list.d/cuda-ubuntu.list 2>/dev/null || true
                    apt-get update 2>/dev/null || true
                fi
                if apt-get install -y cuda-toolkit-12-2 2>/dev/null || apt-get install -y cuda-toolkit-12-0 2>/dev/null; then
                    log_ok "CUDA toolkit installed (nvcc). Add /usr/local/cuda/bin to PATH when using a GPU."
                else
                    log_warn "CUDA toolkit install failed (repo or network)."
                fi
            fi
        elif [ "$HAS_GPU" = true ]; then
            log_info "NVIDIA driver already working (nvidia-smi OK). Skipping driver install."
        elif [ "$HAS_INTERNET" = false ]; then
            log_warn "Skipping NVIDIA install: no internet. Connect and re-run with --install-nvidia."
        elif [ "$PKG_MGR" != "apt" ]; then
            log_warn "Skipping NVIDIA driver install: only supported for apt (Debian/Ubuntu). Install driver manually."
        fi
    fi
fi

# ── DCGM ──
if [ "$HAS_GPU" = true ] && ! has_cmd dcgmi && [ "$HAS_INTERNET" = true ]; then
    log_info "Attempting DCGM install..."
    case "$PKG_MGR" in
        apt)
            # Check if CUDA repo already exists (avoid creating duplicates with conflicting signing)
            if ! ls /etc/apt/sources.list.d/*cuda* 2>/dev/null | head -1 | grep -q .; then
                DCGM_KEY="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(echo $OS_VERSION | tr -d '.')/x86_64/3bf863cc.pub"
                curl -fsSL "$DCGM_KEY" 2>/dev/null | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg 2>/dev/null || true
                echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(echo $OS_VERSION | tr -d '.')/x86_64/ /" \
                    > /etc/apt/sources.list.d/datacenter-gpu-manager.list 2>/dev/null || true
                apt-get update 2>/dev/null || true
            else
                log_info "CUDA repo already configured — skipping DCGM repo addition"
            fi
            apt-get install -y datacenter-gpu-manager 2>/dev/null || log_warn "DCGM install failed (non-fatal)"
            ;;
        dnf|yum)
            $PKG_MGR install -y datacenter-gpu-manager 2>/dev/null || log_warn "DCGM install failed (non-fatal)"
            ;;
    esac
    if has_cmd nv-hostengine; then
        nv-hostengine 2>/dev/null || true
    fi
fi

# ── CUDA toolkit (nvcc) when driver works but nvcc missing (Ubuntu: from same repo as DCGM) ──
if [ "$HAS_GPU" = true ] && [ "$HAS_INTERNET" = true ] && ! command -v nvcc &>/dev/null && [ "$PKG_MGR" = "apt" ]; then
    log_info "Installing CUDA toolkit (nvcc) for GPU benchmarks..."
    if ! ls /etc/apt/sources.list.d/*cuda* 2>/dev/null | head -1 | grep -q .; then
        DCGM_KEY="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(echo "${OS_VERSION:-22.04}" | tr -d '.')/x86_64/3bf863cc.pub"
        curl -fsSL "$DCGM_KEY" 2>/dev/null | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg 2>/dev/null || true
        echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(echo "${OS_VERSION:-22.04}" | tr -d '.')/x86_64/ /" \
            > /etc/apt/sources.list.d/cuda-ubuntu.list 2>/dev/null || true
        apt-get update 2>/dev/null || true
    fi
    # Prefer cuda-toolkit-12-2 for broad compatibility; fallback to 12-0
    if apt-get install -y cuda-toolkit-12-2 2>/dev/null; then
        log_ok "CUDA toolkit installed (cuda-toolkit-12-2)"
    elif apt-get install -y cuda-toolkit-12-0 2>/dev/null; then
        log_ok "CUDA toolkit installed (cuda-toolkit-12-0)"
    else
        log_warn "CUDA toolkit install failed — gpu-burn/nccl-tests may need nvcc; ensure PATH includes /usr/local/cuda/bin"
    fi
fi

# ── NCCL (enables nccl-tests on systems without pre-installed NCCL) ──
if [ "$HAS_GPU" = true ] && [ "$HAS_INTERNET" = true ]; then
    log_info "Attempting NCCL library install..."
    case "$PKG_MGR" in
        apt)
            apt-get install -y libnccl2 libnccl-dev 2>/dev/null || log_warn "NCCL install via apt failed (non-fatal)"
            ;;
        dnf|yum)
            # NCCL is typically in the CUDA repo on RHEL-family
            $PKG_MGR install -y libnccl libnccl-devel 2>/dev/null || log_warn "NCCL install via $PKG_MGR failed (non-fatal)"
            ;;
    esac
fi

# ── InfiniBand tools ──
if ls /sys/class/infiniband/*/ports/*/state 2>/dev/null | head -1 | grep -q . && [ "$HAS_INTERNET" = true ]; then
    log_info "InfiniBand hardware detected, ensuring tools..."
    case "$PKG_MGR" in
        apt) pkg_install ibverbs-utils infiniband-diags perftest rdma-core 2>/dev/null || true ;;
        dnf|yum) pkg_install libibverbs-utils infiniband-diags perftest rdma-core 2>/dev/null || true ;;
    esac
fi

# ── Docker/container runtime check ──
if has_cmd docker; then
    log_info "Docker: $(docker --version 2>/dev/null)"
    if docker info 2>/dev/null | grep -qi nvidia; then
        log_info "NVIDIA Container Toolkit: available"
    else
        log_warn "NVIDIA Container Toolkit not detected"
    fi
elif has_cmd podman; then
    log_info "Podman: $(podman --version 2>/dev/null)"
else
    log_warn "No container runtime found (Docker/Podman) — some benchmarks will be limited"
fi

# ── Git & cmake (needed for building benchmarks) ──
if [ "$HAS_INTERNET" = true ]; then
    has_cmd git || pkg_install git 2>/dev/null || true
    has_cmd cmake || pkg_install cmake 2>/dev/null || true
    # Optional: Boost for nvbandwidth build-from-source
    if [ "$HAS_GPU" = true ]; then
        case "$PKG_MGR" in
            apt) pkg_install libboost-dev 2>/dev/null || log_warn "Boost install failed (nvbandwidth may skip)" ;;
            dnf|yum) pkg_install boost-devel 2>/dev/null || log_warn "Boost install failed (nvbandwidth may skip)" ;;
        esac
    fi
fi

# ── Compiler pre-flight ──
# gpu-burn, nccl-tests, and STREAM all require compilation.
# If we can't compile, many benchmarks will fail — warn loudly.
HAS_COMPILER=false
if has_cmd gcc && has_cmd make; then
    HAS_COMPILER=true
    log_info "Compiler check: gcc=$(gcc -dumpversion 2>/dev/null), make=$(make --version 2>/dev/null | head -1)"
else
    missing_tools=""
    has_cmd gcc || missing_tools="${missing_tools} gcc"
    has_cmd make || missing_tools="${missing_tools} make"
    log_error "COMPILER PRE-FLIGHT FAILED: missing${missing_tools}"
    log_error "Many benchmarks (gpu-burn, nccl-tests, STREAM) require compilation and WILL FAIL."
    if [ "$HAS_INTERNET" = true ]; then
        log_error "Attempted package install but compilers still missing — check package manager config."
    else
        log_error "Air-gapped mode: pre-install build-essential (Debian/Ubuntu) or gcc gcc-c++ make (RHEL) before running."
    fi
    # Don't exit — let individual modules fail with clear errors, but record the issue
fi

# ── Summary ──
SUMMARY=$(jq -n \
    --arg hostname "$(hostname)" \
    --arg os "${OS_ID} ${OS_VERSION}" \
    --arg kernel "$(uname -r)" \
    --arg pkg_manager "${PKG_MGR}" \
    --argjson virtualization "$VIRT_INFO" \
    --argjson has_gpu "$HAS_GPU" \
    --arg gpu_model "$([ "$HAS_GPU" = true ] && echo "$GPU_MODEL" || echo "none")" \
    --argjson gpu_count "$([ "$HAS_GPU" = true ] && echo "$GPU_N" || echo 0)" \
    --argjson has_docker "$(has_cmd docker && echo true || echo false)" \
    --argjson has_dcgm "$(has_cmd dcgmi && echo true || echo false)" \
    --argjson has_infiniband "$(ls /sys/class/infiniband/ 2>/dev/null | head -1 | grep -q . && echo true || echo false)" \
    --argjson has_internet "$HAS_INTERNET" \
    --argjson has_ipmitool "$(has_cmd ipmitool && echo true || echo false)" \
    --argjson has_compiler "$HAS_COMPILER" \
    --arg results_dir "${HPC_RESULTS_DIR}" \
    --arg work_dir "${HPC_WORK_DIR}" \
    '{
        hostname: $hostname, os: $os, kernel: $kernel,
        pkg_manager: $pkg_manager, virtualization: $virtualization,
        has_gpu: $has_gpu, gpu_model: $gpu_model, gpu_count: $gpu_count,
        has_docker: $has_docker, has_dcgm: $has_dcgm,
        has_infiniband: $has_infiniband, has_internet: $has_internet,
        has_ipmitool: $has_ipmitool, has_compiler: $has_compiler,
        results_dir: $results_dir, work_dir: $work_dir
    }')

echo "$SUMMARY" | emit_json "bootstrap" "ok"
log_ok "Bootstrap complete"
echo "$SUMMARY" | jq .
