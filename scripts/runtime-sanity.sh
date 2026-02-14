#!/usr/bin/env bash
# runtime-sanity.sh — Early runtime readiness checks (GPU driver/container runtime/DCGM)
SCRIPT_NAME="runtime-sanity"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== Runtime Sanity Check ==="

AUTO_INSTALL="${HPC_AUTO_INSTALL_CONTAINER_RUNTIME:-0}"
FAIL_FAST="${HPC_FAIL_FAST_RUNTIME:-0}"

has_gpu_driver=false
gpu_count_now=0
if has_cmd nvidia-smi && nvidia-smi >/dev/null 2>&1; then
    has_gpu_driver=true
    gpu_count_now=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d '[:space:]')
fi

has_docker=false
nvidia_runtime=false
if has_cmd docker; then
    has_docker=true
    if docker info 2>/dev/null | grep -qi nvidia; then
        nvidia_runtime=true
    fi
fi

has_dcgm=false
has_cmd dcgmi && has_dcgm=true

if [ "$has_gpu_driver" != true ]; then
    log_info "No working NVIDIA GPU driver detected; runtime checks are not required."
    jq -n \
        --arg note "no working GPU driver" \
        --argjson has_gpu_driver false \
        --argjson gpu_count 0 \
        --argjson has_docker "$has_docker" \
        --argjson has_nvidia_container_runtime "$nvidia_runtime" \
        --argjson has_dcgm "$has_dcgm" \
        '{
            note: $note,
            has_gpu_driver: $has_gpu_driver,
            gpu_count: $gpu_count,
            has_docker: $has_docker,
            has_nvidia_container_runtime: $has_nvidia_container_runtime,
            has_dcgm: $has_dcgm
        }' | emit_json "runtime-sanity" "skipped"
    exit 0
fi

# GPU driver exists: check if container runtime is ready.
if [ "$nvidia_runtime" != true ] && [ "$AUTO_INSTALL" = "1" ] && [ "$(id -u)" -eq 0 ]; then
    log_warn "NVIDIA container runtime missing — auto-install requested, running bootstrap installer..."
    if bash "${HPC_BENCH_ROOT}/scripts/bootstrap.sh" --install-nvidia-container-toolkit; then
        if has_cmd docker && docker info 2>/dev/null | grep -qi nvidia; then
            has_docker=true
            nvidia_runtime=true
            log_ok "Auto-install completed and NVIDIA runtime is now available."
        else
            log_warn "Auto-install finished but NVIDIA runtime still not detected."
        fi
    else
        log_warn "Auto-install attempt failed (non-fatal unless fail-fast enabled)."
    fi
fi

status="ok"
note="runtime ready"
if [ "$nvidia_runtime" != true ]; then
    status="warn"
    note="NVIDIA container runtime missing; run: sudo bash scripts/bootstrap.sh --install-nvidia-container-toolkit"
fi

if [ "$FAIL_FAST" = "1" ] && [ "$nvidia_runtime" != true ]; then
    log_error "Fail-fast enabled and NVIDIA container runtime is missing."
    jq -n \
        --arg note "$note" \
        --argjson has_gpu_driver true \
        --argjson gpu_count "$gpu_count_now" \
        --argjson has_docker "$has_docker" \
        --argjson has_nvidia_container_runtime "$nvidia_runtime" \
        --argjson has_dcgm "$has_dcgm" \
        --arg auto_install "$AUTO_INSTALL" \
        --arg fail_fast "$FAIL_FAST" \
        '{
            note: $note,
            has_gpu_driver: $has_gpu_driver,
            gpu_count: $gpu_count,
            has_docker: $has_docker,
            has_nvidia_container_runtime: $has_nvidia_container_runtime,
            has_dcgm: $has_dcgm,
            auto_install_enabled: ($auto_install == "1"),
            fail_fast_enabled: ($fail_fast == "1")
        }' | emit_json "runtime-sanity" "error"
    exit 1
fi

jq -n \
    --arg note "$note" \
    --argjson has_gpu_driver true \
    --argjson gpu_count "$gpu_count_now" \
    --argjson has_docker "$has_docker" \
    --argjson has_nvidia_container_runtime "$nvidia_runtime" \
    --argjson has_dcgm "$has_dcgm" \
    --arg auto_install "$AUTO_INSTALL" \
    --arg fail_fast "$FAIL_FAST" \
    '{
        note: $note,
        has_gpu_driver: $has_gpu_driver,
        gpu_count: $gpu_count,
        has_docker: $has_docker,
        has_nvidia_container_runtime: $has_nvidia_container_runtime,
        has_dcgm: $has_dcgm,
        auto_install_enabled: ($auto_install == "1"),
        fail_fast_enabled: ($fail_fast == "1")
    }' | emit_json "runtime-sanity" "$status"

if [ "$status" = "ok" ]; then
    log_ok "Runtime sanity: ready (GPU/container runtime/DCGM checks complete)"
else
    log_warn "Runtime sanity: warning — $note"
fi
