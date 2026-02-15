#!/usr/bin/env bash
# gpu-inventory.sh — NVIDIA GPU inventory (driver, CUDA, per-GPU details, NVLink, topology)
SCRIPT_NAME="gpu-inventory"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== GPU Inventory ==="

has_cmd nvidia-smi || skip_module "gpu-inventory" "nvidia-smi not found"

if ! nvidia-smi &>/dev/null; then
    log_error "nvidia-smi failed — driver issue"
    echo '{"gpus":[],"error":"nvidia-smi failed"}' | emit_json "gpu-inventory" "error"
    exit 1
fi

# ── Check virtualization ──
VIRT_INFO=$(detect_virtualization)
VIRT_TYPE=$(echo "$VIRT_INFO" | jq -r '.type')
VIRT_NOTE=""
if [ "$VIRT_TYPE" != "none" ]; then
    VIRT_NOTE="Virtualized environment detected - some GPU metrics may not reflect physical hardware"
    log_warn "$VIRT_NOTE"
fi

# ── Driver & CUDA (query may exit non-zero when fields unsupported, e.g. in VMs) ──
driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:cntrl:]') || true
# cuda_version field may not exist on all drivers — fall back to nvidia-smi header parse
cuda_ver=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:cntrl:]') || true
if [ -z "$cuda_ver" ] || echo "$cuda_ver" | grep -qi "not a valid field\|error"; then
    # Parse CUDA version from nvidia-smi banner (e.g. "CUDA Version: 13.0")
    # Anchor to "CUDA Version:" to avoid greedy .* consuming leading digits
    cuda_ver=$(nvidia-smi 2>/dev/null | grep -i "CUDA Version" | head -1 | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p') || true
    [ -z "$cuda_ver" ] && cuda_ver="unknown"
fi
[ -z "$driver_ver" ] && driver_ver="unknown"

# Count GPUs from driver query (authoritative for this host/session).
# This avoids under/over-counting when downstream parsing drops fields.
gpu_count_detected=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d '[:space:]')
case "$gpu_count_detected" in
    ''|*[!0-9]*) gpu_count_detected=0 ;;
esac

# nvcc version — check PATH then common locations
nvcc_ver="none"
nvcc_path=""
if has_cmd nvcc; then
    nvcc_path=$(cmd_path nvcc)
elif [ -x "${CUDA_HOME:-/nonexistent}/bin/nvcc" ]; then
    nvcc_path="${CUDA_HOME}/bin/nvcc"
elif [ -x "/usr/local/cuda/bin/nvcc" ]; then
    nvcc_path="/usr/local/cuda/bin/nvcc"
else
    for candidate in /usr/local/cuda-*/bin/nvcc; do
        if [ -x "$candidate" ]; then
            nvcc_path="$candidate"
            break
        fi
    done
fi
if [ -n "$nvcc_path" ]; then
    nvcc_ver=$("$nvcc_path" --version 2>/dev/null | grep "release" | awk '{print $NF}' | tr -d ',' | tr -d '[:cntrl:]')
    log_info "Found nvcc at: $nvcc_path (version: $nvcc_ver)"
fi

# ── Detect supported nvidia-smi fields ──
# Some fields (e.g. bar1.total) are not supported on all driver/hardware combos
# Test each optional field before including it in the query
BASE_FIELDS="index,name,uuid,pci.bus_id,memory.total,memory.used,memory.free,power.limit,power.draw,temperature.gpu,clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory,compute_cap,persistence_mode,ecc.mode.current,mig.mode.current,pcie.link.gen.current,pcie.link.width.current"
OPTIONAL_FIELDS=("bar1.total" "bar1.used" "fan.speed" "utilization.gpu" "utilization.memory" "clocks_throttle_reasons.active")

QUERY_FIELDS="$BASE_FIELDS"
ACTIVE_OPTIONAL=()
for field in "${OPTIONAL_FIELDS[@]}"; do
    # Probe without failing script when nvidia-smi exits non-zero (e.g. in VMs or unsupported fields)
    field_out=$(nvidia-smi --query-gpu="$field" --format=csv,noheader 2>&1) || true
    if echo "$field_out" | head -1 | grep -qv "not a valid field"; then
        QUERY_FIELDS="${QUERY_FIELDS},${field}"
        ACTIVE_OPTIONAL+=("$field")
    else
        log_warn "nvidia-smi field not supported: $field"
    fi
done

log_info "Querying fields: $QUERY_FIELDS"

# ── Per-GPU details (allow nvidia-smi to fail in VMs / restricted environments) ──
RAW_GPU_DATA=$(nvidia-smi --query-gpu="$QUERY_FIELDS" --format=csv,noheader,nounits 2>/dev/null | tr -d '[:cntrl:]') || true
if [ -z "$RAW_GPU_DATA" ] || ! echo "$RAW_GPU_DATA" | head -1 | grep -q .; then
    log_warn "Full GPU query returned no data — trying base fields only"
    RAW_GPU_DATA=$(nvidia-smi --query-gpu="$BASE_FIELDS" --format=csv,noheader,nounits 2>/dev/null | tr -d '[:cntrl:]') || true
fi
if [ -z "$RAW_GPU_DATA" ]; then
    log_error "nvidia-smi query failed — no GPU data"
    echo '{"gpus":[],"error":"nvidia-smi query failed"}' | emit_json "gpu-inventory" "error"
    exit 1
fi

# Build optional fields JSON string for python3
OPT_JSON="["
opt_first=1
for f in "${ACTIVE_OPTIONAL[@]}"; do
    [ "$opt_first" -eq 0 ] && OPT_JSON+=","
    opt_first=0
    OPT_JSON+="\"$(echo "$f" | tr '.' '_')\""
done
OPT_JSON+="]"

# Build GPU JSON using python3 for robustness (awk gets unwieldy with variable field counts)
gpu_json=$(echo "$RAW_GPU_DATA" | python3 -c "
import sys, json

# Base field names (always present)
base_fields = ['index','name','uuid','pci_bus_id','memory_total_mb','memory_used_mb','memory_free_mb',
               'power_limit_w','power_draw_w','temperature_c','clock_graphics_mhz','clock_memory_mhz',
               'clock_max_graphics_mhz','clock_max_memory_mhz','compute_capability','persistence_mode',
               'ecc_mode','mig_mode','pcie_gen','pcie_width']

# Optional fields that were detected
try:
    optional_fields = json.loads('$OPT_JSON')
except:
    optional_fields = []

all_fields = base_fields + optional_fields

numeric_fields = {'index','memory_total_mb','memory_used_mb','memory_free_mb','power_limit_w','power_draw_w',
                  'temperature_c','clock_graphics_mhz','clock_memory_mhz','clock_max_graphics_mhz',
                  'clock_max_memory_mhz','pcie_gen','pcie_width','bar1_total','bar1_used',
                  'utilization_gpu','utilization_memory'}

gpus = []
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    parts = [p.strip() for p in line.split(', ')]
    gpu = {}
    for i, field in enumerate(all_fields):
        if i < len(parts):
            val = parts[i]
            if val in ('[Not Supported]', '[Unknown]', 'N/A', ''):
                gpu[field] = None
            elif field in numeric_fields:
                try:
                    gpu[field] = float(val) if '.' in val else int(val)
                except ValueError:
                    gpu[field] = None
            else:
                gpu[field] = val
        else:
            gpu[field] = None
    gpus.append(gpu)

print(json.dumps(gpus))
" 2>/dev/null || echo "[]")

# Validate JSON
if ! echo "$gpu_json" | jq . >/dev/null 2>&1; then
    log_error "Failed to parse GPU data as JSON"
    gpu_json="[]"
fi

# ── NVLink topology ──
nvlink_json='{"available": false}'
if has_cmd nvidia-smi; then
    nvlink_out=$(nvidia-smi nvlink --status 2>/dev/null || echo "")
    if [ -n "$nvlink_out" ] && ! echo "$nvlink_out" | grep -qi "not supported\|error"; then
        link_count=$(count_grep_re 'Link' "$nvlink_out")
        nvlink_json="{\"available\": true, \"total_links\": $link_count}"
    fi
fi

# ── GPU Topology matrix ──
topology_json="{}"
if [ "$(echo "$gpu_json" | jq length)" -gt 1 ] && has_cmd nvidia-smi; then
    topo_output=$(nvidia-smi topo -m 2>/dev/null || echo "")
    if [ -n "$topo_output" ] && ! echo "$topo_output" | grep -qi "not supported\|error"; then
        # Parse topology using python3 for robustness
        topology_json=$(echo "$topo_output" | python3 -c "
import sys, json
lines = sys.stdin.read().strip().split('\n')
topo = {}
headers = None
for line in lines:
    if line.startswith('\t') or line.startswith('GPU'):
        parts = line.split('\t')
        if headers is None:
            headers = [p.strip() for p in parts[1:] if p.strip()]
        else:
            gpu_name = parts[0].strip()
            if gpu_name.startswith('GPU'):
                connections = {}
                for i, val in enumerate(parts[1:]):
                    val = val.strip()
                    if val and i < len(headers):
                        connections[headers[i]] = val
                topo[gpu_name] = connections
print(json.dumps(topo))
" 2>/dev/null || echo "{}")
        topology_json=$(json_compact_or "$topology_json" "{}")
    fi
fi

# ── Validate JSON and normalize ──
nvlink_json=$(echo "$nvlink_json" | jq -c . 2>/dev/null) || true
echo "$nvlink_json" | jq -e . >/dev/null 2>&1 || nvlink_json='{"available":false}'
topology_json=$(echo "$topology_json" | jq -c . 2>/dev/null) || true
echo "$topology_json" | jq -e . >/dev/null 2>&1 || topology_json='{}'
gpu_json=$(echo "$gpu_json" | jq -c . 2>/dev/null) || true
echo "$gpu_json" | jq -e . >/dev/null 2>&1 || gpu_json='[]'

TMP_GPU=$(json_tmpfile "gpu_json" "$gpu_json" "[]")
TMP_NVL=$(json_tmpfile "nvlink" "$nvlink_json" '{"available":false}')
TMP_TOP=$(json_tmpfile "topo" "$topology_json" "{}")

# ── Build result (sanitize string args for JSON safety) ──
driver_ver=$(printf '%s' "$driver_ver" | tr -d '\000-\037')
cuda_ver=$(printf '%s' "$cuda_ver" | tr -d '\000-\037')
nvcc_ver=$(printf '%s' "$nvcc_ver" | tr -d '\000-\037')
RESULT=$(jq -n \
    --arg drv "$driver_ver" \
    --arg cuda "$cuda_ver" \
    --arg nvcc "$nvcc_ver" \
    --slurpfile gpus "$TMP_GPU" \
    --slurpfile nvlink_arr "$TMP_NVL" \
    --slurpfile topo_arr "$TMP_TOP" \
    --argjson virt "$VIRT_INFO" \
    --arg note "$VIRT_NOTE" \
    --argjson detected_count "$gpu_count_detected" \
    '$nvlink_arr[0] as $nvlink | $topo_arr[0] as $topo | {
        driver_version: $drv,
        cuda_version: $cuda,
        nvcc_version: $nvcc,
        gpu_count: (if $detected_count > 0 then $detected_count else ($gpus[0] | length) end),
        gpus: $gpus[0],
        nvlink: $nvlink,
        topology: $topo,
        virtualization: $virt,
        note: (if $note != "" then $note else null end)
    }')

finish_module "gpu-inventory" "ok" "$RESULT" '{gpu_count, driver_version, cuda_version, nvlink}'
