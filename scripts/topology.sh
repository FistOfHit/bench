#!/usr/bin/env bash
# topology.sh — PCIe topology, NVLink, NUMA mapping, CPU-GPU affinity
SCRIPT_NAME="topology"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== Topology ==="

# ── nvidia-smi topo ──
topo_matrix=""
if has_cmd nvidia-smi; then
    topo_matrix=$(nvidia-smi topo -m 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' || echo "N/A")
    log_info "GPU topology matrix captured"
fi

# ── NVLink status ──
nvlink_json="[]"
if has_cmd nvidia-smi; then
    nvlink_json=$(nvidia-smi nvlink --status 2>/dev/null | awk '
    BEGIN { print "["; first=1 }
    /GPU/ { gpu=$2 }
    /Link/ { link=$2; gsub(/:/, "", link) }
    /Link.*active/ {
        if(!first) printf ","; first=0
        printf "{\"gpu\":%s,\"link\":%s,\"active\":true}\n", gpu, link
    }
    END { print "]" }
    ' 2>/dev/null || echo "[]")
fi
nvlink_json=$(json_compact_or "$nvlink_json" "[]")

# ── NVSwitch detection ──
nvswitch_count=0
if has_cmd nvidia-smi; then
    # Avoid grep -P (not always available); take the first integer token.
    nvswitch_count=$(nvidia-smi nvswitch --count 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i ~ /^[0-9]+$/) {print $i; exit}}' || true)
fi
nvswitch_count=$(int_or_default "${nvswitch_count:-0}" 0)

# ── lstopo ──
lstopo_text=""
if has_cmd lstopo; then
    lstopo_text=$(lstopo --of txt 2>/dev/null | head -200) || lstopo_text="N/A"
    # Generate a clean SVG focused on CPU/NUMA/Package hierarchy.
    # --no-io:     hide PCI/disk/NIC clutter
    # --no-caches: hide L1/L2/L3 boxes
    # --no-smt:    hide individual PU (hardware thread) boxes
    # --merge:     collapse levels with no hierarchical impact
    # --no-attrs:  hide verbose size/frequency attributes (keeps labels)
    # lstopo refuses to overwrite — remove stale file first
    rm -f "${HPC_RESULTS_DIR}/topology.svg"
    if lstopo --of svg --no-io --no-caches --no-smt --merge --no-attrs \
             "${HPC_RESULTS_DIR}/topology.svg" 2>/dev/null; then
        log_info "SVG topology saved (clean: no-io, no-caches, no-smt, merged)"
    elif lstopo --of svg --no-io --merge "${HPC_RESULTS_DIR}/topology.svg" 2>/dev/null; then
        log_info "SVG topology saved (fallback: no-io, merged)"
    elif lstopo --of svg "${HPC_RESULTS_DIR}/topology.svg" 2>/dev/null; then
        log_info "SVG topology saved (full — no filters supported)"
    fi
elif has_cmd hwloc-ls; then
    lstopo_text=$(hwloc-ls 2>/dev/null | head -200) || lstopo_text="N/A"
fi

# ── NUMA mapping ──
numa_json="[]"
if has_cmd numactl; then
    _numa_out=$(numactl --hardware 2>/dev/null) || true
    if [ -n "$_numa_out" ]; then
        numa_json=$(echo "$_numa_out" | awk '
        BEGIN { print "[" ; first=1 }
        /^node [0-9]+ cpus:/ {
            node=$2; cpus=$0; sub(/.*cpus: /, "", cpus)
            if(!first) printf ","
            first=0
            printf "{\"node\":%s,\"cpus\":\"%s\"}", node, cpus
        }
        END { print "]" }
        ') || numa_json="[]"
    fi
elif [ -d /sys/devices/system/node ]; then
    # Fallback: read NUMA info from sysfs
    numa_json=$(ls -d /sys/devices/system/node/node* 2>/dev/null | while read ndir; do
        node=$(basename "$ndir" | sed 's/node//')
        cpus=$(cat "$ndir/cpulist" 2>/dev/null || echo "")
        echo "{\"node\":$node,\"cpus\":\"$cpus\"}"
    done | json_slurp_objects_or "[]")
fi
numa_json=$(json_compact_or "$numa_json" "[]")

# ── CPU-GPU affinity ──
affinity_json="[]"
if has_cmd nvidia-smi; then
    affinity_json=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null | while IFS=', ' read -r idx bus; do
        numa_node=$(cat "/sys/bus/pci/devices/${bus,,}/numa_node" 2>/dev/null || echo "-1")
        echo "{\"gpu\":$idx,\"pci_bus\":\"$bus\",\"numa_node\":$numa_node}"
    done | json_slurp_objects_or "[]")
fi
affinity_json=$(json_compact_or "$affinity_json" "[]")

# ── PCIe link details for GPUs ──
pcie_json="[]"
if has_cmd nvidia-smi; then
    pcie_json=$(nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '
    BEGIN { print "[" }
    NR>1 { printf "," }
    {
        printf "{\"gpu\":%s,\"pcie_gen_current\":%s,\"pcie_gen_max\":%s,\"pcie_width_current\":%s,\"pcie_width_max\":%s}", $1, $2, $3, $4, $5
    }
    END { print "]" }
    ' 2>/dev/null || echo "[]")
fi
pcie_json=$(json_compact_or "$pcie_json" "[]")

RESULT=$(jq -n \
    --arg topo "$topo_matrix" \
    --argjson nvlink "$nvlink_json" \
    --arg nvsw "$nvswitch_count" \
    --arg lstopo "$lstopo_text" \
    --argjson numa "$numa_json" \
    --argjson affinity "$affinity_json" \
    --argjson pcie "$pcie_json" \
    '{
        gpu_topology_matrix: $topo,
        nvlink: $nvlink,
        nvlink_count: ($nvlink | length),
        nvswitch_count: ($nvsw | tonumber),
        lstopo: $lstopo,
        numa_node_count: ($numa | length),
        numa_nodes: $numa,
        cpu_gpu_affinity: $affinity,
        gpu_affinity_count: ($affinity | length),
        pcie_links: $pcie,
        pcie_link_count: ($pcie | length)
    }')

finish_module "topology" "ok" "$RESULT"
