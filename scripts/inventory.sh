#!/usr/bin/env bash
# inventory.sh — CPU, RAM, storage, kernel, OS, network basics
SCRIPT_NAME="inventory"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== System Inventory ==="

# ── CPU ──
if has_cmd lscpu; then
    _lscpu_out=$(lscpu 2>/dev/null || true)
else
    _lscpu_out=""
fi
cpu_model=$(echo "$_lscpu_out" | awk -F: '/Model name/ {gsub(/^[ \t]+/,"",$2); print $2; exit}')
cpu_sockets=$(echo "$_lscpu_out" | awk -F: '/^Socket\(s\)/ {gsub(/[ \t]/,"",$2); print $2}')
cpu_cores_per_socket=$(echo "$_lscpu_out" | awk -F: '/Core\(s\) per socket/ {gsub(/[ \t]/,"",$2); print $2}')
cpu_threads=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo "0")
cpu_arch=$(uname -m)
numa_nodes=$(echo "$_lscpu_out" | awk -F: '/NUMA node\(s\)/ {gsub(/[ \t]/,"",$2); print $2}')
cpu_flags=$(echo "$_lscpu_out" | awk -F: '/Flags/ {print $2}' | xargs)
cpu_sockets_num=$(int_or_default "$cpu_sockets" 0)
cpu_cores_per_socket_num=$(int_or_default "$cpu_cores_per_socket" 0)
cpu_threads_num=$(int_or_default "$cpu_threads" 0)
numa_nodes_num=$(int_or_default "${numa_nodes:-1}" 1)
total_cores_num=$((cpu_sockets_num * cpu_cores_per_socket_num))

cpu_json=$(jq -n \
    --arg model "${cpu_model:-unknown}" \
    --arg arch "$cpu_arch" \
    --argjson sockets "$cpu_sockets_num" \
    --argjson cores_per_socket "$cpu_cores_per_socket_num" \
    --argjson total_cores "$total_cores_num" \
    --argjson threads "$cpu_threads_num" \
    --argjson numa "$numa_nodes_num" \
    --argjson avx512 "$(echo "$cpu_flags" | grep -qw avx512f && echo true || echo false)" \
    --argjson amx "$(echo "$cpu_flags" | grep -qw amx && echo true || echo false)" \
    '{
        model: $model,
        architecture: $arch,
        sockets: $sockets,
        cores_per_socket: $cores_per_socket,
        total_cores: $total_cores,
        threads: $threads,
        numa_nodes: $numa,
        has_avx512: $avx512,
        has_amx: $amx
    }')

# ── RAM ──
if [ -r /proc/meminfo ]; then
    total_mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo "0")
elif has_cmd sysctl; then
    # macOS fallback for local dev/testing; Linux path above is primary.
    total_mem_kb=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1024}' || echo "0")
else
    total_mem_kb="0"
fi
total_mem_kb=$(int_or_default "$total_mem_kb" 0)
total_mem_gb=$(awk -v kb="$total_mem_kb" 'BEGIN { printf "%.1f", kb/1048576 }')

# DIMM details via dmidecode (requires root)
dimm_json="[]"
if has_cmd dmidecode; then
    _dmidecode_out=$(try_sudo dmidecode -t memory 2>/dev/null) || true
    if [ -n "$_dmidecode_out" ]; then
        dimm_json=$(echo "$_dmidecode_out" | awk '
        BEGIN { print "["; first=1; size=""; type=""; speed=""; mfr=""; loc=""; bank="" }
        /^Memory Device$/ {
            if(size!="" && size!="No Module Installed" && size!="Not Installed") {
                if(!first) printf ","; first=0;
                full_loc = (bank != "" && bank != loc) ? bank " / " loc : loc
                printf "{\"size\":\"%s\",\"type\":\"%s\",\"speed\":\"%s\",\"manufacturer\":\"%s\",\"locator\":\"%s\"}\n", size, type, speed, mfr, full_loc
            }
            size=""; type=""; speed=""; mfr=""; loc=""; bank=""
        }
        /^\tSize:/ { sub(/^\t*Size: /,""); size=$0 }
        /^\tType:/ && !/Type Detail/ { sub(/^\t*Type: /,""); type=$0 }
        /^\tConfigured Memory Speed:/ { sub(/^\t*Configured Memory Speed: /,""); speed=$0 }
        /^\tManufacturer:/ { sub(/^\t*Manufacturer: /,""); mfr=$0 }
        /^\tLocator:/ && !/Bank Locator/ { sub(/^\t*Locator: /,""); loc=$0 }
        /^\tBank Locator:/ { sub(/^\t*Bank Locator: /,""); bank=$0 }
        END {
            if(size!="" && size!="No Module Installed" && size!="Not Installed") {
                if(!first) printf ","; first=0;
                full_loc = (bank != "" && bank != loc) ? bank " / " loc : loc
                printf "{\"size\":\"%s\",\"type\":\"%s\",\"speed\":\"%s\",\"manufacturer\":\"%s\",\"locator\":\"%s\"}\n", size, type, speed, mfr, full_loc
            }
            print "]"
        }
        ' 2>/dev/null) || true
        dimm_json=$(json_compact_or "$dimm_json" "[]")
    fi
    if [ "$dimm_json" = "[]" ] && ! is_root; then
        log_warn "dmidecode requires root for DIMM details — run as root for full inventory"
    fi
fi

ram_json=$(jq -n \
    --argjson total_gb "$total_mem_gb" \
    --argjson dimms "$dimm_json" \
    '{total_gb: $total_gb, dimm_count: ($dimms | length), dimms: $dimms}')

# ── Storage (filter out loop/ram devices) ──
if has_cmd lsblk; then
    storage_json=$(lsblk -Jd -o NAME,SIZE,TYPE,MODEL,ROTA,TRAN,SERIAL 2>/dev/null | jq '[.blockdevices // [] | .[] | select(.type != "loop" and .type != "ram")]' 2>/dev/null || echo "[]")
else
    storage_json="[]"
fi
storage_json=$(json_compact_or "$storage_json" "[]")

# SMART data for each disk (requires root)
smart_arr="[]"
if has_cmd smartctl; then
    _smart_out=$(lsblk -dn -o NAME,TYPE 2>/dev/null | awk '$2=="disk" {print "/dev/"$1}' | while read dev; do
        health=$(try_sudo smartctl -H "$dev" 2>/dev/null | grep -i "result" | awk -F: '{gsub(/^[ \t]+/,"",$2); print $2}')
        temp=$(try_sudo smartctl -A "$dev" 2>/dev/null | awk '/Temperature_Celsius|Airflow_Temperature/ {print $10; exit}')
        echo "{\"device\":\"$dev\",\"health\":\"${health:-unknown}\",\"temp_c\":\"${temp:-unknown}\"}"
    done) || true
    if [ -n "$_smart_out" ]; then
        smart_arr=$(printf '%s\n' "$_smart_out" | json_slurp_objects_or "[]")
    fi
fi
smart_arr=$(json_compact_or "$smart_arr" "[]")

# ── OS / Kernel ──
_hostname=$(hostname -f 2>/dev/null || hostname)
_os_pretty=$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || uname -s)
_kernel=$(uname -r)
_kernel_arch=$(uname -m)
if [ -r /proc/uptime ]; then
    _uptime=$(awk '{print int($1)}' /proc/uptime 2>/dev/null || echo "0")
else
    _uptime="0"
fi
_uptime=$(int_or_default "$_uptime" 0)
if has_cmd ip; then
    _ips=$(ip -j addr show 2>/dev/null | jq '[.[] | select(.ifname != "lo") | {iface: .ifname, addrs: [.addr_info[] | .local]}]' 2>/dev/null || echo "[]")
else
    _ips="[]"
fi
_ips=$(json_compact_or "$_ips" "[]")

os_json=$(jq -n \
    --arg hostname "$_hostname" \
    --arg os "$_os_pretty" \
    --arg kernel "$_kernel" \
    --arg kernel_arch "$_kernel_arch" \
    --argjson uptime_seconds "$_uptime" \
    --argjson ips "$_ips" \
    '{
        hostname: $hostname,
        os: $os,
        kernel: $kernel,
        kernel_arch: $kernel_arch,
        uptime_seconds: $uptime_seconds,
        ips: $ips
    }')

# ── Assemble ──
RESULT=$(jq -n \
    --argjson cpu "$cpu_json" \
    --argjson ram "$ram_json" \
    --argjson storage "$storage_json" \
    --argjson smart "$smart_arr" \
    --argjson os "$os_json" \
    '{cpu: $cpu, ram: $ram, storage: {device_count: ($storage | length), devices: $storage, smart: $smart}, os: $os}')

finish_module "inventory" "ok" "$RESULT"
