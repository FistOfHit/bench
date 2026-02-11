#!/usr/bin/env bash
# inventory.sh — CPU, RAM, storage, kernel, OS, network basics
SCRIPT_NAME="inventory"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== System Inventory ==="

# ── CPU ──
cpu_model=$(lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^[ \t]+/,"",$2); print $2; exit}')
cpu_sockets=$(lscpu 2>/dev/null | awk -F: '/^Socket\(s\)/ {gsub(/[ \t]/,"",$2); print $2}')
cpu_cores_per_socket=$(lscpu 2>/dev/null | awk -F: '/Core\(s\) per socket/ {gsub(/[ \t]/,"",$2); print $2}')
cpu_threads=$(nproc 2>/dev/null || echo "unknown")
cpu_arch=$(uname -m)
numa_nodes=$(lscpu 2>/dev/null | awk -F: '/NUMA node\(s\)/ {gsub(/[ \t]/,"",$2); print $2}')
cpu_flags=$(lscpu 2>/dev/null | awk -F: '/Flags/ {print $2}' | xargs)

cpu_json=$(jq -n \
    --arg model "${cpu_model:-unknown}" \
    --arg arch "$cpu_arch" \
    --argjson sockets "${cpu_sockets:-0}" \
    --argjson cores_per_socket "${cpu_cores_per_socket:-0}" \
    --argjson total_cores "$(( ${cpu_sockets:-0} * ${cpu_cores_per_socket:-0} ))" \
    --argjson threads "${cpu_threads:-0}" \
    --argjson numa "${numa_nodes:-1}" \
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
total_mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
total_mem_gb=$(echo "scale=1; $total_mem_kb / 1048576" | bc)

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
        # Validate JSON
        echo "$dimm_json" | jq . >/dev/null 2>&1 || dimm_json="[]"
    fi
    if [ "$dimm_json" = "[]" ] && ! is_root; then
        log_warn "dmidecode requires root for DIMM details — run as root for full inventory"
    fi
fi

ram_json=$(jq -n \
    --argjson total_gb "$total_mem_gb" \
    --argjson dimms "$dimm_json" \
    '{total_gb: $total_gb, dimms: $dimms}')

# ── Storage (filter out loop/ram devices) ──
storage_json=$(lsblk -Jd -o NAME,SIZE,TYPE,MODEL,ROTA,TRAN,SERIAL 2>/dev/null | jq '[.blockdevices // [] | .[] | select(.type != "loop" and .type != "ram")]')

# SMART data for each disk (requires root)
smart_arr="[]"
if has_cmd smartctl; then
    _smart_out=$(lsblk -dn -o NAME,TYPE 2>/dev/null | awk '$2=="disk" {print "/dev/"$1}' | while read dev; do
        health=$(try_sudo smartctl -H "$dev" 2>/dev/null | grep -i "result" | awk -F: '{gsub(/^[ \t]+/,"",$2); print $2}')
        temp=$(try_sudo smartctl -A "$dev" 2>/dev/null | awk '/Temperature_Celsius|Airflow_Temperature/ {print $10; exit}')
        echo "{\"device\":\"$dev\",\"health\":\"${health:-unknown}\",\"temp_c\":\"${temp:-unknown}\"}"
    done) || true
    if [ -n "$_smart_out" ]; then
        smart_arr=$(echo "$_smart_out" | jq -s '.' 2>/dev/null) || smart_arr="[]"
    fi
fi

# ── OS / Kernel ──
_hostname=$(hostname -f 2>/dev/null || hostname)
_os_pretty=$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || uname -s)
_kernel=$(uname -r)
_kernel_arch=$(uname -m)
_uptime=$(awk '{print int($1)}' /proc/uptime)
_ips=$(ip -j addr show 2>/dev/null | jq '[.[] | select(.ifname != "lo") | {iface: .ifname, addrs: [.addr_info[] | .local]}]' 2>/dev/null || echo "[]")

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
    '{cpu: $cpu, ram: $ram, storage: {devices: $storage, smart: $smart}, os: $os}')

echo "$RESULT" | emit_json "inventory" "ok"
log_ok "Inventory complete"
echo "$RESULT" | jq .
