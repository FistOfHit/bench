#!/usr/bin/env bash
# bmc-inventory.sh — IPMI/BMC inventory (firmware, network, sensors); skips in VMs
SCRIPT_NAME="bmc-inventory"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== BMC Inventory ==="

# ── Check virtualization first ──
VIRT_INFO=$(detect_virtualization)
VIRT_TYPE=$(echo "$VIRT_INFO" | jq -r '.type')

if [ "$VIRT_TYPE" != "none" ]; then
    log_warn "Running in virtualized environment ($VIRT_TYPE) — BMC not accessible"
    jq -n --arg note "BMC not available in virtualized environment" --argjson virtualization "$VIRT_INFO" '{note: $note, virtualization: $virtualization}' | emit_json "bmc-inventory" "skipped"
    exit 0
fi

if ! has_cmd ipmitool; then
    log_warn "ipmitool not found — skipping BMC inventory"
    echo '{"note":"ipmitool not available"}' | emit_json "bmc-inventory" "skipped"
    exit 0
fi

# Load IPMI kernel modules
modprobe ipmi_devintf 2>/dev/null || true
modprobe ipmi_si 2>/dev/null || true
sleep 1

# Test if IPMI works
if ! ipmitool mc info &>/dev/null; then
    log_warn "IPMI not responding — no BMC or not accessible"
    echo '{"note":"IPMI not responding"}' | emit_json "bmc-inventory" "skipped"
    exit 0
fi

# ── BMC Info ──
bmc_info=$(ipmitool mc info 2>/dev/null)
fw_version=$(echo "$bmc_info" | awk -F: '/Firmware Revision/ {gsub(/[ \t]/,"",$2); print $2}')
mfr=$(echo "$bmc_info" | awk -F: '/Manufacturer Name/ {gsub(/^[ \t]+/,"",$2); print $2}')
product=$(echo "$bmc_info" | awk -F: '/Product Name/ {gsub(/^[ \t]+/,"",$2); print $2}')

# ── BMC Network ──
bmc_ip=$(ipmitool lan print 2>/dev/null | awk -F: '/IP Address[^S]/ {gsub(/[ \t]/,"",$2); print $2; exit}')
bmc_mac=$(ipmitool lan print 2>/dev/null | awk -F: '/MAC Address/ {gsub(/^[ \t]+/,"",$2); print $2}' | head -1)

# ── Chassis Status ──
chassis=$(ipmitool chassis status 2>/dev/null)
power_state=$(echo "$chassis" | awk -F: '/System Power/ {gsub(/[ \t]/,"",$2); print $2}')

# ── Sensors (temps, fans, PSU) ──
sensors_json=$(ipmitool sensor list 2>/dev/null | awk -F'|' '
BEGIN { print "[" ; first=1 }
NF>=3 {
    name=$1; value=$2; unit=$3; status=$4
    gsub(/^[ \t]+|[ \t]+$/, "", name)
    gsub(/^[ \t]+|[ \t]+$/, "", value)
    gsub(/^[ \t]+|[ \t]+$/, "", unit)
    gsub(/^[ \t]+|[ \t]+$/, "", status)
    if (value != "na" && value != "") {
        if(!first) printf ","
        first=0
        printf "{\"name\":\"%s\",\"value\":\"%s\",\"unit\":\"%s\",\"status\":\"%s\"}\n", name, value, unit, status
    }
}
END { print "]" }
' 2>/dev/null || echo "[]")

# Extract key categories
temp_sensors=$(echo "$sensors_json" | jq '[.[] | select(.unit | test("degrees C"; "i"))]' 2>/dev/null || echo "[]")
fan_sensors=$(echo "$sensors_json" | jq '[.[] | select(.unit | test("RPM"; "i"))]' 2>/dev/null || echo "[]")
power_sensors=$(echo "$sensors_json" | jq '[.[] | select(.unit | test("Watts"; "i"))]' 2>/dev/null || echo "[]")

RESULT=$(jq -n \
    --arg fw "$fw_version" \
    --arg mfr "$mfr" \
    --arg product "$product" \
    --arg ip "$bmc_ip" \
    --arg mac "$bmc_mac" \
    --arg power "$power_state" \
    --argjson temps "$temp_sensors" \
    --argjson fans "$fan_sensors" \
    --argjson psu "$power_sensors" \
    --argjson all_sensors "$sensors_json" \
    '{
        bmc: {firmware: $fw, manufacturer: $mfr, product: $product, ip: $ip, mac: $mac},
        chassis: {power_state: $power},
        sensors: {temperatures: $temps, fans: $fans, power: $psu},
        all_sensors: $all_sensors
    }')

echo "$RESULT" | emit_json "bmc-inventory" "ok"
log_ok "BMC inventory complete"
echo "$RESULT" | jq .
