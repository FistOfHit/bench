#!/usr/bin/env bash
# network-inventory.sh — NICs, InfiniBand HCAs, link speeds, MTU, bonding, RoCE
SCRIPT_NAME="network-inventory"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== Network Inventory ==="

# ── Ethernet NICs ──
nics_json=$(ip -j link show 2>/dev/null | jq '[.[] | select(.ifname != "lo") | {
    name: .ifname,
    mac: .address,
    state: .operstate,
    mtu: .mtu,
    type: .link_type
}]' 2>/dev/null || echo "[]")

# Enrich with ethtool speed
nics_enriched=$(echo "$nics_json" | jq -c '.[]' | while read -r nic; do
    ifname=$(echo "$nic" | jq -r '.name')
    speed=$(ethtool "$ifname" 2>/dev/null | awk '/Speed:/ {print $2}' || echo "unknown")
    driver=$(ethtool -i "$ifname" 2>/dev/null | awk '/driver:/ {print $2}' || echo "unknown")
    echo "$nic" | jq --arg s "$speed" --arg d "$driver" '. + {speed: $s, driver: $d}'
done | jq -s '.' 2>/dev/null || echo "$nics_json")

# ── Bonding ──
bond_json="[]"
if ls /proc/net/bonding/* 2>/dev/null | head -1 | grep -q .; then
    bond_json=$(for f in /proc/net/bonding/*; do
        name=$(basename "$f")
        mode=$(awk '/Bonding Mode:/ {$1=$2=""; print substr($0,3)}' "$f")
        slaves=$(awk '/Slave Interface:/ {print $3}' "$f" | tr '\n' ',' | sed 's/,$//')
        echo "{\"name\":\"$name\",\"mode\":\"$mode\",\"slaves\":\"$slaves\"}"
    done | jq -s '.' 2>/dev/null || echo "[]")
fi

# ── InfiniBand ──
ib_json="[]"
if has_cmd ibstat; then
    ib_json=$(ibstat 2>/dev/null | awk '
    BEGIN { print "["; first=1 }
    /^CA / { if(ca!="") { if(!first) printf ","; first=0; printf "{\"ca\":\"%s\",\"type\":\"%s\",\"ports\":%s,\"fw\":\"%s\",\"state\":\"%s\",\"rate\":\"%s\"}\n", ca, catype, numports, fw, state, rate }; ca=$2; gsub(/'\''/,"",ca); catype=""; numports=0; fw=""; state=""; rate="" }
    /CA type:/ { catype=$0; sub(/.*CA type: */,"",catype) }
    /Number of ports:/ { numports=$NF }
    /Firmware version:/ { fw=$NF }
    /State:/ { state=$NF }
    /Rate:/ { rate=$NF }
    END { if(ca!="") { if(!first) printf ","; printf "{\"ca\":\"%s\",\"type\":\"%s\",\"ports\":%s,\"fw\":\"%s\",\"state\":\"%s\",\"rate\":\"%s\"}\n", ca, catype, numports, fw, state, rate }; print "]" }
    ' 2>/dev/null || echo "[]")
fi

# ── ibv_devinfo ──
ibv_json="[]"
if has_cmd ibv_devinfo; then
    ibv_json=$(ibv_devinfo 2>/dev/null | awk '
    BEGIN { print "["; first=1 }
    /hca_id:/ { if(hca!="") { if(!first) printf ","; first=0; printf "{\"hca\":\"%s\",\"transport\":\"%s\",\"fw\":\"%s\",\"node_guid\":\"%s\"}\n", hca, transport, fw, guid }; hca=$NF; transport=""; fw=""; guid="" }
    /transport:/ { transport=$NF }
    /fw_ver:/ { fw=$NF }
    /node_guid:/ { guid=$NF }
    END { if(hca!="") { if(!first) printf ","; printf "{\"hca\":\"%s\",\"transport\":\"%s\",\"fw\":\"%s\",\"node_guid\":\"%s\"}\n", hca, transport, fw, guid }; print "]" }
    ' 2>/dev/null || echo "[]")
fi

# ── RoCE detection ──
roce_detected=false
if has_cmd ibv_devinfo; then
    if ibv_devinfo 2>/dev/null | grep -qi "RoCE\|Ethernet"; then
        roce_detected=true
    fi
fi

RESULT=$(jq -n \
    --argjson nics "$nics_enriched" \
    --argjson bonds "$bond_json" \
    --argjson ib "$ib_json" \
    --argjson ibv "$ibv_json" \
    --argjson roce "$roce_detected" \
    '{nics: $nics, bonding: $bonds, infiniband: $ib, ibv_devices: $ibv, roce_detected: $roce}')

echo "$RESULT" | emit_json "network-inventory" "ok"
log_ok "Network inventory complete"
echo "$RESULT" | jq .
