#!/usr/bin/env bash
# network-diag.sh — Firewall rules, open ports, routing, DNS
SCRIPT_NAME="network-diag"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== Network Diagnostics ==="

# ── Firewall rules ──
fw_type="none"
fw_rules=""
if has_cmd nft && nft list ruleset &>/dev/null; then
    fw_type="nftables"
    fw_rules=$(nft list ruleset 2>/dev/null | head -200) || true
elif has_cmd iptables; then
    fw_rules=$(try_sudo iptables -L -n -v 2>/dev/null | head -100) || true
    [ -n "$fw_rules" ] && fw_type="iptables"
fi

# ── Open ports ──
# ss -tulnp columns: State Recv-Q Send-Q Local-Address:Port Peer-Address:Port Process
_ss_out=$(ss -tulnp 2>/dev/null) || true
listening="[]"
if [ -n "$_ss_out" ]; then
    listening=$(echo "$_ss_out" | awk '
    BEGIN { print "[" ; first=1 }
    NR>1 {
        if(!first) printf ","
        first=0
        # Escape any quotes in process field
        proc=$NF; gsub(/"/, "\\\"", proc)
        printf "{\"state\":\"%s\",\"local\":\"%s\",\"peer\":\"%s\",\"process\":\"%s\"}", $1, $4, $5, proc
    }
    END { print "]" }
    ') || listening="[]"
    listening=$(json_compact_or "$listening" "[]")
fi

# ── Routing ──
routes=$(ip -j route show 2>/dev/null) || routes="[]"
routes=$(json_compact_or "$routes" "[]")
default_gw=$(ip route | awk '/default/ {print $3; exit}')

# ── DNS ──
dns_servers=$(awk '/^nameserver/ {print $2}' /etc/resolv.conf 2>/dev/null | json_array_from_lines "[]")
dns_search=$(awk '/^search/ {$1=""; print}' /etc/resolv.conf 2>/dev/null | xargs)

RESULT=$(jq -n \
    --arg fw_type "$fw_type" \
    --arg fw_rules "$fw_rules" \
    --argjson listen "$listening" \
    --argjson routes "$routes" \
    --arg gw "$default_gw" \
    --argjson dns "$dns_servers" \
    --arg search "$dns_search" \
    '{
        firewall: {type: $fw_type, rules_excerpt: $fw_rules},
        listening_ports: $listen,
        routing: {routes: $routes, default_gateway: $gw},
        dns: {servers: $dns, search_domain: $search}
    }')

finish_module "network-diag" "ok" "$RESULT" '{firewall_type: .firewall.type, listening_count: (.listening_ports | length), default_gw: .routing.default_gateway}'
