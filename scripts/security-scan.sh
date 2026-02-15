#!/usr/bin/env bash
# security-scan.sh — SSH config audit, services, SUID, kernel params, ports
SCRIPT_NAME="security-scan"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== Security Scan ==="

warnings=()

# ── SSH Config ──
ssh_config="/etc/ssh/sshd_config"
ssh_findings="[]"
if [ -f "$ssh_config" ]; then
    root_login=$(grep -i "^PermitRootLogin" "$ssh_config" 2>/dev/null | awk '{print $2}' || echo "default")
    password_auth=$(grep -i "^PasswordAuthentication" "$ssh_config" 2>/dev/null | awk '{print $2}' || echo "default")
    x11_fwd=$(grep -i "^X11Forwarding" "$ssh_config" 2>/dev/null | awk '{print $2}' || echo "default")
    ssh_port=$(grep -i "^Port " "$ssh_config" 2>/dev/null | awk '{print $2}' || echo "22")

    [ "$root_login" = "yes" ] && warnings+=("SSH: PermitRootLogin=yes")
    [ "$password_auth" != "no" ] && warnings+=("SSH: PasswordAuth not disabled")

    ssh_findings=$(jq -n \
        --arg rl "$root_login" --arg pa "$password_auth" --arg x11 "$x11_fwd" --arg port "$ssh_port" \
        '[{check:"PermitRootLogin",value:$rl},{check:"PasswordAuthentication",value:$pa},{check:"X11Forwarding",value:$x11},{check:"Port",value:$port}]')
fi

# ── Running services ──
_svc_out=$(systemctl list-units --type=service --state=running --no-pager --no-legend 2>/dev/null | awk '{print $1}') || true
if [ -n "$_svc_out" ]; then
    services_json=$(printf '%s\n' "$_svc_out" | json_array_from_lines "[]")
else
    services_json="[]"
fi

# ── SUID binaries ──
# Restrict to known paths to avoid unbounded scan on large filesystems (e.g. multi-TB /)
# -xdev: don't cross filesystem boundaries; timeout: safety net for slow NFS/Lustre mounts.
_suid_out=$(timeout "${SUID_SCAN_TIMEOUT:-30}" find /usr /bin /sbin /opt /var -xdev -perm -4000 -type f 2>/dev/null | head -50) || true
if [ -n "$_suid_out" ]; then
    suid_json=$(printf '%s\n' "$_suid_out" | json_array_from_lines "[]")
else
    suid_json="[]"
fi
suid_count=$(echo "$suid_json" | jq 'length' 2>/dev/null) || suid_count=0

# ── Key kernel security params ──
kernel_params=$(jq -n \
    --arg aslr "$(cat /proc/sys/kernel/randomize_va_space 2>/dev/null)" \
    --arg ptrace "$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo N/A)" \
    --arg dmesg "$(cat /proc/sys/kernel/dmesg_restrict 2>/dev/null)" \
    --arg sysrq "$(cat /proc/sys/kernel/sysrq 2>/dev/null)" \
    --arg ipfwd "$(cat /proc/sys/net/ipv4/ip_forward 2>/dev/null)" \
    '{aslr: $aslr, ptrace_scope: $ptrace, dmesg_restrict: $dmesg, sysrq: $sysrq, ip_forward: $ipfwd}')

[ "$(cat /proc/sys/kernel/randomize_va_space 2>/dev/null)" != "2" ] && warnings+=("ASLR not fully enabled")

# ── Open ports (external-facing) ──
_ports_out=$(ss -tulnp 2>/dev/null | awk 'NR>1 && $4 !~ /127\.0\.0/ && $4 !~ /::1/ && $4 !~ /\[::1\]/ {print $4}' | sort -u) || true
if [ -n "$_ports_out" ]; then
    open_ports=$(printf '%s\n' "$_ports_out" | json_array_from_lines "[]")
else
    open_ports="[]"
fi

# ── Assess ──
# Status uses suite-standard values: ok / warn / error / skipped
status="ok"
[ ${#warnings[@]} -gt 3 ] && status="warn"
[ ${#warnings[@]} -gt 6 ] && status="error"

if [ ${#warnings[@]} -gt 0 ]; then
    warnings_json=$(printf '%s\n' "${warnings[@]}" | json_array_from_lines "[]")
else
    warnings_json="[]"
fi

RESULT=$(jq -n \
    --argjson ssh "$ssh_findings" \
    --argjson services "$services_json" \
    --argjson suid "$suid_json" \
    --argjson kernel "$kernel_params" \
    --argjson ports "$open_ports" \
    --argjson warnings "$warnings_json" \
    --arg status "$status" \
    '{
        status: $status,
        ssh_config: $ssh,
        running_services_count: ($services | length),
        running_services: $services,
        suid_binaries_count: ($suid | length),
        suid_binaries: $suid,
        kernel_security: $kernel,
        external_open_ports: $ports,
        warnings: $warnings
    }')

finish_module "security-scan" "$status" "$RESULT" '{status, warning_count: (.warnings|length), warnings}'
