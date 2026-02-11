#!/usr/bin/env bash
# filesystem-diag.sh — Mount points, FS types, RAID, LVM, NFS/Lustre/GPFS
SCRIPT_NAME="filesystem-diag"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== Filesystem Diagnostics ==="

# ── Mount points ──
_df_out=$(df -hT 2>/dev/null) || true
mounts_json="[]"
if [ -n "$_df_out" ]; then
    mounts_json=$(echo "$_df_out" | awk '
    BEGIN { print "["; first=1 }
    NR>1 {
        if(!first) printf ","
        first=0
        printf "{\"filesystem\":\"%s\",\"type\":\"%s\",\"size\":\"%s\",\"used\":\"%s\",\"avail\":\"%s\",\"use_pct\":\"%s\",\"mountpoint\":\"%s\"}", $1,$2,$3,$4,$5,$6,$7
    }
    END { print "]" }
    ') || mounts_json="[]"
    echo "$mounts_json" | jq . >/dev/null 2>&1 || mounts_json="[]"
fi

# ── Parallel/network filesystems ──
pfs_json="[]"
pfs_found=()
# Lustre
if mount 2>/dev/null | grep -q lustre 2>/dev/null; then
    pfs_found+=("lustre")
fi
# GPFS/Spectrum Scale
if has_cmd mmlsfs || mount 2>/dev/null | grep -q gpfs 2>/dev/null; then
    pfs_found+=("gpfs")
fi
# NFS
nfs_mounts=$(mount | grep -c "type nfs" || true)
if [ "${nfs_mounts:-0}" -gt 0 ] 2>/dev/null; then
    pfs_found+=("nfs")
fi
# BeeGFS
if mount | grep -q beegfs; then
    pfs_found+=("beegfs")
fi
if [ ${#pfs_found[@]} -gt 0 ]; then
    pfs_json=$(printf '%s\n' "${pfs_found[@]}" | jq -R . | jq -s '.') || pfs_json="[]"
else
    pfs_json="[]"
fi

# ── RAID ──
raid_json="{}"
if has_cmd mdadm; then
    md_arrays=$(cat /proc/mdstat 2>/dev/null | grep "^md" | wc -l)
    md_detail=$(mdadm --detail --scan 2>/dev/null || echo "none")
    raid_json=$(jq -n --arg n "$md_arrays" --arg d "$md_detail" '{type:"mdadm",arrays:($n|tonumber),detail:$d}')
elif has_cmd megacli; then
    raid_json=$(jq -n '{type:"megacli",detail:"present"}')
elif has_cmd storcli; then
    raid_json=$(jq -n '{type:"storcli",detail:"present"}')
fi

# ── LVM ──
lvm_json="[]"
if has_cmd lvs; then
    _lvs_out=$(try_sudo lvs --noheadings --reportformat json 2>/dev/null) || true
    if [ -n "$_lvs_out" ]; then
        lvm_json=$(echo "$_lvs_out" | jq '.report[0].lv // []' 2>/dev/null) || lvm_json="[]"
    fi
fi

RESULT=$(jq -n \
    --argjson mounts "$mounts_json" \
    --argjson pfs "$pfs_json" \
    --argjson raid "$raid_json" \
    --argjson lvm "$lvm_json" \
    '{mounts: $mounts, parallel_filesystems: $pfs, raid: $raid, lvm: $lvm}')

echo "$RESULT" | emit_json "filesystem-diag" "ok"
log_ok "Filesystem diagnostics complete"
echo "$RESULT" | jq '{mount_count: (.mounts|length), parallel_fs: .parallel_filesystems, raid_type: .raid.type}'
