#!/usr/bin/env bash
# filesystem-diag.sh -- Mount points, FS types, RAID, LVM, NFS/Lustre/GPFS
# Phase: 4 (diagnostic)
# Requires: jq, awk
# Emits: filesystem-diag.json
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
    mounts_json=$(json_compact_or "$mounts_json" "[]")
fi
mounts_json=$(json_compact_or "$mounts_json" "[]")

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
    pfs_json=$(printf '%s\n' "${pfs_found[@]}" | json_array_from_lines "[]")
else
    pfs_json="[]"
fi

# ── RAID ──
raid_json="{}"
if has_cmd mdadm; then
    md_arrays=0
    if [ -r /proc/mdstat ]; then
        md_arrays=$(count_grep_re '^md' < /proc/mdstat)
    fi
    md_detail=$(mdadm --detail --scan 2>/dev/null || echo "none")
    md_detail=$(printf '%s' "$md_detail" | tr -d '\000-\037"' | head -c 2000)
    raid_json=$(jq -n --arg n "$md_arrays" --arg d "$md_detail" '{type:"mdadm",arrays:($n|tonumber),detail:$d}' 2>/dev/null) || true
fi
raid_json=$(json_compact_or "$raid_json" "{}")
if [ "$raid_json" = '{}' ] && has_cmd megacli; then
    raid_json=$(jq -n '{type:"megacli",detail:"present"}')
elif [ "$raid_json" = '{}' ] && has_cmd storcli; then
    raid_json=$(jq -n '{type:"storcli",detail:"present"}')
fi

# ── LVM ──
lvm_json="[]"
if has_cmd lvs; then
    _lvs_out=$(try_sudo lvs --noheadings --reportformat json 2>/dev/null) || true
    if [ -n "$_lvs_out" ]; then
        lvm_json=$(echo "$_lvs_out" | jq '.report[0].lv // []' 2>/dev/null) || true
    fi
fi
lvm_json=$(json_compact_or "$lvm_json" "[]")
pfs_json=$(json_compact_or "$pfs_json" "[]")

# Use temp files for large/embedded JSON to avoid arg length and escaping with --argjson
_tmp_m=$(json_tmpfile "fs_mounts" "$mounts_json" "[]")
_tmp_p=$(json_tmpfile "fs_pfs" "$pfs_json" "[]")
_tmp_r=$(json_tmpfile "fs_raid" "$raid_json" "{}")
_tmp_l=$(json_tmpfile "fs_lvm" "$lvm_json" "[]")

RESULT=$(jq -n \
    --slurpfile mounts "$_tmp_m" \
    --slurpfile pfs "$_tmp_p" \
    --slurpfile raid "$_tmp_r" \
    --slurpfile lvm "$_tmp_l" \
    '{mount_count: ($mounts[0] | length), mounts: $mounts[0], parallel_filesystem_count: ($pfs[0] | length), parallel_filesystems: $pfs[0], raid: $raid[0], lv_count: ($lvm[0] | length), lvm: $lvm[0]}')

finish_module "filesystem-diag" "ok" "$RESULT" '{mount_count: (.mounts|length), parallel_fs: .parallel_filesystems, raid_type: .raid.type}'
