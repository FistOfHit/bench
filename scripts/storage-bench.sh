#!/usr/bin/env bash
# storage-bench.sh — fio storage benchmarks (sequential, random, mixed profiles)
SCRIPT_NAME="storage-bench"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== Storage Benchmark (fio) ==="

if ! has_cmd fio; then
    log_warn "fio not found — skipping"
    echo '{"note":"fio not available"}' | emit_json "storage-bench" "skipped"
    exit 0
fi

# ── Test directory ──
TEST_DIR="${FIO_TEST_DIR:-/tmp/hpc-fio-test}"
mkdir -p "$TEST_DIR"
register_cleanup "$TEST_DIR"

# Check available space
avail_kb=$(df "$TEST_DIR" | awk 'NR==2 {print $4}')
avail_gb=$((avail_kb / 1048576))
if [ "$avail_gb" -lt 2 ]; then
    log_warn "Only ${avail_gb}GB available, need 2GB+ for fio. Using smaller files."
    FIO_SIZE="512M"
else
    FIO_SIZE="4G"
fi

# Quick mode: 5s per profile and only 2 profiles (seq-read + rand-4k-read) to verify fio fast. Full: 60s, 7 profiles.
if [ "${HPC_QUICK:-0}" = "1" ]; then
    FIO_RUNTIME=5
    FIO_QUICK_PROFILES=1   # 1 = run only 2 minimal profiles
else
    FIO_RUNTIME=${HPC_QUICK:+15}
    FIO_RUNTIME=${FIO_RUNTIME:-60}
    FIO_QUICK_PROFILES=0
fi
FIO_COMMON="--directory=$TEST_DIR --size=$FIO_SIZE --runtime=$FIO_RUNTIME --time_based --group_reporting"

# Placeholder for skipped profiles in quick mode (valid JSON for jq)
FIO_PLACEHOLDER='{"read_bw_mbps":0,"read_iops":0,"read_lat_usec":0,"write_bw_mbps":0,"write_iops":0,"write_lat_usec":0}'

# ── Run fio profiles ──
run_fio() {
    local name="$1"; shift
    log_info "fio: $name"
    local output_file="/tmp/fio-${name}-$$.json"
    rm -f "$output_file"

    # Use fio's native --output flag to write JSON directly to file
    # This avoids stdout contamination from progress/status messages
    timeout 120 fio --name="$name" $FIO_COMMON --output-format=json --output="$output_file" "$@" \
        2>>"${HPC_LOG_DIR}/fio-${name}.log" || true

    # Parse the JSON output
    if [ -f "$output_file" ] && [ -s "$output_file" ]; then
        local parsed
        # fio JSON may have trailing text; extract the JSON object
        parsed=$(jq '{
            read_bw_mbps: ((.jobs[0].read.bw // 0) / 1024),
            read_iops: (.jobs[0].read.iops // 0),
            read_lat_usec: ((.jobs[0].read.lat_ns.mean // 0) / 1000),
            write_bw_mbps: ((.jobs[0].write.bw // 0) / 1024),
            write_iops: (.jobs[0].write.iops // 0),
            write_lat_usec: ((.jobs[0].write.lat_ns.mean // 0) / 1000)
        }' "$output_file" 2>/dev/null)

        if [ -n "$parsed" ] && echo "$parsed" | jq . >/dev/null 2>&1; then
            rm -f "$output_file"
            echo "$parsed"
            return
        fi
    fi

    rm -f "$output_file"
    log_warn "fio $name: failed to parse output"
    echo '{"error":"parse failed"}'
}

# Quick mode: run only 2 profiles (seq-read + rand-4k-read) to verify fio with minimal time
if [ "${FIO_QUICK_PROFILES:-0}" = "1" ]; then
    log_info "Quick mode — running 2 fio profiles (${FIO_RUNTIME}s each)"
    seq_read=$(run_fio "seq-read" --rw=read --bs=1M --iodepth=32 --numjobs=1 --direct=1)
    rand_read=$(run_fio "rand-4k-read" --rw=randread --bs=4k --iodepth=64 --numjobs=4 --direct=1)
    seq_write="$FIO_PLACEHOLDER"
    rand_write="$FIO_PLACEHOLDER"
    mixed="$FIO_PLACEHOLDER"
    seq_read_deep="$FIO_PLACEHOLDER"
else
    seq_read=$(run_fio "seq-read" --rw=read --bs=1M --iodepth=32 --numjobs=1 --direct=1)
    seq_write=$(run_fio "seq-write" --rw=write --bs=1M --iodepth=32 --numjobs=1 --direct=1)
    rand_read=$(run_fio "rand-4k-read" --rw=randread --bs=4k --iodepth=64 --numjobs=4 --direct=1)
    rand_write=$(run_fio "rand-4k-write" --rw=randwrite --bs=4k --iodepth=64 --numjobs=4 --direct=1)
    mixed=$(run_fio "mixed-randrw" --rw=randrw --rwmixread=70 --bs=4k --iodepth=64 --numjobs=4 --direct=1)
    seq_read_deep=$(run_fio "seq-read-qd128" --rw=read --bs=128k --iodepth=128 --numjobs=1 --direct=1)
fi

# ── Detect storage type ──
test_device=$(df "$TEST_DIR" | awk 'NR==2 {print $1}')
base_device=$(lsblk -no PKNAME "$test_device" 2>/dev/null | head -1)
[ -z "$base_device" ] && base_device=$(basename "$test_device")
rotational=$(cat "/sys/block/${base_device}/queue/rotational" 2>/dev/null || echo "unknown")
scheduler=$(cat "/sys/block/${base_device}/queue/scheduler" 2>/dev/null || echo "unknown")

RESULT=$(jq -n \
    --argjson seq_r "$seq_read" \
    --argjson seq_w "$seq_write" \
    --argjson rand_r "$rand_read" \
    --argjson rand_w "$rand_write" \
    --argjson mixed "$mixed" \
    --argjson seq_deep "$seq_read_deep" \
    --arg fio_size "$FIO_SIZE" \
    --arg fio_runtime "$FIO_RUNTIME" \
    --arg test_dir "$TEST_DIR" \
    --arg device "$test_device" \
    --arg rota "$rotational" \
    --arg sched "$scheduler" \
    '{
        test_config: {size: $fio_size, runtime_sec: ($fio_runtime | tonumber), test_dir: $test_dir, device: $device, rotational: $rota, scheduler: $sched},
        sequential_read_1M: $seq_r,
        sequential_write_1M: $seq_w,
        random_4k_read: $rand_r,
        random_4k_write: $rand_w,
        mixed_randrw_70_30: $mixed,
        sequential_read_128k_qd128: $seq_deep
    }')

echo "$RESULT" | emit_json "storage-bench" "ok"
log_ok "Storage benchmarks complete"
echo "$RESULT" | jq '{
    "seq_read_MB/s": .sequential_read_1M.read_bw_mbps,
    "seq_write_MB/s": .sequential_write_1M.write_bw_mbps,
    "rand_4k_read_IOPS": .random_4k_read.read_iops,
    "rand_4k_write_IOPS": .random_4k_write.write_iops
}'
