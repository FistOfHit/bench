#!/usr/bin/env bash
# ib-tests.sh — InfiniBand perftest benchmarks (loopback/single-node) (V1.1)
# Added: Virtualization detection - skips gracefully in VMs
SCRIPT_NAME="ib-tests"
source "$(dirname "$0")/../lib/common.sh"

log_info "=== InfiniBand Tests ==="

# ── Check virtualization first ──
VIRT_INFO=$(detect_virtualization)
VIRT_TYPE=$(echo "$VIRT_INFO" | jq -r '.type')

if [ "$VIRT_TYPE" != "none" ]; then
    log_warn "Running in virtualized environment ($VIRT_TYPE) — InfiniBand not accessible"
    echo '{"note":"InfiniBand not available in virtualized environment","virtualization":'$VIRT_INFO'}' | emit_json "ib-tests" "skipped"
    exit 0
fi

# Check for IB hardware
if ! ls /sys/class/infiniband/*/ports/*/state 2>/dev/null | head -1 | grep -q .; then
    log_warn "No InfiniBand hardware detected"
    echo '{"note":"no IB hardware"}' | emit_json "ib-tests" "skipped"
    exit 0
fi

# Check for perftest tools
for tool in ib_write_bw ib_read_bw ib_send_lat; do
    if ! has_cmd "$tool"; then
        log_warn "$tool not found — install perftest package"
        echo '{"note":"perftest not installed"}' | emit_json "ib-tests" "skipped"
        exit 0
    fi
done

# Find active IB device
IB_DEV=$(ibstat 2>/dev/null | awk '/^CA / {dev=$2; gsub(/'\''/, "", dev)} /State:.*Active/ {print dev; exit}')
if [ -z "$IB_DEV" ]; then
    log_warn "No active IB port found"
    echo '{"note":"no active IB port"}' | emit_json "ib-tests" "skipped"
    exit 0
fi
log_info "Using IB device: $IB_DEV"

# ── Check and fix memlock ulimit ──
memlock_check="ok"
original_memlock_kb=$(ulimit -l 2>/dev/null || echo "0")
memlock_value="$original_memlock_kb"
if [ "$original_memlock_kb" = "unlimited" ]; then
    log_info "memlock ulimit: unlimited (good)"
    memlock_check="ok"
else
    log_info "Original memlock ulimit: ${original_memlock_kb} KB"
    # Since we run as root, elevate memlock for this benchmark session
    log_info "Elevating memlock to unlimited for benchmark session"
    ulimit -l unlimited 2>/dev/null && {
        log_info "memlock ulimit set to unlimited (was ${original_memlock_kb} KB)"
        memlock_check="ok_elevated"
    } || {
        log_warn "Failed to set memlock unlimited — RDMA performance may be degraded"
        log_warn "Configure /etc/security/limits.conf: * memlock unlimited"
        memlock_check="warn"
    }
fi

# Get IB link info
ib_rate=$(ibstat "$IB_DEV" 2>/dev/null | awk '/Rate:/ {print $2; exit}')
ib_state=$(ibstat "$IB_DEV" 2>/dev/null | awk '/State:/ {print $2; exit}')

# ── Run loopback tests (server + client on same host) ──
run_ib_test() {
    local test_cmd="$1" test_name="$2"
    local port=$((19000 + RANDOM % 1000))

    log_info "Running $test_name (loopback on $IB_DEV, port $port)..."

    # Start server in background
    $test_cmd -d "$IB_DEV" -p "$port" --report_gbits &>/dev/null &
    local srv_pid=$!
    sleep 2

    # Run client
    local output
    output=$(run_with_timeout 120 "$test_name" \
        $test_cmd -d "$IB_DEV" -p "$port" --report_gbits localhost 2>&1) || true

    kill "$srv_pid" 2>/dev/null; wait "$srv_pid" 2>/dev/null

    # Parse: look for the peak BW line (last data line)
    # Format varies but generally: #bytes #iterations BW_peak[Gb/sec] BW_average[Gb/sec] MsgRate[Mpps]
    local peak_bw avg_bw
    peak_bw=$(echo "$output" | awk '/^[0-9]/ {bw=$(NF-1)} END {print bw+0}')
    avg_bw=$(echo "$output" | awk '/^[0-9]/ {bw=$NF} END {print bw+0}')

    # For latency tests: look for avg latency
    local avg_lat
    avg_lat=$(echo "$output" | awk '/^[0-9]/ {lat=$(NF-2)} END {print lat+0}')

    echo "{\"peak_gbps\":$peak_bw,\"avg_gbps\":$avg_bw,\"avg_lat_usec\":$avg_lat}"
}

write_bw=$(run_ib_test "ib_write_bw" "ib_write_bw")
read_bw=$(run_ib_test "ib_read_bw" "ib_read_bw")
send_lat=$(run_ib_test "ib_send_lat" "ib_send_lat")

# Theoretical for context
theo_gbps=0
case "$ib_rate" in
    100) theo_gbps=100 ;;
    200) theo_gbps=200 ;;
    400) theo_gbps=400 ;;
    56)  theo_gbps=56 ;;
    *) theo_gbps=${ib_rate:-0} ;;
esac

RESULT=$(jq -n \
    --arg dev "$IB_DEV" \
    --arg rate "$ib_rate" \
    --arg state "$ib_state" \
    --argjson write "$write_bw" \
    --argjson read "$read_bw" \
    --argjson lat "$send_lat" \
    --arg theo "$theo_gbps" \
    --arg memlock_check "$memlock_check" \
    --arg memlock_value "$memlock_value" \
    --arg original_memlock "$original_memlock_kb" \
    '{
        device: $dev,
        link_rate_gbps: $rate,
        link_state: $state,
        ib_write_bw: $write,
        ib_read_bw: $read,
        ib_send_lat: $lat,
        theoretical_rate_gbps: ($theo | tonumber),
        memlock_ulimit_kb: $memlock_value,
        original_memlock_kb: $original_memlock,
        memlock_check: $memlock_check,
        loopback_disclaimer: "WARNING: These are loopback (single-node) tests only. They validate NIC hardware and driver health but do NOT test switch fabric, cabling, inter-node connectivity, or network congestion. Multi-node perftest (ib_write_bw between hosts) is required to validate the full fabric.",
        test_mode: "loopback"
    }')

echo "$RESULT" | emit_json "ib-tests" "ok"
log_ok "IB tests complete on $IB_DEV (rate: ${ib_rate} Gb/s)"
echo "$RESULT" | jq .
