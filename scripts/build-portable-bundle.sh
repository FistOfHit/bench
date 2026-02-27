#!/usr/bin/env bash
# build-portable-bundle.sh — Create a portable directory bundle for air-gapped runs
# Run from repo root. Requires: curl, tar. Optional: gcc (for pre-built STREAM), makeself (for single .run file).
# Output: dist/hpc-bench-portable-<VERSION>-linux-<arch>.tar.gz
# With --makeself: also dist/hpc-bench-portable-<VERSION>-linux-<arch>.run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

WANT_MAKESELF=0
for arg in "$@"; do
    case "$arg" in
        --makeself) WANT_MAKESELF=1 ;;
    esac
done
[ "${HPC_BUNDLE_MAKESELF:-0}" = "1" ] && WANT_MAKESELF=1

VERSION=$(tr -d '[:space:]' < VERSION 2>/dev/null || echo "unknown")
ARCH="${HPC_BUNDLE_ARCH:-$(uname -m)}"
OUTPUT_DIR="${HPC_BUNDLE_OUTPUT_DIR:-dist}"
SKIP_STREAM="${HPC_BUNDLE_SKIP_STREAM:-0}"

# jqlang/jq release: map kernel arch to jq binary name
case "$ARCH" in
    x86_64)  JQ_ARCH=amd64 ;;
    aarch64) JQ_ARCH=arm64 ;;
    armv7l)  JQ_ARCH=armhf ;;
    *)       JQ_ARCH="$ARCH" ;;
esac

JQ_VERSION="1.7.1"
JQ_URL="https://github.com/jqlang/jq/releases/download/jq-${JQ_VERSION}/jq-linux-${JQ_ARCH}"
BUNDLE_NAME="hpc-bench-portable-${VERSION}-linux-${ARCH}"
BUNDLE_DIR="${OUTPUT_DIR}/${BUNDLE_NAME}"

echo "[build-portable-bundle] Building ${BUNDLE_NAME}"
echo "[build-portable-bundle] Output directory: ${OUTPUT_DIR}"

mkdir -p "$OUTPUT_DIR"
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR"

# Copy repo tree (exclude .git, tests, dist, dev files)
for item in scripts lib specs conf src reporting examples VERSION README.md LICENSE; do
    if [ -e "$REPO_ROOT/$item" ]; then
        cp -a "$REPO_ROOT/$item" "$BUNDLE_DIR/"
    fi
done

mkdir -p "$BUNDLE_DIR/bin"

# Download static jq
echo "[build-portable-bundle] Downloading jq ${JQ_VERSION} (${JQ_ARCH})..."
if ! curl -fsSL "$JQ_URL" -o "$BUNDLE_DIR/bin/jq"; then
    echo "ERROR: Failed to download jq from $JQ_URL (build host needs network)" >&2
    exit 1
fi
chmod +x "$BUNDLE_DIR/bin/jq"
echo "[build-portable-bundle] jq installed at bin/jq"

# Optional: pre-build STREAM (quick-mode defaults)
if [ "$SKIP_STREAM" != "1" ] && command -v gcc &>/dev/null; then
    STREAM_SRC="$BUNDLE_DIR/src/stream.c"
    if [ -f "$STREAM_SRC" ]; then
        echo "[build-portable-bundle] Building STREAM (quick-mode size)..."
        ARRAY_SIZE=1000000
        NTIMES=3
        if gcc -O3 -march=native -fopenmp \
            -DSTREAM_ARRAY_SIZE=$ARRAY_SIZE -DNTIMES=$NTIMES \
            "$STREAM_SRC" -o "$BUNDLE_DIR/bin/stream" 2>/dev/null; then
            chmod +x "$BUNDLE_DIR/bin/stream"
            echo "[build-portable-bundle] STREAM installed at bin/stream"
        else
            gcc -O3 -DSTREAM_ARRAY_SIZE=$ARRAY_SIZE -DNTIMES=$NTIMES \
                "$STREAM_SRC" -o "$BUNDLE_DIR/bin/stream" 2>/dev/null && {
                chmod +x "$BUNDLE_DIR/bin/stream"
                echo "[build-portable-bundle] STREAM installed at bin/stream (no OpenMP)"
            } || echo "[build-portable-bundle] STREAM build failed (non-fatal)"
        fi
    fi
elif [ "$SKIP_STREAM" = "1" ]; then
    echo "[build-portable-bundle] Skipping STREAM build (HPC_BUNDLE_SKIP_STREAM=1)"
else
    echo "[build-portable-bundle] gcc not found — skipping STREAM (suite will build on target if gcc present)"
fi

# Launcher script
cat > "$BUNDLE_DIR/run.sh" << 'LAUNCHER'
#!/usr/bin/env bash
# Launcher for portable bundle. Usage: ./run.sh [--quick] [--smoke] ...
ROOT="$(cd "$(dirname "$0")" && pwd)"
export HPC_BENCH_ROOT="$ROOT"
export PATH="${ROOT}/bin:${PATH}"
export HPC_PORTABLE=1
exec bash "$ROOT/scripts/run-all.sh" "$@"
LAUNCHER
chmod +x "$BUNDLE_DIR/run.sh"
echo "[build-portable-bundle] Launcher written to run.sh"

# Create tarball
TARBALL="${OUTPUT_DIR}/${BUNDLE_NAME}.tar.gz"
tar czf "$TARBALL" -C "$OUTPUT_DIR" "$BUNDLE_NAME"
echo "[build-portable-bundle] Created: $TARBALL"

# Optional: single-file self-extracting archive (makeself)
if [ "$WANT_MAKESELF" = "1" ]; then
    MAKESELF_CMD=""
    for cmd in makeself makeself.sh; do
        if command -v "$cmd" &>/dev/null; then
            MAKESELF_CMD="$cmd"
            break
        fi
    done
    if [ -z "$MAKESELF_CMD" ]; then
        echo "ERROR: --makeself requested but makeself not found. Install with: apt install makeself" >&2
        echo "Or download from https://github.com/megastep/makeself" >&2
        exit 1
    fi
    RUN_FILE="${OUTPUT_DIR}/${BUNDLE_NAME}.run"
    echo "[build-portable-bundle] Creating single-file .run (makeself)..."
    "$MAKESELF_CMD" --notemp "$BUNDLE_DIR" "$RUN_FILE" "HPC Bench portable bundle ${BUNDLE_NAME}" "./run.sh"
    chmod +x "$RUN_FILE"
    echo "[build-portable-bundle] Created: $RUN_FILE"
    echo ""
    echo "Single-file usage on target:"
    echo "  sh ${BUNDLE_NAME}.run           # extracts to ./${BUNDLE_NAME}/ and runs suite"
    echo "  sh ${BUNDLE_NAME}.run --quick   # same, passes --quick to run.sh"
    echo "  (To re-run without re-extracting: cd ${BUNDLE_NAME} && ./run.sh --quick)"
    echo ""
fi

echo "Done. To use on target (tarball):"
echo "  1. Copy $TARBALL to the server (e.g. via USB)"
echo "  2. Extract: tar xzf ${BUNDLE_NAME}.tar.gz && cd ${BUNDLE_NAME}"
echo "  3. Run:     ./run.sh          # full run"
echo "              ./run.sh --quick  # short benchmarks"
echo "              ./run.sh --smoke  # bootstrap + inventory + report only"
echo ""
