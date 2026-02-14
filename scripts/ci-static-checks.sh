#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[ci-static] Running bash syntax checks..."
for f in scripts/*.sh lib/*.sh; do
    bash -n "$f"
done

echo "[ci-static] Installing pre-commit..."
_venv_dir="$(mktemp -d /tmp/hpc-bench-ci-venv.XXXXXX)"
trap 'rm -rf "$_venv_dir"' EXIT
python3 -m venv "$_venv_dir"
# shellcheck disable=SC1091
source "${_venv_dir}/bin/activate"
python3 -m pip install --upgrade pip >/dev/null
python3 -m pip install pre-commit >/dev/null

echo "[ci-static] Running pre-commit hooks..."
pre-commit run --all-files

echo "[ci-static] Static checks passed."
