# HPC Benchmark Suite — Agent guide

This doc tells an AI agent how to work with the HPC Bench Suite: how to run it, what context to assume, and how to debug. For **version history and changelog**, see [CHANGELOG.md](CHANGELOG.md).

---

## What this suite is

- **Purpose:** Benchmark and inventory HPC servers (GPU, CPU, network, storage, topology). Produces JSON per module and a markdown report.
- **Target:** Bare-metal HPC. VMs are supported for convenience; many modules skip or degrade (DCGM, nvbandwidth, HPL-MxP, InfiniBand, BMC). See README “Target environment” and CHANGELOG for VM-related behavior.

---

## How to use it (like a human would)

- **Full run (root):** `sudo bash scripts/bootstrap.sh` then `sudo bash scripts/run-all.sh`.
- **Quick validation:** `sudo bash scripts/run-all.sh --quick` (short benchmarks).
- **Smoke (no benchmarks):** `sudo bash scripts/run-all.sh --smoke` (bootstrap + discovery + report).
- **NVIDIA driver + CUDA on Ubuntu:** `sudo bash scripts/bootstrap.sh --install-nvidia` (internet required; reboot after driver install, then run bootstrap again).
- **Check deps only:** `bash scripts/bootstrap.sh --check-only` (no root; lists present/missing).
- **Single module:** e.g. `bash scripts/gpu-inventory.sh`, `bash scripts/gpu-burn.sh` — run from repo root; most need prior bootstrap and root for full behavior.
- **Results:** Default `$HPC_RESULTS_DIR` is `/var/log/hpc-bench/results` (root) or `$HOME/.local/var/hpc-bench/results` (non-root). Report: `report.md`; logs: `logs/`; per-module JSON: `bootstrap.json`, `gpu-burn.json`, etc.

**Context to assume:** User may be on a VM, air-gapped, or without NVIDIA GPU. Scripts skip modules when prerequisites are missing (e.g. no IB hardware, no nvidia-smi) and emit JSON with `status: "skipped"` and `.note`/`.skip_reason`. Exit codes: 0 = success or conditional pass; 1 = at least one non-skipped module failed; 2 = lockfile held (another run in progress).

---

## Debugging

- **Logs:** Each module logs to `$HPC_LOG_DIR/<SCRIPT_NAME>.log` (e.g. `logs/bootstrap.log`). Stdout from benchmarks often in `logs/<module>-stdout.log` or similar.
- **Lock:** Concurrent runs use `$HPC_RESULTS_DIR/.hpc-bench.lock`. Exit 2 if locked; remove lock only if no other run is active.
- **Crash records:** If a module exits without writing its JSON (e.g. crash, kill), the EXIT trap in common.sh writes a crash record for that module so the report still has an entry.
- **JSON validity:** Many scripts pipe data into `jq`. Control characters or invalid JSON in variables (e.g. from `systemd-detect-virt`, DMI, or nvidia-smi) can cause “parse error: Invalid string: control characters…”. Fix by sanitizing before building JSON (e.g. strip U+0000–U+001F or use `sanitize_json_str` / temp files + `--slurpfile`).
- **Bootstrap “no internet”:** Connectivity check tries several URLs; on failure it logs a “Connectivity check hint” (first line of curl error). If the user has a proxy, they may need `sudo -E` or to set HTTP_PROXY/HTTPS_PROXY for root. If “no NVIDIA GPU” but the machine has GPUs, detection uses `lspci` then fallback `/sys/bus/pci/devices/*/vendor` (0x10de); ensure pciutils is installed or rely on the sys fallback.
- **Module-specific:** See CHANGELOG for known bugs and fixes (e.g. gpu-inventory VM-safe behavior, nccl-tests parser, storage-bench fio output, report field names). Specs and GPU lookup: `specs/hardware-specs.json` and `lookup_gpu_spec()` in `lib/common.sh` (fuzzy match by GPU name).

---

## Key files for agents

| Need | File / location |
|------|------------------|
| User-facing usage | [README.md](README.md) |
| Version number | [VERSION](VERSION) |
| Changelog / version history | [CHANGELOG.md](CHANGELOG.md) |
| Shared helpers, logging, emit_json, detect_virtualization | [lib/common.sh](lib/common.sh) |
| Entry points | [scripts/bootstrap.sh](scripts/bootstrap.sh), [scripts/run-all.sh](scripts/run-all.sh) |
| GPU spec lookup | [lib/common.sh](lib/common.sh) `lookup_gpu_spec`, [specs/hardware-specs.json](specs/hardware-specs.json) |
