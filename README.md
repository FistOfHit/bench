# HPC Bench Suite

A comprehensive benchmarking and diagnostics suite for high-performance computing (HPC) systems. Runs hardware discovery, GPU/CPU/network/storage benchmarks, and produces structured JSON results plus a markdown report.

**Version:** 1.7 (see [VERSION](VERSION))

## Features

- **Bootstrap** — Detects OS and hardware, installs dependencies (jq, dmidecode, DCGM, NCCL, InfiniBand tools, etc.), optional `--check-only` for dry-run
- **Discovery & inventory** — CPU, GPU, topology, network, BMC, software audit
- **Benchmarks** — DCGM diagnostics, GPU burn-in, NCCL tests, NVLink bandwidth, STREAM, storage (fio), HPL (CPU/GPU), InfiniBand tests
- **Diagnostics** — Network, filesystem, thermal/power, security scan
- **Report** — Single markdown report and portable results archive

Results are written as JSON per module and can be consumed by the bundled report generator or external tooling.

## Requirements

- **OS:** Linux (tested on Ubuntu and RHEL/CentOS)
- **Privilege:** Root (or sudo) for bootstrap and full suite
- **Tools:** `jq`, `bash` 4+, optional: NVIDIA driver/CUDA, DCGM, NCCL, InfiniBand stack, Docker (for HPL-MxP)

Bootstrap will attempt to install core packages when run as root with network access; use `--check-only` to see what’s missing without installing.

## Target environment

**This suite is designed for bare-metal HPC servers.** Virtual machines (VMs) are not the intended target: several benchmarks (DCGM, nvbandwidth, HPL-MxP, InfiniBand, BMC) require real hardware or full GPU/PCIe topology and will skip or may fail in VMs. If you run on a VM anyway, the suite will skip unsupported modules and complete with a reduced set of results; use `HPC_QUICK=1` for shorter storage runs when iterating (e.g. CI or smoke tests). See [SKILL.md](SKILL.md) version history for VM-related behavior and fixes.

## Quick start

```bash
# Clone or unpack the suite, then from the repo root:
cd /path/to/hpc-bench

# 1. Bootstrap (install dependencies, detect hardware) — requires root
sudo bash scripts/bootstrap.sh

# 2. Run full suite (all phases: bootstrap, inventory, benchmarks, diagnostics, report)
sudo bash scripts/run-all.sh

# Or run in quick mode (short benchmarks: tiny HPL, 3s GPU burn, DCGM r1 only, etc.) to verify the suite end-to-end:
sudo bash scripts/run-all.sh --quick
```

Results go to `/var/log/hpc-bench/results/` by default (override with `HPC_RESULTS_DIR`). See **Viewing results** below for where to find the report and logs.

## Viewing results

**All results and logs live in one folder.** Default path: `/var/log/hpc-bench/results/`.

| What you want | Where to look |
|---------------|----------------|
| **Quick summary** — human-readable report | **`report.md`** (at the root of the results folder). Open this first for pass/fail, scores, and key numbers. |
| **Per-module JSON** — machine-readable results | One file per module: `bootstrap.json`, `gpu-burn.json`, `dcgm-diag.json`, `run-all.json`, etc., in the same folder. |
| **Logs** — if you need to debug or inspect stdout | **`logs/`** subfolder: e.g. `logs/gpu-burn-stdout.log`, `logs/dcgm-diag.log`, `logs/fio-seq-read.log`. |
| **Portable bundle** — to copy off the server | A timestamped tarball in the results folder: `hpc-bench-<hostname>-<timestamp>.tar.gz` (contains all JSON, report, and logs). |

**Example (default path):**

```text
/var/log/hpc-bench/results/
├── report.md          ← start here: quick inspection
├── report.json
├── run-all.json
├── bootstrap.json
├── gpu-burn.json
├── dcgm-diag.json
├── ...                 (one .json per module)
├── logs/               ← module logs when you need them
│   ├── bootstrap-stdout.log
│   ├── gpu-burn-stdout.log
│   ├── gpu-burn.log
│   └── ...
└── hpc-bench-<hostname>-<timestamp>.tar.gz
```

To use a different output directory and regenerate the report from existing results:

```bash
HPC_RESULTS_DIR=/path/to/results bash scripts/report.sh
```

## Running individual modules

You can run any script under `scripts/` on its own. Each script sources `lib/common.sh` and expects to be run from the repo (or with `HPC_BENCH_ROOT` set).

```bash
# Examples
bash scripts/bootstrap.sh --check-only   # Dependency check only
bash scripts/gpu-inventory.sh
bash scripts/gpu-burn.sh
bash scripts/report.sh                  # Regenerate report from existing results
```

For report-only, point at existing results:

```bash
HPC_RESULTS_DIR=/path/to/results bash scripts/report.sh
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HPC_BENCH_ROOT` | Script-derived | Root directory of the suite |
| `HPC_RESULTS_DIR` | `/var/log/hpc-bench/results` | Where JSON results and report are written |
| `HPC_LOG_DIR` | `$HPC_RESULTS_DIR/logs` | Module log files |
| `HPC_WORK_DIR` | `/tmp/hpc-bench-work` | Build and temporary working files |
| `MAX_MODULE_TIME` | `1800` | Timeout in seconds per module in `run-all.sh` |
| `HPC_KEEP_TOOLS` | `0` | Set to `1` to skip cleanup of built tools in work dir |
| `HPC_QUICK` | *(unset)* | Set to `1` or use `run-all.sh --quick` for quick benchmark mode: short runs (DCGM r1 only, 3s GPU burn, tiny HPL, short NCCL/STREAM/fio) to verify the suite end-to-end |

## Repository layout

```
├── VERSION              # Single source of truth for version (e.g. 1.4)
├── README.md            # This file
├── .pre-commit-config.yaml   # Optional lint/pre-commit hooks
├── lib/
│   ├── common.sh        # Shared bash helpers (logging, JSON, GPU spec lookup, etc.)
│   └── report-common.sh # Report helpers and scorecard logic (used by report.sh)
├── scripts/             # Executable modules
│   ├── bootstrap.sh     # Bootstrap and dependency install
│   ├── run-all.sh       # Master orchestrator (all phases)
│   ├── report.sh        # Report generator
│   ├── inventory.sh     # General / CPU inventory
│   ├── gpu-inventory.sh
│   ├── topology.sh
│   ├── network-inventory.sh
│   ├── bmc-inventory.sh
│   ├── software-audit.sh
│   ├── dcgm-diag.sh
│   ├── gpu-burn.sh
│   ├── nccl-tests.sh
│   ├── nvbandwidth.sh
│   ├── stream-bench.sh
│   ├── storage-bench.sh
│   ├── hpl-cpu.sh
│   ├── hpl-mxp.sh
│   ├── ib-tests.sh
│   ├── network-diag.sh
│   ├── filesystem-diag.sh
│   ├── thermal-power.sh
│   └── security-scan.sh
├── specs/
│   └── hardware-specs.json   # GPU/PCIe/NVLink reference specs
└── src/                  # Bundled benchmark sources
    ├── stream.c          # STREAM memory benchmark
    ├── gpu-burn/         # GPU burn-in (CUDA)
    └── nccl-tests/       # Minimal NCCL test binaries
```

## Linting and pre-commit

Optional: run [pre-commit](https://pre-commit.com/) to enforce trailing-whitespace removal, end-of-file newlines, YAML/JSON checks, and [ShellCheck](https://www.shellcheck.net/) on shell scripts.

```bash
pipx install pre-commit   # or: python3 -m venv .venv && source .venv/bin/activate && pip install pre-commit
pre-commit install       # hook runs on git commit
pre-commit run --all-files   # run once on entire repo
```

Config: [.pre-commit-config.yaml](.pre-commit-config.yaml). ShellCheck is skipped for `src/` (C/CUDA build trees).

## Concurrency and locking

`run-all.sh` uses an exclusive lock file at `$HPC_RESULTS_DIR/.hpc-bench.lock`. If another run is in progress, it exits with code 2. Remove the lock file only if you are sure no other instance is running.

## Exit codes (run-all.sh)

- **0** — All modules passed; report may still show warnings (conditional pass).
- **1** — One or more modules failed (acceptance gate).
- **2** — Another instance is running (lock held).

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE). Third-party code in `src/` (e.g. gpu-burn, nccl-tests, STREAM) may have their own licenses; refer to comments and upstream sources.
