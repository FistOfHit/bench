# HPC Bench Suite

A comprehensive benchmarking and diagnostics suite for high-performance computing (HPC) systems. Runs hardware discovery, GPU/CPU/network/storage benchmarks, and produces structured JSON results plus a markdown report.

**Version:** 1.10 (see [VERSION](VERSION))

## Features

- **Bootstrap** — Detects OS and hardware, installs dependencies (jq, curl, dmidecode, DCGM, NCCL, InfiniBand tools, Boost for nvbandwidth when GPU present, etc.). Use `--check-only` for a dry-run. On Ubuntu with an NVIDIA GPU, use `--install-nvidia` to install the NVIDIA driver and CUDA toolkit (internet required; reboot after driver install). Use `--install-nvidia-container-toolkit` to install/configure Docker + NVIDIA container runtime (needed for `hpl-mxp`).
- **Discovery & inventory** — CPU, GPU, topology, network, BMC, software audit
- **Benchmarks** — DCGM diagnostics, GPU burn-in, NCCL tests, NVLink bandwidth, STREAM, storage (fio), HPL (CPU/GPU), InfiniBand tests
- **Diagnostics** — Network, filesystem, thermal/power, security scan
- **Report** — Single markdown report and portable results archive

Results are written as JSON per module and can be consumed by the bundled report generator or external tooling.

## Requirements

- **OS:** Linux (tested on Ubuntu and RHEL/CentOS)
- **Privilege:** Root (or sudo) for bootstrap and full suite. Non-root runs are supported: results go to `$HOME/.local/var/hpc-bench/results` and root-only modules (bootstrap, bmc-inventory) are skipped.
- **Tools:** `jq` (1.6+), `bash` (4+), `python3` (required by several helpers in `lib/common.sh`). Bootstrap installs jq when missing. Optional: NVIDIA driver/CUDA, DCGM, NCCL, InfiniBand stack, Docker (for HPL-MxP). **DCGM is optional:** if it is not available or all diagnostic levels fail (e.g. in some VMs), the `dcgm-diag` module is skipped with a clear note rather than failing the suite.

Bootstrap installs jq and other core packages when run as root with network access; use `--check-only` to see what’s missing without installing. When GPUs are detected, Boost (libboost-dev / boost-devel) is also installed to improve nvbandwidth build-from-source; nvbandwidth can still skip on some distros and is non-fatal. **Installing NVIDIA stack:** On a fresh Ubuntu system with an NVIDIA GPU, run `sudo bash scripts/bootstrap.sh --install-nvidia`. Internet is required. After the driver is installed, reboot and run bootstrap again (with or without `--install-nvidia`) to complete CUDA toolkit, DCGM, and NCCL setup. To enable GPU-capable containers for `hpl-mxp`, run `sudo bash scripts/bootstrap.sh --install-nvidia-container-toolkit`. You can also combine both flags in one workflow, then reboot when prompted, and run the same combined command once more.

## Target environment

**This suite is designed for bare-metal HPC servers.** Virtual machines (VMs) are not the intended target: several benchmarks (DCGM, nvbandwidth, HPL-MxP, InfiniBand, BMC) require real hardware or full GPU/PCIe topology and will skip or may fail in VMs. If you run on a VM anyway, the suite will skip unsupported modules and complete with a reduced set of results; use `HPC_QUICK=1` for shorter storage runs when iterating (e.g. CI or smoke tests). See [CHANGELOG.md](CHANGELOG.md) for version history and VM-related behavior and fixes.

### VM behavior (why some modules skip)

- **DCGM diagnostics (`dcgm-diag`):** The script runs `dcgmi diag` with a shorter per-level timeout (120s) when virtualized. DCGM is built for bare-metal datacenter GPUs; in VMs with GPU passthrough, diagnostics often take much longer, hang, or fail with errors such as "unsupported Cuda version" (exit 226). If every attempted level (3, 2, 1) fails or times out, the module is skipped with a note that all DCGM levels failed in the VM.
- **HPL-MxP (`hpl-mxp`):** The benchmark runs inside a GPU-capable container. In VMs, the container process often exits with **SIGPIPE (exit 141)** when writing to stdout—e.g. the pipe to the host breaks due to timeout/container runtime behavior with GPU passthrough. The script treats that as a VM-specific skip so the suite can still pass; use `HPC_HPL_MXP_VM_STRICT=1` to treat it as a hard failure instead.

## Quick start

```bash
# Clone or unpack the suite, then from the repo root:
cd /path/to/hpc-bench

# 1. Bootstrap (install dependencies, detect hardware) — requires root
sudo bash scripts/bootstrap.sh

# On fresh Ubuntu with an NVIDIA GPU, install driver + CUDA (internet required; reboot after driver install, then run again):
#   sudo bash scripts/bootstrap.sh --install-nvidia
#   # Or pass to run-all: sudo bash scripts/run-all.sh --quick --install-nvidia
# Optional: install Docker + NVIDIA container runtime (for hpl-mxp):
#   sudo bash scripts/bootstrap.sh --install-nvidia-container-toolkit
# Combined flow (recommended when you want hpl-mxp too):
#   sudo bash scripts/bootstrap.sh --install-nvidia --install-nvidia-container-toolkit
#   # reboot when prompted
#   sudo bash scripts/bootstrap.sh --install-nvidia --install-nvidia-container-toolkit

# 2. Run full suite (all phases: bootstrap, inventory, benchmarks, diagnostics, report)
sudo bash scripts/run-all.sh

# Or run in quick mode (short benchmarks: tiny HPL, 3s GPU burn, DCGM r1 only, etc.) to verify the suite end-to-end:
sudo bash scripts/run-all.sh --quick

# Or smoke mode (bootstrap + inventory + report only, no benchmarks; under ~1 min):
sudo bash scripts/run-all.sh --smoke

# CI-friendly mode (implies quick mode, compacts module stdout, emits failure snippets):
sudo bash scripts/run-all.sh --ci

# Install NVIDIA stack (driver + CUDA) when GPU present; run-all exits for reboot, then re-run:
sudo bash scripts/run-all.sh --quick --install-nvidia

# Runtime sanity controls:
# - auto-install runtime if missing (Docker + NVIDIA container runtime)
sudo bash scripts/run-all.sh --quick --auto-install-runtime
# - fail fast immediately when runtime sanity fails
sudo bash scripts/run-all.sh --quick --fail-fast-runtime
# - optional: in VM, force hpl-mxp to fail instead of auto-skip on VM-specific container issues
sudo env HPC_HPL_MXP_VM_STRICT=1 bash scripts/hpl-mxp.sh
```

**Run on a remote host (single command):**

```bash
rsync -az --exclude .git /path/to/hpc-bench user@host:/tmp/hpc-bench
ssh user@host 'cd /tmp/hpc-bench && sudo ./scripts/run-all.sh --quick'
# Then fetch the report and archive:
scp user@host:/var/log/hpc-bench/results/report.md .
scp user@host:/var/log/hpc-bench/results/hpc-bench-*.tar.gz .
```

If SSH reports `REMOTE HOST IDENTIFICATION HAS CHANGED`, remove only the stale host entry (safer than deleting all known hosts):

```bash
ssh-keygen -R <host-or-ip>
# Example:
ssh-keygen -R 38.128.232.215
```

Results go to `/var/log/hpc-bench/results/` by default when run as root (override with `HPC_RESULTS_DIR`). When run as non-root, results go to `$HOME/.local/var/hpc-bench/results`. See **Viewing results** below for where to find the report and logs.

## Viewing results

**All results and logs live in one folder.** Default path: `/var/log/hpc-bench/results/`.

| What you want | Where to look |
| ------------- | -------------- |
| **Quick summary** — human-readable report | **`report.md`** (at the root of the results folder). Open this first for pass/fail, scores, and key numbers. |
| **Per-module JSON** — machine-readable results | One file per module: `bootstrap.json`, `gpu-burn.json`, `dcgm-diag.json`, `run-all.json`, etc., in the same folder. |
| **Logs** — if you need to debug or inspect stdout | **`logs/`** subfolder: e.g. `logs/gpu-burn-stdout.log`, `logs/dcgm-diag.log`, `logs/fio-seq-read.log`. |
| **Portable bundle** — to copy off the server | A timestamped tarball in the results folder: `hpc-bench-<hostname>-<timestamp>.tar.gz` (contains all JSON, report, and logs). Transfer with `scp` or `rsync`, e.g. `scp user@host:$HPC_RESULTS_DIR/hpc-bench-*.tar.gz .` or `rsync -av user@host:$HPC_RESULTS_DIR/ ./results/`. |
| **Sample report (sanitized)** — see format before running | [examples/report.md](examples/report.md) in the repo. |
| **Optional HTML report** — single-file, print-friendly | Generate with `python3 reporting/generate_html_report.py -i $HPC_RESULTS_DIR -o $HPC_RESULTS_DIR/report.html` (after running the suite). See below. |

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

To use a different output directory and regenerate the report from existing results (e.g. after copying results from another host or an air-gapped run):

```bash
HPC_RESULTS_DIR=/path/to/results bash scripts/report.sh
```

This **offline or air-gapped report regeneration** uses only the JSON files in the results directory; no network or re-run of benchmarks is required.

**Optional HTML report:** To generate a single-file HTML report (handy for sharing or printing), run the Python generator after the suite (or after regenerating `report.md`):

```bash
python3 reporting/generate_html_report.py -i "$HPC_RESULTS_DIR" -o "$HPC_RESULTS_DIR/report.html"
```

If `HPC_RESULTS_DIR` is set, you can omit `-i`. The HTML report includes device result, scorecard, hardware summary, and issues; it does not replace the default Markdown report.

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
| -------- | ------- | ----------- |
| `HPC_BENCH_ROOT` | Script-derived | Root directory of the suite |
| `HPC_RESULTS_DIR` | `/var/log/hpc-bench/results` | Where JSON results and report are written |
| `HPC_LOG_DIR` | `$HPC_RESULTS_DIR/logs` | Module log files |
| `HPC_WORK_DIR` | `/tmp/hpc-bench-work` | Build and temporary working files |
| `MAX_MODULE_TIME` | `1800` | Timeout in seconds per module in `run-all.sh` |
| `HPC_KEEP_TOOLS` | `0` | Set to `1` to keep gpu-burn, nccl-tests, STREAM builds in work dir; subsequent runs skip rebuilds (faster iteration). |
| `HPC_QUICK` | *(unset)* | Set to `1` or use `run-all.sh --quick` for quick benchmark mode: short runs (DCGM r1 only, 3s GPU burn, tiny HPL, short NCCL/STREAM/fio) to verify the suite end-to-end |
| `HPC_SMOKE` | *(unset)* | Set by `run-all.sh --smoke`: run only bootstrap, discovery/inventory, and report (no benchmarks); under ~1 min for “did it install and detect?” |
| `HPC_CI` | `0` | Set to `1` or use `run-all.sh --ci` for CI mode (enables quick-mode defaults, reduces interleaved stdout noise, prints log snippets on failure). |
| `HPC_AUTO_INSTALL_CONTAINER_RUNTIME` | `0` | Set to `1` to let `run-all.sh` auto-run `bootstrap.sh --install-nvidia-container-toolkit` during early runtime sanity if GPU is present but NVIDIA container runtime is missing (requires root + internet). |
| `HPC_FAIL_FAST_RUNTIME` | `0` | Set to `1` to make early runtime sanity fail immediately when GPU driver is present but NVIDIA container runtime is missing. |
| `HPC_HPL_MXP_VM_STRICT` | `0` | Set to `1` to disable VM auto-skip conversion in `hpl-mxp.sh`; VM-specific container failures (e.g., SIGPIPE/no usable output) are treated as hard errors. |
| `HPC_ASCII_OUTPUT` | `0` | Set to `1` to use ASCII-only status symbols in the report and run-all checklist (e.g. `[OK]`, `[FAIL]`, `[SKIP]`) instead of Unicode; use when terminals or CI mangle UTF-8. |

## Repository layout

```text
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
│   ├── runtime-sanity.sh
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
└── src/                  # Bundled benchmark sources
    ├── stream.c          # STREAM memory benchmark
    ├── gpu-burn/         # GPU burn-in (CUDA)
    └── nccl-tests/       # Minimal NCCL test binaries
```

## Linting and pre-commit

Optional: run [pre-commit](https://pre-commit.com/) to enforce trailing-whitespace removal, end-of-file newlines, YAML/JSON checks, and [ShellCheck](https://www.shellcheck.net/) on shell scripts.
This repo also includes [shfmt](https://github.com/mvdan/sh) via pre-commit to keep shell formatting consistent.

```bash
pipx install pre-commit   # or: python3 -m venv .venv && source .venv/bin/activate && pip install pre-commit
pre-commit install       # hook runs on git commit
pre-commit run --all-files   # run once on entire repo
```

Config: [.pre-commit-config.yaml](.pre-commit-config.yaml). ShellCheck is skipped for `src/` (C/CUDA build trees).

If you prefer a `make` entrypoint (Linux only):

```bash
make lint
make static-checks
```

## CI

- GitHub Actions workflow: `.github/workflows/ci.yml`
- Static gate runs `scripts/ci-static-checks.sh` (`bash -n` + `pre-commit run --all-files`)
- Ubuntu VM job runs `run-all.sh --smoke --ci` and `run-all.sh --quick --ci`
- Optional GPU job runs on self-hosted runners labeled `self-hosted,linux,x64,gpu,nvidia` and is enabled by repo variable `HPC_ENABLE_GPU_CI=1`

## Concurrency and locking

`run-all.sh` uses an exclusive lock file at `$HPC_RESULTS_DIR/.hpc-bench.lock` (and falls back to lock directory `$HPC_RESULTS_DIR/.hpc-bench.lock.d` when `flock` is unavailable). If another run is in progress, it exits immediately with **exit code 2** (another instance is running). Remove stale lock artifacts only after confirming no other `run-all.sh` is active.

## Exit codes (run-all.sh)

- **0** — All modules passed; report may still show warnings (conditional pass).
- **1** — One or more modules failed (acceptance gate).
- **2** — Another instance is running (lock held). Wait for it to finish or remove the lock only after confirming no other instance is running.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE). Third-party code in `src/` (e.g. gpu-burn, nccl-tests, STREAM) may have their own licenses; refer to comments and upstream sources.
