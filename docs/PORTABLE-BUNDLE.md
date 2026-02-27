# Portable bundle (air-gapped / no-network runs)

This document describes how to build and run the HPC Bench Suite as a **portable directory bundle** for datacenters with no network access or minimal tooling (e.g. USB-only deployment). It is intended for both human operators and for agents/AI that need to reason about or modify this feature.

---

## How to make a single file for USB

To produce **one `.run` file** you can copy to a USB and execute on the server (no extract step on target):

1. **On the build host** (office, with network): install makeself (`apt install makeself` on Debian/Ubuntu), then from the repo root run:
   ```bash
   bash scripts/build-portable-bundle.sh --makeself
   ```
2. **Output:** `dist/hpc-bench-portable-<VERSION>-linux-<arch>.run` (and the `.tar.gz`).
3. **On the target:** Copy the `.run` to the server, then run e.g. `sh hpc-bench-portable-1.11-linux-x86_64.run` or `sh hpc-bench-portable-1.11-linux-x86_64.run --quick`. The first run extracts to `./hpc-bench-portable-.../` and runs the suite; arguments like `--quick` are passed through. To run again without re-extracting: `cd hpc-bench-portable-... && ./run.sh --quick`.

See **Single-file (makeself) build** and **Run from single file** below for details and options.

---

## Purpose

- **Build once** (in the office, with network and tools).
- **Run on target** (e.g. from USB) with **no network** and **no package installs**.
- Target is assumed to be Ubuntu (or similar) with Bash 4+ and standard userland (awk, grep, timeout, coreutils). The bundle supplies **jq** (static binary) and optionally a pre-built **STREAM** binary so the suite can run without installing anything.

---

## Quick example (end-to-end)

### 1. Build the bundle (office, with network)

From the repo root:

```bash
bash scripts/build-portable-bundle.sh
```

Output: `dist/hpc-bench-portable-<VERSION>-linux-<arch>.tar.gz` (e.g. `1.11` and `x86_64` from `VERSION` and `uname -m`).

### 2. Transfer to target

Copy the tarball to a USB (or other removable media), then to the server. Example:

```bash
# On your workstation (after build)
cp dist/hpc-bench-portable-1.11-linux-x86_64.tar.gz /media/usb/

# On the target server (after mounting USB)
cp /media/usb/hpc-bench-portable-1.11-linux-x86_64.tar.gz /tmp/
cd /tmp
tar xzf hpc-bench-portable-1.11-linux-x86_64.tar.gz
cd hpc-bench-portable-1.11-linux-x86_64
```

### 3. Run on target (no network required)

```bash
# Full run (all phases; may take a long time)
sudo ./run.sh

# Short benchmarks (recommended for first try)
sudo ./run.sh --quick

# Smoke only: bootstrap + inventory + report (no benchmarks, ~1 min)
sudo ./run.sh --smoke
```

Results go to `/var/log/hpc-bench/results/` (root) or `$HOME/.local/var/hpc-bench/results` (non-root). Open `report.md` in the results folder for the summary.

### Run from single file (`.run`)

If you built with `--makeself`, copy the `.run` file to the server (e.g. via USB) and run it directly. The first run extracts the bundle to `./hpc-bench-portable-<VERSION>-linux-<arch>/` in the current directory and then runs the suite. Any arguments you pass to the `.run` file are forwarded to `run.sh` (e.g. `--quick`, `--smoke`).

```bash
# On the target server (e.g. /tmp, after copying the .run from USB)
sh hpc-bench-portable-1.11-linux-x86_64.run           # full run
sh hpc-bench-portable-1.11-linux-x86_64.run --quick   # short benchmarks
sh hpc-bench-portable-1.11-linux-x86_64.run --smoke   # bootstrap + inventory + report only
```

After extraction, the directory `hpc-bench-portable-1.11-linux-x86_64/` remains. To run again without re-extracting: `cd hpc-bench-portable-1.11-linux-x86_64 && sudo ./run.sh --quick`.

---

## Build options

Run the build script from the **repo root**. Optional environment variables:

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `HPC_BUNDLE_OUTPUT_DIR` | `dist` | Directory where the bundle folder and tarball are written |
| `HPC_BUNDLE_ARCH` | `$(uname -m)` | Target architecture (e.g. `x86_64`, `aarch64`) for jq download and STREAM build |
| `HPC_BUNDLE_SKIP_STREAM` | `0` | Set to `1` to skip building STREAM (suite will use bundled source and compile on target if gcc is present) |

**Examples:**

```bash
# Custom output directory
HPC_BUNDLE_OUTPUT_DIR=/opt/packages bash scripts/build-portable-bundle.sh

# Build for aarch64 on an x86_64 host (jq will be correct arch; STREAM will be x86_64 unless you cross-compile)
HPC_BUNDLE_ARCH=aarch64 bash scripts/build-portable-bundle.sh

# No STREAM in bundle (smaller tarball; stream-bench will compile on target when gcc available)
HPC_BUNDLE_SKIP_STREAM=1 bash scripts/build-portable-bundle.sh
```

**Build host requirements:** `curl`, `tar`. Optional: `gcc` (to include pre-built STREAM).

### Single-file (makeself) build

To get a **single `.run` file** for USB (instead of or in addition to the tarball), pass `--makeself` and have **makeself** installed on the build host:

```bash
bash scripts/build-portable-bundle.sh --makeself
```

Or set `HPC_BUNDLE_MAKESELF=1`. The script creates the tarball as usual, then if makeself is found (or required by the flag), produces `dist/hpc-bench-portable-<VERSION>-linux-<arch>.run`. Install makeself with `apt install makeself` (Debian/Ubuntu) or download from [megastep/makeself](https://github.com/megastep/makeself). If `--makeself` is passed but makeself is not in PATH, the script exits with an error.

---

## What the bundle contains

```
hpc-bench-portable-<VERSION>-linux-<arch>/
├── run.sh              # Launcher: sets HPC_BENCH_ROOT, PATH (bin first), HPC_PORTABLE=1, runs run-all.sh
├── bin/
│   ├── jq              # Static binary (required; from jqlang/jq releases)
│   └── stream          # Optional; pre-built STREAM (quick-mode size) if gcc was available at build time
├── scripts/            # All suite scripts (run-all.sh, bootstrap.sh, etc.)
├── lib/
├── specs/
├── conf/
├── src/                # Bundled sources (stream.c, gpu-burn, nccl-tests)
├── reporting/
├── examples/
├── VERSION
├── README.md
└── LICENSE
```

When you run `./run.sh`, the launcher:

1. Sets `HPC_BENCH_ROOT` to the bundle root
2. Prepends `$HPC_BENCH_ROOT/bin` to `PATH` (so `jq` and `stream` are found)
3. Sets `HPC_PORTABLE=1`
4. Execs `bash scripts/run-all.sh "$@"` (all flags like `--quick`, `--smoke` are passed through)

---

## Portable-mode behavior in the suite

When `HPC_PORTABLE=1`:

- **Bootstrap:** Skips connectivity check and all package installation; only hardware/OS detection and JSON emit. No `apt-get` / `dnf` / `yum`.
- **jq:** Found via `PATH` (bundled `bin/jq`). No code change needed.
- **STREAM (stream-bench):**
  - If `bin/stream` exists and is executable, it is used as-is (no download, no compile). Pre-built binary uses fixed size (1M elements, 3 iterations).
  - If not, only the bundled `src/stream.c` is used (no curl); if gcc is present on target, STREAM is compiled there.

GPU benchmarks (gpu-burn, nccl-tests, nvbandwidth, dcgm-diag, hpl-mxp), storage (fio), and other optional tools are **not** bundled. If the target has NVIDIA driver, CUDA, gcc, make, fio, etc., those modules run; otherwise they skip as in a normal run. The portable bundle only guarantees **jq** and (optionally) **STREAM** so the core flow and at least some benchmarks work with zero installs.

---

## Target requirements

- **OS:** Ubuntu or similar (Bash 4+, standard userland).
- **Privilege:** Root (or sudo) for full suite; non-root runs are supported with reduced scope (bootstrap and bmc-inventory skipped).
- **Assumed present:** `bash`, `awk`, `grep`, `timeout` (coreutils). No network or package manager needed.
- **Optional on target:** NVIDIA driver/CUDA, gcc/make, python3, fio, Docker, etc. — for full benchmark coverage; missing tools cause the relevant modules to skip.

---

## Troubleshooting

| Issue | What to check |
| ----- | -------------- |
| Build fails: "Failed to download jq" | Build host needs network and access to `https://github.com/jqlang/jq/releases/`. |
| On target: "jq: not found" | Run via `./run.sh` (so PATH includes `bin/`). Do not invoke `bash scripts/run-all.sh` directly without setting PATH and HPC_PORTABLE=1. |
| Bootstrap still tries to install packages | Ensure you run `./run.sh`, which sets `HPC_PORTABLE=1`. |
| STREAM not used from bundle | Confirm `bin/stream` exists in the extracted bundle and is executable (`ls -la bin/stream`). If you built with `HPC_BUNDLE_SKIP_STREAM=1` or without gcc, there is no `bin/stream`; stream-bench will compile from `src/stream.c` on target if gcc is present. |
| Results location | Default: `/var/log/hpc-bench/results/` (root) or `$HOME/.local/var/hpc-bench/results` (non-root). Override with `HPC_RESULTS_DIR`. |

---

## For agents / AI

**Quick reference for reasoning or code changes:**

- **Build script:** [scripts/build-portable-bundle.sh](scripts/build-portable-bundle.sh). Creates bundle dir, downloads static jq (jqlang/jq 1.7.1), optionally compiles STREAM from `src/stream.c` into `bin/stream`, writes `run.sh` launcher, tars to `dist/hpc-bench-portable-<VERSION>-linux-<arch>.tar.gz`. With `--makeself` (or `HPC_BUNDLE_MAKESELF=1`), if makeself is in PATH, also produces a single `.run` file (makeself self-extracting archive with `--notemp`, startup script `./run.sh`); the tarball remains the primary payload.
- **Launcher:** Bundle root `run.sh` sets `HPC_BENCH_ROOT`, `PATH=$ROOT/bin:$PATH`, `HPC_PORTABLE=1`, then `exec bash "$ROOT/scripts/run-all.sh" "$@"`.
- **Bootstrap portable behavior:** [scripts/bootstrap.sh](scripts/bootstrap.sh): when `HPC_PORTABLE=1`, sets `HAS_INTERNET=false` and skips the connectivity loop; existing logic then skips all package installation.
- **STREAM portable behavior:** [scripts/stream-bench.sh](scripts/stream-bench.sh): if `HPC_PORTABLE=1` and `-x "${HPC_BENCH_ROOT}/bin/stream"`, use that binary and skip download/compile; else if `HPC_PORTABLE=1`, do not call curl (use only `HPC_BENCH_ROOT/src/stream.c` for compile path).
- **jq:** No suite code change; launcher PATH ensures `bin/jq` is found.
- **Commands:**
  - Build: `bash scripts/build-portable-bundle.sh` (from repo root; needs network). Add `--makeself` for a single `.run` file (requires makeself on build host).
  - Run on target: from extracted bundle root `./run.sh` or `./run.sh --quick` or `./run.sh --smoke`; or from single file `sh <bundle>.run` / `sh <bundle>.run --quick`.
- **Env vars (build):** `HPC_BUNDLE_OUTPUT_DIR`, `HPC_BUNDLE_ARCH`, `HPC_BUNDLE_SKIP_STREAM`, `HPC_BUNDLE_MAKESELF` (or use `--makeself`).
- **Env vars (runtime):** Launcher exports `HPC_BENCH_ROOT`, `PATH`, `HPC_PORTABLE=1`; other suite vars (e.g. `HPC_RESULTS_DIR`, `HPC_QUICK`) work as in normal runs.
