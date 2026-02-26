# HPC Benchmark Suite — Agent Guide

This doc tells an AI agent how to work with the HPC Bench Suite: how to run it, what context to assume, how to make changes, and how to debug. For **version history and changelog**, see [CHANGELOG.md](CHANGELOG.md).

---

## What this suite is

- **Purpose:** Benchmark and inventory HPC servers (GPU, CPU, network, storage, topology). Produces JSON per module and a markdown report.
- **Target:** Bare-metal HPC. VMs are supported for convenience; many modules skip or degrade (DCGM, nvbandwidth, HPL-MxP, InfiniBand, BMC). See README "Target environment" and CHANGELOG for VM-related behavior.
- **Language:** Pure Bash. Designed for minimal dependencies and portability across HPC environments.

---

## Codebase structure

```
bench/
├── conf/
│   ├── defaults.sh          # Central tunables (timeouts, thresholds, container images)
│   └── local.sh             # Optional user overrides (gitignored)
├── docs/
│   └── DEVELOPMENT.md       # Detailed dev conventions, JSON contract, config guide
├── lib/
│   ├── common.sh            # Core shared library (logging, JSON, GPU, cleanup, etc.)
│   └── report-common.sh     # Report scoring logic, module status helpers
├── scripts/
│   ├── bootstrap.sh          # Phase 0: hardware detection, tool install
│   ├── runtime-sanity.sh     # Phase 1: pre-flight checks
│   ├── inventory.sh          # Phase 2: CPU/RAM/storage/OS discovery
│   ├── gpu-inventory.sh      # Phase 2: NVIDIA GPU inventory
│   ├── topology.sh           # Phase 2: NUMA/PCIe/NVLink topology
│   ├── network-inventory.sh  # Phase 2: NICs, InfiniBand, link info
│   ├── bmc-inventory.sh      # Phase 2: IPMI/BMC sensors/firmware
│   ├── software-audit.sh     # Phase 2: CUDA, cuDNN, NCCL, MPI, etc.
│   ├── dcgm-diag.sh          # Phase 3: DCGM diagnostics
│   ├── gpu-burn.sh           # Phase 3: GPU stress test
│   ├── nccl-tests.sh         # Phase 3: NCCL collective benchmarks
│   ├── nvbandwidth.sh        # Phase 3: GPU memory/PCIe/NVLink bandwidth
│   ├── stream-bench.sh       # Phase 3: STREAM memory bandwidth
│   ├── storage-bench.sh      # Phase 3: fio storage benchmarks
│   ├── hpl-cpu.sh            # Phase 3: HPL CPU Linpack
│   ├── hpl-mxp.sh            # Phase 3: HPL-MxP GPU mixed-precision
│   ├── ib-tests.sh           # Phase 3: InfiniBand loopback perftest
│   ├── network-diag.sh       # Phase 4: firewall, ports, routing, DNS
│   ├── filesystem-diag.sh    # Phase 4: mounts, FS types, RAID, NFS/Lustre
│   ├── thermal-power.sh      # Phase 4: thermals, fans, PSU, throttling
│   ├── security-scan.sh      # Phase 4: SSH audit, services, SUID, kernel
│   ├── report.sh             # Phase 5: generates report.md from results
│   ├── run-all.sh            # Orchestrator (runs all phases)
│   ├── ci-static-checks.sh   # CI-only: shellcheck + syntax checks
│   └── check-updates.sh      # Dependency update checker (not a module)
├── specs/
│   ├── modules.json          # Module manifest (single source of truth)
│   ├── dependencies.json     # Tracked external dependency versions
│   └── update-history.json   # Dependency update audit log
├── src/                       # Bundled sources (gpu-burn, nccl-tests, STREAM)
├── tests/
│   ├── helpers.bash           # Shared BATS test helpers
│   ├── common_helpers.bats    # Unit tests for lib/common.sh
│   ├── report_helpers.bats    # Unit tests for lib/report-common.sh
│   ├── module_integration.bats # Integration tests (syntax, manifest, source-gate)
│   └── check_updates.bats    # Tests for dependency checker + manifest
├── .editorconfig              # Formatting rules
├── .pre-commit-config.yaml    # Pre-commit hooks (shfmt, shellcheck)
├── Makefile                   # Quality gates: make lint, make test, make smoke
├── CONTRIBUTING.md            # Contributor guidelines
├── SKILL.md                   # This file (agent guide)
└── README.md                  # User-facing documentation
```

---

## How to use it (like a human would)

- **Full run (root):** `sudo bash scripts/bootstrap.sh` then `sudo bash scripts/run-all.sh`.
- **Quick validation:** `sudo bash scripts/run-all.sh --quick` (short benchmarks).
- **Smoke (no benchmarks):** `sudo bash scripts/run-all.sh --smoke` (bootstrap + discovery + report).
- **NVIDIA driver + CUDA on Ubuntu:** `sudo bash scripts/bootstrap.sh --install-nvidia` (internet required; reboot after driver install, then run bootstrap again).
- **Check deps only:** `bash scripts/bootstrap.sh --check-only` (no root; lists present/missing).
- **Single module:** e.g. `bash scripts/gpu-inventory.sh`, `bash scripts/gpu-burn.sh` — run from repo root; most need prior bootstrap and root for full behavior.
- **Results:** Default `$HPC_RESULTS_DIR` is `/var/log/hpc-bench/results` (root) or `$HOME/.local/var/hpc-bench/results` (non-root). Report: `report.md`; logs: `logs/`; per-module JSON: `bootstrap.json`, `gpu-burn.json`, etc.
- **Quality checks:** `make check` (shellcheck + bats + static), `make lint` (shellcheck only), `make test` (bats only).

**Context to assume:** User may be on a VM, air-gapped, or without NVIDIA GPU. Scripts skip modules when prerequisites are missing (e.g. no IB hardware, no nvidia-smi) and emit JSON with `status: "skipped"` and `.note`/`.skip_reason`. Exit codes: 0 = success or conditional pass; 1 = at least one non-skipped module failed; 2 = lockfile held (another run in progress).

---

## Making changes

### Adding a new module

1. Create `scripts/<name>.sh` with the standard header:
   ```bash
   #!/usr/bin/env bash
   # <name>.sh -- <one-line description>
   # Phase: <N> (<phase-name>)
   # Requires: <cmd1>, <cmd2>
   # Emits: <name>.json
   SCRIPT_NAME="<name>"
   source "$(dirname "$0")/../lib/common.sh"
   ```
2. Register in `specs/modules.json` with `name`, `script`, `phase`, `order`, and `required_cmds`.
3. Use `skip_module` or `skip_module_with_data` when prerequisites are missing (exit 0).
4. Use `finish_module` or `emit_json` to write results (the EXIT trap writes a crash record if you forget).
5. Use `status: "error"` + `exit 1` for failures; never `status: "error"` + `exit 0`.
6. Add scoring logic in `lib/report-common.sh` `score_module()` if the module has custom pass/fail criteria.
7. Add report rendering in `scripts/report.sh` `emit_benchmark_results()` or the appropriate emit function.

### Modifying an existing module

- All tunables (timeouts, thresholds, durations, container images) live in `conf/defaults.sh`. Override in `conf/local.sh` for site-specific values.
- Shared utilities (CUDA detection, container runtime, HPL.dat generation, JSON helpers) are in `lib/common.sh`. Use them instead of duplicating logic.
- Follow the naming convention: `lower_snake_case` for local variables, `UPPER_SNAKE_CASE` for exported/global variables.
- Wrap lines at ~120 characters (see `.editorconfig`).

### Running tests

- `make test` — runs all BATS tests.
- `make lint` — runs shellcheck on all `.sh` files.
- `make check` — runs lint + test + static checks.
- `make smoke` — runs `run-all.sh --smoke` end-to-end.
- `make quick` — runs `run-all.sh --quick` end-to-end.
- `make check-updates` — checks 14 tracked dependencies for upstream updates.

### JSON contract

Every module MUST emit a JSON file to `$HPC_RESULTS_DIR/<module>.json` with at least:
```json
{
  "module": "<module-name>",
  "status": "ok|warn|error|skipped",
  "timestamp": "ISO8601"
}
```
- **`status` values:** `ok`, `warn`, `error`, `skipped` — these are the ONLY valid values.
- Skipped modules must include `skip_reason` (human-readable string).
- Error modules must include `error` (human-readable string).
- Additional fields are module-specific.

---

## Common pitfalls

Things an AI agent should **never** do:

1. **Use `pass` or `fail` as status values.** The only valid values are `ok`, `warn`, `error`, `skipped`.
2. **Forget to register a new module in `specs/modules.json`.** The orchestrator (`run-all.sh`) reads this manifest; unregistered scripts won't run.
3. **Emit `status: "error"` and then `exit 0`.** Error status must pair with `exit 1`.
4. **Use `eval` for dynamic variable access.** Use the existing JSON helpers (`json_compact_or`, `sanitize_json_str`, etc.) instead.
5. **Omit `local` in function variables.** All lowercase variables inside functions must be declared `local`.
6. **Break the JSON output structure.** Always validate JSON before writing (use `emit_json_safe` for untrusted data, or `json_compact_or` for validation).
7. **Hardcode CUDA paths.** Use `detect_cuda_home()` from `lib/common.sh`.
8. **Hardcode container runtime.** Use `detect_container_runtime()` from `lib/common.sh`.
9. **Use `function name()` syntax.** Always use `name() {` style.
10. **Skip writing JSON on error.** The EXIT trap writes a crash record if you forget, but it's better to explicitly call `emit_json` with `status: "error"`.

---

## Debugging

- **Logs:** Each module logs to `$HPC_LOG_DIR/<SCRIPT_NAME>.log` (e.g. `logs/bootstrap.log`). Stdout from benchmarks often in `logs/<module>-stdout.log` or similar.
- **Lock:** Concurrent runs use `$HPC_RESULTS_DIR/.hpc-bench.lock`. Exit 2 if locked; remove lock only if no other run is active.
- **Crash records:** If a module exits without writing its JSON (e.g. crash, kill), the EXIT trap in common.sh writes a crash record for that module so the report still has an entry.
- **JSON validity:** Many scripts pipe data into `jq`. Control characters or invalid JSON in variables (e.g. from `systemd-detect-virt`, DMI, or nvidia-smi) can cause "parse error: Invalid string: control characters…". Fix by sanitizing before building JSON (e.g. strip U+0000–U+001F or use `sanitize_json_str` / temp files + `--slurpfile`).
- **Bootstrap "no internet":** Connectivity check tries several URLs; on failure it logs a "Connectivity check hint" (first line of curl error). If the user has a proxy, they may need `sudo -E` or to set HTTP_PROXY/HTTPS_PROXY for root. If "no NVIDIA GPU" but the machine has GPUs, detection uses `lspci` then fallback `/sys/bus/pci/devices/*/vendor` (0x10de); ensure pciutils is installed or rely on the sys fallback.
- **Module-specific:** See CHANGELOG for known bugs and fixes (e.g. gpu-inventory VM-safe behavior, nccl-tests parser, storage-bench fio output, report field names). Specs and GPU lookup: `specs/hardware-specs.json` and `lookup_gpu_spec()` in `lib/common.sh` (fuzzy match by GPU name).

---

## Key files for agents

| Need | File / location |
|------|------------------|
| User-facing usage | [README.md](README.md) |
| Version number | [VERSION](VERSION) |
| Changelog / version history | [CHANGELOG.md](CHANGELOG.md) |
| Shared helpers, logging, emit_json, GPU/CUDA/container detection | [lib/common.sh](lib/common.sh) |
| Report scoring logic, module status helpers | [lib/report-common.sh](lib/report-common.sh) |
| Module manifest (single source of truth for modules) | [specs/modules.json](specs/modules.json) |
| Central defaults (timeouts, thresholds, container images) | [conf/defaults.sh](conf/defaults.sh) |
| Entry points | [scripts/bootstrap.sh](scripts/bootstrap.sh), [scripts/run-all.sh](scripts/run-all.sh) |
| GPU spec lookup | [lib/common.sh](lib/common.sh) `lookup_gpu_spec`, [specs/hardware-specs.json](specs/hardware-specs.json) |
| Dev conventions, JSON contract, config guide | [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) |
| Contributor guidelines | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Quality gates | [Makefile](Makefile) |
| Dependency version manifest | [specs/dependencies.json](specs/dependencies.json) |
| Dependency update checker | [scripts/check-updates.sh](scripts/check-updates.sh) |
| Dependency update history | [specs/update-history.json](specs/update-history.json) |
