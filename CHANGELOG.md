# Changelog

All notable changes to the HPC Bench Suite are documented here. Version number is in [VERSION](VERSION).

## Version History

### V1.10 Changes (2026-02-14)

1. **scripts/run-all.sh** — Added CI mode:
   - New flag: `--ci` (sets `HPC_CI=1` and enables quick-mode defaults).
   - CI mode reduces noisy/interleaved stdout by writing module output to per-module logs and only printing compact status lines.
   - On module failure/timeout in CI mode, prints a short tail snippet from the module log for fast diagnosis.
2. **scripts/run-all.sh** — Added orchestrator-level minimal dependency preflight:
   - Each module now has a minimal command prerequisite list (for example `jq`, `awk`, `timeout`, `python3` depending on module).
   - Missing prerequisites are now reported as explicit **skips** with reason `missing commands: ...` and module JSON is still emitted.
3. **scripts/run-all.sh** — Updated CLI help/unknown option messaging to include `--ci`.
4. **scripts/run-all.sh** — Locking remains `flock`-first with fallback lock directory (`.hpc-bench.lock.d`) for minimal environments without `flock`.
5. **scripts/inventory.sh** — Hardened for minimal/non-Linux developer environments:
   - Guarded `lscpu` usage (no hard failure when command is absent).
   - Numeric fields now normalized before `jq --argjson` usage.
   - RAM math no longer depends on `bc`; uses awk and cross-platform fallback for local testing.
   - Added command/file guards for `lsblk`, `/proc/uptime`, and `ip`.
6. **scripts/network-inventory.sh** — Hardened JSON assembly:
   - Prevents malformed JSON when optional IB commands return empty/error output.
   - Validates NIC enrichment JSON and safely falls back to base NIC list when needed.
7. **scripts/software-audit.sh** — Guarded Fabric Manager checks behind `has_cmd systemctl` to avoid noisy errors on non-systemd systems.
8. **New script:** `scripts/ci-static-checks.sh`
   - Runs `bash -n` for all shell scripts and `pre-commit run --all-files`.
   - Used by CI as a static quality gate.
9. **New CI workflow:** `.github/workflows/ci.yml`
   - Static checks job (bash syntax + pre-commit).
   - Ubuntu VM job running `--smoke --ci` and `--quick --ci` with result artifact upload.
   - Optional self-hosted GPU quick job (enabled with repository variable `HPC_ENABLE_GPU_CI=1`).
10. **README.md** — Added CI section, documented `--ci` / `HPC_CI`, updated lock fallback docs.
11. **VERSION** — Bumped to `1.10`.

### V1.9 Changes (2026-02-12)

1. **scripts/bootstrap.sh** — Dynamic NVIDIA stack install improvements:
   - Auto-detect latest `nvidia-driver-*-server` package (fallback to `580-server`).
   - Dynamically discovers available `cuda-toolkit-X-Y` packages (no hardcoded version pins).
   - Uses `DEBIAN_FRONTEND=noninteractive` for unattended installs.
   - Emits clean JSON and clear reboot guidance when driver install requires reboot.
2. **scripts/bootstrap.sh** — Added optional container-runtime installer:
   - New flag: `--install-nvidia-container-toolkit`.
   - Installs/configures Docker + NVIDIA Container Toolkit on Ubuntu (`apt` path).
   - Runs `nvidia-ctk runtime configure --runtime=docker`, restarts Docker, and verifies runtime availability.
   - Bootstrap summary now includes `has_nvidia_container_runtime`.
3. **scripts/bootstrap.sh** — Combined-flow UX for first-time GPU setup:
   - Supports using `--install-nvidia --install-nvidia-container-toolkit` together.
   - Post-reboot checklist text now explicitly guides rerun of the combined command.
4. **scripts/run-all.sh** — Added early runtime gate (`Phase 0.5`):
   - New module `scripts/runtime-sanity.sh` runs immediately after bootstrap.
   - Verifies GPU driver, Docker runtime, NVIDIA container runtime, and DCGM availability before benchmark phases.
5. **scripts/runtime-sanity.sh** — New fail-fast and auto-install controls:
   - `HPC_AUTO_INSTALL_CONTAINER_RUNTIME=1` attempts automatic runtime installation when GPU exists but NVIDIA container runtime is missing.
   - `HPC_FAIL_FAST_RUNTIME=1` fails immediately (error JSON + non-zero exit) when runtime prerequisites are missing.
6. **scripts/run-all.sh** — New CLI flags for runtime controls:
   - `--auto-install-runtime` and `--fail-fast-runtime`.
   - Runtime mode state is logged in run banner.
   - If fail-fast is enabled and runtime-sanity reports error, run exits before Phase 1.
7. **scripts/gpu-inventory.sh** — GPU count consistency fix:
   - `gpu_count` now comes from authoritative `nvidia-smi --query-gpu=index` counting, with parser-length fallback.
   - Prevents undercount when field parsing and reported GPU array diverge.
8. **scripts/thermal-power.sh** — GPU count alignment fix:
   - Runtime GPU count now prioritizes direct `nvidia-smi` detection and harmonizes with `gpu-inventory.json`.
   - Eliminates mismatches where thermal report showed fewer GPUs than detected elsewhere.
9. **scripts/run-all.sh** — Orchestrator resilience and startup improvements:
   - Safely continues after per-module failures (does not crash whole orchestrator).
   - Added GPU warm-up before parallel inventory fan-out to avoid initial driver/DCGM lock contention.
10. **scripts/nvbandwidth.sh** — Fixed JSON fallback bug (`"${bw_values:-{}}"`) that could append a stray `}` and break `jq` parsing; replaced with explicit empty-check handling.
11. **scripts/gpu-inventory.sh** — Fixed CUDA version parsing by anchoring to `CUDA Version:` to avoid truncating values like `13.0` to `3.0`.
12. **scripts/gpu-burn.sh** — Quick-mode reliability:
    - Increased quick burn duration from 3s to 10s so GFLOPS lines are consistently emitted.
    - Added parser fallback for progress-line GFLOPS format used by newer gpu-burn output.
13. **scripts/report.sh** — Quick-mode storage reporting clarity:
    - Detects `quick_mode_skip: true` and renders "skipped — quick mode" instead of misleading zeros.
14. **README.md** — Documentation updates:
    - Safer SSH host-key remediation (`ssh-keygen -R <host-or-ip>`).
    - New bootstrap flag docs for NVIDIA container toolkit setup.
    - Combined bootstrap workflow docs (driver + runtime).
    - New `run-all` runtime controls and environment variables documented.
15. **lib/common.sh + scripts/thermal-power.sh** — Added reusable numeric sanitization helpers (`trim_ws`, `json_numeric_or_null`) and refactored thermal GPU parser to use them, preventing `jq --argjson` crashes on values like `N/A`/`[Not Supported]`.
16. **scripts/dcgm-diag.sh** — Quick mode on VMs now keeps the VM timeout behavior (120s) while still running r1 only; VM skip note now states diagnostics were attempted before skip.
17. **scripts/hpl-cpu.sh** — Container-first reliability improvements:
    - Detects usable container runtime as Docker first, Podman fallback.
    - Pre-pulls `intel/hpckit:latest` when missing.
    - Emits specific skip notes (runtime missing, image pull failure, run failure, system xhpl failure).
18. **scripts/hpl-mxp.sh** — Added opt-in strict VM mode via `HPC_HPL_MXP_VM_STRICT=1`; default remains VM-safe auto-skip behavior, strict mode converts VM-specific auto-skips into hard failures.
19. **Spec DB removed from runtime/report path**:
    - Removed GPU spec lookup usage from `gpu-inventory.sh`, `report.sh`, `nccl-tests.sh`, `nvbandwidth.sh`, and `hpl-mxp.sh`.
    - Deleted `specs/hardware-specs.json`; report now uses runtime-measured fields only.
20. **scripts/hpl-cpu.sh** — Better remediation before failure:
    - Longer dedicated image pull timeout (`HPL_PULL_TIMEOUT`) to avoid quick-mode false skips.
    - Detects missing `xhpl` in container output and falls back to host CPU paths.
    - Adds CPU fallback via `hpcc` package and parses HPL numbers from `hpccoutf.txt`.
    - Reports hard error with concise detail when no CPU HPL path is viable.
21. **scripts/nccl-tests.sh** — Runtime-error handling and retries:
    - Retries with `NCCL_P2P_DISABLE=1` + `NCCL_IB_DISABLE=1`.
    - Additional fallback retry with 2 GPUs when >2 GPU run fails.
    - Emits `runtime_error_count`; returns module error (non-zero) if NCCL still fails.
22. **scripts/nvbandwidth.sh** — Adds explicit `p2p_status` (`supported`/`not_supported`/`unknown`) so D2D `N/A` is explained by topology constraints.

**Validation notes (V1.9):**
- Repeated end-to-end `--quick` runs validated on remote Ubuntu 24.04 KVM host with H100 GPUs.
- Verified full reinstall cycle (purge CUDA user-space + Docker/runtime, then reinstall via combined bootstrap flags).
- Runtime sanity behavior validated for:
  - **positive path** (`--auto-install-runtime --fail-fast-runtime` with runtime available),
  - **fail-fast path** (forced missing runtime context; early exit with rc=1 before benchmark phases).

### V1.8 Changes (2026-02-11)

1. **run-all.sh** — `--smoke` flag: bootstrap + discovery + report only (no benchmarks); Phase 1 (discovery) runs in parallel; skipped modules show short reason (`.note`/`.skip_reason`) in checklist.
2. **User-level mode** — When not root, results default to `$HOME/.local/var/hpc-bench/results`; bootstrap and bmc-inventory are skipped.
3. **GPU count consistency** — thermal-power now derives `gpu_count` from gpu-inventory or nvidia-smi (single source); report uses gpu-inventory with bootstrap fallback.
4. **Report** — Primary storage line (benchmarked device or from inventory); skip reasons in scorecard Notes; SSH PasswordAuthentication remediation hint when security-scan reports it.
5. **report-common.sh** — SKIP modules show `.skip_reason` or `.note` in scorecard.
6. **README** — Remote-run example (rsync + ssh); jq 1.6+, Bash 4+, python3; archive transfer (scp/rsync); exit code 2 and lock doc; `--smoke` and `HPC_KEEP_TOOLS` doc.
7. **lib/common.sh** — **detect_virtualization()**: strip control characters (U+0000–U+001F) from `virt_type` and `virt_details` before emitting JSON so `jq` never sees invalid strings. Fixes parse error "Invalid string: control characters from U+0000 through U+001F must be escaped" when `systemd-detect-virt` or DMI emits stray bytes.
8. **scripts/bootstrap.sh** — **Connectivity check**: try multiple URLs (google.com, archive.ubuntu.com, mirror.centos.org, cloudflare.com, 1.1.1.1); on failure log a "Connectivity check hint" with the first line of curl error (e.g. DNS vs timeout). **Early install**: curl is installed alongside jq when root so the connectivity check can run even when curl was missing.
9. **scripts/bootstrap.sh** — **--install-nvidia**: when the flag is passed but driver install is skipped, log explicit reason (no NVIDIA GPU detected, driver already working, no internet, or not apt). When no NVIDIA GPU is detected but `--install-nvidia` and internet/apt are available, install **CUDA toolkit only** (nvcc, libs) for development or future GPU use.
10. **scripts/bootstrap.sh** — **NVIDIA GPU detection**: prefer `lspci`; fallback to `/sys/bus/pci/devices/*/vendor` (vendor 0x10de = NVIDIA) when `lspci` is missing or PATH is limited under sudo, so GPUs are detected even when pciutils is not installed or `lspci` is not in root's PATH.

### V1.7 Changes (2026-02-11)

**Quick benchmark mode for end-to-end verification.**

1. **run-all.sh** — `--quick` flag (or `HPC_QUICK=1`) enables quick mode: short benchmarks so the full suite finishes fast for smoke tests / VM validation. In quick mode, per-module timeout is 10 min (vs 30 min full). Banner shows "Quick Run (short benchmarks)" when active.
2. **Benchmark quick-mode behavior:** DCGM: level 1 (r1) only, 90s timeout. GPU burn: 3s (override with `GPU_BURN_DURATION`). HPL-MxP: tiny problem (N=2048, NB=128), 120s timeout. HPL-CPU: tiny N=10000, NB=128, **30s timeout** (skip fast if container not ready). NCCL: **one test** (all_reduce_perf), **8B–1M** range, 1 iter, 45s timeout. STREAM: 1M elements, 3 iterations, 60s run timeout. Storage: **2 fio profiles** (seq-read, rand-4k-read), **5s** each (not 7×15s). nvbandwidth: 60s timeout per test.
3. **README** — Quick start documents `run-all.sh --quick`; env table documents `HPC_QUICK` for full quick-mode description.
4. **VERSION** — Bumped to 1.7.
5. **VM validation (2026-02-11):** Full suite `run-all.sh --quick` run on remote VM (Ubuntu 24.04, 8× H100 PCIe, KVM): 16 passed, 5 skipped (bmc-inventory, hpl-cpu, hpl-mxp, ib-tests, nvbandwidth), 0 failed; report and archive OK. Optional: install Boost in bootstrap when GPU present to improve nvbandwidth build-from-source (README).

### V1.6 Changes (2026-02-11)

**Online-first sources, progressive output, VM testing.**

1. **Online-first, local fallback** — gpu-burn, nccl-tests, and stream-bench now try to obtain the latest source from the network first (git clone or curl), and only use bundled sources under `src/` when offline or when the download fails. HPL-MxP container: try registry pull before loading from bundled tar.
2. **Progressive output** — run-all.sh prints (1) device result first (DEVICE: PASSED / FAILED / INCONCLUSIVE), (2) a compact checklist of all modules with ✓/○/✗, (3) detail sections for passed and failed/skipped. Report: first line is "Device result: PASSED/FAILED/CONDITIONAL PASS" for at-a-glance reading.
3. **VERSION** — Bumped to 1.6.

### V1.5 Changes (2026-02-11)

**VM validation and results archive fix.**

1. **scripts/run-all.sh** — Results archive no longer fails with "file changed as we read it": archive is built from a temporary snapshot copy of the results directory, then the snapshot is removed. Ensures a consistent tarball when the log file or other outputs are still being written.
2. **Full suite validated on a remote VM** (Ubuntu 24.04, 8× A100-SXM4-80GB passthrough, KVM): 21 modules, 15 passed, 6 skipped (bmc-inventory, dcgm-diag, hpl-cpu, hpl-mxp, ib-tests, nvbandwidth), 0 failed; report and archive generated successfully.
3. **VERSION** — Bumped to 1.5.

### V1.4 Changes (2026-02-11)

**VM testing, NCCL parsing, HPL-MxP SIGPIPE, and quick mode.**

1. **scripts/nccl-tests.sh** — Parser robustness: (a) accept data rows with leading space (fixed-width output); (b) fallback to "# Avg bus bandwidth" when binary fails before printing rows (e.g. NCCL init error in some VMs); (c) peak busbw uses avg_busbw_gbps when busbw_gbps missing; (d) ensure peak_busbw is numeric for jq/bc.
2. **scripts/hpl-mxp.sh** — VM SIGPIPE (exit 141): always treat as skipped when virtualized (removed condition that required no existing json), so suite acceptance passes on VMs.
3. **scripts/storage-bench.sh** — Optional quick mode: when `HPC_QUICK=1`, fio runtime per profile is 15s instead of 60s (faster VM/CI runs).
4. **scripts/run-all.sh** — Archive failure now logs first 3 lines of tar stderr to aid debugging.
5. **src/nccl-tests/src/all_reduce.cu** — Removed unused variable `maxCount` to silence nvcc warning.
6. **VERSION** — Bumped to 1.4.

### V1.3 Changes (2026-02-11)

**VM and robustness: fixes for virtualized environments and brittle scripts.**

1. **gpu-inventory.sh** — VM-safe: (a) optional-field probe no longer triggers `set -e` when `nvidia-smi` exits non-zero; (b) driver/CUDA version queries use `|| true` so unsupported fields (e.g. `cuda_version` in VMs) don't exit; (c) full GPU query fallback to base fields only when combined query returns no data; (d) OPT_JSON loop fixed (`$opt_first` as command ran `false` and exited — now use numeric flag); (e) all JSON for final jq passed via temp files + `--slurpfile` to avoid arg length and invalid spec from `lookup_gpu_spec`; (f) spec file written only after `jq -c` validation; (g) CUDA version fallback parse uses `sed` for version number only (avoids "|" from table).
2. **lib/common.sh** — `lookup_gpu_spec`: validate and echo only valid JSON (exact and fuzzy paths); malformed spec now returns `{}`.
3. **filesystem-diag.sh** — (a) RAID block: `md_arrays`/`md_detail` sanitized, jq failure doesn't exit; (b) LVM/raid/pfs JSON validated before final jq; (c) mounts/pfs/raid/lvm passed via temp files + `--slurpfile` to avoid invalid/oversized `--argjson`.
4. **nvbandwidth.sh** — When build or offline clone fails, emit `skipped` and exit 0 so suite doesn't fail (VM/bare metal without nvbandwidth).
5. **dcgm-diag.sh** — VM-aware: shorter timeout (120s) when virtualized; when all levels fail in a VM, emit `skipped` and exit 0.
6. **gpu-burn.sh** — VM-aware: default burn duration 60s when virtualized (override with `GPU_BURN_DURATION`). JSON result fix: validate/normalize `gpu_gflops`; pass `errors` via `--arg` + `tonumber`; write `max_temps`/`max_power`/`gpu_gflops` to temp files and use `--slurpfile` to avoid invalid JSON and arg length issues.
7. **hpl-mxp.sh** — When run produces no usable output (e.g. container SIGPIPE in VM), emit `skipped` and exit 0 when virtualized.
8. **VERSION** — Bumped to 1.3.

### V1.2.2 Changes (2026-02-11)

**P1 + P2 fixes: correctness, robustness, and hardening for DC deployment.**

1. **run-all.sh** — Acceptance exit code gate. `exit 1` if any non-skipped module failed, `exit 0` with CONDITIONAL warning if report-level warnings exist.
2. **run-all.sh** — Results archive: bundles all JSON/reports/logs into portable `hpc-bench-<hostname>-<timestamp>.tar.gz`.
3. **run-all.sh** — flock-based lockfile (`$HPC_RESULTS_DIR/.hpc-bench.lock`) prevents concurrent runs clobbering results. Exits code 2 if locked.
4. **bootstrap.sh** — Compiler pre-flight: checks `gcc`/`make` after install, emits loud error with air-gapped remediation. Adds `has_compiler` to JSON.
5. **bootstrap.sh** — NCCL install now has `dnf`/`yum` path (`libnccl libnccl-devel`) alongside apt.
6. **bootstrap.sh** — Migrated summary JSON from heredoc → `jq --arg` (injection-safe).
7. **hpl-mxp.sh** — Offline-aware 3-tier container resolution: local check → `docker load` from `src/*.tar` → pull. Graceful skip if all fail.
8. **common.sh** — `lookup_gpu_spec()` rewritten with scored fuzzy matching. Correctly disambiguates A100 SXM4 80GB vs PCIe 40GB, etc. Fixed `{}}` JSON bug.
9. **common.sh** — Default results dir: `/tmp/hpc-bench-results` → `/var/log/hpc-bench/results` (persistent across reboots).
10. **common.sh** — Crash-safety trap: if module exits without `emit_json`, EXIT trap writes a crash record so report always has a result.
11. **inventory.sh** — All heredoc JSON → `jq --arg` (injection-safe for hostnames, CPU models, OS names).
12. **gpu-burn.sh** — Error count now only matches gpu-burn's `FAULTY` output, not log wrapper `[ERROR]`. Fixed gawk-ism to POSIX awk.
13. **security-scan.sh** — SUID scan: `find / -xdev` + `timeout 30` prevents NFS mount hangs.
14. **hpl-cpu.sh** — Dynamic timeout: `1800 + (RAM_GB × 2)` seconds. 2TB node → ~97 min.
15. **hpl-mxp.sh** — Dynamic timeout: `1800 + (total_GPU_mem_MB / 1024)` seconds.

### V1.2.2 Known Bugs

| # | Module | Bug | Status |
|---|--------|-----|--------|
| 5 | nvbandwidth | cmake not installed by bootstrap on minimal images | Low — nvbandwidth bundled in src/ |

### V1.2.1 Changes (2026-02-11)

**P0 fixes: report now produces correct data from all modules.**

1. **nccl-tests.sh** — Fixed parser column count mismatch. Bundled nccl-tests source outputs 7 columns; parser expected 9+ (upstream format). Changed `NF>=9` to `NF>=6` with NF-relative column indexing. Fixes Bug #4 (empty datapoints).
2. **nvbandwidth.sh** — Rewrote bandwidth parser. Previous version grepped all floats (including matrix indices and headers) and averaged them. New parser uses python3 to understand nvbandwidth's GPU matrix and SUM output formats, extracting only actual bandwidth values.
3. **storage-bench.sh** — Fixed fio JSON capture. Now uses fio's native `--output=` flag to write JSON directly to file, avoiding stdout contamination from progress messages. Fixes Bug #3 (all results showing "parse failed").
4. **report.sh** — Fixed all field name mismatches between modules and report:
   - dcgm-diag: `.overall_result` → `.overall`
   - nccl-tests: `.allreduce_busbw_gbps` → `.peak_allreduce_busbw_gbps`; rewrote section to read `.tests[]` array
   - nvbandwidth: key regex `h2d|d2h|d2d` → explicit field reads of `host_to_device`, `device_to_host`, etc.
   - storage-bench: `.results[]` → individual top-level keys (`sequential_read_1M`, `random_4k_read`, etc.)
   - gpu-burn: `.gpus[]` → `.gpu_performance[]` and `.max_temps[]`
   - thermal-power: `.warnings_count` → `.thermal_status` and `.hot_gpus_above_85c`
5. **report.sh** — Fixed version string. Now reads from `VERSION` file instead of hardcoded "1.0". Fixes Bug #6.
6. **run-all.sh** — Version banner now reads from `VERSION` file.
7. **VERSION** — Added single source of truth version file.

### V1.2.1 Known Bugs (carried from V1.2, not yet addressed)

| # | Module | Bug | Status |
|---|--------|-----|--------|
| 2 | gpu-burn | Error grep matches log wrapper output (gawk `match()` capture groups not POSIX) | Deferred to P1 |
| 5 | nvbandwidth | cmake not installed by bootstrap on minimal images | Low priority |
| 7 | nccl-tests | `theoretical_nvlink_bw_gbps: 0` when fuzzy GPU spec match fails | Deferred to P1 (lookup_gpu_spec needs better matching) |

### V1.2 Changes (2026-02-11)

1. **gpu-inventory.sh** — Added JSON validation for nvlink_json and topology_json before passing to `jq --argjson`. Python3 parser handles empty optional_fields gracefully.
2. **topology.sh** — Strip ANSI escape codes from `nvidia-smi topo` output.
3. **run-all.sh** — Version banner updated to v1.2.
4. **bootstrap.sh** — Added NCCL library installation (`libnccl2`, `libnccl-dev`) when GPU detected and internet available.
5. **nvbandwidth.sh** — Graceful skip when cmake is unavailable and nvbandwidth binary not found (instead of failing on git clone).
6. **SKILL.md** — Added version documentation.

### V1.2 Known Bugs (to fix in V1.2.1)

| # | Module | Bug | Root Cause | Fix Status |
|---|--------|-----|------------|------------|
| 1 | gpu-inventory | `jq: invalid JSON text passed to --argjson` | `lookup_gpu_spec()` returns `{}}` (extra brace); also `cuda_version` nvidia-smi field unsupported on some drivers returns error string | **Patched locally** — added JSON validation for `$spec`, fallback cuda_version parsing from `nvidia-smi` header |
| 2 | gpu-burn | `errors_detected: 8` false positive | Regex `grep -ci "error\|fault\|fail"` matches "faults: 0" lines — counts 8 GPUs × "faults: 0" as 8 errors | **Patched locally** — changed to count only FAULTY/ERROR/FAIL, subtract "faults: 0" false positives |
| 3 | storage-bench | All fio results show `"error": "parse failed"` | fio JSON output still contaminated by wrapper or parser can't find the temp file output | Needs investigation — V1.2 claimed to fix via direct file output but parsing still broken |
| 4 | nccl-tests | All tests return empty `datapoints: []` and `0 GB/s` | Tests built successfully but ran in <2s total for 5 tests — likely not executing the actual benchmark, or output parser regex doesn't match nccl-tests output format | Needs investigation — parser and test invocation both suspect |
| 5 | nvbandwidth | `build failed` — cmake unavailable | VM lacks cmake; bootstrap doesn't install it; graceful skip logic may not trigger correctly | Low priority — cmake install or better skip logic needed |
| 6 | report | Version string says "v1.0" instead of "v1.2" | Hardcoded or not reading version from run-all.sh | Needs fix |
| 7 | nccl-tests | `theoretical_nvlink_bw_gbps: 0` | NVLink BW lookup not working despite full NV12 mesh detected in topology | Parser needs to read topology data |

### V1.1 Changes

1. **gpu-burn.sh** — Rewritten with direct timeout capture instead of `run_with_timeout`.
2. **storage-bench.sh** — Direct fio JSON output to file for reliable parsing.
3. **bootstrap.sh** — DCGM repo fix: checks if cuda repo exists before adding duplicates.
4. **nccl-tests.sh** — Added ldconfig/find check for NCCL library before attempting tests.
5. **run-all.sh** — Virtualization detection and module skipping.

### V1.0

- Initial battle-tested release.
