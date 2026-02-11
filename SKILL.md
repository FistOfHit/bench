# HPC Benchmark Suite

Comprehensive HPC server benchmarking suite — GPU, CPU, network, storage, topology.

## Usage

```bash
# Full suite (as root)
bash scripts/bootstrap.sh   # Install deps
bash scripts/run-all.sh     # Run everything

# Individual modules
bash scripts/gpu-inventory.sh
bash scripts/gpu-burn.sh
# etc.
```

## Version History

### V1.3 Changes (2026-02-11)

**VM and robustness: fixes for virtualized environments and brittle scripts.**

1. **gpu-inventory.sh** — VM-safe: (a) optional-field probe no longer triggers `set -e` when `nvidia-smi` exits non-zero; (b) driver/CUDA version queries use `|| true` so unsupported fields (e.g. `cuda_version` in VMs) don’t exit; (c) full GPU query fallback to base fields only when combined query returns no data; (d) OPT_JSON loop fixed (`$opt_first` as command ran `false` and exited — now use numeric flag); (e) all JSON for final jq passed via temp files + `--slurpfile` to avoid arg length and invalid spec from `lookup_gpu_spec`; (f) spec file written only after `jq -c` validation; (g) CUDA version fallback parse uses `sed` for version number only (avoids "|" from table).
2. **lib/common.sh** — `lookup_gpu_spec`: validate and echo only valid JSON (exact and fuzzy paths); malformed spec now returns `{}`.
3. **filesystem-diag.sh** — (a) RAID block: `md_arrays`/`md_detail` sanitized, jq failure doesn’t exit; (b) LVM/raid/pfs JSON validated before final jq; (c) mounts/pfs/raid/lvm passed via temp files + `--slurpfile` to avoid invalid/oversized `--argjson`.
4. **nvbandwidth.sh** — When build or offline clone fails, emit `skipped` and exit 0 so suite doesn’t fail (VM/bare metal without nvbandwidth).
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
