# AGENTS.md

## Cursor Cloud specific instructions

### Overview

HPC Bench Suite is a pure-Bash benchmarking/diagnostics CLI for HPC servers. There are no long-running services — just shell scripts executed on demand. See `SKILL.md` for the full agent guide and `README.md` for usage.

### Quality gates (single service)

| Check | Command |
|-------|---------|
| Lint (pre-commit: shfmt + shellcheck) | `make lint` |
| ShellCheck only | `make shellcheck` |
| Unit tests (bats) | `make test` |
| Static checks (bash -n + pre-commit in venv) | `make static-checks` |
| All quality gates | `make check` |

### Running the suite

- **Smoke mode** (inventory + report only, no benchmarks; ~10s): `sudo bash scripts/run-all.sh --smoke --ci`
- **Quick mode** (short benchmarks): `sudo bash scripts/run-all.sh --quick --ci`
- The `--ci` flag compacts stdout and enables quick-mode defaults.
- This VM has no GPU; GPU modules (`gpu-inventory`, `gpu-burn`, `dcgm-diag`, `nccl-tests`, `nvbandwidth`, `hpl-mxp`) and `bmc-inventory` will be skipped automatically.

### Non-obvious caveats

- `make static-checks` creates a temporary Python venv internally via `scripts/ci-static-checks.sh` — requires `python3.12-venv` to be installed.
- `pre-commit` is installed to `~/.local/bin` — ensure that directory is on `PATH`.
- The suite uses an exclusive lock file; if a previous run was killed, remove `/var/log/hpc-bench/results/.hpc-bench.lock` (or `.hpc-bench.lock.d`) before re-running.
- Results default to `/var/log/hpc-bench/results` (root) or `$HOME/.local/var/hpc-bench/results` (non-root).
