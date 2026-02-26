# AGENTS.md

## Cursor Cloud specific instructions

This is a pure-Bash HPC benchmarking CLI suite (no web services, databases, or background daemons). Development happens entirely in the terminal.

### Quick reference

| Task | Command |
|------|---------|
| Lint (pre-commit: shfmt + shellcheck) | `make lint` |
| Unit tests (BATS, 98 tests) | `make test` |
| All quality gates | `make check` |
| CI static checks | `make static-checks` |
| Smoke run (bootstrap + inventory + report, ~5s) | `sudo bash scripts/run-all.sh --smoke --ci` |
| Quick run (short benchmarks) | `sudo bash scripts/run-all.sh --quick --ci` |

See `Makefile` for all targets and `README.md` / `SKILL.md` for full documentation.

### Gotchas

- `pre-commit` is installed as a user package (`pip install --user`). Ensure `$HOME/.local/bin` is on `PATH` (the update script handles this).
- `make static-checks` invokes `scripts/ci-static-checks.sh` which creates its own virtualenv — requires `python3-venv` to be installed.
- The suite is designed for bare-metal HPC servers. In a VM/container, GPU modules (`gpu-inventory`, `dcgm-diag`, `gpu-burn`, `nccl-tests`, `nvbandwidth`, `hpl-mxp`), BMC, and InfiniBand modules skip gracefully with `status: "skipped"`.
- The lock file at `/var/log/hpc-bench/results/.hpc-bench.lock` prevents concurrent runs. If a previous run was killed, remove the lock manually before re-running.
- Results default to `/var/log/hpc-bench/results/` (root) or `$HOME/.local/var/hpc-bench/results` (non-root). Override with `HPC_RESULTS_DIR`.
