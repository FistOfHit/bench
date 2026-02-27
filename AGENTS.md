# AGENTS.md

## Cursor Cloud specific instructions

This is a pure-Bash HPC benchmarking CLI suite (no web services, databases, or background daemons). Development happens entirely in the terminal.

### Quick reference

| Task | Command |
|------|---------|
| Lint (pre-commit: shfmt + shellcheck) | `make lint` |
| Unit tests (BATS, 126 tests) | `make test` |
| All quality gates | `make check` |
| CI static checks | `make static-checks` |
| Smoke run (bootstrap + inventory + report, ~5s) | `sudo bash scripts/run-all.sh --smoke --ci` |
| Quick run (short benchmarks) | `sudo bash scripts/run-all.sh --quick --ci` |

| Check dependency updates | `make check-updates` |
| Preview dependency updates | `bash scripts/check-updates.sh --apply --dry-run` |
| Build portable bundle (tarball) | `bash scripts/build-portable-bundle.sh` |
| Build single .run file (for USB) | `bash scripts/build-portable-bundle.sh --makeself` (requires makeself on build host) |

See `Makefile` for all targets and `README.md` / `SKILL.md` for full documentation. **Portable bundle (tarball or single .run):** full how-to, single-file steps, build options, and agent reference in [docs/PORTABLE-BUNDLE.md](docs/PORTABLE-BUNDLE.md).

### Dependency update system

`scripts/check-updates.sh` tracks 14 external dependencies (container images, NVIDIA packages, upstream repos, pre-commit hooks) via `specs/dependencies.json`. It queries nvcr.io, Docker Hub, GitHub, and NVIDIA apt repos. Key modes:
- `--json` for CI consumption
- `--apply` to update version pins in source files
- `--apply --dry-run` to preview without modifying
- `--category <cat>` to filter (container_image, nvidia_package, pre_commit_hook, upstream_source)

The weekly GitHub Actions workflow (`.github/workflows/dependency-update.yml`) runs this automatically, validates with lint/tests/smoke, and opens a PR. CUDA↔driver compatibility constraints are checked before applying.

### Gotchas

- `pre-commit` is installed as a user package (`pip install --user`). Ensure `$HOME/.local/bin` is on `PATH` (the update script handles this).
- `make static-checks` invokes `scripts/ci-static-checks.sh` which creates its own virtualenv — requires `python3-venv` to be installed.
- The suite is designed for bare-metal HPC servers. In a VM/container, GPU modules (`gpu-inventory`, `dcgm-diag`, `gpu-burn`, `nccl-tests`, `nvbandwidth`, `hpl-mxp`), BMC, and InfiniBand modules skip gracefully with `status: "skipped"`.
- The lock file at `/var/log/hpc-bench/results/.hpc-bench.lock` prevents concurrent runs. If a previous run was killed, remove the lock manually before re-running.
- Results default to `/var/log/hpc-bench/results/` (root) or `$HOME/.local/var/hpc-bench/results` (non-root). Override with `HPC_RESULTS_DIR`.
