# Contributing to HPC Bench Suite

Thank you for your interest in contributing. This document provides a quick overview; see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for full conventions and implementation details.

## Development Setup

1. Clone the repository and ensure you have:
   - `bash` 4+, `jq` 1.6+, `python3`
   - Optional: `pre-commit` for linting (`pipx install pre-commit`)

2. Run static checks locally:

   ```bash
   make lint          # pre-commit (shfmt, shellcheck, etc.)
   make static-checks # Full CI checks
   ```

## Key Conventions

- **Modules** live under `scripts/`, each with `SCRIPT_NAME` and `source lib/common.sh`
- **Status values**: Use only `ok`, `warn`, `error`, `skipped` (never `pass`, `fail`, `timeout`)
- **Module manifest**: `specs/modules.json` is the single source of truth; add new modules there
- **Config**: Thresholds in `conf/defaults.sh`; overrides via `conf/local.sh` (gitignored)
- **POSIX**: Prefer portable `awk`; avoid gawk-specific features like `match($0, re, arr)`

## Adding a New Module

1. Create `scripts/<module>.sh` (executable)
2. Register in `specs/modules.json` (phase, order, required_cmds)
3. Use `finish_module` or `emit_json` for output; `skip_module` for hardware gaps
4. Run `make smoke` or `make quick` to verify

## Submitting Changes

- Run `make lint` and `make static-checks` before pushing
- Keep commits focused; reference issues where applicable
