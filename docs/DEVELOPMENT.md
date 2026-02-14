# Development Notes

This repository is intentionally shell-heavy: it is designed to run on fresh Linux hosts with minimal dependencies.
The goal is to keep the suite easy to copy onto a server and run with `bash`.

## Conventions

- Scripts under `scripts/` are "modules". Each module should:
  - `source lib/common.sh`
  - set `SCRIPT_NAME="<module-name>"` (used for logging + crash records)
  - write a single JSON result file via `emit_json "<module-name>" "<status>"` where status is one of:
    - `ok`, `warn`, `error`, `skipped`
- If a module cannot run due to missing hardware, prefer emitting `skipped` with a `note` or `skip_reason`.
- If a module exits unexpectedly before emitting JSON, `lib/common.sh` writes a crash record on exit so the report does not silently omit the module.

## Formatting and linting

Formatting and basic hygiene checks are handled by `pre-commit`:

- `shfmt` for shell formatting
- `shellcheck` for shell static analysis (errors only)
- whitespace, EOF newline, YAML/JSON checks

Run locally:

```bash
pre-commit run --all-files
```

Or via `make` (Linux):

```bash
make lint
make static-checks
```

## Adding a new module

1. Create `scripts/<module>.sh` and ensure it is executable.
2. Register it in `specs/modules.json` (phase, ordering, root requirement, required commands).
3. Add it to `ALL_MODULES` in `lib/report-common.sh` so the report includes it.
4. Decide how it should score (PASS/WARN/FAIL/SKIP). If needed, extend `score_module()` with module-specific rules.
5. Ensure it emits JSON consistently (prefer `emit_json_safe` when generating JSON from other tools).

## Results contract

The report generator expects:

- One JSON file per module at `$HPC_RESULTS_DIR/<module>.json`
- A generated report at `$HPC_RESULTS_DIR/report.md`
- Optional module logs under `$HPC_LOG_DIR/` (defaults to `$HPC_RESULTS_DIR/logs`)

Avoid renaming JSON fields without updating the report and any external consumers.
