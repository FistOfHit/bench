# Development Notes

This repository is intentionally shell-heavy: it is designed to run on fresh Linux hosts with minimal dependencies.
The goal is to keep the suite easy to copy onto a server and run with `bash`.

## Conventions

- Scripts under `scripts/` are "modules". Each module should:
  - `source lib/common.sh`
  - set `SCRIPT_NAME="<module-name>"` (used for logging + crash records)
  - write a single JSON result file via `emit_json "<module-name>" "<status>"` where status is one of:
    - `ok`, `warn`, `error`, `skipped`
  - **Do NOT use** `pass`, `fail`, or `timeout` as status values. The canonical set is **ok / warn / error / skipped**.
- If a module cannot run due to missing hardware, use the `skip_module` helper:
  ```bash
  skip_module "module-name" "reason text"
  ```
  This emits a standard "skipped" JSON record with both `note` and `skip_reason` fields and exits cleanly.
- Use `require_gpu` for GPU modules — it calls `skip_module` internally.
- Use `finish_module` for the standard 3-line ending pattern:
  ```bash
  finish_module "module-name" "status" "$RESULT" '{optional_jq_summary}'
  ```
  This calls `emit_json`, `log_ok`, and prints a compact summary to stdout.
- If a module exits unexpectedly before emitting JSON, `lib/common.sh` writes a crash record on exit so the report does not silently omit the module.

## Single source of truth

- **Module list**: `specs/modules.json` is the single source of truth for which modules exist, their phases, ordering, and required commands. `lib/report-common.sh` derives `ALL_MODULES` from this manifest at runtime.
- **Status values**: Always use `ok` / `warn` / `error` / `skipped`. The report scoring handles these in `lib/report-common.sh`.
- **Thresholds and tunables**: Centralized in `conf/defaults.sh`. Override by exporting variables before running, or by creating a `conf/local.sh` file (gitignored).
- **Phase numbering**: The manifest uses phases 0–5. `run-all.sh` comments match these numbers.

## Configuration

`conf/defaults.sh` contains all configurable thresholds and tunables:
- Module timeouts (`MAX_MODULE_TIME_QUICK`, `MAX_MODULE_TIME_FULL`)
- GPU burn duration (`GPU_BURN_DURATION_QUICK`, `GPU_BURN_DURATION_FULL`)
- Thermal thresholds (`GPU_THERMAL_WARN_C`)
- Container images (`HPL_IMAGE`, `HPL_IMAGE_ALT`)
- NCCL test timeouts, STREAM benchmark parameters, etc.

To override, either:
1. Export variables: `export GPU_THERMAL_WARN_C=90 && sudo bash scripts/run-all.sh`
2. Create `conf/local.sh` with your overrides (gitignored via `*.local` pattern)

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
3. The report will automatically include it via the manifest — no need to edit `report-common.sh`.
4. If the module needs custom scoring (beyond the default status-based scoring), extend `score_module()` in `lib/report-common.sh`.
5. If the module needs custom report sections, add them to `scripts/report.sh`.
6. Ensure it emits JSON consistently (prefer `emit_json_safe` when generating JSON from other tools).

## Output conventions

- When `HPC_ASCII_OUTPUT=1`, report and run-all checklist use ASCII labels (`[OK]`, `[WARN]`, `[FAIL]`, `[SKIP]`) instead of Unicode symbols; use `status_display_string PASS|WARN|FAIL|SKIP` from `lib/common.sh` for any new user-facing status output.

## Variable naming

- `UPPER_SNAKE_CASE` for exported/global variables and constants
- `lower_snake_case` for local variables within functions
- Prefix internal library variables with `_` (e.g., `_output`, `_tmp`)

## Results contract

The report generator expects:

- One JSON file per module at `$HPC_RESULTS_DIR/<module>.json`
- A generated report at `$HPC_RESULTS_DIR/report.md`
- Optional module logs under `$HPC_LOG_DIR/` (defaults to `$HPC_RESULTS_DIR/logs`)

Avoid renaming JSON fields without updating the report and any external consumers. Each module JSON includes `suite_version` (from the VERSION file) for traceability; see `emit_json` in `lib/common.sh`.
