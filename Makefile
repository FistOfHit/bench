.PHONY: help lint shellcheck test static-checks check smoke quick ci-smoke ci-quick version

# ── Meta ──

help:
	@echo "HPC Bench Suite — Make targets"
	@echo ""
	@echo "Quality:"
	@echo "  make check          Run all quality gates (shellcheck + unit tests + static checks)"
	@echo "  make lint           Run pre-commit on all files (format + shellcheck)"
	@echo "  make shellcheck     Run shellcheck on all shell scripts"
	@echo "  make test           Run unit tests (requires bats)"
	@echo "  make static-checks  Run CI static checks script"
	@echo ""
	@echo "Execution:"
	@echo "  make smoke          Run suite in smoke mode (bootstrap + inventory + report)"
	@echo "  make quick          Run suite in quick mode (short benchmarks)"
	@echo "  make ci-smoke       Run smoke with --ci (compact output for CI)"
	@echo "  make ci-quick       Run quick with --ci (compact output for CI)"
	@echo ""
	@echo "Info:"
	@echo "  make version        Show suite version"
	@echo ""
	@echo "Notes:"
	@echo "  - These targets are intended to run on Linux."
	@echo "  - Many modules require sudo/root; prefix with sudo if needed."

# ── Quality gates ──

check: shellcheck test static-checks
	@echo ""
	@echo "All quality gates passed."

lint:
	@command -v pre-commit >/dev/null 2>&1 || { echo "pre-commit not found. Install: pipx install pre-commit"; exit 1; }
	pre-commit run --all-files

shellcheck:
	@command -v shellcheck >/dev/null 2>&1 || { echo "shellcheck not found. Install: sudo apt-get install shellcheck"; exit 1; }
	shellcheck -s bash -S error scripts/*.sh lib/*.sh

test:
	@command -v bats >/dev/null 2>&1 || { echo "bats not found. Install: sudo apt-get install bats"; exit 1; }
	bats tests/

static-checks:
	bash scripts/ci-static-checks.sh

# ── Execution ──

smoke:
	bash scripts/run-all.sh --smoke

quick:
	bash scripts/run-all.sh --quick

ci-smoke:
	bash scripts/run-all.sh --smoke --ci

ci-quick:
	bash scripts/run-all.sh --quick --ci

# ── Info ──

version:
	@cat VERSION
