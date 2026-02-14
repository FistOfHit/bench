.PHONY: help lint static-checks smoke quick ci-smoke ci-quick

help:
	@echo "HPC Bench Suite helper targets"
	@echo ""
	@echo "Targets:"
	@echo "  make lint          Run pre-commit on all files (format + shellcheck)"
	@echo "  make static-checks Run CI static checks script"
	@echo "  make smoke         Run suite (smoke mode)"
	@echo "  make quick         Run suite (quick mode)"
	@echo "  make ci-smoke      Run smoke mode with --ci"
	@echo "  make ci-quick      Run quick mode with --ci"
	@echo ""
	@echo "Notes:"
	@echo "  - These targets are intended to run on Linux."
	@echo "  - Many modules require sudo/root; prefix with sudo if needed."

lint:
	@command -v pre-commit >/dev/null 2>&1 || { echo "pre-commit not found. Install: pipx install pre-commit"; exit 1; }
	pre-commit run --all-files

static-checks:
	bash scripts/ci-static-checks.sh

smoke:
	bash scripts/run-all.sh --smoke

quick:
	bash scripts/run-all.sh --quick

ci-smoke:
	bash scripts/run-all.sh --smoke --ci

ci-quick:
	bash scripts/run-all.sh --quick --ci
