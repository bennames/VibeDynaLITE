.PHONY: setup test test-all lint format bench clean help

PYTHON ?= python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy

help: ## Show this help message
	@echo "KevlarGrid Explicit Solver v2.0 - Development Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv, install deps, detect hardware
	@echo "==> Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "==> Upgrading pip..."
	$(PIP) install --upgrade pip
	@echo "==> Installing core + dev dependencies..."
	$(PIP) install -e ".[dev]"
	@echo ""
	@echo "==> Attempting to install GUI dependencies..."
	-$(PIP) install -e ".[gui]" 2>/dev/null || echo "  [SKIP] GUI deps not available on this platform"
	@echo ""
	@echo "==> Attempting to install compute dependencies..."
	-$(PIP) install -e ".[compute]" 2>/dev/null || echo "  [SKIP] Compute deps not available (JAX/Numba)"
	@echo ""
	@echo "==> Attempting to install visualization dependencies..."
	-$(PIP) install -e ".[viz]" 2>/dev/null || echo "  [SKIP] Visualization deps not available"
	@echo ""
	@echo "==> Detecting hardware backend..."
	$(VENV)/bin/python scripts/detect_backend.py
	@echo ""
	@echo "✅ Setup complete! Activate with: source $(VENV)/bin/activate"

test: ## Run unit tests (fast)
	OMP_NUM_THREADS=1 $(PYTEST) tests/unit/ -v --tb=short

test-all: ## Run all tests including integration and slow
	OMP_NUM_THREADS=1 $(PYTEST) tests/ -v --tb=short

test-unit: ## Run only unit tests
	OMP_NUM_THREADS=1 $(PYTEST) tests/unit/ -v --tb=short

test-integration: ## Run integration tests
	OMP_NUM_THREADS=1 $(PYTEST) tests/integration/ -v --tb=short -m "not slow"

test-slow: ## Run slow integration tests
	OMP_NUM_THREADS=1 $(PYTEST) tests/ -v --tb=short -m slow

lint: ## Run ruff linter and mypy type checker
	@echo "==> Running ruff check..."
	$(RUFF) check src/ tests/
	@echo "==> Running mypy..."
	$(MYPY) src/

format: ## Format code with ruff
	$(RUFF) format src/ tests/
	$(RUFF) check --fix src/ tests/

bench: ## Run performance benchmarks
	$(VENV)/bin/python benchmarks/bench_solver.py

clean: ## Remove build artifacts, caches, autosave
	rm -rf build/ dist/ *.egg-info/
	rm -rf .mypy_cache/ .ruff_cache/ .pytest_cache/
	rm -rf htmlcov/ .coverage
	rm -rf .autosave/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleaned build artifacts and caches"

clean-all: clean ## Remove venv and all generated files
	rm -rf $(VENV)
	@echo "✅ Removed virtual environment"
