.PHONY: help setup lint format type test coverage rust-check rust-develop parity validation data evaluation report demos diagram docker-build docker-up clean all

DATA ?= ../.bench-data/ad-auction-lab

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies (builds the Rust core) and pre-commit hooks
	uv sync --dev
	uv run pre-commit install

format: ## Format automatically
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Lint (no fixes)
	uv run ruff format --check .
	uv run ruff check .

type: ## Type-check
	uv run mypy src

test: ## Run tests
	uv run pytest

coverage: ## Tests with coverage gate
	uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=80

rust-check: ## Rust core: fmt, clippy (with and without Python), unit tests
	cd rust && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo clippy --all-targets --features python -- -D warnings && cargo test

rust-develop: ## Rebuild the Rust core into .venv after editing rust/ (uv sync also builds it)
	uv sync --dev --group rust
	uv run --no-sync maturin develop --uv --release

parity: ## Rust vs numpy reference parity tests
	AAL_REQUIRE_RUST=1 uv run --no-sync pytest tests/test_rust_parity.py -v

validation: ## Tests against obp and scikit-learn (obp is installed without its pinned dependencies)
	uv sync --dev --group validation
	uv pip install --no-deps obp==0.5.7
	uv run --no-sync pytest -m validation -v

data: ## Stream Criteo (1 row in 4, 1.35 GB) and the Open Bandit Dataset into $(DATA)
	scripts/download_data.sh $(DATA)

diagram: ## Render architecture diagram to SVG
	npx -y @mermaid-js/mermaid-cli -i docs/architecture/architecture.mmd \
	     -o docs/architecture/architecture.svg \
	     -c mermaid.config.json -b transparent

docker-build: ## Build the image
	docker build -t ad-auction-lab .

docker-up: ## Run via docker compose
	docker compose up --build

clean: ## Remove caches and artifacts
	rm -rf .pytest_cache .pytest-tmp .mypy_cache .ruff_cache dist build *.egg-info coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +

all: lint type coverage ## Full local gate (mirror of continuous integration)
