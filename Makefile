.DEFAULT_GOAL := help
.PHONY: help setup check lint format typecheck test up down ingest eval report

CONFIG ?= configs/e3_hybrid.yaml

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

setup: ## Install dependencies (locked) and pre-commit hooks
	uv sync --locked
	uv run pre-commit install

check: lint typecheck test ## ruff + mypy + pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-fix lint and formatting
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest --cov --cov-report=term-missing

up: ## Start Qdrant (API and UI are added in Phase 7)
	docker compose up -d --wait qdrant

down: ## Stop services
	docker compose down

ingest: ## Download, parse and chunk the corpus (embedding and indexing come in Phase 2)
	uv run python -m eurolex.ingestion --config configs/ingest.yaml

eval: ## Run an experiment: make eval CONFIG=configs/e3_hybrid.yaml
	@echo "make eval CONFIG=$(CONFIG): not implemented yet (Phase 4)" && exit 1

report: ## Regenerate docs/results.md from saved runs
	@echo "make report: not implemented yet (Phase 4)" && exit 1
