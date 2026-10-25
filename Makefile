.DEFAULT_GOAL := help
.PHONY: help setup check lint format typecheck test up down ingest eval report candidates review

CONFIG ?= configs/e3_hybrid.yaml
SPLIT ?= dev

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

ingest: ## Download, parse, chunk, embed and index the corpus (needs `make up`)
	uv run python -m eurolex.ingestion --config configs/ingest.yaml

eval: ## Run an experiment: make eval CONFIG=configs/e3_hybrid.yaml SPLIT=dev
	uv run python -m eurolex.evaluation.runner --config $(CONFIG) --split $(SPLIT)

report: ## Regenerate docs/results.md from saved runs
	uv run python -m eurolex.evaluation.report

candidates: ## Generate pending LLM question candidates (MODEL=mistral:latest)
	uv run python -m eurolex.evaluation.candidates $(if $(MODEL),--model $(MODEL))

review: ## Review pending questions in the terminal
	uv run python -m eurolex.evaluation.review_cli review
