# CLAUDE.md

Project: **EuroLex-RAG**, an English-language, self-hostable RAG assistant over EU law (GDPR first, then the AI Act), built as a junior AI portfolio project. The full specification is in `SPEC.md`. Read it before doing anything, and re-read the section for the phase you are working on.

## How to work

- Work **one phase at a time**, in the order given in `SPEC.md` section 6. Do not start a phase until the previous phase's acceptance checks pass.
- At the end of each phase: run `make check`, commit with a clear message, and print a short summary of what was done and what the human must do next.
- **Stop and ask** at every "HUMAN GATE" in `SPEC.md`. Never skip a gate or fill in for the human.
- Prefer small, typed, tested modules. No notebooks as the source of truth; notebooks are for exploration only.
- If something in `SPEC.md` is unclear or turns out to be wrong (for example a URL, a file format or a library API), verify it, tell the human, and propose a fix. Do not guess silently.

## Hard rules

1. **Never fabricate results.** Every number in `docs/results.md` and the README must come from an actual run of `make eval`. If a run has not happened, write `TODO` instead of a number.
2. **Never fabricate data.** Do not invent articles, questions, gold labels or citations. Only questions with `status: validated` are used for reported metrics.
3. **No secrets in the repo.** Use environment variables; keep `.env.example` current; `.env` stays in `.gitignore`.
4. **Do not commit** raw data, indices, model weights or large files. Provide scripts to recreate them.
5. **Do not log raw user queries** by default (see `SPEC.md` section 9).
6. Keep the project reproducible: pinned dependencies, fixed random seeds, saved run configs.
7. The project gives **information, not legal advice**. Keep the disclaimer in the UI and README.

## Stack (decided, do not change without asking)

Python 3.11+, `uv` for dependencies, FastAPI, Pydantic v2, Streamlit, Qdrant (Docker), BGE-M3 for embeddings, `bge-reranker-v2-m3` as the reranker, a local LLM through Ollama (default `qwen2.5:7b-instruct`, configurable), pytest, ruff, mypy, pre-commit, GitHub Actions, Docker Compose, Make.

## Common commands (create these in the Makefile in Phase 0)

```
make setup      # install dependencies and pre-commit hooks
make check      # ruff + mypy + pytest
make up         # start Qdrant (and the API/UI when they exist)
make ingest     # download/parse/chunk/embed/index the corpus
make eval CONFIG=configs/e3_hybrid.yaml
make report     # regenerate docs/results.md from saved runs
```

## Code style

- Type hints everywhere; `mypy --strict` on `src/`.
- Pydantic models for every data contract in `SPEC.md` section 5.
- Config through `pydantic-settings` and YAML experiment files; no magic constants in code.
- Docstrings explain **why**, not what. Add an ADR in `docs/adr/` for each real design decision.
