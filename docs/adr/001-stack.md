# ADR 001: Technology stack

- Status: accepted
- Date: 2026-10-22

## Context

EuroLex-RAG must be self-hostable, reproducible by a stranger from a clean clone, and cheap to run
on a single developer machine. The main deliverable is a measured comparison of retrieval
strategies (SPEC.md section 6, experiments E0 to E7), so every component must be swappable by
configuration and runnable offline.

## Decision

| Area | Choice | Why |
|---|---|---|
| Language / tooling | Python 3.11+, `uv` with a committed `uv.lock` | Fast, reproducible installs with exact pins |
| Embeddings | `BAAI/bge-m3` (dense + sparse) | One model gives both vector types needed for hybrid search. E6 compares it with `bge-large-en-v1.5` and `e5-large-v2` |
| Reranker | `BAAI/bge-reranker-v2-m3` on the fused top-50 | Open cross-encoder; same family as the embedder |
| Vector DB | Qdrant in Docker, named vectors `dense` and `sparse` | Native sparse vectors and payload filtering (e.g. by `act_id`) |
| Lexical baseline | `rank_bm25` | Independent BM25 baseline for E2, separate from Qdrant sparse |
| LLM | Ollama, default `qwen2.5:7b-instruct`, set by `LLM_MODEL` | Local and free; one `LLMClient` wrapper makes models swappable |
| API / UI | FastAPI + Pydantic v2, Streamlit | Typed contracts and OpenAPI for free; UI calls the API only |
| Evaluation | Own metrics code, optional RAGAS for faithfulness | Metrics are small and must be unit-tested by hand; judge model differs from the answering model |
| Observability | Structured JSON logs, optional Langfuse | No raw queries logged by default (privacy) |
| Quality | ruff, mypy `--strict`, pytest, pre-commit, GitHub Actions | Same `make check` locally and in CI |

## Ollama: host or container

Ollama runs on the **host** by default because GPU passthrough into Docker is awkward on Windows
and macOS. Containers reach it at `http://host.docker.internal:11434`. A `docker compose
--profile ollama` service is provided for Linux or CPU-only setups.

## Consequences

- Heavy ML dependencies (PyTorch, `FlagEmbedding` or `sentence-transformers`) are added in Phase 2,
  not Phase 0, so CI for the skeleton stays fast. Their exact library API is verified then.
- Changing any row above requires asking the project owner and a new ADR.
