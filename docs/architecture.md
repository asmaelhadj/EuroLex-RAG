# Architecture

## Overview

```mermaid
flowchart LR
    subgraph Offline["Offline: make ingest"]
        CELLAR["CELLAR API<br/>(official EU publications)"] --> DL["download.py<br/>raw XHTML"]
        DL --> PARSE["parse.py<br/>articles + recitals"]
        PARSE --> CHUNK["chunk.py<br/>article-aware chunks"]
        CHUNK --> EMB1["BGE-M3<br/>dense + sparse vectors"]
        EMB1 --> QD[("Qdrant<br/>named vectors<br/>dense / sparse")]
    end

    subgraph Online["Online: one question"]
        UI["Streamlit UI<br/>ui/app.py"] -- "POST /ask<br/>X-API-Key" --> API["FastAPI<br/>api/main.py"]
        API --> AUTH["auth, length check,<br/>rate limit"]
        AUTH --> EMB2["BGE-M3<br/>(CPU)"]
        EMB2 --> RET["dense search<br/>articles first,<br/>3 recitals as support"]
        QD --> RET
        RET --> THR{"top score<br/>≥ 0.5003?"}
        THR -- no --> REFUSE["fixed refusal,<br/>no LLM call"]
        THR -- yes --> PROMPT["prompt: passages as<br/>numbered data blocks"]
        PROMPT --> LLM["Ollama<br/>qwen2.5:7b-instruct"]
        LLM --> CHECK["citation check,<br/>disclaimer ending"]
        CHECK --> API
        REFUSE --> API
    end

    API -. "hash + length, latencies,<br/>scores, token counts" .-> LOGS["JSON logs<br/>(optional Langfuse)"]
```

## Pieces

| Piece | Where | Notes |
|---|---|---|
| Download | `ingestion/download.py` | CELLAR, because EUR-Lex blocks scripts with a bot check (ADR 002) |
| Parse and chunk | `ingestion/parse.py`, `chunk.py` | one chunk per article, split by paragraph when long (ADR 003) |
| Embed and index | `embeddings.py`, `ingestion/index.py` | BGE-M3 at a pinned revision; idempotent upserts (ADR 005) |
| Retrieval | `retrieval/pipeline.py` | dense, sparse, BM25, RRF, reranker and rewriting are all config options (ADR 006); the served config ranks articles only and adds recitals as support (ADR 008) |
| Generation | `generation/` | prompt with passages as untrusted data, `[n]` citations checked, evidence threshold (ADR 009) |
| Evaluation | `evaluation/` | runner, metrics, report, LLM judge, question review tools (ADR 007) |
| API | `api/` | the evaluated `e7_threshold` config; key auth, limits, JSON logs (ADR 010) |
| UI | `ui/` | Streamlit, talks to the API only |

## Running it

- `make up` starts Qdrant, the API and the UI in Docker. Ollama runs on the host (better GPU
  support on Windows and macOS); the containers reach it at `host.docker.internal:11434`.
  To run Ollama in Docker instead: `docker compose --profile ollama up -d ollama` and set
  `DOCKER_OLLAMA_URL=http://ollama:11434` in `.env`.
- Ingestion and evaluation run on the host (`make ingest`, `make eval`), so they can use a
  local GPU. The containers are CPU-only.

## Why these boundaries

- The API serves exactly one evaluated config, so the UI behaves like `docs/results.md` says.
- The UI never imports the pipeline. It could be replaced by any HTTP client.
- Questions are never written to disk by the API: no answer cache, and logs carry only a
  hash and a length unless `LOG_QUERIES=true`.
