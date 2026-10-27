# ADR 010: API and UI

- Status: accepted
- Date: 2026-10-27

## What the API serves

The API loads `API_CONFIG`, by default `configs/e7_threshold.yaml`: the exact pipeline that
was evaluated (E1s retrieval, qwen2.5:7b-instruct, threshold 0.5003). So answers in the UI
match what `docs/results.md` measured. The judge in that config is not used.

The LLM disk cache used by evaluation is **off** in the API, because it would write users'
questions to disk (SPEC.md section 9). Uvicorn's access log has only the method, path and
status, never the question.

## Contract

`POST /ask` and `GET /health` follow SPEC.md section 6. Two small choices:

- Cited recitals have no article id. They get their chunk id (e.g. `gdpr:rct-85`) as
  `article_id` and "Recital 85" as `article_title`, so the field is never empty.
- `latency_ms.rerank` is 0, because the served pipeline has no reranker (ADR 008).
- The EUR-Lex link is the whole act, not the article: our source URL is per act.

`POST /feedback` (a stretch goal) is not built.

## Security

- `X-API-Key` checked with `secrets.compare_digest`; missing or wrong key is 401, and it is
  checked before the body, so an unauthorised caller learns nothing about validation.
- The app will not start without `API_KEY` (at least 16 characters).
- Question length: 1 to `MAX_QUESTION_CHARS` (500) after trimming, else 422. `act_id` must
  match `^[a-z_]+$`.
- Rate limit: `RATE_LIMIT_PER_MINUTE` (10) per client address, sliding window, 429 with
  `Retry-After`. It is in memory, so it is per process; several workers would need a shared
  store. Requests from the UI all come from one address, so they share one budget.
- One question at a time (a lock): the models share one 4 GB GPU. On this laptop an answer
  takes about a minute, so a second user waits.
- Ollama down or failing gives 503, not a stack trace.

## UI

Streamlit calls the API over HTTP with the key from `.env`, never the pipeline directly.
The HTTP part (`ui/client.py`) is separate from the page so it can be tested. The page shows
a not-legal-advice banner, the answer with `[n]` markers, one expander per citation with the
exact text and the EUR-Lex link, and a separate look for refusals.

## Tests

`tests/test_api.py` runs the real pipeline over a five-chunk in-memory Qdrant index with a
fake embedder and a mocked LLM: auth, length limits, rate limit, threshold refusal, LLM
failure, OpenAPI docs, and the UI client against the API. Docker Compose for the API and UI
is Phase 7.
