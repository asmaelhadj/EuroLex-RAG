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

`POST /ask` and `GET /health` follow SPEC.md section 6. Some small choices:

- Cited recitals have no article id. They get their chunk id (e.g. `gdpr:rct-85`) as
  `article_id` and "Recital 85" as `article_title`, so the field is never empty.
- `latency_ms.rerank` is 0, because the served pipeline has no reranker (ADR 008).
- The EUR-Lex link jumps to the cited article or recital (`#art_33`, `#rct_85`). The stored
  source URL is per act, so the API adds the anchor; checked on the live EUR-Lex page
  (2026-10-27), which uses the same ids as the CELLAR copy we ingest.

Two fields go beyond the SPEC contract, both added for the UI: `citations[].score` (the
passage's retrieval similarity, shown as a badge and labelled as not a probability) and
`passages` (how many passages the model was given).

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
The HTTP part (`ui/client.py`) and the HTML helpers (`ui/render.py`) are separate from the
page so they can be tested. Layout: sidebar with the corpus and example questions, header
with an API status pill and the not-legal-advice banner, chat cards with `[n]` badges and a
stats row, and a sources panel with the exact text and EUR-Lex link per citation. Refusals
and errors have their own look. Model output is HTML-escaped before it is shown, and
markdown images are dropped, so an answer cannot inject tags or load outside URLs. Custom
CSS on top of Streamlit, not a separate frontend, so the stack stays Python-only. The chat
history is kept only in the browser session.

## Tests

`tests/test_api.py` runs the real pipeline over a five-chunk in-memory Qdrant index with a
fake embedder and a mocked LLM: auth, length limits, rate limit, threshold refusal, LLM
failure, OpenAPI docs, and the UI client against the API. Docker Compose for the API and UI
is Phase 7.
