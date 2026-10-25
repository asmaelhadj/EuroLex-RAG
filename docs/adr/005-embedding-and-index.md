# ADR 005: Embedding runtime and Qdrant index layout

- Status: accepted
- Date: 2026-10-24

## Embedding: BGE-M3 through FlagEmbedding

`src/eurolex/embeddings.py` wraps `FlagEmbedding.BGEM3FlagModel` (1.4.x), the model authors'
reference implementation, and returns, per text:

- dense = the 1024-d L2-normalised CLS vector;
- sparse = the lexical weights (`{token_id: weight}`), converted to sorted indices for Qdrant.

**Alternative considered:** reimplementing the same computation directly on `transformers`
(about 60 lines, verified against FlagEmbedding's source). It avoided FlagEmbedding's
training-only dependencies (`datasets`, `peft`, `ir-datasets`, `sentence-transformers`). The
project owner chose FlagEmbedding: no risk of drifting from the reference, less custom code,
and `FlagReranker` is available for Phase 3. The cost is a few hundred MB of extra packages.

**Model files.** Given a repo name, FlagEmbedding downloads the whole Hugging Face repo,
including a 2.3 GB ONNX export. We download only the files inference needs at a **pinned
revision** (`configs/ingest.yaml`) and pass the local folder. `colbert_linear.pt` must be among
them: FlagEmbedding loads the trained `sparse_linear.pt` only if **both** files exist, and
otherwise initialises the sparse head at random with only a log message.

ColBERT multi-vectors are not used. SPEC.md's hybrid search is dense + sparse.

## PyTorch: CUDA on Windows, CPU on Linux

`pyproject.toml` takes `torch` from the CUDA 12.6 index on Windows, the CPU index on Linux and
PyPI on macOS.

- First we used CPU wheels everywhere. On the dev laptop (i5-11400H) that meant ~2 s per
  model forward pass: 17 min to index the GDPR and ~2 min per query to rerank 50 candidates,
  which is too slow for evaluation runs and a demo (2026-10-24).
- The dev machine has a GTX 1650 (4 GB, compute capability 7.5). cu126 supports it; the wheel
  is 2.6 GB.
- CI runs on Linux with the CPU wheel. Its tests use fake models, so they stay fast.
- `embedding.device: auto` picks CUDA when available and then runs in **fp16**. fp16 vectors
  differ slightly from fp32 ones, so the precision is part of `embedding_model` in the payload
  (e.g. `BAAI/bge-m3@5617a9f61b02+fp16`). Switching device therefore re-embeds the
  collection on the next `make ingest`.

## Index layout

- One Qdrant collection per chunking strategy: `eurolex_article`, `eurolex_fixed`.
- Named vectors: `dense` (1024-d, cosine) and `sparse` (BGE-M3 lexical weights; no IDF modifier,
  because the weights are already learned).
- Point id = UUIDv5(fixed namespace, `chunk_id`).
- Payload = the full `Chunk` plus `article_id`, `embedding_model` (name@revision) and the
  SHA-256 of the exact text embedded. Keyword payload indexes are created on `act_id`, `kind`
  and `article_id` for filtering.
- Indexing is a **sync**: unchanged chunks are skipped, changed ones re-embedded, and points
  whose chunk disappeared are deleted. A collection with an incompatible vector layout is an
  error unless `--recreate` is passed.

## What is embedded

Article-aware chunks get a header line, e.g. `GDPR Article 6(1) - Lawfulness of processing`,
because paragraph chunks rarely restate their article's subject. Fixed-size chunks (the E0
baseline) are embedded as plain text, so that baseline stays naive. The header is part of the
hashed input, so toggling it re-embeds the affected collection.

Chunk sizes are still counted in words (ADR 003). The indexer counts real BGE-M3 tokens for
every chunk and reports any that exceed `max_length` (1024) and would be truncated.
