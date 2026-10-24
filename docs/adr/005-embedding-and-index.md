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

## PyTorch: CPU wheels by default

`pyproject.toml` takes `torch` from the PyTorch CPU index (PyPI on macOS). This keeps installs
small (the CUDA wheels are several GB), and CI and laptops get the same numerics. The corpus is
a few hundred chunks, so CPU indexing takes minutes. To use a GPU, point `torch` at a CUDA index
and set `embedding.device: cuda` (`auto` already picks CUDA when it is available, and enables
fp16 there).

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
