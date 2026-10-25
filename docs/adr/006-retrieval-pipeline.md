# ADR 006: Retrieval pipeline

- Status: accepted
- Date: 2026-10-24

## Decisions

- **One pipeline, many configs.** `retrieval/pipeline.py` builds rewrite -> retrieve -> fuse ->
  rerank from an experiment YAML (`configs/e0_naive.yaml` ... `e5_rewrite.yaml`). Each stage is
  on or off by config. Every result carries per-stage latency in milliseconds.
- **Fusion.** RRF with `k = 60` and optional per-retriever weights. Only ranks are used.
  Ties go to the better best rank, then the smaller chunk id, so runs are repeatable. With a
  single retriever there is no fusion stage.
- **Output size.** Retrieval returns `top_k = 10` for every experiment, so recall@10 and
  nDCG@10 compare like with like. The 8 passages for generation (SPEC.md) are chosen in
  Phase 5 from these 10.
- **Reranking** uses `FlagReranker` with sigmoid-normalised scores (0..1), so Phase 5 can put
  an evidence threshold on them. Article chunks get the same header line as at indexing.
  The reranker scores the **original** question, not the rewrite.
- **Query rewriting** asks the LLM for a search query. `expand` (default) searches with the
  question plus the rewrite; `replace` uses the rewrite only. Any LLM failure falls back to
  the original question.
- **LLM client pulled forward.** Rewriting needs an LLM, so a small `OllamaClient` (timeout,
  retries, temperature 0, fixed seed) lives in `generation/llm.py` now. Phase 5 extends it.
- **BM25** tokenises on `\w+` lower-cased, with no stop-word list and no stemming.

## Measured cost on the development laptop (i5-11400H, GTX 1650 Max-Q 4 GB, plugged in)

| Stage | Time per query |
|---|---|
| Embed query (BGE-M3, CPU fp32) | ~0.2 s (first call ~3 s) |
| Dense / sparse search in Qdrant | ~5-35 ms each |
| RRF | < 1 ms |
| Rerank 50 candidates (bge-reranker-v2-m3, GPU fp32) | **~10-12 s** |

For comparison, reranking 50 candidates took ~110-150 s on the CPU and ~60 s on the GPU in
fp16 (see ADR 005 for why fp16 is slower on this card). The reranker runs on
`rerank.device` (`auto`) in `rerank.precision` (`fp32`). Fewer candidates would be faster;
any change to `candidates` is tuned on the dev split only.
