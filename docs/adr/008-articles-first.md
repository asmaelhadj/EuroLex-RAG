# ADR 008: Rank articles alone, return recitals as support

- Status: accepted (dev results; final numbers come from the test split)
- Date: 2026-10-26

## Context

Gold labels are articles (SPEC.md 5.1): an answer must cite the binding rule. The index also
holds the 173 recitals, which often restate an article in plainer words. On the dev split
(60 in-scope questions), E1 (article-aware chunks, dense) ranked a recital first for 26
questions, and on average 5.3 of the top 10 results were recitals. Reranking (E4) barely
changed this (23 questions with a recital first). If recitals are skipped, the first article
retrieved by E1 was already the gold one for 46 of 60 questions, against R@1 = 0.42.

## Decision

New setting `retrieval.recitals`:

- `include`: recitals and articles are ranked together (E0 to E5);
- `support`: only articles are ranked and scored; the top `support_k` (3) recitals are
  fetched separately and returned in `RetrievalResult.support`, to be given to the answer
  model as context in Phase 5;
- `exclude`: articles only.

The metric is unchanged. Recitals were never counted as hits; this only stops them taking
ranking slots.

## Results (dev split, 60 in-scope questions)

| Run | R@1 | R@10 | MRR | nDCG@10 | FAQ core nDCG@10 | p50 latency |
|---|---|---|---|---|---|---|
| E1 dense, recitals ranked | 0.422 | 0.869 | 0.610 | 0.669 | 0.610 | 0.3 s |
| E1s dense, articles first | 0.711 | 0.961 | 0.842 | 0.856 | 0.792 | 0.3 s |
| E4 hybrid + rerank, recitals ranked | 0.425 | 0.903 | 0.628 | 0.685 | 0.676 | 14.5 s |
| E4s hybrid + rerank, articles first | 0.725 | 0.967 | 0.813 | 0.847 | 0.820 | 15.4 s |

Paired bootstrap over the same 60 questions (5,000 resamples, 95% CI of the difference):

| Comparison | R@1 | nDCG@10 | Questions better / worse |
|---|---|---|---|
| E1s - E1 | +0.289 [+0.183, +0.406] | +0.187 [+0.133, +0.243] | 29 / 0 (nDCG) |
| E4s - E4 | +0.300 [+0.183, +0.417] | +0.162 [+0.113, +0.214] | 32 / 1 (nDCG) |
| E4s - E1s | +0.014 [-0.106, +0.136] | -0.009 [-0.075, +0.056] | 9 / 15 (nDCG) |

## Consequences

- Articles-first is the largest gain found so far, with both retrievers, and almost no
  question gets worse.
- Once recitals are out of the ranking, the reranker adds nothing measurable on dev, while
  taking about 50 times longer. The dense-only E1s is the leading candidate for the final
  pipeline. This is a dev result on 60 questions; the choice is confirmed on the test split
  only once, after Phase 5.
- Recitals are not thrown away: Phase 5 passes them to the answer model as explanation, and
  answers cite articles.
- Hybrid search (E3) and query rewriting (E5) did not beat their baselines on dev either.
