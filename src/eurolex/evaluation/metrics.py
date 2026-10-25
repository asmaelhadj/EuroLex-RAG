"""Article-level retrieval metrics.

Retrieved chunks are mapped to article ids. Recital chunks have no article id, so they are
never relevant but still take a rank. If several chunks of one article are retrieved, only
the first one counts, so splitting an article into paragraphs is not rewarded.
"""

import math
from collections.abc import Sequence


def first_hits(retrieved: Sequence[str | None]) -> list[str | None]:
    """Keep each article id at its first rank; later repeats become None."""
    seen: set[str] = set()
    out: list[str | None] = []
    for article in retrieved:
        if article is None or article in seen:
            out.append(None)
        else:
            seen.add(article)
            out.append(article)
    return out


def recall_at_k(retrieved: Sequence[str | None], gold: Sequence[str], k: int) -> float:
    if not gold:
        raise ValueError("recall needs gold articles")
    found = {a for a in retrieved[:k] if a is not None} & set(gold)
    return len(found) / len(set(gold))


def reciprocal_rank(retrieved: Sequence[str | None], gold: Sequence[str]) -> float:
    gold_set = set(gold)
    for rank, article in enumerate(retrieved, start=1):
        if article is not None and article in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str | None], gold: Sequence[str], k: int = 10) -> float:
    """Binary relevance; the ideal list has min(|gold|, k) relevant items at the top."""
    if not gold:
        raise ValueError("nDCG needs gold articles")
    gold_set = set(gold)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, article in enumerate(first_hits(retrieved)[:k], start=1)
        if article is not None and article in gold_set
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold_set), k) + 1))
    return dcg / ideal


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile, q in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(q / 100 * len(ordered)) - 1)
    return ordered[index]
