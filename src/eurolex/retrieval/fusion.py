"""Reciprocal rank fusion: score(d) = sum_i w_i / (k + rank_i(d)), ranks starting at 1."""

from collections.abc import Sequence

from eurolex.schemas import Chunk, ScoredChunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[ScoredChunk]:
    """Fuse ranked lists. Only ranks matter; the original scores are ignored.

    A chunk missing from a list gets nothing from that list. Ties go to the chunk with
    the better best rank, then to the smaller chunk id.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"got {len(weights)} weights for {len(rankings)} rankings")

    fused: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    chunks: dict[str, Chunk] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        seen: set[str] = set()
        for rank, hit in enumerate(ranking, start=1):
            cid = hit.chunk.chunk_id
            if cid in seen:  # count a duplicate only once per list
                continue
            seen.add(cid)
            fused[cid] = fused.get(cid, 0.0) + weight / (k + rank)
            best_rank[cid] = min(best_rank.get(cid, rank), rank)
            chunks[cid] = hit.chunk

    order = sorted(fused, key=lambda cid: (-fused[cid], best_rank[cid], cid))
    return [ScoredChunk(chunk=chunks[cid], score=fused[cid]) for cid in order]
