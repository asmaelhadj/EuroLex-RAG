"""Lexical retrievers: BGE-M3 sparse vectors in Qdrant, and plain BM25 for experiment E2."""

import re
from collections.abc import Sequence

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from eurolex.embeddings import Embedder
from eurolex.ingestion.index import SPARSE
from eurolex.retrieval.base import act_filter, chunk_from_payload
from eurolex.schemas import Chunk, ScoredChunk

_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class SparseRetriever:
    name = "sparse"

    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder) -> None:
        self._client = client
        self._collection = collection
        self._embedder = embedder

    def retrieve(self, query: str, k: int, act_id: str | None = None) -> list[ScoredChunk]:
        (vec,) = self._embedder.embed([query])
        if not vec.sparse.indices:
            return []
        points = self._client.query_points(
            self._collection,
            query=models.SparseVector(indices=vec.sparse.indices, values=vec.sparse.values),
            using=SPARSE,
            query_filter=act_filter(act_id),
            limit=k,
            with_payload=True,
        ).points
        return [ScoredChunk(chunk=chunk_from_payload(p.payload), score=p.score) for p in points]


class BM25Retriever:
    """In-memory BM25 over the chunk file, independent of Qdrant."""

    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self._chunks = list(chunks)
        self._bm25 = BM25Okapi([tokenize(c.text) for c in self._chunks], k1=k1, b=b)

    def retrieve(self, query: str, k: int, act_id: str | None = None) -> list[ScoredChunk]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            (i for i, c in enumerate(self._chunks) if act_id is None or c.act_id == act_id),
            # ties broken by chunk id so runs are repeatable
            key=lambda i: (-scores[i], self._chunks[i].chunk_id),
        )
        return [
            ScoredChunk(chunk=self._chunks[i], score=float(scores[i]))
            for i in ranked[:k]
            if scores[i] > 0
        ]
