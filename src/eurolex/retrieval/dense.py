"""Dense top-k search on the `dense` named vector."""

from qdrant_client import QdrantClient

from eurolex.embeddings import Embedder
from eurolex.ingestion.index import DENSE
from eurolex.retrieval.base import chunk_from_payload, payload_filter
from eurolex.schemas import ChunkKind, ScoredChunk


class DenseRetriever:
    name = "dense"

    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder) -> None:
        self._client = client
        self._collection = collection
        self._embedder = embedder

    def retrieve(
        self, query: str, k: int, act_id: str | None = None, kind: ChunkKind | None = None
    ) -> list[ScoredChunk]:
        (vec,) = self._embedder.embed([query])
        points = self._client.query_points(
            self._collection,
            query=vec.dense,
            using=DENSE,
            query_filter=payload_filter(act_id, kind),
            limit=k,
            with_payload=True,
        ).points
        return [ScoredChunk(chunk=chunk_from_payload(p.payload), score=p.score) for p in points]
