"""Retriever interface and Qdrant helpers."""

from collections.abc import Mapping
from typing import Any, Protocol

from qdrant_client import models

from eurolex.schemas import Chunk, ChunkKind, ScoredChunk

_CHUNK_FIELDS = set(Chunk.model_fields)


class Retriever(Protocol):
    name: str

    def retrieve(
        self, query: str, k: int, act_id: str | None = None, kind: ChunkKind | None = None
    ) -> list[ScoredChunk]: ...


def chunk_from_payload(payload: Mapping[str, Any] | None) -> Chunk:
    # payload also holds index bookkeeping fields, drop them
    return Chunk.model_validate({k: v for k, v in (payload or {}).items() if k in _CHUNK_FIELDS})


def payload_filter(act_id: str | None, kind: ChunkKind | None = None) -> models.Filter | None:
    conditions: list[models.Condition] = [
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in (("act_id", act_id), ("kind", kind))
        if value is not None
    ]
    return models.Filter(must=conditions) if conditions else None
