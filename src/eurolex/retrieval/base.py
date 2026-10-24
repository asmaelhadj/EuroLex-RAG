"""Retriever interface and Qdrant helpers."""

from collections.abc import Mapping
from typing import Any, Protocol

from qdrant_client import models

from eurolex.schemas import Chunk, ScoredChunk

_CHUNK_FIELDS = set(Chunk.model_fields)


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int, act_id: str | None = None) -> list[ScoredChunk]: ...


def chunk_from_payload(payload: Mapping[str, Any] | None) -> Chunk:
    # payload also holds index bookkeeping fields, drop them
    return Chunk.model_validate({k: v for k, v in (payload or {}).items() if k in _CHUNK_FIELDS})


def act_filter(act_id: str | None) -> models.Filter | None:
    if act_id is None:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="act_id", match=models.MatchValue(value=act_id))]
    )
