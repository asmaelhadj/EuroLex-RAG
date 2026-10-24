"""Embed chunks and sync them into Qdrant collections with `dense` and `sparse` named vectors.

Indexing is a sync, not an append: point ids are UUIDv5 of the chunk id, unchanged chunks are
skipped (by a hash of the exact embedding input and the model id), changed chunks are
re-embedded, and points whose chunk no longer exists are deleted. Re-running `make ingest`
therefore leaves the collection identical, and changing the chunker cannot leave stale points
behind to pollute an experiment.
"""

import hashlib
import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from eurolex.embeddings import Embedder
from eurolex.schemas import Chunk

logger = logging.getLogger(__name__)

# Fixed namespace: changing it would change every point id.
POINT_NAMESPACE = uuid.UUID("0b6f7c3e-5a2d-5d4e-9f1a-3c8b2e7d6a10")
DENSE = "dense"
SPARSE = "sparse"
_HASH_FIELD = "embedding_input_sha256"
_MODEL_FIELD = "embedding_model"


class CollectionMismatchError(RuntimeError):
    """Raised when an existing collection is incompatible with the current embedder."""


@dataclass(frozen=True)
class IndexReport:
    collection: str
    points: int
    embedded: int
    unchanged: int
    deleted: int
    truncated: int


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def embedding_text(chunk: Chunk, act_names: Mapping[str, str], *, prepend_header: bool) -> str:
    """Text actually embedded for a chunk.

    The header gives the model the article number and title, which the chunk body often lacks
    (paragraph 3 of an article rarely restates what the article is about).
    """
    if not prepend_header:
        return chunk.text
    act = act_names.get(chunk.act_id, chunk.act_id)
    if chunk.kind == "recital":
        header = f"{act} Recital {chunk.recital_number}"
    else:
        locator = f"Article {chunk.article_number}"
        if chunk.paragraph:
            locator += f"({chunk.paragraph})"
        header = f"{act} {locator} - {chunk.article_title}"
    return f"{header}\n{chunk.text}"


def ensure_collection(client: QdrantClient, name: str, dense_dim: int, *, recreate: bool) -> None:
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if client.collection_exists(name):
        params = client.get_collection(name).config.params
        vectors = params.vectors if isinstance(params.vectors, dict) else {}
        dense = vectors.get(DENSE)
        if dense is None or dense.size != dense_dim or SPARSE not in (params.sparse_vectors or {}):
            raise CollectionMismatchError(
                f"Collection {name!r} does not match the embedder (dense size {dense_dim}, "
                "sparse vectors). Re-run with --recreate."
            )
        return
    client.create_collection(
        name,
        vectors_config={
            DENSE: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={SPARSE: models.SparseVectorParams()},
    )
    for field in ("act_id", "kind", "article_id"):
        client.create_payload_index(name, field, field_schema=models.PayloadSchemaType.KEYWORD)
    logger.info("Created collection %s", name)


def _existing_points(client: QdrantClient, name: str) -> dict[str, tuple[str, str]]:
    existing: dict[str, tuple[str, str]] = {}
    offset: models.ExtendedPointId | None = None
    while True:
        records, offset = client.scroll(
            name,
            limit=256,
            offset=offset,
            with_payload=[_HASH_FIELD, _MODEL_FIELD],
            with_vectors=False,
        )
        for record in records:
            payload = record.payload or {}
            existing[str(record.id)] = (
                str(payload.get(_HASH_FIELD, "")),
                str(payload.get(_MODEL_FIELD, "")),
            )
        if offset is None:
            return existing


def sync_collection(
    client: QdrantClient,
    name: str,
    chunks: Sequence[Chunk],
    embedder: Embedder,
    act_names: Mapping[str, str],
    *,
    prepend_header: bool,
    batch_size: int,
) -> IndexReport:
    existing = _existing_points(client, name)
    pending: list[tuple[str, Chunk, str, str]] = []
    wanted: set[str] = set()
    truncated = 0
    for chunk in chunks:
        pid = point_id(chunk.chunk_id)
        wanted.add(pid)
        text = embedding_text(chunk, act_names, prepend_header=prepend_header)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if existing.get(pid) == (digest, embedder.model_id):
            continue
        if embedder.count_tokens(text) > embedder.max_length:
            truncated += 1
            logger.warning(
                "%s exceeds %d tokens and is truncated", chunk.chunk_id, embedder.max_length
            )
        pending.append((pid, chunk, text, digest))

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = embedder.embed([text for _, _, text, _ in batch])
        client.upsert(
            name,
            points=[
                models.PointStruct(
                    id=pid,
                    vector={
                        DENSE: vec.dense,
                        SPARSE: models.SparseVector(
                            indices=vec.sparse.indices, values=vec.sparse.values
                        ),
                    },
                    payload={
                        **chunk.model_dump(),
                        "article_id": chunk.article_id,
                        _HASH_FIELD: digest,
                        _MODEL_FIELD: embedder.model_id,
                    },
                )
                for (pid, chunk, _, digest), vec in zip(batch, vectors, strict=True)
            ],
            wait=True,
        )
        logger.info("%s: embedded %d/%d", name, start + len(batch), len(pending))

    stale = sorted(set(existing) - wanted)
    if stale:
        client.delete(name, points_selector=models.PointIdsList(points=list(stale)), wait=True)
    report = IndexReport(
        collection=name,
        points=client.count(name, exact=True).count,
        embedded=len(pending),
        unchanged=len(chunks) - len(pending),
        deleted=len(stale),
        truncated=truncated,
    )
    logger.info("%s", report)
    return report


@dataclass(frozen=True)
class SearchHit:
    score: float
    chunk: Chunk


def search(
    client: QdrantClient, name: str, embedder: Embedder, query: str, *, using: str, limit: int
) -> list[SearchHit]:
    """Minimal single-vector search for smoke tests; the real retrievers arrive in Phase 3."""
    (vec,) = embedder.embed([query])
    q: list[float] | models.SparseVector = (
        vec.dense
        if using == DENSE
        else models.SparseVector(indices=vec.sparse.indices, values=vec.sparse.values)
    )
    points = client.query_points(name, query=q, using=using, limit=limit, with_payload=True).points
    fields = set(Chunk.model_fields)
    return [
        SearchHit(
            score=p.score,
            chunk=Chunk.model_validate({k: v for k, v in (p.payload or {}).items() if k in fields}),
        )
        for p in points
    ]
