import uuid

import pytest
from qdrant_client import QdrantClient

from eurolex.embeddings import to_sparse
from eurolex.ingestion.index import (
    DENSE,
    SPARSE,
    CollectionMismatchError,
    embedding_text,
    ensure_collection,
    point_id,
    search,
    sync_collection,
)
from eurolex.schemas import Chunk
from tests.fakes import FakeEmbedder

URL = "https://example.test/act"
NAMES = {"gdpr": "GDPR"}


def article_chunk(cid: str, text: str, number: str = "6", paragraph: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=cid,
        act_id="gdpr",
        celex="32016R0679",
        kind="article",
        article_number=number,
        article_title="Lawfulness of processing",
        paragraph=paragraph,
        chapter="Chapter II",
        text=text,
        source_url=URL,
        char_count=len(text),
    )


def recital_chunk(number: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"gdpr:rct-{number}",
        act_id="gdpr",
        celex="32016R0679",
        kind="recital",
        article_number=None,
        article_title=None,
        paragraph=None,
        chapter=None,
        recital_number=number,
        text=text,
        source_url=URL,
        char_count=len(text),
    )


CHUNKS = [
    article_chunk("gdpr:art-6:p-1", "processing shall be lawful only if consent", paragraph="1"),
    article_chunk("gdpr:art-4", "personal data means any information", number="4"),
    recital_chunk("26", "principles of data protection apply to identifiable persons"),
]


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


def sync(client: QdrantClient, chunks: list[Chunk], embedder: FakeEmbedder):  # type: ignore[no-untyped-def]
    ensure_collection(client, "c", embedder.dense_dim, recreate=False)
    return sync_collection(client, "c", chunks, embedder, NAMES, prepend_header=True, batch_size=2)


def test_point_id_is_deterministic_uuid5() -> None:
    assert point_id("gdpr:art-6") == point_id("gdpr:art-6")
    assert point_id("gdpr:art-6") != point_id("gdpr:art-7")
    assert uuid.UUID(point_id("gdpr:art-6")).version == 5


def test_embedding_text_headers() -> None:
    para, art, rct = CHUNKS
    assert embedding_text(para, NAMES, prepend_header=True).startswith(
        "GDPR Article 6(1) - Lawfulness of processing\nprocessing shall"
    )
    assert embedding_text(art, NAMES, prepend_header=True).startswith("GDPR Article 4 - ")
    assert embedding_text(rct, NAMES, prepend_header=True).startswith("GDPR Recital 26\n")
    assert embedding_text(para, NAMES, prepend_header=False) == para.text


def test_reindexing_is_idempotent(client: QdrantClient) -> None:
    embedder = FakeEmbedder()
    first = sync(client, CHUNKS, embedder)
    assert (first.points, first.embedded, first.unchanged, first.deleted) == (3, 3, 0, 0)
    second = sync(client, CHUNKS, embedder)
    assert (second.points, second.embedded, second.unchanged, second.deleted) == (3, 0, 3, 0)
    assert len(embedder.embedded_texts) == 3


def test_changed_and_removed_chunks_are_synced(client: QdrantClient) -> None:
    embedder = FakeEmbedder()
    sync(client, CHUNKS, embedder)
    changed = [article_chunk("gdpr:art-4", "personal data means any information relating", "4")]
    report = sync(client, [CHUNKS[0], *changed], embedder)
    assert (report.points, report.embedded, report.unchanged, report.deleted) == (2, 1, 1, 1)


def test_new_model_reembeds_everything(client: QdrantClient) -> None:
    sync(client, CHUNKS, FakeEmbedder())
    report = sync(client, CHUNKS, FakeEmbedder(model_id="fake@111111111111"))
    assert (report.points, report.embedded) == (3, 3)


def test_truncation_is_counted(client: QdrantClient) -> None:
    long = article_chunk("gdpr:art-9", " ".join(["word"] * 100), number="9")
    report = sync(client, [long], FakeEmbedder(max_length=50))
    assert report.truncated == 1


def test_payload_and_search(client: QdrantClient) -> None:
    embedder = FakeEmbedder()
    sync(client, CHUNKS, embedder)
    for using in (DENSE, SPARSE):
        hits = search(
            client, "c", embedder, "personal data means any information", using=using, limit=3
        )
        assert hits[0].chunk.chunk_id == "gdpr:art-4"
        assert hits[0].chunk == CHUNKS[1]
    record = client.retrieve("c", [point_id("gdpr:art-4")], with_payload=True)[0]
    assert record.payload is not None
    assert record.payload["article_id"] == "gdpr:art-4"
    assert record.payload["embedding_model"] == embedder.model_id


def test_incompatible_collection_needs_recreate(client: QdrantClient) -> None:
    ensure_collection(client, "c", 16, recreate=False)
    with pytest.raises(CollectionMismatchError, match="--recreate"):
        ensure_collection(client, "c", 32, recreate=False)
    ensure_collection(client, "c", 32, recreate=True)
    vectors = client.get_collection("c").config.params.vectors
    assert isinstance(vectors, dict)
    assert vectors[DENSE].size == 32


def test_flagembedding_lexical_weights_become_sorted_sparse_vector() -> None:
    sparse = to_sparse({"17": 0.7, "5": 0.4, "1203": 0.05})
    assert sparse.indices == [5, 17, 1203]
    assert sparse.values == [0.4, 0.7, 0.05]
