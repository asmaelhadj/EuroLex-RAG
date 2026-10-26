from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient

from eurolex.config import (
    IngestConfig,
    RerankConfig,
    RetrievalConfig,
    RetrieverSpec,
    RewriteConfig,
    load_experiment_config,
    load_ingest_config,
)
from eurolex.ingestion.__main__ import write_jsonl
from eurolex.ingestion.index import ensure_collection, sync_collection
from eurolex.retrieval.pipeline import CachingEmbedder, build_pipeline
from eurolex.retrieval.rerank import BGEReranker
from eurolex.retrieval.sparse import BM25Retriever, tokenize
from eurolex.schemas import Chunk, ScoredChunk
from tests.fakes import FakeEmbedder
from tests.test_llm import FakeLLM

ROOT = Path(__file__).parents[1]


def chunk(cid: str, text: str, act_id: str = "gdpr") -> Chunk:
    return Chunk(
        chunk_id=cid,
        act_id=act_id,
        celex="x",
        kind="article",
        article_number=cid.rsplit("-", 1)[-1],
        article_title="t",
        paragraph=None,
        chapter=None,
        text=text,
        source_url="https://example.test",
        char_count=len(text),
    )


# synthetic text, only the word overlap matters
CHUNKS = [
    chunk("gdpr:art-1", "alpha bravo charlie"),
    chunk("gdpr:art-2", "delta echo foxtrot"),
    chunk("gdpr:art-3", "alpha delta golf"),
    chunk("gdpr:art-4", "hotel india juliet"),
    chunk("other:art-5", "alpha alpha alpha", act_id="other"),
]


class ReverseReranker:
    """Puts candidates in reverse order so we can see that reranking happened."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, int]] = []

    def rerank(self, query: str, hits: Sequence[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        self.seen.append((query, len(hits)))
        return [ScoredChunk(chunk=h.chunk, score=float(i)) for i, h in enumerate(hits)][::-1][
            :top_n
        ]


@pytest.fixture
def ingest(tmp_path: Path) -> IngestConfig:
    config = load_ingest_config(ROOT / "configs" / "ingest.yaml")
    write_jsonl(tmp_path / "chunks_article.jsonl", CHUNKS)
    return config.model_copy(update={"processed_dir": tmp_path})


@pytest.fixture
def client() -> QdrantClient:
    qdrant = QdrantClient(":memory:")
    embedder = FakeEmbedder()
    ensure_collection(qdrant, "eurolex_article", embedder.dense_dim, recreate=False)
    sync_collection(
        qdrant, "eurolex_article", CHUNKS, embedder, {}, prepend_header=False, batch_size=10
    )
    return qdrant


def config(*types: str, **extra: Any) -> RetrievalConfig:
    return RetrievalConfig(
        collection="eurolex_article",
        chunker="article",
        retrievers=[RetrieverSpec(type=t) for t in types],  # type: ignore[arg-type]
        top_k=3,
        **extra,
    )


def ids(hits: Sequence[ScoredChunk]) -> list[str]:
    return [h.chunk.chunk_id for h in hits]


def test_tokenize() -> None:
    assert tokenize("Article 6(1)(a) of the GDPR!") == [
        "article",
        "6",
        "1",
        "a",
        "of",
        "the",
        "gdpr",
    ]


def test_bm25_ranking_filter_and_empty_query() -> None:
    bm25 = BM25Retriever(CHUNKS)
    assert ids(bm25.retrieve("delta foxtrot", 2)) == ["gdpr:art-2", "gdpr:art-3"]
    assert "other:art-5" not in ids(bm25.retrieve("alpha", 10, act_id="gdpr"))
    assert bm25.retrieve("!!!", 5) == []
    assert bm25.retrieve("zulu", 5) == []  # no matching term


def test_caching_embedder_embeds_each_text_once() -> None:
    inner = FakeEmbedder()
    cached = CachingEmbedder(inner, size=2)
    cached.embed(["a"])
    cached.embed(["a", "b"])
    cached.embed(["c"])  # evicts "a"
    cached.embed(["a"])
    assert inner.calls == [["a"], ["b"], ["c"], ["a"]]


def test_dense_only_has_no_fusion_stage(ingest: IngestConfig, client: QdrantClient) -> None:
    pipeline = build_pipeline(config("dense"), ingest, client=client, embedder=FakeEmbedder())
    result = pipeline.run("alpha bravo charlie")
    assert ids(result.hits)[0] == "gdpr:art-1"
    assert len(result.hits) == 3
    assert set(result.latency_ms) == {"embed_query", "dense", "total"}


def test_hybrid_fuses_and_embeds_query_once(ingest: IngestConfig, client: QdrantClient) -> None:
    embedder = FakeEmbedder()
    pipeline = build_pipeline(config("dense", "sparse"), ingest, client=client, embedder=embedder)
    result = pipeline.run("delta echo foxtrot")
    assert ids(result.hits)[0] == "gdpr:art-2"
    assert set(result.latency_ms) == {"embed_query", "dense", "sparse", "fusion", "total"}
    assert embedder.calls == [["delta echo foxtrot"]]
    assert result.latency_ms["total"] == pytest.approx(
        sum(v for k, v in result.latency_ms.items() if k != "total"), abs=0.05
    )


def test_bm25_only_needs_no_qdrant_or_embedder(ingest: IngestConfig) -> None:
    def no_qdrant() -> QdrantClient:
        raise AssertionError("BM25 must not connect to Qdrant")

    pipeline = build_pipeline(config("bm25"), ingest, qdrant_factory=no_qdrant)
    result = pipeline.run("hotel")
    assert ids(result.hits) == ["gdpr:art-4"]
    assert set(result.latency_ms) == {"bm25", "total"}


def test_act_filter_applies_to_every_retriever(ingest: IngestConfig, client: QdrantClient) -> None:
    pipeline = build_pipeline(
        config("dense", "sparse", "bm25"), ingest, client=client, embedder=FakeEmbedder()
    )
    assert all(h.chunk.act_id == "other" for h in pipeline.run("alpha", act_id="other").hits)


def test_rerank_stage(ingest: IngestConfig, client: QdrantClient) -> None:
    reranker = ReverseReranker()
    cfg = config("dense", "sparse", rerank=RerankConfig(enabled=True, candidates=4))
    pipeline = build_pipeline(
        cfg, ingest, client=client, embedder=FakeEmbedder(), reranker=reranker
    )
    plain = build_pipeline(
        config("dense", "sparse"), ingest, client=client, embedder=FakeEmbedder()
    )
    result = pipeline.run("alpha")
    assert reranker.seen == [("alpha", 4)]
    assert "rerank" in result.latency_ms
    assert len(result.hits) == 3
    assert ids(result.hits) != ids(plain.run("alpha").hits)


def test_rerank_disabled_ignores_given_reranker(ingest: IngestConfig, client: QdrantClient) -> None:
    reranker = ReverseReranker()
    pipeline = build_pipeline(
        config("dense"), ingest, client=client, embedder=FakeEmbedder(), reranker=reranker
    )
    assert "rerank" not in pipeline.run("alpha").latency_ms
    assert reranker.seen == []


def test_rewrite_stage(ingest: IngestConfig, client: QdrantClient) -> None:
    llm = FakeLLM("hotel india")
    reranker = ReverseReranker()
    cfg = config(
        "bm25",
        rewrite=RewriteConfig(enabled=True, mode="replace"),
        rerank=RerankConfig(enabled=True),
    )
    pipeline = build_pipeline(cfg, ingest, llm=llm, reranker=reranker)
    result = pipeline.run("where is the hotel?")
    assert result.rewritten_query == "hotel india"
    assert ids(result.hits) == ["gdpr:art-4"]
    assert "rewrite" in result.latency_ms
    # the reranker judges against the user's question, not the rewrite
    assert reranker.seen[0][0] == "where is the hotel?"


def test_repo_experiment_configs_load() -> None:
    names = sorted(p.stem for p in (ROOT / "configs").glob("e*.yaml"))
    assert names == [
        "e0_naive",
        "e1_article_dense",
        "e1s_articles_first",
        "e2_bm25",
        "e3_hybrid",
        "e4_rerank",
        "e4s_articles_first",
        "e5_rewrite",
        "e7_generation",
        "e7_threshold",
    ]
    for name in names:
        experiment = load_experiment_config(ROOT / "configs" / f"{name}.yaml")
        assert experiment.name == name


class FakeFlagReranker:
    def __init__(self, scores: list[float] | float) -> None:
        self.scores = scores

    def compute_score(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float] | float:
        assert kwargs["normalize"] is True
        return self.scores


def bge_reranker(scores: list[float] | float) -> BGEReranker:
    reranker = object.__new__(BGEReranker)
    reranker._model = FakeFlagReranker(scores)
    reranker._batch_size = 4
    reranker._max_length = 512
    reranker._passage_text = lambda c: c.text
    return reranker


def test_bge_reranker_sorts_by_model_score() -> None:
    hits = [ScoredChunk(chunk=c, score=0.0) for c in CHUNKS[:3]]
    reranked = bge_reranker([0.1, 0.9, 0.5]).rerank("q", hits, top_n=2)
    assert ids(reranked) == ["gdpr:art-2", "gdpr:art-3"]
    assert [h.score for h in reranked] == [0.9, 0.5]
    single = bge_reranker(0.7).rerank("q", hits[:1], top_n=5)
    assert [h.score for h in single] == [0.7]
    assert bge_reranker([]).rerank("q", [], top_n=5) == []


def test_failed_rewrite_is_flagged_and_original_used(ingest: IngestConfig) -> None:
    cfg = config("bm25", rewrite=RewriteConfig(enabled=True, mode="replace"))
    result = build_pipeline(cfg, ingest, llm=FakeLLM(None)).run("hotel")
    assert result.rewrite_failed is True
    assert result.rewritten_query is None
    assert ids(result.hits) == ["gdpr:art-4"]  # searched with the original question


def test_runner_marks_failed_rewrites_not_reportable(ingest: IngestConfig) -> None:
    from eurolex.config import ExperimentConfig
    from eurolex.evaluation.runner import run_experiment
    from eurolex.schemas import Question

    cfg = config("bm25", rewrite=RewriteConfig(enabled=True))
    pipeline = build_pipeline(cfg, ingest, llm=FakeLLM(None))
    q = Question(
        id="q-0001",
        question="hotel?",
        gold_article_ids=["gdpr:art-4"],
        type="factual",
        source="human",
        status="validated",
    )
    experiment = ExperimentConfig(name="e5_rewrite", description="d", retrieval=cfg)
    run = run_experiment(experiment, ingest, [q], "dev", pipeline)
    assert run["reportable"] is False
    assert "query rewriting failed for 1 questions" in run["not_reportable_because"]


def test_missing_rewrite_model_fails_fast(
    ingest: IngestConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eurolex.config import ConfigError
    from eurolex.generation.llm import OllamaClient

    monkeypatch.setattr(OllamaClient, "has_model", lambda self: False)
    cfg = config("bm25", rewrite=RewriteConfig(enabled=True, model="not-pulled"))
    with pytest.raises(ConfigError, match="not-pulled is not installed"):
        build_pipeline(cfg, ingest)


def recital(number: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"gdpr:rct-{number}",
        act_id="gdpr",
        celex="x",
        kind="recital",
        article_number=None,
        article_title=None,
        paragraph=None,
        chapter=None,
        recital_number=number,
        text=text,
        source_url="https://example.test",
        char_count=len(text),
    )


@pytest.fixture
def with_recitals(tmp_path: Path) -> tuple[IngestConfig, QdrantClient]:
    chunks = [*CHUNKS[:4], recital("1", "alpha bravo charlie delta"), recital("2", "echo foxtrot")]
    ingest = load_ingest_config(ROOT / "configs" / "ingest.yaml").model_copy(
        update={"processed_dir": tmp_path}
    )
    write_jsonl(tmp_path / "chunks_article.jsonl", chunks)
    qdrant = QdrantClient(":memory:")
    embedder = FakeEmbedder()
    ensure_collection(qdrant, "eurolex_article", embedder.dense_dim, recreate=False)
    sync_collection(
        qdrant, "eurolex_article", chunks, embedder, {}, prepend_header=False, batch_size=10
    )
    return ingest, qdrant


@pytest.mark.parametrize("types", [("dense",), ("dense", "sparse"), ("bm25",)])
def test_recital_modes(
    with_recitals: tuple[IngestConfig, QdrantClient], types: tuple[str, ...]
) -> None:
    ingest, qdrant = with_recitals
    query = "alpha bravo charlie delta"

    def run(**extra: Any) -> Any:
        cfg = config(*types, **extra)
        return build_pipeline(cfg, ingest, client=qdrant, embedder=FakeEmbedder()).run(query)

    included = run()
    assert "gdpr:rct-1" in ids(included.hits)  # recitals compete with articles
    assert included.support == []

    supported = run(recitals="support", support_k=2)
    assert all(h.chunk.kind == "article" for h in supported.hits)
    assert ids(supported.hits)[0] == "gdpr:art-1"
    assert supported.support
    assert all(h.chunk.kind == "recital" for h in supported.support)
    assert "gdpr:rct-1" in ids(supported.support)
    assert "support" in supported.latency_ms

    excluded = run(recitals="exclude")
    assert all(h.chunk.kind == "article" for h in excluded.hits)
    assert excluded.support == []
