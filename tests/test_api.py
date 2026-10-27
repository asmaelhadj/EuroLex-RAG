"""API integration tests: real pipeline over a tiny in-memory index, mocked LLM."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from eurolex.api.main import Answerer, create_app
from eurolex.api.ratelimit import RateLimiter
from eurolex.config import ConfigError, IngestConfig, Settings, load_ingest_config
from eurolex.generation.answer import AnswerGenerator
from eurolex.ingestion.__main__ import write_jsonl
from eurolex.ingestion.index import ensure_collection, sync_collection
from eurolex.retrieval.pipeline import build_pipeline
from eurolex.ui.client import ask
from tests.fakes import FakeEmbedder
from tests.test_llm import FakeLLM
from tests.test_pipeline import CHUNKS, ROOT, config

KEY = "k" * 32
REPLY = "A processor needs a contract [1]. This is general information, not legal advice."


def settings(**extra: Any) -> Settings:
    return Settings(_env_file=None, api_key=KEY, **extra)  # type: ignore[call-arg]


def generator(
    ingest: IngestConfig, client: QdrantClient, llm: FakeLLM, threshold: float | None = None
) -> AnswerGenerator:
    pipeline = build_pipeline(config("dense"), ingest, client=client, embedder=FakeEmbedder())
    return AnswerGenerator(pipeline, llm, context_k=3, threshold=threshold)


def make_client(answerer: Answerer, **extra: Any) -> TestClient:
    return TestClient(create_app(settings(**extra), lambda s: answerer))


@pytest.fixture
def ingest(tmp_path: Path) -> IngestConfig:
    write_jsonl(tmp_path / "chunks_article.jsonl", CHUNKS)
    config = load_ingest_config(ROOT / "configs" / "ingest.yaml")
    return config.model_copy(update={"processed_dir": tmp_path})


@pytest.fixture
def client() -> QdrantClient:
    """The tiny test index: five synthetic chunks in an in-memory Qdrant."""
    qdrant = QdrantClient(":memory:")
    embedder = FakeEmbedder()
    ensure_collection(qdrant, "eurolex_article", embedder.dense_dim, recreate=False)
    sync_collection(
        qdrant, "eurolex_article", CHUNKS, embedder, {}, prepend_header=False, batch_size=10
    )
    return qdrant


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM(REPLY)


@pytest.fixture
def api(ingest: IngestConfig, client: QdrantClient, llm: FakeLLM) -> Iterator[TestClient]:
    with make_client(generator(ingest, client, llm)) as test_client:
        yield test_client


def post(api: TestClient, question: str, key: str | None = KEY, **body: Any) -> httpx.Response:
    headers = {"X-API-Key": key} if key is not None else {}
    return api.post("/ask", json={"question": question, **body}, headers=headers)


def test_health_needs_no_key(api: TestClient) -> None:
    response = api.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_returns_answer_with_citations(api: TestClient, llm: FakeLLM) -> None:
    response = post(api, "alpha bravo", filters={"act_id": "gdpr"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == REPLY
    assert body["refused"] is False
    assert body["citations"] == [
        {
            "n": 1,
            "article_id": "gdpr:art-1",
            "article_title": "t",
            "text": "alpha bravo charlie",
            "url": "https://example.test",
        }
    ]
    assert set(body["latency_ms"]) == {"retrieval", "rerank", "generation"}
    assert "alpha bravo" in llm.prompts[0]


@pytest.mark.parametrize("key", [None, "", "wrong-key-of-some-length", KEY[:-1]])
def test_missing_or_wrong_key_is_401(api: TestClient, llm: FakeLLM, key: str | None) -> None:
    response = post(api, "alpha", key=key)
    assert response.status_code == 401
    assert llm.prompts == []


def test_bad_key_is_rejected_before_body_validation(api: TestClient) -> None:
    assert api.post("/ask", json={}, headers={"X-API-Key": "nope"}).status_code == 401


@pytest.mark.parametrize("question", ["x" * 501, "", "   "])
def test_bad_question_is_422(api: TestClient, llm: FakeLLM, question: str) -> None:
    assert post(api, question).status_code == 422
    assert llm.prompts == []


def test_question_at_the_limit_is_accepted(api: TestClient) -> None:
    assert post(api, "alpha " + "x" * 494).status_code == 200


def test_bad_act_filter_is_422(api: TestClient) -> None:
    assert post(api, "alpha", filters={"act_id": "../etc"}).status_code == 422


def test_rate_limit_is_429(ingest: IngestConfig, client: QdrantClient, llm: FakeLLM) -> None:
    with make_client(generator(ingest, client, llm), rate_limit_per_minute=2) as api:
        assert post(api, "alpha").status_code == 200
        assert post(api, "alpha").status_code == 200
        response = post(api, "alpha")
        assert response.status_code == 429
        assert int(response.headers["Retry-After"]) >= 1
    assert len(llm.prompts) == 2


def test_low_evidence_refuses_without_llm(
    ingest: IngestConfig, client: QdrantClient, llm: FakeLLM
) -> None:
    with make_client(generator(ingest, client, llm, threshold=0.99)) as api:
        body = post(api, "zulu yankee").json()
    assert body["refused"] is True
    assert body["citations"] == []
    assert llm.prompts == []


def test_llm_down_is_503(ingest: IngestConfig, client: QdrantClient) -> None:
    with make_client(generator(ingest, client, FakeLLM(None))) as api:
        assert post(api, "alpha").status_code == 503


def test_openapi_docs(api: TestClient) -> None:
    schema = api.get("/openapi.json").json()
    assert "/ask" in schema["paths"]
    assert "not legal advice" in schema["info"]["description"]
    assert api.get("/docs").status_code == 200


def test_app_refuses_to_start_without_key() -> None:
    with pytest.raises(ConfigError):
        create_app(Settings(_env_file=None, api_key=None), lambda s: None)  # type: ignore[call-arg, arg-type, return-value]


def test_rate_limiter_window() -> None:
    now = [0.0]
    limiter = RateLimiter(2, window_seconds=60, clock=lambda: now[0])
    assert limiter.check("a") is None
    assert limiter.check("a") is None
    assert limiter.check("a") == 60
    assert limiter.check("b") is None  # counted per client
    now[0] = 60.0
    assert limiter.check("a") is None


def test_ui_client_against_the_api(api: TestClient) -> None:
    reply = ask("alpha bravo", api_url="http://testserver", api_key=KEY, client=api)
    assert reply.ok
    assert reply.answer == REPLY
    assert reply.citations[0]["article_id"] == "gdpr:art-1"

    bad = ask("alpha", api_url="http://testserver", api_key="wrong", client=api)
    assert not bad.ok
    assert "API key" in bad.error

    long = ask("x" * 600, api_url="http://testserver", api_key=KEY, client=api)
    assert not long.ok
    assert "500 characters" in long.error


def test_ui_client_when_api_is_down() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    http = httpx.Client(transport=httpx.MockTransport(refuse))
    reply = ask("q", api_url="http://nowhere.test", api_key=KEY, client=http)
    assert not reply.ok
    assert "Could not reach" in reply.error


def test_ui_client_maps_llm_errors() -> None:
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    reply = ask("q", api_url="http://x.test", api_key=KEY, client=http)
    assert "Ollama" in reply.error


def test_recital_citation_gets_an_id_and_title() -> None:
    from eurolex.api.main import to_response
    from eurolex.schemas import Answer, Chunk, Citation, RetrievalResult, ScoredChunk

    recital = Chunk(
        chunk_id="gdpr:rct-85",
        act_id="gdpr",
        celex="x",
        kind="recital",
        article_number=None,
        article_title=None,
        paragraph=None,
        chapter=None,
        recital_number="85",
        text="A breach may cause damage.",
        source_url="https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679",
        char_count=26,
    )
    answer = Answer(
        question="q",
        answer="a [1]",
        refused=False,
        citations=[
            Citation(
                n=1,
                chunk_id="gdpr:rct-85",
                article_id=None,
                article_title=None,
                text=recital.text,
                url=recital.source_url,
            )
        ],
        passages=[ScoredChunk(chunk=recital, score=0.5)],
        retrieval=RetrievalResult(query="q", hits=[], latency_ms={}),
        latency_ms={"retrieval": 1.4, "generation": 2.6},
    )
    out = to_response(answer)
    assert out.citations[0].article_id == "gdpr:rct-85"
    assert out.citations[0].article_title == "Recital 85"
    assert out.citations[0].url.endswith("CELEX:32016R0679#rct_85")
    assert out.latency_ms.model_dump() == {"retrieval": 1, "rerank": 0, "generation": 3}


def test_deep_link_to_article() -> None:
    from eurolex.api.main import deep_link

    act = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679"
    article = CHUNKS[0].model_copy(update={"article_number": "33", "source_url": act})
    assert deep_link(act, article) == act + "#art_33"
    assert deep_link(act + "#x", article) == act + "#x"  # already has an anchor
    assert deep_link("https://example.test", article) == "https://example.test"  # not EUR-Lex
    assert deep_link(act, None) == act
