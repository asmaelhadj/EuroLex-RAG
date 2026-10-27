"""FastAPI app: POST /ask and GET /health.

Run with `make api`. The answerer is built from `settings.api_config`, the config evaluated
in docs/results.md. Tests pass their own answerer (tiny index, mocked LLM) to `create_app`.
"""

import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from eurolex.api.auth import check_key, key_from_header
from eurolex.api.ratelimit import RateLimiter
from eurolex.config import Settings, get_settings, load_experiment_config, load_ingest_config
from eurolex.generation.llm import LLMError
from eurolex.generation.prompts import DISCLAIMER
from eurolex.schemas import Answer, Chunk

VERSION = version("eurolex-rag")

DESCRIPTION = (
    f"Questions about EU law (GDPR), answered from the official text with citations. {DISCLAIMER}"
)


class Answerer(Protocol):
    def answer(self, question: str, act_id: str | None = None) -> Answer: ...


class Filters(BaseModel):
    act_id: str | None = Field(default=None, pattern=r"^[a-z_]+$")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    filters: Filters = Filters()


class CitationOut(BaseModel):
    n: int
    article_id: str
    article_title: str
    text: str
    url: str


class Latency(BaseModel):
    retrieval: int
    rerank: int
    generation: int


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[CitationOut]
    latency_ms: Latency


class Health(BaseModel):
    status: str
    version: str


def deep_link(url: str, chunk: Chunk | None) -> str:
    """Jump to the article or recital. EUR-Lex pages use ids like art_33 and rct_85."""
    if chunk is None or "#" in url or "eur-lex.europa.eu" not in url:
        return url
    if chunk.article_number:
        return f"{url}#art_{chunk.article_number}"
    if chunk.recital_number:
        return f"{url}#rct_{chunk.recital_number}"
    return url


def to_response(answer: Answer) -> AskResponse:
    chunks = {p.chunk.chunk_id: p.chunk for p in answer.passages}
    citations = []
    for c in answer.citations:
        chunk = chunks.get(c.chunk_id)
        recital = chunk.recital_number if chunk else None
        # recitals have no article id; use the chunk id (e.g. gdpr:rct-85) so the field is set
        citations.append(
            CitationOut(
                n=c.n,
                article_id=c.article_id or c.chunk_id,
                article_title=c.article_title or (f"Recital {recital}" if recital else ""),
                text=c.text,
                url=deep_link(c.url, chunk),
            )
        )
    lat = answer.latency_ms
    return AskResponse(
        answer=answer.answer,
        refused=answer.refused,
        citations=citations,
        latency_ms=Latency(
            retrieval=round(lat.get("retrieval", 0)),
            rerank=round(answer.retrieval.latency_ms.get("rerank", 0)),
            generation=round(lat.get("generation", 0)),
        ),
    )


def build_answerer(settings: Settings) -> Answerer:
    """Load the evaluated pipeline. Imported here so tests never load models."""
    from eurolex.generation.answer import AnswerGenerator
    from eurolex.generation.llm import OllamaClient
    from eurolex.retrieval.pipeline import build_pipeline

    experiment = load_experiment_config(settings.api_config)
    ingest = load_ingest_config(experiment.ingest_config)
    gen = experiment.generation
    pipeline = build_pipeline(experiment.retrieval, ingest)
    # no disk cache here: it would store users' questions (SPEC.md section 9)
    llm = OllamaClient(
        settings.ollama_base_url,
        gen.model or settings.llm_model,
        seed=settings.random_seed,
        num_gpu=gen.gpu_layers,
        num_ctx=gen.num_ctx,
        timeout_seconds=gen.timeout_seconds,
    )
    return AnswerGenerator(pipeline, llm, context_k=gen.context_k, threshold=gen.threshold)


def create_app(
    settings: Settings | None = None,
    answerer_factory: Callable[[Settings], Answerer] = build_answerer,
) -> FastAPI:
    settings = settings or get_settings()
    api_key = settings.require_api_key().get_secret_value()
    limiter = RateLimiter(settings.rate_limit_per_minute)
    # one question at a time: the models share one GPU and are not thread-safe
    lock = threading.Lock()
    state: dict[str, Answerer] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state["answerer"] = answerer_factory(settings)
        yield
        state.clear()

    app = FastAPI(title="EuroLex-RAG", version=VERSION, description=DESCRIPTION, lifespan=lifespan)

    def authorised(key: Annotated[str | None, Depends(key_from_header)]) -> None:
        check_key(key, api_key)

    @app.get("/health")
    def health() -> Health:
        return Health(status="ok" if "answerer" in state else "starting", version=VERSION)

    @app.post(
        "/ask",
        dependencies=[Depends(authorised)],
        responses={401: {}, 422: {}, 429: {}, 503: {}},
    )
    def ask(body: AskRequest, request: Request) -> AskResponse:
        question = body.question.strip()
        if not question:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Question is empty.")
        if len(question) > settings.max_question_chars:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"Question is longer than {settings.max_question_chars} characters.",
            )
        client = request.client.host if request.client else "unknown"
        wait = limiter.check(client)
        if wait is not None:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "Too many questions, try again shortly.",
                headers={"Retry-After": str(max(1, round(wait)))},
            )
        with lock:
            try:
                answer = state["answerer"].answer(question, act_id=body.filters.act_id)
            except LLMError as exc:
                raise HTTPException(
                    status.HTTP_503_SERVICE_UNAVAILABLE, "The language model is not available."
                ) from exc
        return to_response(answer)

    return app


def app() -> FastAPI:
    """For `uvicorn --factory eurolex.api.main:app`, so importing this module reads no env."""
    return create_app()
