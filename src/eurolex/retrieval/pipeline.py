"""Retrieval pipeline built from an experiment config: rewrite -> retrieve -> fuse -> rerank."""

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

from eurolex.config import (
    IngestConfig,
    RetrievalConfig,
    RetrieverSpec,
    get_settings,
)
from eurolex.embeddings import Embedder, Embedding
from eurolex.generation.llm import LLMClient, OllamaClient
from eurolex.ingestion.index import embedding_text
from eurolex.retrieval.base import Retriever
from eurolex.retrieval.dense import DenseRetriever
from eurolex.retrieval.fusion import reciprocal_rank_fusion
from eurolex.retrieval.rerank import BGEReranker, Reranker
from eurolex.retrieval.rewrite import QueryRewriter
from eurolex.retrieval.sparse import BM25Retriever, SparseRetriever
from eurolex.schemas import Chunk, RetrievalResult, ScoredChunk


class CachingEmbedder:
    """Remembers the last few queries so dense and sparse don't embed the same text twice."""

    def __init__(self, inner: Embedder, size: int = 32) -> None:
        self._inner = inner
        self._size = size
        self._cache: dict[str, Embedding] = {}
        self.model_id = inner.model_id
        self.dense_dim = inner.dense_dim
        self.max_length = inner.max_length

    def count_tokens(self, text: str) -> int:
        return self._inner.count_tokens(text)

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for text, vec in zip(missing, self._inner.embed(missing), strict=True):
                self._cache[text] = vec
            while len(self._cache) > self._size:
                self._cache.pop(next(iter(self._cache)))
        return [self._cache[t] for t in texts]


@dataclass(frozen=True)
class Stage:
    retriever: Retriever
    spec: RetrieverSpec


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class RetrievalPipeline:
    def __init__(
        self,
        stages: Sequence[Stage],
        *,
        top_k: int,
        fusion_k: int = 60,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        rerank_candidates: int = 50,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        if not stages:
            raise ValueError("need at least one retriever")
        self._stages = list(stages)
        self._top_k = top_k
        self._fusion_k = fusion_k
        self._embedder = embedder
        self._reranker = reranker
        self._rerank_candidates = rerank_candidates
        self._rewriter = rewriter

    def run(self, question: str, act_id: str | None = None) -> RetrievalResult:
        latency: dict[str, float] = {}
        query = question
        rewritten = None
        if self._rewriter is not None:
            start = time.perf_counter()
            rewritten = query = self._rewriter.rewrite(question)
            latency["rewrite"] = _ms(start)

        if self._embedder is not None:
            # embed once up front so retriever timings are search time only
            start = time.perf_counter()
            self._embedder.embed([query])
            latency["embed_query"] = _ms(start)

        rankings: list[list[ScoredChunk]] = []
        for stage in self._stages:
            start = time.perf_counter()
            rankings.append(stage.retriever.retrieve(query, stage.spec.top_k, act_id))
            latency[stage.retriever.name] = _ms(start)

        if len(rankings) == 1:
            candidates = rankings[0]
        else:
            start = time.perf_counter()
            candidates = reciprocal_rank_fusion(
                rankings, k=self._fusion_k, weights=[s.spec.weight for s in self._stages]
            )
            latency["fusion"] = _ms(start)

        if self._reranker is not None:
            start = time.perf_counter()
            # the reranker scores the original question, not the rewrite
            hits = self._reranker.rerank(
                question, candidates[: self._rerank_candidates], self._top_k
            )
            latency["rerank"] = _ms(start)
        else:
            hits = list(candidates[: self._top_k])

        latency["total"] = round(sum(latency.values()), 2)
        return RetrievalResult(
            query=question, rewritten_query=rewritten, hits=hits, latency_ms=latency
        )


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as fh:
        return [Chunk.model_validate(json.loads(line)) for line in fh if line.strip()]


def build_pipeline(
    config: RetrievalConfig,
    ingest: IngestConfig,
    *,
    client: QdrantClient | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    llm: LLMClient | None = None,
    qdrant_factory: Callable[[], QdrantClient] | None = None,
) -> RetrievalPipeline:
    """Wire up the stages named in the config. Real models are only loaded if needed."""
    stages: list[Stage] = []
    cached: CachingEmbedder | None = None
    bm25: BM25Retriever | None = None
    for spec in config.retrievers:
        retriever: Retriever
        if spec.type == "bm25":
            if bm25 is None:
                path = ingest.processed_dir / f"chunks_{config.chunker}.jsonl"
                bm25 = BM25Retriever(load_chunks(path))
            retriever = bm25
        else:
            if cached is None:
                if embedder is None:
                    from eurolex.embeddings import BGEM3Embedder

                    embedder = BGEM3Embedder(ingest.embedding, device=config.query_device)
                cached = CachingEmbedder(embedder)
            if client is None:
                client = (qdrant_factory or default_qdrant_client)()
            cls = DenseRetriever if spec.type == "dense" else SparseRetriever
            retriever = cls(client, config.collection, cached)
        stages.append(Stage(retriever, spec))

    if config.rerank.enabled and reranker is None:
        act_names = {act.act_id: act.short_name for act in ingest.acts}
        prepend = config.rerank.prepend_header
        reranker = BGEReranker(
            config.rerank.model,
            config.rerank.revision,
            max_length=config.rerank.max_length,
            batch_size=config.rerank.batch_size,
            device=config.rerank.device,
            fp16=config.rerank.precision == "fp16",
            passage_text=lambda c: embedding_text(c, act_names, prepend_header=prepend),
        )

    rewriter = None
    if config.rewrite.enabled:
        if llm is None:
            settings = get_settings()
            llm = OllamaClient(
                settings.ollama_base_url,
                config.rewrite.model or settings.llm_model,
                seed=settings.random_seed,
            )
        rewriter = QueryRewriter(llm, config.rewrite.mode)

    return RetrievalPipeline(
        stages,
        top_k=config.top_k,
        fusion_k=config.fusion.k,
        embedder=cached,
        reranker=reranker if config.rerank.enabled else None,
        rerank_candidates=config.rerank.candidates,
        rewriter=rewriter,
    )


def default_qdrant_client() -> QdrantClient:
    settings = get_settings()
    key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return QdrantClient(url=settings.qdrant_url, api_key=key)
