"""Cross-encoder reranking with bge-reranker-v2-m3."""

import logging
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from huggingface_hub import snapshot_download

from eurolex.embeddings import resolve_device
from eurolex.schemas import Chunk, ScoredChunk

logger = logging.getLogger(__name__)

_RERANKER_FILES = [
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[ScoredChunk], top_n: int) -> list[ScoredChunk]: ...


class BGEReranker:
    def __init__(
        self,
        model: str,
        revision: str,
        *,
        max_length: int,
        batch_size: int,
        device: str = "auto",
        passage_text: Callable[[Chunk], str] = lambda c: c.text,
    ) -> None:
        from FlagEmbedding import FlagReranker

        path = snapshot_download(model, revision=revision, allow_patterns=_RERANKER_FILES)
        resolved = resolve_device(device)
        self._model: Any = FlagReranker(
            path, use_fp16=resolved != "cpu", devices=resolved, max_length=max_length
        )
        self._batch_size = batch_size
        self._max_length = max_length
        self._passage_text = passage_text
        logger.info("Loaded reranker %s on %s", model, resolved)

    def rerank(self, query: str, hits: Sequence[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not hits:
            return []
        pairs = [(query, self._passage_text(h.chunk)) for h in hits]
        scores = self._model.compute_score(
            pairs, batch_size=self._batch_size, max_length=self._max_length, normalize=True
        )
        if isinstance(scores, float):  # FlagEmbedding unwraps a single pair
            scores = [scores]
        return sort_by_score(
            [ScoredChunk(chunk=h.chunk, score=float(s)) for h, s in zip(hits, scores, strict=True)],
            top_n,
        )


def sort_by_score(hits: Sequence[ScoredChunk], top_n: int) -> list[ScoredChunk]:
    # stable tie-break on chunk id
    return sorted(hits, key=lambda h: (-h.score, h.chunk.chunk_id))[:top_n]
