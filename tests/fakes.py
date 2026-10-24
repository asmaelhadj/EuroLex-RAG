"""Deterministic stand-ins so tests never download models or need a running Qdrant."""

import hashlib
import math
from collections.abc import Sequence

from eurolex.embeddings import Embedding, SparseVector


def _hash(word: str) -> int:
    return int.from_bytes(hashlib.sha256(word.lower().encode()).digest()[:4], "big")


class FakeEmbedder:
    """Bag-of-words embedder: texts sharing words get similar dense and sparse vectors."""

    def __init__(self, dim: int = 256, model_id: str = "fake@000000000000", max_length: int = 64):
        self.dense_dim = dim
        self.model_id = model_id
        self.max_length = max_length
        self.calls: list[list[str]] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        self.calls.append(list(texts))
        out = []
        for text in texts:
            dense = [0.0] * self.dense_dim
            sparse: dict[int, float] = {}
            for word in text.split():
                h = _hash(word)
                dense[h % self.dense_dim] += 1.0
                sparse[h % 50_000] = 1.0
            norm = math.sqrt(sum(x * x for x in dense)) or 1.0
            indices = sorted(sparse)
            out.append(
                Embedding(
                    dense=[x / norm for x in dense],
                    sparse=SparseVector(indices=indices, values=[sparse[i] for i in indices]),
                )
            )
        return out

    @property
    def embedded_texts(self) -> list[str]:
        return [t for call in self.calls for t in call]
