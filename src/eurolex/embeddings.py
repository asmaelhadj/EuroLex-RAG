"""Text embedders shared by indexing (Phase 2) and query-time retrieval (Phase 3).

BGE-M3 runs through FlagEmbedding, its reference implementation, so the dense and sparse
vectors are exactly what the model authors intended (ADR 005). We download only the files
inference needs at a pinned revision and hand FlagEmbedding the local folder; given a repo name
it would also fetch a 2.3 GB ONNX export we never use.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from huggingface_hub import snapshot_download

from eurolex.config import EmbeddingConfig

logger = logging.getLogger(__name__)

# Only the files inference needs; the repo also holds an ONNX export and images.
_BGE_M3_FILES = [
    "config.json",
    "pytorch_model.bin",
    "sparse_linear.pt",
    "colbert_linear.pt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
]


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class Embedding:
    dense: list[float]
    sparse: SparseVector


class Embedder(Protocol):
    model_id: str
    dense_dim: int
    max_length: int

    def embed(self, texts: Sequence[str]) -> list[Embedding]: ...

    def count_tokens(self, text: str) -> int: ...


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return "cuda:0" if device == "cuda" else device


def to_sparse(lexical_weights: dict[str, float]) -> SparseVector:
    """FlagEmbedding returns `{token_id_as_str: weight}`; Qdrant wants sorted int indices."""
    pairs = sorted((int(token_id), float(w)) for token_id, w in lexical_weights.items())
    return SparseVector(indices=[i for i, _ in pairs], values=[w for _, w in pairs])


class BGEM3Embedder:
    def __init__(self, config: EmbeddingConfig) -> None:
        # Imported here: FlagEmbedding pulls in heavy modules that tests and CLI help don't need.
        from FlagEmbedding import BGEM3FlagModel

        path = snapshot_download(
            config.model, revision=config.revision, allow_patterns=_BGE_M3_FILES
        )
        device = resolve_device(config.device)
        self.model_id = config.model_id
        self.max_length = config.max_length
        self._batch_size = config.batch_size
        # fp16 only on GPU; on CPU FlagEmbedding runs fp32 regardless.
        self._model: Any = BGEM3FlagModel(path, use_fp16=device != "cpu", devices=device)
        self.dense_dim = int(self._model.model.model.config.hidden_size)
        logger.info("Loaded %s on %s", self.model_id, device)

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer(text, add_special_tokens=True)["input_ids"])

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        if not texts:
            return []
        out = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return [
            Embedding(dense=[float(x) for x in dense], sparse=to_sparse(weights))
            for dense, weights in zip(out["dense_vecs"], out["lexical_weights"], strict=True)
        ]
