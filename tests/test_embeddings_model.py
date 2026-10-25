"""Opt-in checks against the real BGE-M3 weights (2.3 GB download, CPU inference).

Run with `RUN_MODEL_TESTS=1 uv run pytest tests/test_embeddings_model.py`. Skipped in CI.
"""

import math
import os
from pathlib import Path

import pytest

from eurolex.config import load_ingest_config

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 to load BGE-M3"
)

CONFIG = Path(__file__).parents[1] / "configs" / "ingest.yaml"


@pytest.fixture(scope="module")
def embedder():  # type: ignore[no-untyped-def]
    from eurolex.embeddings import BGEM3Embedder

    return BGEM3Embedder(load_ingest_config(CONFIG).embedding)


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_output_shapes(embedder) -> None:  # type: ignore[no-untyped-def]
    (vec,) = embedder.embed(["The controller shall notify the supervisory authority."])
    assert embedder.dense_dim == len(vec.dense) == 1024
    # 1e-3 because fp16 on GPU only keeps ~3 significant digits
    assert math.isclose(math.sqrt(sum(x * x for x in vec.dense)), 1.0, rel_tol=1e-3)
    assert vec.sparse.indices == sorted(set(vec.sparse.indices))
    assert all(v > 0 for v in vec.sparse.values)


def test_batching_does_not_change_vectors(embedder) -> None:  # type: ignore[no-untyped-def]
    texts = ["short text", "a much longer text " * 20, "right to erasure"]
    together = embedder.embed(texts)
    alone = [embedder.embed([t])[0] for t in texts]
    for a, b in zip(together, alone, strict=True):
        assert _cos(a.dense, b.dense) > 0.9999


def test_semantic_ordering(embedder) -> None:  # type: ignore[no-untyped-def]
    query, related, unrelated = embedder.embed(
        [
            "When can personal data be deleted?",
            "The data subject shall have the right to obtain the erasure of personal data.",
            "Member States shall lay down rules on penalties for infringements.",
        ]
    )
    assert _cos(query.dense, related.dense) > _cos(query.dense, unrelated.dense)
