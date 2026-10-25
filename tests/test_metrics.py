"""Hand-computed toy examples for the retrieval metrics."""

import math

import pytest
from pydantic import ValidationError

from eurolex.evaluation.metrics import (
    first_hits,
    mean,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from eurolex.schemas import Question

# ranked chunks mapped to article ids; None = recital
RETRIEVED = ["gdpr:art-5", None, "gdpr:art-6", "gdpr:art-6", "gdpr:art-9", "gdpr:art-17"]
GOLD = ["gdpr:art-6", "gdpr:art-17"]


def test_recall_at_k() -> None:
    assert recall_at_k(RETRIEVED, GOLD, 1) == 0.0
    assert recall_at_k(RETRIEVED, GOLD, 3) == 0.5  # art-6 at rank 3
    assert recall_at_k(RETRIEVED, GOLD, 5) == 0.5
    assert recall_at_k(RETRIEVED, GOLD, 6) == 1.0  # art-17 at rank 6
    assert recall_at_k(RETRIEVED, GOLD, 100) == 1.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(RETRIEVED, GOLD) == pytest.approx(1 / 3)
    assert reciprocal_rank(RETRIEVED, ["gdpr:art-99"]) == 0.0
    assert reciprocal_rank([], GOLD) == 0.0


def test_ndcg_hand_computed() -> None:
    # relevant at ranks 3 and 6 (the repeat of art-6 at rank 4 does not count)
    dcg = 1 / math.log2(4) + 1 / math.log2(7)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RETRIEVED, GOLD, 10) == pytest.approx(dcg / ideal)
    # cut at 5: only rank 3 counts, ideal still has two slots
    assert ndcg_at_k(RETRIEVED, GOLD, 5) == pytest.approx((1 / math.log2(4)) / ideal)


def test_ndcg_perfect_and_zero() -> None:
    assert ndcg_at_k(["gdpr:art-6", "gdpr:art-17"], GOLD) == pytest.approx(1.0)
    assert ndcg_at_k(["gdpr:art-17", "gdpr:art-6", "gdpr:art-1"], GOLD) == pytest.approx(1.0)
    assert ndcg_at_k([None, "gdpr:art-1"], GOLD) == 0.0


def test_ndcg_ideal_is_capped_at_k() -> None:
    gold = [f"gdpr:art-{i}" for i in range(1, 13)]
    assert ndcg_at_k(gold[:10], gold, 10) == pytest.approx(1.0)


def test_first_hits_drops_repeats() -> None:
    assert first_hits(["a", None, "a", "b", "b"]) == ["a", None, None, "b", None]


def test_metrics_need_gold() -> None:
    with pytest.raises(ValueError, match="gold"):
        recall_at_k(RETRIEVED, [], 5)
    with pytest.raises(ValueError, match="gold"):
        ndcg_at_k(RETRIEVED, [])


def test_mean_and_percentile() -> None:
    assert mean([]) is None
    assert mean([1.0, 2.0, 6.0]) == 3.0
    assert percentile([], 50) is None
    assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], 50) == 3.0
    assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], 95) == 5.0
    assert percentile([7.0], 0) == 7.0


def q(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "q-0001",
        "question": "Can a company process personal data without consent?",
        "gold_article_ids": ["gdpr:art-6"],
        "type": "factual",
        "source": "human",
        "status": "validated",
        "split": "dev",
        "notes": "",
    }
    return {**row, **overrides}


def test_question_contract() -> None:
    assert Question.model_validate(q()).in_scope
    oos = Question.model_validate(q(type="out_of_scope", gold_article_ids=[]))
    assert not oos.in_scope
    for bad in (
        q(gold_article_ids=[]),
        q(type="out_of_scope"),
        q(gold_article_ids=["art-6"]),
        q(gold_article_ids=["gdpr:art-6", "gdpr:art-6"]),
        q(id="question-1"),
        q(status="approved"),
        q(extra_field=1),
    ):
        with pytest.raises(ValidationError):
            Question.model_validate(bad)
