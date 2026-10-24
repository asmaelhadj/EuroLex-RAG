import pytest

from eurolex.retrieval.fusion import reciprocal_rank_fusion
from eurolex.schemas import Chunk, ScoredChunk


def hit(cid: str, score: float = 1.0) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=cid,
        act_id="gdpr",
        celex="32016R0679",
        kind="article",
        article_number="1",
        article_title="t",
        paragraph=None,
        chapter=None,
        text=cid,
        source_url="https://example.test",
        char_count=len(cid),
    )
    return ScoredChunk(chunk=chunk, score=score)


def ranking(*ids: str) -> list[ScoredChunk]:
    return [hit(i) for i in ids]


def ids(hits: list[ScoredChunk]) -> list[str]:
    return [h.chunk.chunk_id for h in hits]


def test_hand_computed_example() -> None:
    fused = reciprocal_rank_fusion([ranking("a", "b", "c"), ranking("b", "c", "d")], k=60)
    assert ids(fused) == ["b", "c", "a", "d"]
    scores = {h.chunk.chunk_id: h.score for h in fused}
    assert scores["b"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 62 + 1 / 63)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["d"] == pytest.approx(1 / 63)


def test_document_missing_from_a_list_gets_nothing_from_it() -> None:
    fused = reciprocal_rank_fusion([ranking("a", "b"), ranking("a")], k=10)
    scores = {h.chunk.chunk_id: h.score for h in fused}
    assert scores == pytest.approx({"a": 1 / 11 + 1 / 11, "b": 1 / 12})


def test_original_scores_are_ignored() -> None:
    fused = reciprocal_rank_fusion([[hit("a", 0.001), hit("b", 999.0)]], k=60)
    assert ids(fused) == ["a", "b"]


def test_equal_scores_tie_break_on_chunk_id() -> None:
    # x and y swap places, so both get 1/61 + 1/62
    fused = reciprocal_rank_fusion([ranking("y", "x"), ranking("x", "y")], k=60)
    assert fused[0].score == pytest.approx(fused[1].score)
    assert ids(fused) == ["x", "y"]


def test_equal_scores_tie_break_on_best_rank_first() -> None:
    # k=1: "z" gets 1/(1+1) = 0.5, "a" gets 2/(1+3) = 0.5; z has the better best rank
    fused = reciprocal_rank_fusion([ranking("z"), ranking("p", "q", "a")], k=1, weights=[1.0, 2.0])
    by_id = {h.chunk.chunk_id: h.score for h in fused}
    assert by_id["z"] == pytest.approx(by_id["a"])
    assert ids(fused).index("z") < ids(fused).index("a")


def test_weights() -> None:
    fused = reciprocal_rank_fusion([ranking("a"), ranking("b")], k=60, weights=[1.0, 3.0])
    assert ids(fused) == ["b", "a"]
    assert fused[0].score == pytest.approx(3 / 61)


def test_duplicate_in_one_list_counts_once() -> None:
    fused = reciprocal_rank_fusion([ranking("a", "a", "b")], k=60)
    scores = {h.chunk.chunk_id: h.score for h in fused}
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 63)


def test_single_list_keeps_order_and_empty_lists() -> None:
    assert ids(reciprocal_rank_fusion([ranking("c", "a", "b")])) == ["c", "a", "b"]
    assert reciprocal_rank_fusion([[], []]) == []
    assert ids(reciprocal_rank_fusion([[], ranking("a")])) == ["a"]


def test_bad_arguments() -> None:
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([ranking("a")], weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion([ranking("a")], k=0)
