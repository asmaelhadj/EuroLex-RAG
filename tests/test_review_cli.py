"""Review CLI tests with scripted input. Question texts here are test data only."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from eurolex.evaluation.questions import (
    candidate_rejection_rate,
    hash_split,
    load_questions,
    next_id,
    read_log,
    save_questions,
)
from eurolex.evaluation.review_cli import Reviewer, assign_splits, stats
from eurolex.schemas import Article, Paragraph, Question


def article(n: str, title: str) -> Article:
    return Article(
        act_id="gdpr",
        celex="x",
        article_number=n,
        title=title,
        chapter=None,
        chapter_title=None,
        paragraphs=[Paragraph(number=None, text=f"text of article {n}")],
        source_url="https://example.test",
    )


ARTICLES = {a.article_id: a for a in (article("6", "Lawfulness"), article("17", "Erasure"))}


def question(qid: str, **kw: object) -> Question:
    row: dict[str, object] = {
        "id": qid,
        "question": f"question {qid}?",
        "gold_article_ids": ["gdpr:art-6"],
        "type": "factual",
        "source": "llm_candidate",
        "status": "pending",
    }
    return Question.model_validate({**row, **kw})


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Path, Path]:
    qpath, log = tmp_path / "questions.jsonl", tmp_path / "log.jsonl"
    save_questions([question("q-0001"), question("q-0002"), question("q-0003")], qpath)
    return qpath, log


def reviewer(paths: tuple[Path, Path], answers: list[str], out: list[str]) -> Reviewer:
    it: Iterator[str] = iter(answers)
    return Reviewer(paths[0], paths[1], ARTICLES, ask=lambda _: next(it), say=out.append)


def test_validate_reject_and_skip(paths: tuple[Path, Path]) -> None:
    out: list[str] = []
    reviewer(paths, ["v", "r", "s"], out).review()
    statuses = [q.status for q in load_questions(paths[0])]
    assert statuses == ["validated", "rejected", "pending"]
    assert [e["decision"] for e in read_log(paths[1])] == ["validated", "rejected"]
    assert any("Lawfulness" in line for line in out)  # gold article shown


def test_edit_question_gold_and_type(paths: tuple[Path, Path]) -> None:
    answers = ["e", "Can I have my data erased?", "g", "gdpr:art-17", "t", "obligation", "v", "q"]
    reviewer(paths, answers, []).review()
    first = load_questions(paths[0])[0]
    assert first.question == "Can I have my data erased?"
    assert first.gold_article_ids == ["gdpr:art-17"]
    assert first.type == "obligation"
    assert first.status == "validated"
    assert read_log(paths[1])[0]["edited"] == ["question", "gold_article_ids", "type"]


def test_unknown_gold_is_refused(paths: tuple[Path, Path]) -> None:
    out: list[str] = []
    reviewer(paths, ["g", "gdpr:art-999", "q"], out).review()
    assert load_questions(paths[0])[0].gold_article_ids == ["gdpr:art-6"]
    assert any("Unknown article ids" in line for line in out)


def test_empty_gold_makes_it_out_of_scope(paths: tuple[Path, Path]) -> None:
    reviewer(paths, ["g", "", "v", "q"], []).review()
    first = load_questions(paths[0])[0]
    assert (first.type, first.gold_article_ids, first.status) == ("out_of_scope", [], "validated")


def test_quit_keeps_earlier_decisions(paths: tuple[Path, Path]) -> None:
    reviewer(paths, ["v", "q"], []).review()
    assert [q.status for q in load_questions(paths[0])] == ["validated", "pending", "pending"]


def test_add_human_question(paths: tuple[Path, Path]) -> None:
    answers = ["When must I delete data?", "faq", "EDPB FAQ, CC BY 4.0", "gdpr:art-17", ""]
    added = reviewer(paths, answers, []).add()
    assert added is not None
    assert (added.id, added.source, added.status, added.type) == (
        "q-0004",
        "faq",
        "validated",
        "factual",
    )
    assert load_questions(paths[0])[-1] == added


def test_hash_split_is_stable_and_roughly_80_20() -> None:
    splits = [hash_split(f"q-{i:04d}", 0.2, 42) for i in range(1, 1001)]
    assert splits == [hash_split(f"q-{i:04d}", 0.2, 42) for i in range(1, 1001)]
    assert 150 < splits.count("test") < 250


def test_assign_splits_only_touches_validated(paths: tuple[Path, Path]) -> None:
    reviewer(paths, ["v", "v", "v"], []).review()
    save_questions([*load_questions(paths[0]), question("q-0004")], paths[0])
    counts = assign_splits(paths[0], 1.0, 42)  # everything validated goes to test
    assert sum(counts.values()) == 3
    assert [q.split for q in load_questions(paths[0])] == ["test", "test", "test", "dev"]


def test_stats_and_rejection_rate(paths: tuple[Path, Path]) -> None:
    reviewer(paths, ["v", "r", "r"], []).review()
    lines = stats(*paths)
    assert "LLM candidates reviewed: 3, rejected: 2 (67%)" in lines[-1]
    # a later decision on the same question replaces the earlier one
    log = [
        *read_log(paths[1]),
        {"id": "q-0002", "source": "llm_candidate", "decision": "validated"},
    ]
    assert candidate_rejection_rate(log) == (1, 3)


def test_next_id_and_load_errors(tmp_path: Path) -> None:
    assert next_id([]) == "q-0001"
    assert next_id([question("q-0009"), question("q-0002")]) == "q-0010"
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "q-0001"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.jsonl:1"):
        load_questions(bad)
    assert load_questions(tmp_path / "missing.jsonl") == []
