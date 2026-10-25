"""Candidate generator tests with a scripted LLM. The texts are test data only."""

import json
from pathlib import Path

from eurolex.evaluation.candidates import (
    CandidateGenerator,
    Draft,
    append_candidates,
    build_candidates,
    parse_reply,
    reference_pairs,
)
from eurolex.evaluation.questions import load_questions
from eurolex.schemas import Article, ArticleReference, Paragraph


class ScriptedLLM:
    model = "test-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, object] | None = None,
        temperature: float | None = None,
    ) -> str:
        assert json_schema is not None
        enum = json_schema["properties"]["questions"]["items"]["properties"]["type"]["enum"]  # type: ignore[index]
        self.calls.append({"prompt": prompt, "types": enum})
        n = len(self.calls)
        return json.dumps(
            {"questions": [{"question": f"generated question {n}?", "type": enum[0]}]}
        )


def article(n: str, refs: list[str] | None = None) -> Article:
    return Article(
        act_id="gdpr",
        celex="x",
        article_number=n,
        title=f"Title {n}",
        chapter=None,
        chapter_title=None,
        paragraphs=[Paragraph(number=None, text=f"body {n}")],
        references=[ArticleReference(article_number=r) for r in refs or []],
        source_url="https://example.test",
    )


ARTICLES = {
    a.article_id: a for a in (article("6"), article("17", ["6", "99"]), article("21", ["21"]))
}


def test_parse_reply_skips_bad_items() -> None:
    raw = json.dumps(
        {
            "questions": [
                {"question": "Is this a good question?", "type": "factual"},
                {"question": "short", "type": "factual"},
                {"question": "Wrong type question here?", "type": "essay"},
                "not a dict",
                {"question": "Second good question?", "type": "definition"},
            ]
        }
    )
    assert parse_reply(raw, ["factual", "definition"], 5) == [
        ("Is this a good question?", "factual"),
        ("Second good question?", "definition"),
    ]
    assert parse_reply(raw, ["factual", "definition"], 1) == [
        ("Is this a good question?", "factual")
    ]
    assert parse_reply("not json", ["factual"], 5) == []
    assert parse_reply('{"questions": "x"}', ["factual"], 5) == []


def test_reference_pairs_only_link_existing_other_articles() -> None:
    pairs = [(a.article_id, b.article_id) for a, b in reference_pairs(ARTICLES)]
    assert pairs == [("gdpr:art-17", "gdpr:art-6")]  # art-99 missing, art-21 self-reference


def test_build_candidates_covers_all_kinds() -> None:
    llm = ScriptedLLM()
    drafts = build_candidates(
        CandidateGenerator(llm),
        ARTICLES,
        n_articles=2,
        per_article=1,
        n_false_premise=1,
        n_multi=5,
        n_oos=3,
        seed=1,
    )
    kinds = [d.type for d in drafts]
    assert kinds.count("out_of_scope") == 1
    assert kinds.count("multi_article") == 1
    multi = next(d for d in drafts if d.type == "multi_article")
    assert multi.gold == ["gdpr:art-17", "gdpr:art-6"]
    oos = next(d for d in drafts if d.type == "out_of_scope")
    assert oos.gold == []
    assert all("test-model" in d.notes for d in drafts)
    assert "false premise" in drafts[2].notes


def test_append_candidates_is_pending_and_deduplicated(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    drafts = [
        Draft("Can my boss read my email?", "factual", ["gdpr:art-6"], "n"),
        Draft("can my boss READ my e-mail", "factual", ["gdpr:art-6"], "n"),  # near duplicate
        Draft("Is my car insurance tax deductible?", "out_of_scope", [], "n"),
    ]
    added = append_candidates(drafts, path)
    assert [q.id for q in added] == ["q-0001", "q-0002"]
    assert {q.status for q in added} == {"pending"}
    assert {q.source for q in added} == {"llm_candidate"}
    again = append_candidates(drafts, path)
    assert again == []
    assert len(load_questions(path)) == 2
