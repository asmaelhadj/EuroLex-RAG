import pytest

from eurolex.evaluation.import_csv import guess_type, parse_gold, rows_to_questions

KNOWN = {f"gdpr:art-{n}" for n in range(1, 100)}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Article 4(7)", [4]),
        ("Articles 12\u201314", [12, 13, 14]),
        ("Article 21(2)\u2013(3)", [21]),
        ("Articles 5(1)(b) and 6; verify compatibility rules", [5, 6]),
        ("Article 17 and exceptions", [17]),
        ("Article 77; EDPB institutional role", [77]),
        ("Articles 5(1)(f) and 32", [5, 32]),
        ("Articles 37-39", [37, 38, 39]),
        ("Article 2(2)(c) and Recital 18", [2]),
        ("Recitals 26 and 27", []),
        ("", []),
    ],
)
def test_parse_gold(text: str, expected: list[int]) -> None:
    assert parse_gold(text) == [f"gdpr:art-{n}" for n in expected]


def test_guess_type() -> None:
    assert guess_type("What does the GDPR mean by personal data?", ["gdpr:art-4"]) == "definition"
    assert guess_type("When must an organisation appoint a DPO?", ["gdpr:art-37"]) == "obligation"
    assert guess_type("Can a person ask for a copy?", ["gdpr:art-15"]) == "factual"
    assert guess_type("x?", ["gdpr:art-5", "gdpr:art-6"]) == "multi_article"
    assert guess_type("What is the capital of Japan?", []) == "out_of_scope"


def row(qid: str, question: str, scope: str, gold: str = "") -> dict[str, str]:
    return {
        "question_id": qid,
        "question": question,
        "scope": scope,
        "source_organization": "EDPB" if gold else "",
        "source_title": "Guide" if gold else "",
        "source_url": "https://example.test/page" if gold else "",
        "source_license": "unknown" if gold else "",
        "gold_article": gold,
    }


def test_rows_to_questions() -> None:
    rows = [
        row("IN01", "Can I get a copy of my data?", "in_scope", "Article 15"),
        row("IN02", "Can I get a copy of my data?", "in_scope", "Article 15"),  # duplicate
        row("IN03", "Something about article 400?", "in_scope", "Article 400"),  # unknown
        row("IN04", "When must I name a DPO?", "in_scope", "Articles 37\u201339"),
        row("OUT01", "What is the capital of Japan?", "out_of_scope"),
    ]
    added, problems = rows_to_questions(rows, [], KNOWN, "llm_candidate", "drafted by an AI tool")
    assert [q.id for q in added] == ["q-0001", "q-0002", "q-0003"]
    assert {q.status for q in added} == {"pending"}
    assert {q.source for q in added} == {"llm_candidate"}
    assert added[0].gold_article_ids == ["gdpr:art-15"]
    assert "csv IN01" in added[0].notes
    assert "license: unknown" in added[0].notes
    assert "consider narrowing" in added[1].notes
    assert (added[2].type, added[2].gold_article_ids) == ("out_of_scope", [])
    assert len(problems) == 2
