from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from eurolex.ingestion.parse import (
    ParseError,
    check_counts,
    extract_references,
    load_soup,
    parse_articles,
    parse_recitals,
)
from eurolex.schemas import Article, ArticleReference, Recital

FIXTURE = Path(__file__).parent / "fixtures" / "gdpr_sample.html"
META = {"act_id": "gdpr", "celex": "32016R0679", "source_url": "https://example.test/gdpr"}


@pytest.fixture(scope="module")
def articles() -> dict[str, Article]:
    return {a.article_number: a for a in parse_articles(load_soup(FIXTURE), **META)}


@pytest.fixture(scope="module")
def recitals() -> list[Recital]:
    return parse_recitals(load_soup(FIXTURE), **META)


def test_article_numbering_titles_and_chapters(articles: dict[str, Article]) -> None:
    assert list(articles) == ["1", "4", "6"]
    art6 = articles["6"]
    assert art6.article_id == "gdpr:art-6"
    assert art6.title == "Lawfulness of processing"
    assert (art6.chapter, art6.chapter_title) == ("Chapter II", "Principles")
    assert articles["1"].chapter == "Chapter I"


def test_paragraph_splitting(articles: dict[str, Article]) -> None:
    paragraphs = articles["6"].paragraphs
    assert [p.number for p in paragraphs] == ["1", "2", "3", "4"]
    first = paragraphs[0].text.split("\n")
    assert first[0].startswith("1. Processing shall be lawful only if")
    # List points stay attached to their label, one point per line.
    assert [line[:3] for line in first[1:7]] == ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    # The closing subparagraph after the list is its own line.
    assert first[7].startswith("Point (f) of the first subparagraph shall not apply")


def test_unnumbered_article_is_one_paragraph(articles: dict[str, Article]) -> None:
    definitions = articles["4"].paragraphs
    assert len(definitions) == 1
    assert definitions[0].number is None
    lines = definitions[0].text.split("\n")
    assert lines[0] == "For the purposes of this Regulation:"
    assert lines[1].startswith("(1) ")
    assert "personal data" in lines[1]


def test_text_is_normalised(articles: dict[str, Article]) -> None:
    text = articles["6"].text
    assert "\xa0" not in text
    assert "  " not in text


def test_recitals_strip_footnote_markers(recitals: list[Recital]) -> None:
    assert [r.recital_number for r in recitals] == ["1", "3"]
    assert recitals[0].text.startswith("The protection of natural persons")
    assert not recitals[0].text.startswith("(1)")
    # In the source, "Directive 95/46/EC" is followed by a footnote call "(4)".
    assert "95/46/EC of the European Parliament" in recitals[1].text


def test_references_extracted_from_article(articles: dict[str, Article]) -> None:
    assert ArticleReference(article_number="23", paragraph="1") in articles["6"].references


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("as referred to in Article 6(1)(a)", [("6", "1", "a")]),
        ("pursuant to Article 9", [("9", None, None)]),
        ("point (f) of Article 6(1) applies", [("6", "1", "f")]),
        ("Articles 13 and 14", [("13", None, None), ("14", None, None)]),
        ("Articles 15 to 18", [(str(n), None, None) for n in range(15, 19)]),
        ("Article 58(2) of this Regulation", [("58", "2", None)]),
        ("Article 8 of Directive 95/46/EC", []),
        ("Article 16(2) TFEU", []),
        ("Article 8(1) of the Charter", []),
        ("Article 6 and again Article 6", [("6", None, None)]),
    ],
)
def test_extract_references(text: str, expected: list[tuple[str, str | None, str | None]]) -> None:
    got = [(r.article_number, r.paragraph, r.point) for r in extract_references(text)]
    assert got == expected


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def test_paragraph_number_falls_back_to_div_id() -> None:
    html = """<div id="cpt_I"><div id="art_2"><p class="oj-ti-art">Article 2</p>
    <div class="eli-title"><p class="oj-sti-art">Scope</p></div>
    <div id="002.001"><p class="oj-normal">Unnumbered opening text.</p></div>
    <div id="002.002"><p class="oj-normal">2. Second paragraph.</p></div></div></div>"""
    (article,) = parse_articles(_soup(html), **META)
    assert [p.number for p in article.paragraphs] == ["1", "2"]
    assert article.title == "Scope"


def test_empty_article_is_an_error() -> None:
    html = '<div id="art_9"><p class="oj-ti-art">Article 9</p></div>'
    with pytest.raises(ParseError, match="Article 9 has no text"):
        parse_articles(_soup(html), **META)


def test_check_counts(articles: dict[str, Article], recitals: list[Recital]) -> None:
    arts = list(articles.values())
    check_counts("gdpr", arts, recitals, expected_articles=3, expected_recitals=2)
    with pytest.raises(ParseError, match="3 articles, expected 99"):
        check_counts("gdpr", arts, recitals, expected_articles=99, expected_recitals=2)
    with pytest.raises(ParseError, match="duplicate article numbers"):
        check_counts("gdpr", [*arts, arts[0]], recitals, expected_articles=4, expected_recitals=2)
