"""Parse EUR-Lex / CELLAR act HTML into structured articles and recitals.

The parser relies on the ELI ids the Publications Office converter puts in every act page
(`art_6`, `rct_12`, `cpt_II`, paragraph divs like `006.001`) rather than on visual classes or
text patterns, because those ids are stable across the HTML and XHTML renditions. Counts are
checked against the expected numbers in the config so a format change fails loudly.
"""

import re
import warnings
from collections.abc import Iterator
from pathlib import Path

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from eurolex.schemas import Article, ArticleReference, Paragraph, Recital

_ARTICLE_ID = re.compile(r"^art_(\d+[a-z]?)$")
_RECITAL_ID = re.compile(r"^rct_(\d+)$")
_CHAPTER_ID = re.compile(r"^cpt_([IVXLC]+)$")
_PARAGRAPH_ID = re.compile(r"^\d{3}\.\d{3}$")
_PARAGRAPH_NUMBER = re.compile(r"^(\d+[a-z]?)\.\s")
_WHITESPACE = re.compile(r"\s+")

# One reference inside a list, e.g. `6`, `6(1)`, `6(1)(a)`.
_REF = r"\d+[a-z]?(?:\(\d+[a-z]?\))?(?:\([a-z]+\))?"
_REFERENCE = re.compile(
    r"(?:\bpoints?\s+\((?P<point>[a-z]+)\)\s+of\s+)?"
    r"\bArticles?\s+(?P<refs>" + _REF + r"(?:\s*(?:,|and|or|to)\s*" + _REF + r")*)"
)
_SINGLE_REF = re.compile(r"(\d+[a-z]?)(?:\((\d+[a-z]?)\))?(?:\(([a-z]+)\))?")
_RANGE = re.compile(r"(\d+)\s*to\s*(\d+)")
# A reference followed by "of <another instrument>" or a treaty abbreviation is external.
_EXTERNAL_SUFFIX = re.compile(r"^\s+(?:of\s+(?!this\b)|TFEU\b|TEU\b)")


class ParseError(ValueError):
    pass


def load_soup(path: Path) -> BeautifulSoup:
    """Parse with the lenient HTML parser so both CELLAR XHTML and browser-saved HTML work."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        return BeautifulSoup(path.read_bytes(), "lxml")


def clean_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text.replace("\xa0", " ")).strip()


def _strip_footnote_markers(root: Tag) -> None:
    for anchor in root.select('a[href^="#ntr"]'):
        anchor.decompose()


def _render_blocks(element: Tag) -> Iterator[str]:
    """Yield one line per structural block (sentence paragraph or list point), in order.

    List points are two-cell tables (label, body); the label is prefixed to the first line of the
    body so `(a)` stays attached to its text, and nested points become their own lines.
    """
    for child in element.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "p":
            if "oj-ti-art" in (child.get("class") or []):
                continue
            text = clean_text(child.get_text(" "))
            if text:
                yield text
        elif child.name == "table":
            for row in child.find_all("tr", recursive=True):
                if row.find_parent("table") is not child:
                    continue
                cells = row.find_all("td", recursive=False)
                if len(cells) < 2:
                    yield from _render_blocks(row)
                    continue
                label = clean_text(cells[0].get_text(" "))
                lines = list(_render_blocks(cells[1]))
                if lines:
                    lines[0] = f"{label} {lines[0]}".strip()
                    yield from lines
        elif child.name == "div" and "eli-title" in (child.get("class") or []):
            continue
        else:
            yield from _render_blocks(child)


def _paragraphs(article: Tag) -> list[Paragraph]:
    divs = list(article.find_all("div", id=_PARAGRAPH_ID))
    if not divs:
        text = "\n".join(_render_blocks(article))
        return [Paragraph(number=None, text=text)] if text else []
    paragraphs: list[Paragraph] = []
    for div in divs:
        text = "\n".join(_render_blocks(div))
        if not text:
            continue
        match = _PARAGRAPH_NUMBER.match(text)
        number = match.group(1) if match else str(int(str(div["id"]).split(".")[1]))
        paragraphs.append(Paragraph(number=number, text=text))
    return paragraphs


def extract_references(text: str) -> list[ArticleReference]:
    """Find references to articles of the same act, e.g. `Article 6(1)(a)` or `Articles 15 to 22`.

    References that continue with "of <other instrument>" (a directive, a treaty, the Charter) or
    "TFEU" are dropped; "of this Regulation" is kept. Output is de-duplicated in order.
    """
    found: dict[tuple[str, str | None, str | None], None] = {}
    for match in _REFERENCE.finditer(text):
        if _EXTERNAL_SUFFIX.match(text[match.end() :]):
            continue
        refs = match.group("refs")
        prefix_point = match.group("point")
        for start, end in _RANGE.findall(refs):
            for n in range(int(start), int(end) + 1):
                found[(str(n), None, None)] = None
        for number, paragraph, point in _SINGLE_REF.findall(_RANGE.sub("", refs)):
            found[(number, paragraph or None, point or prefix_point or None)] = None
    return [ArticleReference(article_number=a, paragraph=p, point=q) for a, p, q in found]


def _chapter(article: Tag, soup: BeautifulSoup) -> tuple[str | None, str | None]:
    chapter = article.find_parent("div", id=_CHAPTER_ID)
    if chapter is None:
        return None, None
    roman = str(chapter["id"]).removeprefix("cpt_")
    title_div = soup.find("div", id=f"cpt_{roman}.tit_1")
    title = clean_text(title_div.get_text(" ")) if isinstance(title_div, Tag) else None
    return f"Chapter {roman}", title


def parse_articles(
    soup: BeautifulSoup, *, act_id: str, celex: str, source_url: str
) -> list[Article]:
    articles: list[Article] = []
    for div in soup.find_all("div", id=_ARTICLE_ID):
        number = str(div["id"]).removeprefix("art_")
        _strip_footnote_markers(div)
        title_div = div.find("div", class_="eli-title")
        title = clean_text(title_div.get_text(" ")) if isinstance(title_div, Tag) else ""
        paragraphs = _paragraphs(div)
        if not paragraphs:
            raise ParseError(f"{act_id}: Article {number} has no text")
        chapter, chapter_title = _chapter(div, soup)
        body = "\n".join(p.text for p in paragraphs)
        articles.append(
            Article(
                act_id=act_id,
                celex=celex,
                article_number=number,
                title=title,
                chapter=chapter,
                chapter_title=chapter_title,
                paragraphs=paragraphs,
                references=extract_references(body),
                source_url=source_url,
            )
        )
    return articles


def parse_recitals(
    soup: BeautifulSoup, *, act_id: str, celex: str, source_url: str
) -> list[Recital]:
    recitals: list[Recital] = []
    for div in soup.find_all("div", id=_RECITAL_ID):
        number = str(div["id"]).removeprefix("rct_")
        _strip_footnote_markers(div)
        cells = div.find_all("td")
        body = cells[-1] if cells else div
        text = clean_text(body.get_text(" "))
        if not text:
            raise ParseError(f"{act_id}: recital {number} has no text")
        recitals.append(
            Recital(
                act_id=act_id,
                celex=celex,
                recital_number=number,
                text=text,
                source_url=source_url,
            )
        )
    return recitals


def check_counts(
    act_id: str,
    articles: list[Article],
    recitals: list[Recital],
    *,
    expected_articles: int,
    expected_recitals: int,
) -> None:
    """Fail on missing, extra or duplicated articles or recitals."""
    problems: list[str] = []
    numbers = [a.article_number for a in articles]
    if len(set(numbers)) != len(numbers):
        problems.append("duplicate article numbers")
    if len({r.recital_number for r in recitals}) != len(recitals):
        problems.append("duplicate recital numbers")
    if len(articles) != expected_articles:
        problems.append(f"{len(articles)} articles, expected {expected_articles}")
    if len(recitals) != expected_recitals:
        problems.append(f"{len(recitals)} recitals, expected {expected_recitals}")
    if problems:
        raise ParseError(f"{act_id}: " + "; ".join(problems))
