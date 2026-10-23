"""Chunker tests use synthetic placeholder text: they test structure, not legal content."""

import pytest
from pydantic import ValidationError

from eurolex.config import ArticleChunkerConfig, FixedChunkerConfig
from eurolex.ingestion.chunk import ArticleChunker, FixedSizeChunker, count_words, pack_lines
from eurolex.schemas import Article, Chunk, Paragraph, Recital

URL = "https://example.test/act"


def words(n: int, tag: str = "w") -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def make_article(number: str, paragraphs: list[tuple[str | None, str]]) -> Article:
    return Article(
        act_id="gdpr",
        celex="32016R0679",
        article_number=number,
        title=f"Title {number}",
        chapter="Chapter I",
        chapter_title="General",
        paragraphs=[Paragraph(number=n, text=t) for n, t in paragraphs],
        source_url=URL,
    )


def make_recital(number: str, text: str) -> Recital:
    return Recital(
        act_id="gdpr", celex="32016R0679", recital_number=number, text=text, source_url=URL
    )


def test_pack_lines_splits_only_between_lines() -> None:
    text = "\n".join([words(4, "a"), words(4, "b"), words(4, "c")])
    assert pack_lines(text, 8, count_words) == [
        f"{words(4, 'a')}\n{words(4, 'b')}",
        words(4, "c"),
    ]


def test_pack_lines_cuts_an_oversized_line() -> None:
    assert [count_words(p) for p in pack_lines(words(25), 10, count_words)] == [10, 10, 5]


def test_short_article_is_one_chunk() -> None:
    article = make_article("5", [("1", words(10)), ("2", words(10))])
    (chunk,) = ArticleChunker(ArticleChunkerConfig(max_tokens=50)).chunk([article], [])
    assert chunk.chunk_id == "gdpr:art-5"
    assert chunk.paragraph is None
    assert chunk.article_id == "gdpr:art-5"
    assert chunk.text == article.text
    assert chunk.char_count == len(chunk.text)
    assert (chunk.article_title, chunk.chapter) == ("Title 5", "Chapter I")


def test_long_article_splits_per_paragraph_then_parts() -> None:
    long_para = "\n".join(words(20, t) for t in "xyz")
    article = make_article("6", [("1", words(20)), ("2", long_para)])
    chunks = ArticleChunker(ArticleChunkerConfig(max_tokens=45)).chunk([article], [])
    assert [c.chunk_id for c in chunks] == [
        "gdpr:art-6:p-1",
        "gdpr:art-6:p-2:part-1",
        "gdpr:art-6:p-2:part-2",
    ]
    assert [c.paragraph for c in chunks] == ["1", "2", "2"]
    assert {c.article_id for c in chunks} == {"gdpr:art-6"}


def test_long_unnumbered_article_splits_into_parts() -> None:
    article = make_article("4", [(None, "\n".join(words(30, t) for t in "abc"))])
    chunks = ArticleChunker(ArticleChunkerConfig(max_tokens=40)).chunk([article], [])
    assert [c.chunk_id for c in chunks] == [f"gdpr:art-4:part-{i}" for i in (1, 2, 3)]


def test_recital_chunks() -> None:
    (chunk,) = ArticleChunker(ArticleChunkerConfig(max_tokens=40)).chunk(
        [], [make_recital("26", words(5))]
    )
    assert chunk.chunk_id == "gdpr:rct-26"
    assert (chunk.kind, chunk.recital_number, chunk.article_id) == ("recital", "26", None)


def test_article_chunker_is_deterministic() -> None:
    articles = [make_article(str(n), [("1", words(30)), ("2", words(30))]) for n in range(1, 4)]
    chunker = ArticleChunker(ArticleChunkerConfig(max_tokens=40))
    first, second = chunker.chunk(articles, []), chunker.chunk(articles, [])
    assert first == second
    assert len({c.chunk_id for c in first}) == len(first)


def test_fixed_chunker_windows_and_overlap() -> None:
    article = make_article("1", [(None, words(100))])
    chunker = FixedSizeChunker(FixedChunkerConfig(size_tokens=40, overlap_tokens=10))
    chunks = chunker.chunk([article], [])
    assert [c.chunk_id for c in chunks] == [f"gdpr:fixed-40-10:{i:05d}" for i in range(3)]
    tokens = [c.text.split() for c in chunks]
    assert [len(t) for t in tokens] == [40, 40, 40]
    assert tokens[0][-10:] == tokens[1][:10]  # overlap
    assert tokens[-1][-1] == "w99"  # nothing dropped at the end


def test_fixed_chunker_attributes_to_majority_source() -> None:
    recital = make_recital("1", words(15, "r"))
    article = make_article("2", [(None, words(85, "a"))])
    chunks = FixedSizeChunker(FixedChunkerConfig(size_tokens=20, overlap_tokens=0)).chunk(
        [article], [recital]
    )
    assert (chunks[0].kind, chunks[0].recital_number) == ("recital", "1")
    assert chunks[1].kind == "article"
    assert chunks[1].article_id == "gdpr:art-2"
    assert chunks[1].text.split()[:5] == ["a5", "a6", "a7", "a8", "a9"]


def test_fixed_chunker_short_text_gives_one_chunk() -> None:
    article = make_article("1", [(None, words(5))])
    chunks = FixedSizeChunker(FixedChunkerConfig(size_tokens=40, overlap_tokens=10)).chunk(
        [article], []
    )
    assert len(chunks) == 1


def test_fixed_config_rejects_overlap_ge_size() -> None:
    with pytest.raises(ValidationError):
        FixedChunkerConfig(size_tokens=10, overlap_tokens=10)


def test_chunk_validation() -> None:
    base = {
        "chunk_id": "gdpr:art-1",
        "act_id": "gdpr",
        "celex": "32016R0679",
        "kind": "article",
        "article_number": "1",
        "article_title": "t",
        "paragraph": None,
        "chapter": None,
        "text": "abc",
        "source_url": URL,
        "char_count": 3,
    }
    Chunk.model_validate(base)
    with pytest.raises(ValidationError, match="char_count"):
        Chunk.model_validate({**base, "char_count": 4})
    with pytest.raises(ValidationError, match="article_number"):
        Chunk.model_validate({**base, "article_number": None})
    with pytest.raises(ValidationError, match="recital_number"):
        Chunk.model_validate({**base, "kind": "recital"})
