"""Turn parsed acts into retrieval chunks.

Two strategies share one interface so experiment E0 (fixed-size) and E1 (article-aware) differ
only in the chunker. Chunk ids are derived purely from the document structure and the chunker
settings, never from run order or randomness, so re-indexing is idempotent and gold labels stay
valid across runs.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Protocol

from eurolex.config import ArticleChunkerConfig, FixedChunkerConfig
from eurolex.schemas import Article, Chunk, ChunkKind, Recital


class TokenCounter(Protocol):
    def __call__(self, text: str) -> int: ...


def count_words(text: str) -> int:
    """Default token estimate: whitespace words.

    A real subword tokenizer (BGE-M3) arrives with the embedding code in Phase 2; until then this
    keeps chunking dependency-free. Subword counts run roughly 1.3x higher for English legal text.
    """
    return len(text.split())


class Chunker(Protocol):
    name: str

    def chunk(self, articles: Sequence[Article], recitals: Sequence[Recital]) -> list[Chunk]: ...


def _make_chunk(
    chunk_id: str,
    text: str,
    *,
    kind: ChunkKind,
    article: Article | None = None,
    recital: Recital | None = None,
    paragraph: str | None = None,
) -> Chunk:
    source = article or recital
    if source is None:
        raise ValueError("a chunk needs an article or a recital")
    return Chunk(
        chunk_id=chunk_id,
        act_id=source.act_id,
        celex=source.celex,
        kind=kind,
        article_number=article.article_number if article else None,
        article_title=article.title if article else None,
        paragraph=paragraph,
        chapter=article.chapter if article else None,
        recital_number=recital.recital_number if recital else None,
        text=text,
        source_url=source.source_url,
        char_count=len(text),
    )


def _recital_chunks(recitals: Iterable[Recital]) -> list[Chunk]:
    return [
        _make_chunk(f"{r.act_id}:rct-{r.recital_number}", r.text, kind="recital", recital=r)
        for r in recitals
    ]


def pack_lines(text: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Greedily pack lines into parts of at most `max_tokens`, splitting only between lines.

    A single line longer than the limit is cut on word boundaries as a last resort.
    """
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for line in text.split("\n"):
        line_tokens = count(line)
        if line_tokens > max_tokens:
            if current:
                parts.append("\n".join(current))
                current, current_tokens = [], 0
            words = line.split()
            parts.extend(
                " ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)
            )
            continue
        if current and current_tokens + line_tokens > max_tokens:
            parts.append("\n".join(current))
            current, current_tokens = [], 0
        current.append(line)
        current_tokens += line_tokens
    if current:
        parts.append("\n".join(current))
    return parts


class ArticleChunker:
    """One chunk per article; per paragraph when the article is too long; parts as a last resort.

    Chunk ids: `gdpr:art-6`, `gdpr:art-6:p-1`, `gdpr:art-6:p-1:part-2`, `gdpr:art-4:part-3`,
    and `gdpr:rct-1` for recitals.
    """

    name = "article"

    def __init__(self, config: ArticleChunkerConfig, count: TokenCounter = count_words) -> None:
        self._max = config.max_tokens
        self._count = count

    def chunk(self, articles: Sequence[Article], recitals: Sequence[Recital]) -> list[Chunk]:
        chunks = _recital_chunks(recitals)
        for article in articles:
            chunks.extend(self._article_chunks(article))
        return chunks

    def _article_chunks(self, article: Article) -> list[Chunk]:
        base = article.article_id
        if self._count(article.text) <= self._max:
            return [_make_chunk(base, article.text, kind="article", article=article)]
        chunks: list[Chunk] = []
        for para in article.paragraphs:
            prefix = f"{base}:p-{para.number}" if para.number else base
            parts = pack_lines(para.text, self._max, self._count)
            if len(parts) == 1 and para.number:
                chunks.append(
                    _make_chunk(
                        prefix, parts[0], kind="article", article=article, paragraph=para.number
                    )
                )
                continue
            chunks.extend(
                _make_chunk(
                    f"{prefix}:part-{i}",
                    part,
                    kind="article",
                    article=article,
                    paragraph=para.number,
                )
                for i, part in enumerate(parts, start=1)
            )
        return chunks


class FixedSizeChunker:
    """Sliding windows of `size_tokens` words with `overlap_tokens` overlap, ignoring structure.

    This is the naive E0 baseline. Each act is chunked separately (recitals, then articles, in
    document order). Because a window can straddle articles, each chunk is attributed to the
    article (or recital) that contributes most of its words, so article-level metrics stay
    well-defined without crediting one chunk to several articles.
    """

    name = "fixed"

    def __init__(self, config: FixedChunkerConfig) -> None:
        self._size = config.size_tokens
        self._step = config.size_tokens - config.overlap_tokens
        self._tag = f"fixed-{config.size_tokens}-{config.overlap_tokens}"

    def chunk(self, articles: Sequence[Article], recitals: Sequence[Recital]) -> list[Chunk]:
        acts = list(dict.fromkeys([r.act_id for r in recitals] + [a.act_id for a in articles]))
        chunks: list[Chunk] = []
        for act_id in acts:
            owners: list[Article | Recital] = [r for r in recitals if r.act_id == act_id]
            owners += [a for a in articles if a.act_id == act_id]
            words = [(w, i) for i, owner in enumerate(owners) for w in owner.text.split()]
            chunks.extend(self._windows(act_id, words, owners))
        return chunks

    def _windows(
        self, act_id: str, words: list[tuple[str, int]], owners: list[Article | Recital]
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        starts = range(0, max(len(words) - self._size + self._step, 1), self._step)
        for index, start in enumerate(starts):
            window = words[start : start + self._size]
            owner = owners[Counter(i for _, i in window).most_common(1)[0][0]]
            text = " ".join(w for w, _ in window)
            chunk_id = f"{act_id}:{self._tag}:{index:05d}"
            if isinstance(owner, Article):
                chunks.append(_make_chunk(chunk_id, text, kind="article", article=owner))
            else:
                chunks.append(_make_chunk(chunk_id, text, kind="recital", recital=owner))
        return chunks
