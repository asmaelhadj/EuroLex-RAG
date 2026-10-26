"""Pydantic data contracts shared across ingestion, retrieval, evaluation and the API.

Every artefact written to disk (articles, recitals, chunks, questions) round-trips through these
models, so a malformed row fails loudly at the boundary instead of silently skewing metrics.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ChunkKind = Literal["article", "recital"]


def article_id(act_id: str, article_number: str) -> str:
    """Article id used for gold labels and metrics, e.g. `gdpr:art-6` (SPEC.md section 5.1)."""
    return f"{act_id}:art-{article_number}"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ArticleReference(_Frozen):
    """A cross-reference such as `Article 6(1)(a)` found in the text of the same act."""

    article_number: str
    paragraph: str | None = None
    point: str | None = None


class Paragraph(_Frozen):
    """A numbered paragraph of an article; `number` is None for unnumbered article bodies.

    `text` keeps one structural block (intro sentence, list point) per line so chunkers can
    split long paragraphs at natural boundaries.
    """

    number: str | None
    text: str


class Article(_Frozen):
    act_id: str
    celex: str
    article_number: str
    title: str
    chapter: str | None
    chapter_title: str | None
    paragraphs: list[Paragraph] = Field(min_length=1)
    references: list[ArticleReference] = Field(default_factory=list)
    source_url: str

    @property
    def article_id(self) -> str:
        return article_id(self.act_id, self.article_number)

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs)


class Recital(_Frozen):
    act_id: str
    celex: str
    recital_number: str
    text: str
    source_url: str


class Chunk(_Frozen):
    """Unit of retrieval (SPEC.md section 5.1).

    `recital_number` is an addition to the spec contract: without it, recital chunks would carry
    no locator at all. Recital chunks have no article id, so they never count as article hits.
    """

    chunk_id: str
    act_id: str
    celex: str
    kind: ChunkKind
    article_number: str | None
    article_title: str | None
    paragraph: str | None
    chapter: str | None
    recital_number: str | None = None
    text: str = Field(min_length=1)
    source_url: str
    char_count: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> "Chunk":
        if self.char_count != len(self.text):
            raise ValueError("char_count must equal len(text)")
        if self.kind == "article" and self.article_number is None:
            raise ValueError("article chunks need an article_number")
        if self.kind == "recital" and self.recital_number is None:
            raise ValueError("recital chunks need a recital_number")
        return self

    @property
    def article_id(self) -> str | None:
        if self.article_number is None:
            return None
        return article_id(self.act_id, self.article_number)


class ScoredChunk(_Frozen):
    chunk: Chunk
    score: float


class RetrievalResult(_Frozen):
    query: str
    # Set when query rewriting is on.
    rewritten_query: str | None = None
    # rewriting was on but the LLM failed, so the original question was used
    rewrite_failed: bool = False
    hits: list[ScoredChunk]
    # extra context (recitals) that is not part of the ranking or the metrics
    support: list[ScoredChunk] = Field(default_factory=list)
    # Stage name -> milliseconds, e.g. {"dense": 41.2, "fusion": 0.1}.
    latency_ms: dict[str, float]


QuestionType = Literal[
    "factual", "definition", "obligation", "multi_article", "paraphrase", "out_of_scope"
]
QuestionSource = Literal["human", "llm_candidate", "faq"]
QuestionStatus = Literal["pending", "validated", "rejected"]
Split = Literal["dev", "test"]

_ARTICLE_ID = r"^[a-z_]+:art-\d+[a-z]?$"


class Question(BaseModel):
    """One row of data/eval/questions.jsonl (SPEC.md section 5.2)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^q-\d{4,}$")
    question: str = Field(min_length=3)
    gold_article_ids: list[str]
    type: QuestionType
    source: QuestionSource
    status: QuestionStatus = "pending"
    split: Split = "dev"
    notes: str = ""

    @model_validator(mode="after")
    def _check_gold(self) -> "Question":
        for gold in self.gold_article_ids:
            if not re.match(_ARTICLE_ID, gold):
                raise ValueError(f"bad article id {gold!r}, expected e.g. gdpr:art-6")
        if self.type == "out_of_scope" and self.gold_article_ids:
            raise ValueError("out_of_scope questions have no gold articles")
        if self.type != "out_of_scope" and not self.gold_article_ids:
            raise ValueError("in-scope questions need at least one gold article")
        if len(set(self.gold_article_ids)) != len(self.gold_article_ids):
            raise ValueError("duplicate gold article ids")
        return self

    @property
    def in_scope(self) -> bool:
        return self.type != "out_of_scope"


class Citation(_Frozen):
    """A `[n]` marker in an answer, resolved to the passage it points at."""

    n: int
    chunk_id: str
    article_id: str | None
    article_title: str | None
    text: str
    url: str


RefusalReason = Literal["low_evidence", "model"]


class Answer(_Frozen):
    question: str
    answer: str
    refused: bool
    refusal_reason: RefusalReason | None = None
    citations: list[Citation] = Field(default_factory=list)
    # [n] markers that pointed at no passage; they are removed from `answer`
    invalid_citations: list[int] = Field(default_factory=list)
    passages: list[ScoredChunk] = Field(default_factory=list)
    retrieval: RetrievalResult
    latency_ms: dict[str, float]
