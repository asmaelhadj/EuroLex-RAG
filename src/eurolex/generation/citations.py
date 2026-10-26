"""Find `[n]` markers in an answer, map them to passages, drop the ones that point nowhere."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from eurolex.schemas import Citation, ScoredChunk

# [1], [1, 2], [1,2,3]; ranges like [1-3] are expanded
_MARKER = re.compile(r"\[(\d+(?:\s*[-,]\s*\d+)*)\]")


@dataclass(frozen=True)
class CitationCheck:
    text: str  # answer with invalid numbers removed
    valid: list[int]  # in order of first use
    invalid: list[int]


def _numbers(group: str) -> list[int]:
    out: list[int] = []
    for part in group.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            out += range(start, end + 1) if 0 < end - start < 20 else [start, end]
        else:
            out.append(int(part))
    return out


def check_citations(answer: str, n_passages: int) -> CitationCheck:
    valid: dict[int, None] = {}
    invalid: dict[int, None] = {}

    def replace(match: re.Match[str]) -> str:
        keep = []
        for n in _numbers(match.group(1)):
            if 1 <= n <= n_passages:
                valid[n] = None
                keep.append(n)
            else:
                invalid[n] = None
        return "".join(f"[{n}]" for n in dict.fromkeys(keep))

    text = _MARKER.sub(replace, answer)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)  # tidy spaces left by removed markers
    return CitationCheck(text=text, valid=list(valid), invalid=list(invalid))


def resolve(numbers: Sequence[int], passages: Sequence[ScoredChunk]) -> list[Citation]:
    citations = []
    for n in numbers:
        chunk = passages[n - 1].chunk
        citations.append(
            Citation(
                n=n,
                chunk_id=chunk.chunk_id,
                article_id=chunk.article_id,
                article_title=chunk.article_title,
                text=chunk.text,
                url=chunk.source_url,
            )
        )
    return citations
