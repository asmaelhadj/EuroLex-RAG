"""Optional LLM query rewriting (experiment E5)."""

import logging
from typing import Literal

from eurolex.generation.llm import LLMClient, LLMError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You rewrite questions about EU law into search queries. "
    "Use the legal terms the regulation itself would use. "
    "Reply with the search query only, on one line, no explanation."
)
MAX_REWRITE_CHARS = 300


class QueryRewriter:
    """Asks the LLM for a search query. Falls back to the original question on any failure."""

    def __init__(self, llm: LLMClient, mode: Literal["replace", "expand"] = "expand") -> None:
        self._llm = llm
        self._mode = mode

    def rewrite(self, question: str) -> str:
        try:
            raw = self._llm.generate(f"Question: {question}", system=SYSTEM_PROMPT)
        except LLMError:
            logger.warning("query rewriting failed, using the original question")
            return question
        rewritten = clean_rewrite(raw)
        if not rewritten:
            return question
        return rewritten if self._mode == "replace" else f"{question} {rewritten}"


def clean_rewrite(raw: str) -> str:
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    text = lines[0].removeprefix("Search query:").strip().strip("\"'`").strip()
    return text[:MAX_REWRITE_CHARS]
