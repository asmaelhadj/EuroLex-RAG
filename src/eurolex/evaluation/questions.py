"""Read and write data/eval/questions.jsonl and the review log."""

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from eurolex.schemas import Article, Question, Split

QUESTIONS_PATH = Path("data/eval/questions.jsonl")
REVIEW_LOG_PATH = Path("data/eval/review_log.jsonl")


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    if not path.exists():
        return []
    questions = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if line.strip():
                try:
                    questions.append(Question.model_validate_json(line))
                except ValueError as exc:
                    raise ValueError(f"{path}:{n}: {exc}") from exc
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate question ids")
    return questions


def save_questions(questions: Iterable[Question], path: Path = QUESTIONS_PATH) -> None:
    # write to a temp file first so a crash never leaves half a file
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        for q in questions:
            fh.write(q.model_dump_json() + "\n")
    os.replace(tmp, path)


def next_id(questions: Iterable[Question]) -> str:
    numbers = [int(q.id.removeprefix("q-")) for q in questions]
    return f"q-{max(numbers, default=0) + 1:04d}"


def normalise(text: str) -> str:
    """For duplicate checks: lower case, punctuation dropped ("e-mail" == "email")."""
    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return " ".join(text.split())


def hash_split(question_id: str, test_fraction: float, seed: int) -> Split:
    """Stable split: depends only on the id, so adding questions never moves old ones."""
    digest = hashlib.sha256(f"{seed}:{question_id}".encode()).digest()
    return "test" if int.from_bytes(digest[:8], "big") / 2**64 < test_fraction else "dev"


def load_articles(path: Path) -> dict[str, Article]:
    with path.open(encoding="utf-8") as fh:
        articles = [Article.model_validate_json(line) for line in fh if line.strip()]
    return {a.article_id: a for a in articles}


def log_decision(
    question: Question, decision: str, edited: list[str], path: Path = REVIEW_LOG_PATH
) -> None:
    entry = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "id": question.id,
        "source": question.source,
        "decision": decision,
        "edited": edited,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(entry) + "\n")


def read_log(path: Path = REVIEW_LOG_PATH) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def candidate_rejection_rate(log: list[dict[str, object]]) -> tuple[int, int]:
    """(rejected, reviewed) for LLM candidates, using the last decision per question."""
    last: dict[str, str] = {}
    for entry in log:
        if entry.get("source") == "llm_candidate" and entry.get("decision") in (
            "validated",
            "rejected",
        ):
            last[str(entry["id"])] = str(entry["decision"])
    return sum(d == "rejected" for d in last.values()), len(last)
