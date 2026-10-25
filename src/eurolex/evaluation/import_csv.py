"""Import draft questions from a CSV as pending candidates.

    python -m eurolex.evaluation.import_csv path/to/questions.csv --source llm_candidate

Expected columns: question_id, question, scope, source_organization, source_title,
source_url, source_license, gold_article. Everything comes in as `pending`; types are a
guess and the gold ids come from the free-text `gold_article` column, so both are checked
in `review_cli review`.
"""

import argparse
import csv
import re
import sys
from collections.abc import Iterable
from pathlib import Path

from eurolex.evaluation.questions import (
    QUESTIONS_PATH,
    load_articles,
    load_questions,
    next_id,
    normalise,
    save_questions,
)
from eurolex.schemas import Question, QuestionSource

_PARENS = re.compile(r"\([^)]*\)")
_RANGE = re.compile(r"(\d+)\s*[-\u2013\u2014]\s*(\d+)")  # hyphen, en dash, em dash
_NUMBER = re.compile(r"\d+")
_OBLIGATION = re.compile(r"\b(must|should|have to|has to|required|need to|appoint)\b", re.I)
_DEFINITION = re.compile(r"\b(what does .* mean|count as|what role|definition)\b", re.I)


def parse_gold(text: str, act_id: str = "gdpr") -> list[str]:
    """'Articles 12-14' -> art-12..14, 'Article 21(2)-(3)' -> art-21, 'Articles 5 and 6' -> 5, 6.

    Only the part before the first ';' is read, the rest is a note.
    """
    head = _PARENS.sub("", text.split(";")[0])
    numbers: list[int] = []
    for start, end in _RANGE.findall(head):
        numbers += range(int(start), int(end) + 1)
    numbers += [int(n) for n in _NUMBER.findall(_RANGE.sub(" ", head))]
    return [f"{act_id}:art-{n}" for n in dict.fromkeys(sorted(numbers))]


def guess_type(question: str, gold: list[str]) -> str:
    if not gold:
        return "out_of_scope"
    if len(gold) > 1:
        return "multi_article"
    if _DEFINITION.search(question):
        return "definition"
    if _OBLIGATION.search(question):
        return "obligation"
    return "factual"


def rows_to_questions(
    rows: Iterable[dict[str, str]],
    existing: list[Question],
    known_articles: set[str],
    source: QuestionSource,
    tool_note: str,
) -> tuple[list[Question], list[str]]:
    seen = {normalise(q.question) for q in existing}
    added: list[Question] = []
    problems: list[str] = []
    for row in rows:
        text = row["question"].strip()
        if normalise(text) in seen:
            problems.append(f"{row['question_id']}: duplicate, skipped")
            continue
        raw_gold = row.get("gold_article", "").strip()
        gold = parse_gold(raw_gold) if row["scope"] == "in_scope" else []
        unknown = [g for g in gold if g not in known_articles]
        if row["scope"] == "in_scope" and (not gold or unknown):
            problems.append(f"{row['question_id']}: gold {raw_gold!r} not understood, skipped")
            continue
        notes = [f"csv {row['question_id']}", tool_note]
        if row.get("source_url"):
            notes.append(
                f"based on {row.get('source_organization', '')}: {row.get('source_title', '')} "
                f"{row['source_url']} (license: {row.get('source_license') or 'unknown'})"
            )
        if raw_gold:
            notes.append(f"gold as written: {raw_gold}")
        if len(gold) > 2:
            notes.append("gold is a range, consider narrowing it")
        added.append(
            Question.model_validate(
                {
                    "id": next_id([*existing, *added]),
                    "question": text,
                    "gold_article_ids": gold,
                    "type": guess_type(text, gold),
                    "source": source,
                    "status": "pending",
                    "notes": "; ".join(notes),
                }
            )
        )
        seen.add(normalise(text))
    return added, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eurolex.evaluation.import_csv")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--source", choices=["human", "llm_candidate", "faq"], required=True)
    parser.add_argument("--note", default="", help="added to every row, e.g. who drafted it")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--articles", type=Path, default=Path("data/processed/articles.jsonl"))
    args = parser.parse_args(argv)

    with args.csv_path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    existing = load_questions(args.questions)
    added, problems = rows_to_questions(
        rows, existing, set(load_articles(args.articles)), args.source, args.note
    )
    save_questions([*existing, *added], args.questions)
    for problem in problems:
        print(problem, file=sys.stderr)
    print(f"Added {len(added)} pending questions to {args.questions}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
