"""Terminal tool for building the question set by hand.

python -m eurolex.evaluation.review_cli review     # go through pending questions
python -m eurolex.evaluation.review_cli add        # add your own question (human / faq)
python -m eurolex.evaluation.review_cli split      # assign dev/test to validated questions
python -m eurolex.evaluation.review_cli stats      # counts and candidate rejection rate
"""

import argparse
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import get_args

from eurolex.evaluation.questions import (
    QUESTIONS_PATH,
    REVIEW_LOG_PATH,
    candidate_rejection_rate,
    hash_split,
    load_articles,
    load_questions,
    log_decision,
    next_id,
    read_log,
    save_questions,
)
from eurolex.schemas import Article, Question, QuestionType

InputFn = Callable[[str], str]
PrintFn = Callable[[str], None]

PREVIEW_CHARS = 1200
TYPES = list(get_args(QuestionType))


class Reviewer:
    def __init__(
        self,
        questions_path: Path,
        log_path: Path,
        articles: dict[str, Article],
        ask: InputFn = input,
        say: PrintFn = print,
    ) -> None:
        self.questions_path = questions_path
        self.log_path = log_path
        self.articles = articles
        self.ask = ask
        self.say = say

    def check_gold(self, ids: list[str]) -> list[str]:
        return [i for i in ids if i not in self.articles]

    def show(self, q: Question, n: int, total: int, full: bool = False) -> None:
        self.say(f"\n[{n}/{total}] {q.id}  type={q.type}  source={q.source}  split={q.split}")
        self.say(f"Q: {q.question}")
        if q.notes:
            self.say(f"Notes: {q.notes}")
        if not q.gold_article_ids:
            self.say("Gold: none (out of scope)")
        for gold in q.gold_article_ids:
            article = self.articles.get(gold)
            if article is None:
                self.say(f"Gold: {gold}  (NOT FOUND in the corpus)")
                continue
            self.say(f"Gold: {gold}  Article {article.article_number} - {article.title}")
            text = article.text if full else article.text[:PREVIEW_CHARS]
            self.say(text + ("" if full or len(article.text) <= PREVIEW_CHARS else " [...]"))

    def review(self, source: str | None = None) -> None:
        questions = load_questions(self.questions_path)
        pending = [
            q.id
            for q in questions
            if q.status == "pending" and (source is None or q.source == source)
        ]
        if not pending:
            self.say("Nothing pending.")
            return
        for n, qid in enumerate(pending, start=1):
            if not self._review_one(qid, n, len(pending)):
                break
        self.say("Saved.")

    def _review_one(self, qid: str, n: int, total: int) -> bool:
        """Returns False when the user quits."""
        questions = load_questions(self.questions_path)
        index = next(i for i, q in enumerate(questions) if q.id == qid)
        q = questions[index]
        edited: list[str] = []
        self.show(q, n, total)
        while True:
            choice = (
                self.ask(
                    "[v]alidate [r]eject [e]dit question [g]old ids [t]ype [n]otes "
                    "[f]ull text [s]kip [q]uit > "
                )
                .strip()
                .lower()
            )
            if choice in ("v", "r"):
                missing = self.check_gold(q.gold_article_ids)
                if choice == "v" and missing:
                    self.say(f"Cannot validate: unknown article ids {missing}")
                    continue
                status = "validated" if choice == "v" else "rejected"
                q = q.model_copy(update={"status": status})
                questions[index] = q
                save_questions(questions, self.questions_path)
                log_decision(q, status, edited, self.log_path)
                return True
            if choice == "e":
                text = self.ask("New question: ").strip()
                if text:
                    q = q.model_copy(update={"question": text})
                    edited.append("question")
            elif choice == "g":
                q = self._edit_gold(q, edited)
            elif choice == "t":
                q = self._edit_type(q, edited)
            elif choice == "n":
                q = q.model_copy(update={"notes": self.ask("Notes: ").strip()})
                edited.append("notes")
            elif choice == "f":
                self.show(q, n, total, full=True)
                continue
            elif choice == "s":
                return True
            elif choice == "q":
                return False
            else:
                continue
            questions[index] = q
            save_questions(questions, self.questions_path)
            self.show(q, n, total)

    def _edit_gold(self, q: Question, edited: list[str]) -> Question:
        raw = self.ask("Gold ids, comma separated (empty for out of scope): ")
        ids = [i.strip() for i in raw.split(",") if i.strip()]
        missing = self.check_gold(ids)
        if missing:
            self.say(f"Unknown article ids {missing}, not changed")
            return q
        # keep the type unless it no longer fits; [t] changes it afterwards
        new_type = q.type
        if not ids:
            new_type = "out_of_scope"
        elif q.type == "out_of_scope":
            new_type = "factual"
        try:
            q = Question.model_validate(
                {**q.model_dump(), "gold_article_ids": ids, "type": new_type}
            )
        except ValueError as exc:
            self.say(f"Not changed: {exc}")
            return q
        edited.append("gold_article_ids")
        return q

    def _edit_type(self, q: Question, edited: list[str]) -> Question:
        raw = self.ask(f"Type {TYPES}: ").strip()
        try:
            q = Question.model_validate({**q.model_dump(), "type": raw})
        except ValueError as exc:
            self.say(f"Not changed: {exc}")
            return q
        edited.append("type")
        return q

    def add(self) -> Question | None:
        questions = load_questions(self.questions_path)
        text = self.ask("Question: ").strip()
        if not text:
            return None
        source = self.ask("Source [human/faq] (default human): ").strip() or "human"
        notes = self.ask("Notes (for faq: where it is from and its license): ").strip()
        raw = self.ask("Gold ids, comma separated (empty for out of scope): ")
        ids = [i.strip() for i in raw.split(",") if i.strip()]
        missing = self.check_gold(ids)
        if missing:
            self.say(f"Unknown article ids {missing}, nothing added")
            return None
        default_type = (
            "out_of_scope" if not ids else ("multi_article" if len(ids) > 1 else "factual")
        )
        qtype = self.ask(f"Type {TYPES} (default {default_type}): ").strip() or default_type
        try:
            q = Question(
                id=next_id(questions),
                question=text,
                gold_article_ids=ids,
                type=qtype,  # type: ignore[arg-type]
                source=source,  # type: ignore[arg-type]
                status="validated",  # written and checked by the human
                notes=notes,
            )
        except ValueError as exc:
            self.say(f"Not added: {exc}")
            return None
        save_questions([*questions, q], self.questions_path)
        log_decision(q, "added", [], self.log_path)
        self.say(f"Added {q.id}")
        return q


def assign_splits(path: Path, test_fraction: float, seed: int) -> Counter[tuple[str, str]]:
    questions = load_questions(path)
    updated = [
        q.model_copy(update={"split": hash_split(q.id, test_fraction, seed)})
        if q.status == "validated"
        else q
        for q in questions
    ]
    save_questions(updated, path)
    return Counter((q.type, q.split) for q in updated if q.status == "validated")


def stats(path: Path, log_path: Path) -> list[str]:
    questions = load_questions(path)
    lines = [f"{len(questions)} questions in {path}"]
    for field in ("status", "source", "type", "split"):
        counts = Counter(str(getattr(q, field)) for q in questions)
        lines.append(f"  {field}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    validated = [q for q in questions if q.status == "validated"]
    in_scope = sum(q.in_scope for q in validated)
    human_core = sum(q.source in ("human", "faq") for q in validated)
    lines.append(
        f"  validated: {len(validated)} ({in_scope} in scope, {len(validated) - in_scope} "
        f"out of scope, {human_core} human/faq)"
    )
    rejected, reviewed = candidate_rejection_rate(read_log(log_path))
    rate = f"{rejected / reviewed:.0%}" if reviewed else "n/a"
    lines.append(f"  LLM candidates reviewed: {reviewed}, rejected: {rejected} ({rate})")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eurolex.evaluation.review_cli")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--log", type=Path, default=REVIEW_LOG_PATH)
    parser.add_argument("--articles", type=Path, default=Path("data/processed/articles.jsonl"))
    sub = parser.add_subparsers(dest="command", required=True)
    review = sub.add_parser("review")
    review.add_argument("--source", choices=["human", "llm_candidate", "faq"])
    sub.add_parser("add")
    split = sub.add_parser("split")
    split.add_argument("--test-fraction", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=42)
    sub.add_parser("stats")
    args = parser.parse_args(argv)

    if args.command == "stats":
        print("\n".join(stats(args.questions, args.log)))
        return 0
    if args.command == "split":
        counts = assign_splits(args.questions, args.test_fraction, args.seed)
        for (qtype, split_name), n in sorted(counts.items()):
            print(f"  {qtype:<14} {split_name:<5} {n}")
        return 0
    reviewer = Reviewer(args.questions, args.log, load_articles(args.articles))
    try:
        if args.command == "review":
            reviewer.review(args.source)
        else:
            reviewer.add()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Decisions so far are saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
