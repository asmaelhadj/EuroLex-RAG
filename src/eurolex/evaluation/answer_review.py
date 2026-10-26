"""Human check of generated answers (SPEC.md section 10, task 3).

    python -m eurolex.evaluation.answer_review export runs/<e7 run>     # writes the CSV
    python -m eurolex.evaluation.answer_review summary                  # counts your verdicts

The CSV has the question, the answer and the text of every cited passage. Fill in
`citations_correct` and `answer_correct` with yes / partly / no, plus notes if needed.
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REVIEW_PATH = Path("data/eval/answer_review.csv")
VERDICTS = ("yes", "partly", "no")
COLUMNS = [
    "id",
    "type",
    "source",
    "gold",
    "question",
    "answer",
    "cited_passages",
    "invalid_citations",
    "citations_correct",
    "answer_correct",
    "notes",
]


def load_chunks(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {r["chunk_id"]: r for r in rows}


def sample_rows(results: dict[str, Any], n: int, seed: int) -> list[dict[str, Any]]:
    """In-scope questions the model actually answered, picked with a fixed seed."""
    answered = [
        r
        for r in results["questions"]
        if r["type"] != "out_of_scope" and r.get("answer") and not r["answer"]["refused"]
    ]
    rng = random.Random(seed)  # noqa: S311  (sampling, not security)
    return sorted(rng.sample(answered, min(n, len(answered))), key=lambda r: r["id"])


def export(run_dir: Path, out: Path, n: int, seed: int, chunks_file: Path) -> int:
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    if out.exists():
        print(f"{out} exists; not overwriting your review.", file=sys.stderr)
        return 1
    questions = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in Path("data/eval/questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    chunks = load_chunks(chunks_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in sample_rows(results, n, seed):
            cited = [
                f"[{c['n']}] {c['chunk_id']}: {chunks[c['chunk_id']]['text'][:600]}"
                for c in row["answer"]["citations"]
            ]
            writer.writerow(
                {
                    "id": row["id"],
                    "type": row["type"],
                    "source": row["source"],
                    "gold": " ".join(row["gold"]),
                    "question": questions[row["id"]],
                    "answer": row["answer"]["text"],
                    "cited_passages": "\n\n".join(cited) or "(no citations)",
                    "invalid_citations": " ".join(map(str, row["answer"]["invalid_citations"])),
                    "citations_correct": "",
                    "answer_correct": "",
                    "notes": "",
                }
            )
    print(f"Wrote {out} from run {results['run_id']}. Fill in the two verdict columns.")
    return 0


def summary(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    lines = [f"{len(rows)} answers in {path}"]
    for column in ("citations_correct", "answer_correct"):
        values = [r[column].strip().lower() for r in rows]
        bad = [v for v in values if v and v not in VERDICTS]
        if bad:
            raise ValueError(f"{column}: use {'/'.join(VERDICTS)}, got {sorted(set(bad))}")
        counts = Counter(v or "not reviewed" for v in values)
        lines.append(f"  {column}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eurolex.evaluation.answer_review")
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("run_dir", type=Path)
    exp.add_argument("--n", type=int, default=20)
    exp.add_argument("--seed", type=int, default=42)
    exp.add_argument("--out", type=Path, default=REVIEW_PATH)
    exp.add_argument("--chunks", type=Path, default=Path("data/processed/chunks_article.jsonl"))
    summ = sub.add_parser("summary")
    summ.add_argument("--path", type=Path, default=REVIEW_PATH)
    args = parser.parse_args(argv)
    if args.command == "export":
        return export(args.run_dir, args.out, args.n, args.seed, args.chunks)
    print("\n".join(summary(args.path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
