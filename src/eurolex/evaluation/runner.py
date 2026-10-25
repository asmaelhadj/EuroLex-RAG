"""`make eval CONFIG=configs/e3_hybrid.yaml SPLIT=dev`

Runs one experiment over the validated questions of a split and saves
runs/<run_id>/results.json (everything) and runs/<run_id>/summary.json (no per-question data).
"""

import argparse
import hashlib
import json
import logging
import platform
import random
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eurolex.config import (
    ConfigError,
    ExperimentConfig,
    IngestConfig,
    get_settings,
    load_experiment_config,
    load_ingest_config,
)
from eurolex.evaluation.metrics import mean, ndcg_at_k, percentile, recall_at_k, reciprocal_rank
from eurolex.evaluation.questions import QUESTIONS_PATH, load_questions
from eurolex.retrieval.pipeline import RetrievalPipeline, build_pipeline
from eurolex.schemas import Question, Split

logger = logging.getLogger(__name__)

RECALL_KS = (1, 3, 5, 10)
METRICS = [*(f"recall@{k}" for k in RECALL_KS), "mrr", "ndcg@10"]
HUMAN_SOURCES = ("human", "faq")


def select_questions(
    questions: Sequence[Question], split: Split, *, include_pending: bool
) -> list[Question]:
    allowed = {"validated", "pending"} if include_pending else {"validated"}
    return [q for q in questions if q.split == split and q.status in allowed]


def score_question(q: Question, retrieved: Sequence[str | None]) -> dict[str, float]:
    gold = q.gold_article_ids
    scores = {f"recall@{k}": recall_at_k(retrieved, gold, k) for k in RECALL_KS}
    scores["mrr"] = reciprocal_rank(retrieved, gold)
    scores["ndcg@10"] = ndcg_at_k(retrieved, gold, 10)
    return scores


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r["metrics"] is not None]
    out: dict[str, Any] = {"n": len(scored)}
    for name in METRICS:
        value = mean([r["metrics"][name] for r in scored])
        out[name] = round(value, 4) if value is not None else None
    return out


def latency_summary(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    stages = sorted({s for r in rows for s in r["latency_ms"]})
    return {
        s: {
            "p50": percentile([r["latency_ms"][s] for r in rows if s in r["latency_ms"]], 50),
            "p95": percentile([r["latency_ms"][s] for r in rows if s in r["latency_ms"]], 95),
        }
        for s in stages
    }


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        cmd = ["git", *args]
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()  # noqa: S603

    return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}


def device_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {"torch": torch.__version__, "python": platform.python_version()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def set_seeds(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(
    experiment: ExperimentConfig,
    ingest: IngestConfig,
    questions: Sequence[Question],
    split: Split,
    pipeline: RetrievalPipeline,
    *,
    include_pending: bool = False,
    limit: int | None = None,
    seed: int = 42,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    selected = select_questions(questions, split, include_pending=include_pending)
    if limit is not None:
        selected = selected[:limit]
    rows: list[dict[str, Any]] = []
    for i, q in enumerate(selected, start=1):
        result = pipeline.run(q.question)
        retrieved = [h.chunk.article_id for h in result.hits]
        rows.append(
            {
                "id": q.id,
                "type": q.type,
                "source": q.source,
                "status": q.status,
                "gold": q.gold_article_ids,
                "rewritten_query": result.rewritten_query,
                "retrieved": [
                    {
                        "chunk_id": h.chunk.chunk_id,
                        "article_id": h.chunk.article_id,
                        "score": h.score,
                    }
                    for h in result.hits
                ],
                "top_score": result.hits[0].score if result.hits else None,
                "metrics": score_question(q, retrieved) if q.in_scope else None,
                "latency_ms": result.latency_ms,
            }
        )
        if progress:
            progress(i, len(selected))

    in_scope = [r for r in rows if r["type"] != "out_of_scope"]
    types = sorted({r["type"] for r in in_scope})
    not_reportable = []
    if include_pending:
        not_reportable.append("includes unreviewed (pending) questions")
    if limit is not None:
        not_reportable.append(f"limited to {limit} questions")
    if not in_scope:
        not_reportable.append("no in-scope questions")

    timestamp = datetime.now(UTC)
    question_hash = hashlib.sha256(
        "\n".join(q.model_dump_json() for q in selected).encode("utf-8")
    ).hexdigest()
    return {
        "run_id": f"{timestamp:%Y%m%d-%H%M%S}_{experiment.name}_{split}",
        "experiment": experiment.name,
        "description": experiment.description,
        "split": split,
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "reportable": not not_reportable,
        "not_reportable_because": not_reportable,
        "git": git_info(),
        "seed": seed,
        "device": device_info(),
        "config": {
            "experiment": experiment.model_dump(mode="json"),
            "embedding": ingest.embedding.model_dump(mode="json"),
            "chunking": ingest.chunking.model_dump(mode="json"),
        },
        "question_set": {
            "sha256": question_hash,
            "n": len(rows),
            "n_in_scope": len(in_scope),
            "n_out_of_scope": len(rows) - len(in_scope),
            "by_source": {
                s: sum(r["source"] == s for r in rows) for s in sorted({r["source"] for r in rows})
            },
        },
        "metrics": {
            "all": aggregate(in_scope),
            "human_core": aggregate([r for r in in_scope if r["source"] in HUMAN_SOURCES]),
            "by_type": {t: aggregate([r for r in in_scope if r["type"] == t]) for t in types},
        },
        "out_of_scope_top_score": {
            "mean": mean(
                [
                    r["top_score"]
                    for r in rows
                    if r["type"] == "out_of_scope" and r["top_score"] is not None
                ]
            ),
        },
        "latency_ms": latency_summary(rows),
        "questions": rows,
    }


def save_run(results: dict[str, Any], runs_dir: Path) -> Path:
    run_dir: Path = runs_dir / str(results["run_id"])
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in results.items() if k != "questions"}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return run_dir


def format_summary(results: dict[str, Any]) -> list[str]:
    def row(label: str, m: dict[str, Any]) -> str:
        cells = [f"{m[name]:.3f}" if m[name] is not None else "  -  " for name in METRICS]
        return f"{label:<16} {m['n']:>4}  " + "  ".join(f"{c:>9}" for c in cells)

    lines = [
        f"Run {results['run_id']}"
        + (
            ""
            if results["reportable"]
            else "  (NOT REPORTABLE: " + "; ".join(results["not_reportable_because"]) + ")"
        ),
        f"{'':<16} {'n':>4}  " + "  ".join(f"{m:>9}" for m in METRICS),
        row("all", results["metrics"]["all"]),
        row("human/faq core", results["metrics"]["human_core"]),
    ]
    lines += [row(f"  {t}", m) for t, m in results["metrics"]["by_type"].items()]
    total = results["latency_ms"].get("total", {})
    if total:
        lines.append(f"latency total: p50 {total['p50']:.0f} ms, p95 {total['p95']:.0f} ms")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eurolex.evaluation.runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument(
        "--include-pending", action="store_true", help="smoke test only; run is not reportable"
    )
    parser.add_argument("--limit", type=int, help="smoke test only; run is not reportable")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    try:
        experiment = load_experiment_config(args.config)
        ingest = load_ingest_config(experiment.ingest_config)
    except (ConfigError, OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    questions = load_questions(args.questions)
    selected = select_questions(questions, args.split, include_pending=args.include_pending)
    if not selected:
        print(
            f"No usable questions for split '{args.split}' in {args.questions}. Build the question "
            "set first (SPEC.md HUMAN GATE 1): candidates.py, then review_cli.py.",
            file=sys.stderr,
        )
        return 1
    set_seeds(settings.random_seed)
    pipeline = build_pipeline(experiment.retrieval, ingest)

    def progress(i: int, n: int) -> None:
        print(f"\r{i}/{n} questions", end="", file=sys.stderr, flush=True)

    results = run_experiment(
        experiment,
        ingest,
        questions,
        args.split,
        pipeline,
        include_pending=args.include_pending,
        limit=args.limit,
        seed=settings.random_seed,
        progress=progress,
    )
    print(file=sys.stderr)
    run_dir = save_run(results, args.runs_dir or settings.runs_dir)
    print("\n".join(format_summary(results)))
    print(f"Saved to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
