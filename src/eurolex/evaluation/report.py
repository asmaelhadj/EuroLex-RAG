"""`make report`: rebuild docs/results.md from runs/*/summary.json.

Only reportable runs are used (validated questions only, no --limit). For each experiment
and split the latest such run is shown. Missing numbers are written as TODO, never guessed.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eurolex.evaluation.questions import REVIEW_LOG_PATH, candidate_rejection_rate, read_log

COLUMNS = [
    ("recall@1", "R@1"),
    ("recall@5", "R@5"),
    ("recall@10", "R@10"),
    ("mrr", "MRR"),
    ("ndcg@10", "nDCG@10"),
]


def load_summaries(runs_dir: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(runs_dir.glob("*/summary.json")):
        with path.open(encoding="utf-8") as fh:
            summaries.append(json.load(fh))
    return summaries


def latest_reportable(summaries: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for s in sorted(summaries, key=lambda s: str(s["timestamp"])):
        if s.get("reportable"):
            latest[(str(s["experiment"]), str(s["split"]))] = s
    return latest


def _cell(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "TODO"


def _metric_row(label: str, m: dict[str, Any], extra: list[str]) -> str:
    cells = [label, str(m.get("n", 0)), *[_cell(m.get(key)) for key, _ in COLUMNS], *extra]
    return "| " + " | ".join(cells) + " |"


GEN_COLUMNS = [
    ("refusal_accuracy_out_of_scope", "Refusal acc. (out of scope)"),
    ("false_refusal_rate_in_scope", "False refusals (in scope)"),
    ("citation_validity", "Citation validity"),
    ("cites_gold", "Cites a gold article"),
    ("faithfulness_mean", "Faithfulness (1-5)"),
    ("relevance_mean", "Relevance (1-5)"),
]


def generation_table(chosen: list[dict[str, Any]]) -> list[str]:
    with_gen = [s for s in chosen if s.get("generation")]
    if not with_gen:
        return []
    lines = ["", "Generation (judge: see config; scores 1-5 from an LLM judge):", ""]
    lines += ["| Experiment | " + " | ".join(h for _, h in GEN_COLUMNS) + " | judged |"]
    lines += ["|" + "---|" * (len(GEN_COLUMNS) + 2)]
    for s in with_gen:
        g = s["generation"]
        cells = [_cell(g.get(key)) for key, _ in GEN_COLUMNS]
        lines.append(f"| {s['experiment']} | " + " | ".join(cells) + f" | {g.get('judged', 0)} |")
    return lines


def split_section(split: str, runs: dict[tuple[str, str], dict[str, Any]]) -> list[str]:
    chosen = sorted(
        (s for (_, sp), s in runs.items() if sp == split), key=lambda s: s["experiment"]
    )
    title = "Test split (final numbers)" if split == "test" else "Dev split (used for tuning)"
    lines = [f"## {title}", ""]
    if not chosen:
        return [
            *lines,
            f"TODO: no reportable runs on the {split} split yet. Run "
            f"`make eval CONFIG=configs/<experiment>.yaml SPLIT={split}`.",
            "",
        ]
    header = "| Experiment | n | " + " | ".join(h for _, h in COLUMNS) + " | p50 latency |"
    sep = "|" + "---|" * (len(COLUMNS) + 3)
    lines += ["All in-scope questions:", "", header, sep]
    for s in chosen:
        p50 = s["latency_ms"].get("total", {}).get("p50")
        latency = f"{p50 / 1000:.1f} s" if isinstance(p50, int | float) else "TODO"
        lines.append(_metric_row(f"{s['experiment']}", s["metrics"]["all"], [latency]))
    lines += [
        "",
        "Human-written / FAQ core only:",
        "",
        header.replace(" p50 latency |", ""),
        "|" + "---|" * (len(COLUMNS) + 2),
    ]
    for s in chosen:
        core = s["metrics"]["human_core"]
        lines.append(_metric_row(s["experiment"], core, []))
    lines += ["", "By question type (nDCG@10):", ""]
    types = sorted({t for s in chosen for t in s["metrics"]["by_type"]})
    lines += ["| Experiment | " + " | ".join(types) + " |", "|" + "---|" * (len(types) + 1)]
    for s in chosen:
        by_type = s["metrics"]["by_type"]
        cells = [
            f"{_cell(by_type[t]['ndcg@10'])} (n={by_type[t]['n']})" if t in by_type else "-"
            for t in types
        ]
        lines.append(f"| {s['experiment']} | " + " | ".join(cells) + " |")
    lines += generation_table(chosen)
    lines += ["", "Runs used:", ""]
    for s in chosen:
        commit = (s["git"].get("commit") or "unknown")[:8] + (
            " (dirty)" if s["git"].get("dirty") else ""
        )
        gpu = s["device"].get("gpu", "CPU")
        lines.append(
            f"- `{s['run_id']}`: commit {commit}, {gpu}, question set "
            f"`{s['question_set']['sha256'][:12]}`"
        )
    return [*lines, ""]


def question_set_section(runs: dict[tuple[str, str], dict[str, Any]], log_path: Path) -> list[str]:
    lines = ["## Question set", ""]
    for split in ("dev", "test"):
        sets = {
            s["question_set"]["sha256"]: s["question_set"]
            for (_, sp), s in runs.items()
            if sp == split
        }
        if not sets:
            lines.append(f"- {split}: TODO (no reportable runs)")
            continue
        if len(sets) > 1:
            lines.append(f"- {split}: WARNING, runs used {len(sets)} different question sets")
        qs = next(iter(sets.values()))
        sources = ", ".join(f"{k}: {v}" for k, v in qs["by_source"].items())
        lines.append(
            f"- {split}: {qs['n']} questions ({qs['n_in_scope']} in scope, "
            f"{qs['n_out_of_scope']} out of scope); by source: {sources}"
        )
    rejected, reviewed = candidate_rejection_rate(read_log(log_path))
    rate = f"{rejected}/{reviewed} ({rejected / reviewed:.0%})" if reviewed else "TODO"
    lines += [f"- LLM candidates rejected in review: {rate}", "- Annotators: TODO (fill in)", ""]
    return lines


def build_report(runs_dir: Path, log_path: Path) -> str:
    summaries = load_summaries(runs_dir)
    runs = latest_reportable(summaries)
    skipped = sum(1 for s in summaries if not s.get("reportable"))
    lines = [
        "# Results",
        "",
        "Generated by `make report` from `runs/*/summary.json`. Do not edit by hand.",
        "Retrieval metrics are article-level (see `src/eurolex/evaluation/metrics.py`); "
        "out-of-scope questions are excluded from them.",
        "",
        *question_set_section(runs, log_path),
        *split_section("test", runs),
        *split_section("dev", runs),
    ]
    if skipped:
        lines += [
            f"{skipped} smoke-test run(s) were ignored (unreviewed questions or --limit).",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eurolex.evaluation.report")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--log", type=Path, default=REVIEW_LOG_PATH)
    parser.add_argument("--out", type=Path, default=Path("docs/results.md"))
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_report(args.runs_dir, args.log), encoding="utf-8", newline="\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
