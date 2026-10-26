"""Pick the refusal threshold from a dev run and plot the score distributions.

    python -m eurolex.evaluation.threshold runs/<dev run of the final retrieval pipeline>

Only dev runs are accepted: the threshold is a tuned value and the test split stays unseen.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eurolex.generation.refusal import choose_threshold


def score_groups(results: dict[str, Any]) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = {
        "answerable": [],
        "in scope, gold missed": [],
        "out of scope": [],
    }
    for row in results["questions"]:
        if row["top_score"] is None:
            continue
        if row["type"] == "out_of_scope":
            groups["out of scope"].append(row["top_score"])
        elif row["metrics"]["recall@10"] > 0:
            groups["answerable"].append(row["top_score"])
        else:
            groups["in scope, gold missed"].append(row["top_score"])
    return groups


def plot(groups: dict[str, list[float]], threshold: float, title: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.2))
    colors = {
        "answerable": "#2a7a3b",
        "in scope, gold missed": "#c98a00",
        "out of scope": "#b23a3a",
    }
    for y, (name, scores) in enumerate(groups.items()):
        ax.scatter(
            scores,
            [y] * len(scores),
            alpha=0.6,
            s=28,
            color=colors[name],
            label=f"{name} (n={len(scores)})",
        )
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1)
    ax.text(threshold, len(groups) - 0.4, f" threshold {threshold:.3f}", fontsize=9, va="bottom")
    ax.set_yticks(range(len(groups)), list(groups))
    ax.set_xlabel("top retrieval score")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-0.6, len(groups) - 0.1)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eurolex.evaluation.threshold")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/figures/refusal_threshold_dev.png"))
    args = parser.parse_args(argv)
    results = json.loads((args.run_dir / "results.json").read_text(encoding="utf-8"))
    if results["split"] != "dev":
        print("Refusing: the threshold must be chosen on the dev split.", file=sys.stderr)
        return 1
    groups = score_groups(results)
    threshold = choose_threshold(groups["answerable"], groups["out of scope"])
    answered = sum(s >= threshold for s in groups["answerable"]) / len(groups["answerable"])
    refused = sum(s < threshold for s in groups["out of scope"]) / len(groups["out of scope"])
    title = f"{results['experiment']} on dev: top score by group"
    plot(groups, threshold, title, args.out)
    print(f"threshold {threshold:.4f}")
    print(f"answerable answered: {answered:.1%}, out of scope refused: {refused:.1%}")
    print(f"plot: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
