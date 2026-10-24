"""`python -m eurolex.retrieval --config configs/e3_hybrid.yaml --query "..."`"""

import argparse
import logging
import sys
from pathlib import Path

from eurolex.config import ConfigError, load_experiment_config, load_ingest_config
from eurolex.retrieval.pipeline import build_pipeline
from eurolex.schemas import RetrievalResult


def print_result(result: RetrievalResult) -> None:
    print(f"\nQuery: {result.query}")
    if result.rewritten_query:
        print(f"Rewritten: {result.rewritten_query}")
    for rank, hit in enumerate(result.hits, start=1):
        snippet = hit.chunk.text.replace("\n", " ")[:140]
        print(f"{rank:>2}. {hit.score:.4f}  {hit.chunk.chunk_id:<24} {snippet}")
    print("Latency (ms): " + ", ".join(f"{k}={v}" for k, v in result.latency_ms.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eurolex.retrieval")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--query", required=True, action="append", help="can be repeated")
    parser.add_argument("--act-id")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        experiment = load_experiment_config(args.config)
        ingest = load_ingest_config(experiment.ingest_config)
        pipeline = build_pipeline(experiment.retrieval, ingest)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Experiment: {experiment.name} - {experiment.description}")
    for query in args.query:
        print_result(pipeline.run(query, act_id=args.act_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
