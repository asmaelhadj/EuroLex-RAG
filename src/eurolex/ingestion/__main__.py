"""`python -m eurolex.ingestion`: download, parse and chunk the corpus (indexing: Phase 2)."""

import argparse
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from eurolex.config import IngestConfig, load_ingest_config
from eurolex.ingestion.chunk import ArticleChunker, Chunker, FixedSizeChunker
from eurolex.ingestion.download import DownloadError, download_acts
from eurolex.ingestion.parse import (
    ParseError,
    check_counts,
    load_soup,
    parse_articles,
    parse_recitals,
)
from eurolex.schemas import Article, Recital

logger = logging.getLogger("eurolex.ingestion")


def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(row.model_dump_json() + "\n")
            count += 1
    return count


def parse_corpus(config: IngestConfig) -> tuple[list[Article], list[Recital]]:
    articles: list[Article] = []
    recitals: list[Recital] = []
    for act in config.acts:
        path = config.raw_dir / act.raw_filename
        if not path.exists():
            raise ParseError(f"{path} is missing; run the download step or place it manually")
        soup = load_soup(path)
        source_url = config.download.source_url(act.celex)
        act_articles = parse_articles(
            soup, act_id=act.act_id, celex=act.celex, source_url=source_url
        )
        act_recitals = parse_recitals(
            soup, act_id=act.act_id, celex=act.celex, source_url=source_url
        )
        check_counts(
            act.act_id,
            act_articles,
            act_recitals,
            expected_articles=act.expected_articles,
            expected_recitals=act.expected_recitals,
        )
        logger.info(
            "%s: %d articles, %d recitals", act.act_id, len(act_articles), len(act_recitals)
        )
        articles += act_articles
        recitals += act_recitals
    return articles, recitals


def build_chunkers(config: IngestConfig) -> list[Chunker]:
    return [ArticleChunker(config.chunking.article), FixedSizeChunker(config.chunking.fixed)]


def run(config: IngestConfig, *, force_download: bool) -> None:
    download_acts(config.acts, config.raw_dir, config.download, force=force_download)
    articles, recitals = parse_corpus(config)
    out = config.processed_dir
    write_jsonl(out / "articles.jsonl", articles)
    write_jsonl(out / "recitals.jsonl", recitals)
    for chunker in build_chunkers(config):
        chunks = chunker.chunk(articles, recitals)
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ParseError(f"{chunker.name} chunker produced duplicate chunk ids")
        n = write_jsonl(out / f"chunks_{chunker.name}.jsonl", chunks)
        logger.info("%s chunker: %d chunks -> %s", chunker.name, n, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eurolex.ingestion", description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ingest.yaml"))
    parser.add_argument("--force-download", action="store_true", help="re-download raw files")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        run(load_ingest_config(args.config), force_download=args.force_download)
    except (DownloadError, ParseError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
