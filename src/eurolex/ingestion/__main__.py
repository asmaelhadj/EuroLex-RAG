"""`python -m eurolex.ingestion`: download, parse, chunk, embed and index the corpus."""

import argparse
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel
from qdrant_client import QdrantClient

from eurolex.config import ConfigError, IngestConfig, get_settings, load_ingest_config
from eurolex.embeddings import BGEM3Embedder, Embedder
from eurolex.ingestion.chunk import ArticleChunker, Chunker, FixedSizeChunker
from eurolex.ingestion.download import DownloadError, download_acts
from eurolex.ingestion.index import (
    DENSE,
    SPARSE,
    CollectionMismatchError,
    IndexReport,
    SearchHit,
    ensure_collection,
    search,
    sync_collection,
)
from eurolex.ingestion.parse import (
    ParseError,
    check_counts,
    load_soup,
    parse_articles,
    parse_recitals,
)
from eurolex.schemas import Article, Chunk, Recital

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
    for act in config.enabled_acts:
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


def build_corpus(config: IngestConfig, *, force_download: bool) -> dict[str, list[Chunk]]:
    """Download, parse and chunk; returns chunks keyed by chunker name."""
    download_acts(config.enabled_acts, config.raw_dir, config.download, force=force_download)
    articles, recitals = parse_corpus(config)
    out = config.processed_dir
    write_jsonl(out / "articles.jsonl", articles)
    write_jsonl(out / "recitals.jsonl", recitals)
    corpus: dict[str, list[Chunk]] = {}
    for chunker in build_chunkers(config):
        chunks = chunker.chunk(articles, recitals)
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ParseError(f"{chunker.name} chunker produced duplicate chunk ids")
        n = write_jsonl(out / f"chunks_{chunker.name}.jsonl", chunks)
        logger.info("%s chunker: %d chunks -> %s", chunker.name, n, out)
        corpus[chunker.name] = chunks
    return corpus


def index_corpus(
    config: IngestConfig,
    corpus: dict[str, list[Chunk]],
    embedder: Embedder,
    client: QdrantClient,
    *,
    recreate: bool,
) -> list[IndexReport]:
    act_names = {act.act_id: act.short_name for act in config.acts}
    reports = []
    for collection in config.index.collections:
        ensure_collection(client, collection.name, embedder.dense_dim, recreate=recreate)
        reports.append(
            sync_collection(
                client,
                collection.name,
                corpus[collection.chunker],
                embedder,
                act_names,
                prepend_header=collection.prepend_header,
                batch_size=config.index.upsert_batch_size,
            )
        )
    return reports


def _qdrant_client() -> QdrantClient:
    settings = get_settings()
    key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return QdrantClient(url=settings.qdrant_url, api_key=key)


def _print_hits(query: str, collection: str, using: str, hits: list[SearchHit]) -> None:
    print(f"\nQuery: {query}\nCollection: {collection} ({using})")
    for rank, hit in enumerate(hits, start=1):
        snippet = hit.chunk.text.replace("\n", " ")[:160]
        print(f"{rank}. {hit.score:.4f}  {hit.chunk.chunk_id:<24} {snippet}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eurolex.ingestion", description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ingest.yaml"))
    parser.add_argument("--force-download", action="store_true", help="re-download raw files")
    parser.add_argument("--skip-index", action="store_true", help="stop after chunking")
    parser.add_argument("--recreate", action="store_true", help="drop and rebuild collections")
    parser.add_argument("--query", help="only run a smoke-test search against the index")
    parser.add_argument("--collection", default="eurolex_article")
    parser.add_argument("--using", choices=[DENSE, SPARSE], default=DENSE)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        config = load_ingest_config(args.config)
        if args.query:
            embedder = BGEM3Embedder(config.embedding)
            hits = search(
                _qdrant_client(),
                args.collection,
                embedder,
                args.query,
                using=args.using,
                limit=args.top_k,
            )
            _print_hits(args.query, args.collection, args.using, hits)
            return 0
        corpus = build_corpus(config, force_download=args.force_download)
        if not args.skip_index:
            client = _qdrant_client()
            index_corpus(
                config, corpus, BGEM3Embedder(config.embedding), client, recreate=args.recreate
            )
    except (DownloadError, ParseError, CollectionMismatchError, ConfigError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
