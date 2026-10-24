import json
import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from eurolex.config import IngestConfig, load_ingest_config
from eurolex.ingestion.__main__ import main
from eurolex.schemas import Chunk

ROOT = Path(__file__).parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "gdpr_sample.html"


def test_repo_config_is_valid() -> None:
    config = load_ingest_config(ROOT / "configs" / "ingest.yaml")
    assert [a.act_id for a in config.acts] == ["gdpr", "ai_act"]
    assert [a.act_id for a in config.enabled_acts] == ["gdpr"]
    assert config.acts[0].expected_articles == 99


def test_config_needs_an_enabled_act() -> None:
    data = yaml.safe_load((ROOT / "configs" / "ingest.yaml").read_text(encoding="utf-8"))
    data["acts"] = [{**act, "enabled": False} for act in data["acts"]]
    with pytest.raises(ValidationError, match="at least one act must be enabled"):
        IngestConfig.model_validate(data)


def _write_config(tmp_path: Path, expected_articles: int) -> Path:
    data = yaml.safe_load((ROOT / "configs" / "ingest.yaml").read_text(encoding="utf-8"))
    data["raw_dir"] = str(tmp_path / "raw")
    data["processed_dir"] = str(tmp_path / "processed")
    data["acts"] = [
        {**data["acts"][0], "expected_articles": expected_articles, "expected_recitals": 2}
    ]
    path = tmp_path / "ingest.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_end_to_end_on_fixture(tmp_path: Path) -> None:
    # A file already in raw_dir is used as-is, so this runs offline (the manual path).
    (tmp_path / "raw").mkdir()
    shutil.copy(FIXTURE, tmp_path / "raw" / "gdpr_en.html")
    assert main(["--config", str(_write_config(tmp_path, expected_articles=3))]) == 0

    out = tmp_path / "processed"
    assert len((out / "articles.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    for name in ("chunks_article.jsonl", "chunks_fixed.jsonl"):
        rows = [json.loads(line) for line in (out / name).read_text(encoding="utf-8").splitlines()]
        chunks = [Chunk.model_validate(r) for r in rows]
        assert len({c.chunk_id for c in chunks}) == len(chunks)
        assert chunks[0].source_url.startswith("https://eur-lex.europa.eu/")


def test_count_mismatch_fails(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    shutil.copy(FIXTURE, tmp_path / "raw" / "gdpr_en.html")
    assert main(["--config", str(_write_config(tmp_path, expected_articles=99))]) == 1
