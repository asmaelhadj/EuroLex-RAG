"""Runner and report tests on a synthetic BM25 corpus. Questions here are test data only."""

import json
from pathlib import Path

import pytest

from eurolex.config import ExperimentConfig, RetrievalConfig, RetrieverSpec, load_ingest_config
from eurolex.evaluation.report import build_report
from eurolex.evaluation.runner import format_summary, run_experiment, save_run, select_questions
from eurolex.ingestion.__main__ import write_jsonl
from eurolex.retrieval.pipeline import build_pipeline
from eurolex.schemas import Chunk, Question

ROOT = Path(__file__).parents[1]


def chunk(n: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"gdpr:art-{n}",
        act_id="gdpr",
        celex="x",
        kind="article",
        article_number=n,
        article_title="t",
        paragraph=None,
        chapter=None,
        text=text,
        source_url="https://example.test",
        char_count=len(text),
    )


CHUNKS = [chunk("1", "alpha bravo"), chunk("2", "charlie delta"), chunk("3", "echo foxtrot")]


def question(qid: str, text: str, gold: list[str], **kw: object) -> Question:
    row: dict[str, object] = {
        "id": qid,
        "question": text,
        "gold_article_ids": gold,
        "type": "factual" if gold else "out_of_scope",
        "source": "human",
        "status": "validated",
        "split": "dev",
    }
    return Question.model_validate({**row, **kw})


QUESTIONS = [
    question("q-0001", "alpha?", ["gdpr:art-1"]),  # found at rank 1
    question("q-0002", "delta?", ["gdpr:art-1"], source="llm_candidate", type="definition"),  # miss
    question("q-0003", "zulu yankee?", []),  # out of scope
    question("q-0004", "alpha?", ["gdpr:art-1"], status="pending"),
    question("q-0005", "alpha?", ["gdpr:art-1"], status="rejected"),
    question("q-0006", "alpha?", ["gdpr:art-1"], split="test"),
]

EXPERIMENT = ExperimentConfig(
    name="e2_bm25",
    description="BM25 only",
    retrieval=RetrievalConfig(
        collection="unused", chunker="article", retrievers=[RetrieverSpec(type="bm25")], top_k=3
    ),
)


@pytest.fixture
def results(tmp_path: Path) -> dict[str, object]:
    ingest = load_ingest_config(ROOT / "configs" / "ingest.yaml").model_copy(
        update={"processed_dir": tmp_path}
    )
    write_jsonl(tmp_path / "chunks_article.jsonl", CHUNKS)
    pipeline = build_pipeline(EXPERIMENT.retrieval, ingest)
    return run_experiment(EXPERIMENT, ingest, QUESTIONS, "dev", pipeline)


def test_select_questions() -> None:
    assert [q.id for q in select_questions(QUESTIONS, "dev", include_pending=False)] == [
        "q-0001",
        "q-0002",
        "q-0003",
    ]
    with_pending = select_questions(QUESTIONS, "dev", include_pending=True)
    assert [q.id for q in with_pending] == ["q-0001", "q-0002", "q-0003", "q-0004"]


def test_metrics_exclude_out_of_scope(results: dict) -> None:  # type: ignore[type-arg]
    assert results["reportable"] is True
    assert results["question_set"]["n"] == 3
    assert results["question_set"]["n_out_of_scope"] == 1
    overall = results["metrics"]["all"]
    assert overall["n"] == 2
    assert overall["recall@1"] == 0.5  # one hit, one miss
    assert overall["mrr"] == 0.5
    assert results["metrics"]["human_core"]["n"] == 1
    assert results["metrics"]["human_core"]["recall@1"] == 1.0
    assert set(results["metrics"]["by_type"]) == {"factual", "definition"}
    oos = next(r for r in results["questions"] if r["id"] == "q-0003")
    assert oos["metrics"] is None
    assert "bm25" in results["latency_ms"]
    assert results["config"]["experiment"]["name"] == "e2_bm25"
    assert results["seed"] == 42


def test_pending_or_limit_is_not_reportable(tmp_path: Path) -> None:
    ingest = load_ingest_config(ROOT / "configs" / "ingest.yaml").model_copy(
        update={"processed_dir": tmp_path}
    )
    write_jsonl(tmp_path / "chunks_article.jsonl", CHUNKS)
    pipeline = build_pipeline(EXPERIMENT.retrieval, ingest)
    run = run_experiment(
        EXPERIMENT, ingest, QUESTIONS, "dev", pipeline, include_pending=True, limit=2
    )
    assert run["reportable"] is False
    assert len(run["not_reportable_because"]) == 2
    assert run["question_set"]["n"] == 2
    assert "NOT REPORTABLE" in format_summary(run)[0]


def test_save_run_and_report(tmp_path: Path, results: dict) -> None:  # type: ignore[type-arg]
    run_dir = save_run(results, tmp_path / "runs")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert "questions" not in summary
    full = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert len(full["questions"]) == 3

    smoke = {**summary, "run_id": "smoke", "reportable": False, "timestamp": "2099-01-01"}
    (tmp_path / "runs" / "smoke").mkdir()
    (tmp_path / "runs" / "smoke" / "summary.json").write_text(json.dumps(smoke), encoding="utf-8")

    report = build_report(tmp_path / "runs", tmp_path / "no_log.jsonl")
    assert "| e2_bm25 | 2 | 0.500 |" in report  # dev table, real numbers from the run
    assert "no reportable runs on the test split" in report
    assert "1 smoke-test run(s) were ignored" in report
    assert "LLM candidates rejected in review: TODO" in report


def test_report_without_runs_is_all_todo(tmp_path: Path) -> None:
    report = build_report(tmp_path / "runs", tmp_path / "log.jsonl")
    assert "TODO: no reportable runs on the test split" in report
    assert "TODO: no reportable runs on the dev split" in report
    assert "0." not in report  # no numbers at all
