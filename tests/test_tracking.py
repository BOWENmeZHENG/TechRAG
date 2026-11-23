import math
from pathlib import Path

import mlflow
import pytest

from techrag.benchmarks.bench import BenchQuestion
from techrag.common.config import AgentConfig, DataConfig, RetrievalConfig, RunConfig, RunMetadata
from techrag.common.tracking import (
    EVAL_EXPERIMENT,
    agent_trace,
    flatten_metrics,
    flatten_params,
    log_results,
    tracked_eval,
    tracked_run,
)
from techrag.eval.judge import Verdict
from techrag.reasoning.harness import Record


def _config(**overrides) -> RunConfig:
    meta = RunMetadata(run_name="r1", git_sha="abc123", author="Ada")
    return RunConfig(metadata=meta, **overrides)


def _record(qid, qtype, met, gold=1.0, **extra) -> Record:
    verdict = Verdict(tuple(met), "", "") if met is not None else None
    return Record(qid, qtype, "a", ("d1",), gold, 2.0, verdict, **extra)


@pytest.fixture
def store(tmp_path: Path, monkeypatch) -> str:
    """Point MLflow at a throwaway store, so tests never write ``mlflow.db`` in the repo."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(tmp_path)
    return uri


def _runs(store: str, experiment: str = EVAL_EXPERIMENT):
    mlflow.set_tracking_uri(store)
    return mlflow.search_runs(experiment_names=[experiment], output_format="list")


def test_params_are_flat_and_leave_out_metadata_and_question_ids():
    cfg = _config(
        retrieval=RetrievalConfig(mode="bm25", k=10),
        data=DataConfig(seed=3, question_ids=("q1", "q2")),
    )
    params = flatten_params(cfg)
    assert params["retrieval.mode"] == "bm25"
    assert params["retrieval.k"] == 10
    assert params["data.seed"] == 3
    assert params["data.sample_size"] is None
    assert not any(k.startswith("metadata") for k in params)
    assert "data.question_ids" not in params


def test_metrics_cover_overall_groups_and_agent_extras():
    records = [
        _record("a", "basic", [True, True], 1.0),
        _record("b", "basic", [True, False], 0.5),
        _record("c", "info_not_found", [True], None),
        _record("d", "basic", None, 1.0, error="judge: bad"),
    ]
    m = flatten_metrics(records)
    assert m["n_questions"] == 4 and m["errors"] == 1
    assert m["score"] == pytest.approx((1.0 + 0.5 + 1.0) / 3)
    assert m["by_type/basic/score"] == pytest.approx(0.75)
    assert m["by_type/basic/n"] == 3
    assert m["by_retrieval/no_gold_documents/score"] == 1.0
    assert m["gold_found"] == pytest.approx((1.0 + 0.5 + 1.0) / 3)
    assert m["answer_seconds"] == 2.0
    assert not any(k.startswith("stop/") or k == "steps" for k in m)  # single-shot run


def test_metrics_for_agent_records_count_stop_reasons():
    records = [
        _record("a", "basic", [True], steps=3, tokens=100, stop="answered"),
        _record("b", "basic", [False], steps=5, tokens=300, stop="max_steps"),
        _record("c", "basic", [True], steps=1, tokens=50, stop="answered"),
    ]
    m = flatten_metrics(records)
    assert m["steps"] == 3 and m["tokens"] == 150
    assert m["stop/answered"] == 2 and m["stop/max_steps"] == 1


def test_metrics_skip_means_over_nothing():
    m = flatten_metrics([_record("a", "basic", None, error="timeout")])
    assert "score" not in m and m["errors"] == 1
    assert not any(math.isnan(v) for v in m.values())
    assert flatten_metrics([])["n_questions"] == 0


def test_eval_run_records_manifest_params_tags_metrics_and_files(store: str, tmp_path: Path):
    cfg = _config(agent=AgentConfig(max_steps=4), data=DataConfig(question_ids=("q1",)))
    results = tmp_path / "agent.jsonl"
    results.write_text("{}\n")
    with tracked_eval(cfg, "agent", resumed=True) as tracker:
        assert tracker.enabled
        tracker.log_metrics(flatten_metrics([_record("q1", "basic", [True])]))
        tracker.log_artifact(results)

    (run,) = _runs(store)
    assert run.info.run_name == "agent-llama3.2:3b-hybrid"
    assert run.info.status == "FINISHED"
    assert run.data.params["agent.max_steps"] == "4"
    assert run.data.tags["system"] == "agent"
    assert run.data.tags["resumed"] == "true"
    assert run.data.tags["git_sha"] == "abc123"
    assert run.data.metrics["score"] == 1.0
    client = mlflow.MlflowClient(store)
    names = {a.path for a in client.list_artifacts(run.info.run_id)}
    assert {"manifest.json", "question_ids.json", "agent.jsonl"} <= names


def test_a_missing_file_is_a_warning_not_an_error(store: str, tmp_path: Path, caplog):
    with tracked_run("x", "r") as tracker:
        tracker.log_artifact(tmp_path / "nope.jsonl")
    assert "no file to log" in caplog.text


def test_run_status_follows_how_the_body_ends(store: str):
    with pytest.raises(KeyboardInterrupt), tracked_run("x", "interrupted"):
        raise KeyboardInterrupt
    with pytest.raises(RuntimeError), tracked_run("x", "broken"):
        raise RuntimeError("boom")
    status = {r.info.run_name: r.info.status for r in _runs(store, "x")}
    assert status == {"interrupted": "KILLED", "broken": "FAILED"}
    assert mlflow.active_run() is None


def test_disabled_tracking_writes_nothing(store: str, tmp_path: Path):
    with tracked_eval(_config(), "baseline", enabled=False) as tracker:
        assert not tracker.enabled
        tracker.log_metrics({"score": 1.0})
        tracker.log_artifact(tmp_path / "x")
        tracker.log_dict({}, "x.json")
    assert not (tmp_path / "mlflow.db").exists()


def _question(qid: str) -> BenchQuestion:
    return BenchQuestion(qid, "basic", "?", "a", (), ())


def test_agent_trace_is_keyed_by_question():
    call = {"step": 1, "thought": "look", "tool": "search", "args": {"q": "x"}, "docs": ["d1"]}
    trace = agent_trace([_record("a", "basic", [True], steps=2, stop="answered", trace=(call,))])
    assert trace["a"]["steps"] == 2 and trace["a"]["stop"] == "answered"
    assert trace["a"]["score"] == 1.0
    assert trace["a"]["calls"] == [call]


def test_results_are_read_back_from_the_file_for_the_runs_questions_only(
    store: str, tmp_path: Path
):
    results = tmp_path / "agent.jsonl"
    old = _record("old", "basic", [False], steps=1, stop="answered")
    mine = _record("q1", "basic", [True], steps=2, stop="max_steps", trace=({"step": 1},))
    results.write_text(old.to_json() + "\n" + mine.to_json() + "\n")

    with tracked_eval(_config(), "agent") as tracker:
        log_results(tracker, [_question("q1"), _question("unanswered")], results, agent=True)

    (run,) = _runs(store)
    assert run.data.metrics["n_questions"] == 1
    assert run.data.metrics["score"] == 1.0
    assert run.data.metrics["stop/max_steps"] == 1
    client = mlflow.MlflowClient(store)
    assert {"agent.jsonl", "agent_trace.json"} <= {
        a.path for a in client.list_artifacts(run.info.run_id)
    }


def test_baseline_results_log_no_agent_trace(store: str, tmp_path: Path):
    results = tmp_path / "baseline.jsonl"
    results.write_text(_record("q1", "basic", [True]).to_json() + "\n")
    with tracked_eval(_config(), "baseline") as tracker:
        log_results(tracker, [_question("q1")], results)
    (run,) = _runs(store)
    names = {a.path for a in mlflow.MlflowClient(store).list_artifacts(run.info.run_id)}
    assert "agent_trace.json" not in names and "baseline.jsonl" in names


def test_an_unreadable_results_file_does_not_end_the_run(store: str, tmp_path: Path, caplog):
    results = tmp_path / "bad.jsonl"
    results.write_text("not json\n")
    with tracked_eval(_config(), "baseline") as tracker:
        log_results(tracker, [_question("q1")], results)
    assert "could not read" in caplog.text
    assert _runs(store)[0].info.status == "FINISHED"
