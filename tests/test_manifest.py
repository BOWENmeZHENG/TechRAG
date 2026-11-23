import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from techrag.common.config import (
    JUDGE_MODEL,
    MAX_STEPS,
    AgentConfig,
    RetrievalConfig,
    RunConfig,
    RunMetadata,
)
from techrag.reasoning.harness import data_config, run_metadata, save_manifest


def _config(**overrides) -> RunConfig:
    return RunConfig(metadata=RunMetadata(run_name="r1"), **overrides)


def test_defaults_match_the_shared_config():
    config = _config()
    assert config.agent.max_steps == MAX_STEPS
    assert config.evaluation.judge_model == JUDGE_MODEL
    assert config.retrieval.mode == "hybrid"
    assert config.metadata.timestamp.tzinfo is not None


def test_round_trips_through_a_file(tmp_path: Path):
    config = _config(retrieval=RetrievalConfig(mode="bm25", k=10), agent=AgentConfig(think=True))
    path = tmp_path / "sub" / "run.manifest.json"
    config.save(path)
    assert RunConfig.load(path) == config


def test_run_name_is_required():
    with pytest.raises(ValidationError):
        RunConfig.model_validate({})
    with pytest.raises(ValidationError):
        RunMetadata(run_name="")


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        AgentConfig(max_step=3)
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"metadata": {"run_name": "r"}, "retreival": {}})


@pytest.mark.parametrize(
    "build",
    [
        lambda: RetrievalConfig(mode="semantic"),
        lambda: RetrievalConfig(k=0),
        lambda: AgentConfig(max_steps=0),
    ],
)
def test_out_of_range_values_are_rejected(build):
    with pytest.raises(ValidationError):
        build()


def test_sections_are_immutable():
    with pytest.raises(ValidationError):
        AgentConfig().max_steps = 3


def test_data_config_hash_follows_the_subset_contents(tmp_path: Path):
    subset = tmp_path / "mini"
    subset.mkdir()
    (subset / "questions.jsonl").write_text('{"id": "q1"}\n')
    first = data_config(subset)
    assert first.subset_name == "mini"

    (subset / "manifest.json").write_text("{}")
    with_manifest = data_config(subset)
    assert with_manifest.dataset_hash != first.dataset_hash
    assert data_config(subset) == with_manifest

    (subset / "questions.jsonl").write_text('{"id": "q2"}\n')
    assert data_config(subset).dataset_hash != with_manifest.dataset_hash


def test_data_config_records_how_questions_were_sampled(tmp_path: Path):
    (tmp_path / "questions.jsonl").write_text("{}\n")
    data = data_config(tmp_path, seed=7, sample_size=20, per_type=3, question_ids=["b", "a"])
    assert (data.seed, data.sample_size, data.per_type) == (7, 20, 3)
    assert data.question_ids == ("b", "a")
    plain = data_config(tmp_path)
    assert (plain.seed, plain.sample_size, plain.per_type, plain.question_ids) == (
        0,
        None,
        None,
        None,
    )
    with pytest.raises(ValidationError):
        data_config(tmp_path, sample_size=0)


def test_data_config_needs_questions(tmp_path: Path):
    with pytest.raises(OSError):
        data_config(tmp_path)


def test_run_metadata_names_the_run():
    meta = run_metadata("agent-1")
    assert meta.run_name == "agent-1"
    # git may be absent; a value, when there is one, is a full commit id
    assert meta.git_sha is None or len(meta.git_sha) == 40


def test_save_manifest_sits_next_to_the_results(tmp_path: Path):
    path = save_manifest(_config(), tmp_path / "eval" / "agent.jsonl")
    assert path == tmp_path / "eval" / "agent.manifest.json"
    assert json.loads(path.read_text())["metadata"]["run_name"] == "r1"
