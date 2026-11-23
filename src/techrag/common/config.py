"""Defaults shared across the pipeline, evaluation and reasoning packages.

It also defines the run manifest (:class:`RunConfig`): one validated record of every setting that
determines a run's results, which the baseline and agent commands save next to their output.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Retrieval
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
INDEX_DIR = Path(".index/dev")
Mode = Literal["dense", "bm25", "hybrid"]  # ranking methods

# Benchmark
SUBSET_DIR = Path("data/subsets/dev")

# Generation (Ollama)
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2:3b"

# Judging
JUDGE_MODEL = "qwen3.5:9b"
# Room for the prompt and a verdict with one entry per fact: the longest question has 46 facts,
# which needs about 8k tokens, and Ollama's default window of 4096 cuts the reply off mid-JSON.
NUM_CTX = 16384

# Agent loop
MAX_STEPS = 8  # model turns per question


# --- Run manifest -------------------------------------------------------------------------------
# The sections group settings by concern, so two runs can be compared field by field. Every model
# is frozen and rejects unknown fields: a typo in a saved manifest is an error, not a silently
# ignored setting.


class _Section(BaseModel):
    """Base for every manifest section: immutable, and strict about field names."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class LLMConfig(_Section):
    """The model that writes answers, and how it samples."""

    model_name: str = LLM_MODEL
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0, description="reply cap; None is no cap")
    num_ctx: int = Field(default=NUM_CTX, gt=0, description="context window in tokens")


class RetrievalConfig(_Section):
    """The retriever and the index it reads."""

    mode: Mode = "hybrid"
    k: int = Field(default=5, ge=1, description="chunks returned per query")
    embedding_model: str = EMBEDDING_MODEL
    index_version: str | None = Field(
        default=None, description="label or hash of the built index, to tell rebuilds apart"
    )


class AgentConfig(_Section):
    """The limits on the agentic search loop."""

    max_steps: int = Field(default=MAX_STEPS, ge=1, description="model turns per question")
    think: bool = Field(default=False, description="let the model reason before each reply")


class EvaluationConfig(_Section):
    """The model that grades answers."""

    judge_model: str = JUDGE_MODEL
    judge_temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class DataConfig(_Section):
    """The benchmark data a run was scored on."""

    subset_name: str = Field(default=SUBSET_DIR.name, min_length=1)
    dataset_hash: str | None = Field(
        default=None, description="content hash or DVC version of the subset"
    )


class RunMetadata(_Section):
    """Who started the run, when, and from which code. These never change a result."""

    run_name: str = Field(min_length=1)
    git_sha: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    author: str | None = None


class RunConfig(_Section):
    """Everything needed to describe, reproduce and compare one run.

    A run without a search loop, such as the single-shot baseline, records ``max_steps=1``.
    """

    metadata: RunMetadata
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    def save(self, path: Path) -> None:
        """Write the manifest to ``path`` as indented JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        """Read and validate a manifest written by :meth:`save`.

        Raises:
            pydantic.ValidationError: If the file is missing a field, has an unknown one, or
                holds an out-of-range value.
        """
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
