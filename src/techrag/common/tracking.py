"""MLflow runs for the evaluation commands.

This module depends only on the run manifest, so any command can import it, the judge included.
What to log from a baseline or agent run's records is in :mod:`techrag.reasoning.metrics`.

:func:`tracked_eval` wraps a baseline or agent run in an MLflow run: it logs the run manifest
(:class:`~techrag.common.config.RunConfig`) as parameters and an artifact when the run starts, and
hands back a :class:`Tracker` for logging metrics and files as the run goes. :func:`tracked_run` is
the same without a manifest, for commands such as the judge self-test whose settings are not a
``RunConfig``.

``mlflow`` is imported only when tracking is on, so ``enabled=False`` costs nothing at start-up.
Logging calls never end a long evaluation: if the tracking store fails, the error is logged as a
warning and the results file remains the record of the run.
"""

import logging
import os
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from techrag.common.config import RunConfig

logger = logging.getLogger(__name__)

# Where runs are stored unless ``MLFLOW_TRACKING_URI`` says otherwise. Relative to the working
# directory, like the ``eval/`` results.
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
# Baseline and agent runs share an experiment, so the MLflow UI can put them side by side.
EVAL_EXPERIMENT = "techrag"
JUDGE_EXPERIMENT = "techrag-judge"


class Tracker:
    """Logs to the active MLflow run, or does nothing if tracking is off.

    Attributes:
        enabled: Whether calls are sent to MLflow.
    """

    def __init__(self, mlflow_module: Any | None = None) -> None:
        """Create a tracker that logs through ``mlflow_module``, or a no-op one if ``None``."""
        self._mlflow = mlflow_module

    @property
    def enabled(self) -> bool:
        """Return whether calls are sent to MLflow."""
        return self._mlflow is not None

    def _call(self, what: str, name: str, *args: Any, **kwargs: Any) -> None:
        """Call ``mlflow.<name>``, turning a tracking failure into a warning."""
        if self._mlflow is None:
            return
        try:
            getattr(self._mlflow, name)(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - tracking must not end the evaluation it records
            logger.warning("mlflow: could not log %s: %s", what, e)

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log ``params``; MLflow does not let a run change one once logged."""
        self._call("params", "log_params", dict(params))

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Log ``metrics``, a map from metric name to value."""
        self._call("metrics", "log_metrics", dict(metrics), step=step)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        """Log the file ``path``, or warn if it does not exist, under ``artifact_path``."""
        if not path.is_file():
            if self.enabled:
                logger.warning("mlflow: no file to log at %s", path)
            return
        self._call(f"artifact {path.name}", "log_artifact", str(path), artifact_path)

    def log_dict(self, obj: Any, name: str) -> None:
        """Log the JSON-serializable ``obj`` as the artifact ``name``."""
        self._call(f"artifact {name}", "log_dict", obj, name)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Set ``tags`` on the run."""
        self._call("tags", "set_tags", dict(tags))


def _git_dirty() -> bool | None:
    """Return whether the working tree has uncommitted or untracked files, ``None`` if unknown."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(out.strip())


def _git_sha() -> str | None:
    """Return the commit of the working directory's repository, ``None`` if unknown."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.strip() or None


def git_tags() -> dict[str, str]:
    """Return tags for the code a run started from: ``git_sha`` and ``git_dirty``.

    Either is left out if git cannot say.
    """
    sha, dirty = _git_sha(), _git_dirty()
    return {
        **({"git_sha": sha} if sha else {}),
        **({"git_dirty": str(dirty).lower()} if dirty is not None else {}),
    }


def flatten_params(cfg: RunConfig) -> dict[str, Any]:
    """Return the settings of ``cfg`` as MLflow parameters, with dotted names.

    The metadata is left out because it never changes a result (it becomes tags), and so are the
    question ids, which can be longer than a parameter may be (they are logged as an artifact).

    Args:
        cfg: The run's manifest.

    Returns:
        A map such as ``{"llm.model_name": "qwen3.5:9b", "retrieval.k": 10, ...}``.
    """
    dump = cfg.model_dump(mode="json", exclude={"metadata": True, "data": {"question_ids"}})
    params: dict[str, Any] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                walk(f"{prefix}.{key}" if prefix else key, inner)
        else:
            params[prefix] = value

    walk("", dump)
    return params


@contextmanager
def tracked_run(
    experiment: str,
    run_name: str,
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, str] | None = None,
    enabled: bool = True,
) -> Iterator[Tracker]:
    """Run the body inside an MLflow run, and yield a :class:`Tracker` to log to it.

    Runs are stored at ``$MLFLOW_TRACKING_URI``, or :data:`DEFAULT_TRACKING_URI` if it is unset.
    The run ends as ``FINISHED``, as ``KILLED`` on Ctrl-C (an interrupted evaluation resumes in a
    later run), or as ``FAILED`` on any other exception.

    Args:
        experiment: Name of the experiment, created if it does not exist.
        run_name: Name of the run.
        params: Parameters to log at the start. MLflow does not let a run change them later.
        tags: Tags to set at the start.
        enabled: Whether to track at all. If not, nothing is imported or written.

    Yields:
        The tracker for the run; a no-op one if ``enabled`` is false.

    Raises:
        mlflow.exceptions.MlflowException: If the run cannot be started.
    """
    if not enabled:
        yield Tracker()
        return

    import mlflow

    # Set explicitly, even from the environment: a URI set earlier in the process would win over it.
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI)
    mlflow.set_experiment(experiment)
    mlflow.start_run(run_name=run_name)
    status = "FINISHED"
    try:
        tracker = Tracker(mlflow)
        if params:
            tracker.log_params(params)
        if tags:
            tracker.set_tags(tags)
        yield tracker
    except KeyboardInterrupt:
        status = "KILLED"
        raise
    except BaseException:
        status = "FAILED"
        raise
    finally:
        mlflow.end_run(status)


@contextmanager
def tracked_eval(
    cfg: RunConfig,
    system: str,
    run_name: str | None = None,
    resumed: bool = False,
    enabled: bool = True,
) -> Iterator[Tracker]:
    """Run the body inside an MLflow run that records the manifest ``cfg``.

    Logs the settings as parameters (:func:`flatten_params`), the metadata as tags, the whole
    manifest as ``manifest.json`` and, if the run is restricted to some questions, their ids as
    ``question_ids.json``. See :func:`tracked_run` for where runs go and how they end.

    Args:
        cfg: The run's manifest.
        system: What is being evaluated, ``baseline`` or ``agent``. It is tagged ``system``.
        run_name: Name of the run; defaults to ``<system>-<model>-<retrieval mode>``.
        resumed: Whether the run continues an earlier results file.
        enabled: Whether to track at all.

    Yields:
        The tracker for the run; a no-op one if ``enabled`` is false.

    Raises:
        mlflow.exceptions.MlflowException: If the run cannot be started.
    """
    meta = cfg.metadata
    dirty = _git_dirty()
    tags = {
        "system": system,
        "resumed": str(resumed).lower(),
        "started_at": meta.timestamp.isoformat(),
        **({"git_sha": meta.git_sha} if meta.git_sha else {}),
        **({"author": meta.author} if meta.author else {}),
        **({"git_dirty": str(dirty).lower()} if dirty is not None else {}),
    }
    name = run_name or f"{system}-{cfg.llm.model_name}-{cfg.retrieval.mode}"
    with tracked_run(EVAL_EXPERIMENT, name, flatten_params(cfg), tags, enabled) as tracker:
        tracker.log_dict(cfg.model_dump(mode="json"), "manifest.json")
        if cfg.data.question_ids is not None:
            tracker.log_dict(list(cfg.data.question_ids), "question_ids.json")
        yield tracker
