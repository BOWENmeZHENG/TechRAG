from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point MLflow at a throwaway store for every test, and return its URI.

    A run recorded without it would leave ``mlflow.db`` and ``mlruns/`` (where artifacts go, next
    to the working directory) in the repository, so the working directory moves as well.
    """
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(tmp_path)
    return uri
