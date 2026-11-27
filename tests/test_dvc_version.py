from pathlib import Path

import pytest

from techrag.common.config import DvcVersion
from techrag.common.dvc_version import dvc_dir_md5, dvc_version
from techrag.reasoning.harness import data_config

# What `dvc add` recorded in d.dvc for a directory d holding a.txt ("alpha") and sub/b.txt
# ("beta\n"), made with DVC 3.67. If DVC changes its format, this is the test that fails.
PINNED = "3a7597dcc503218c03020aceacf32b73.dir"


def _fixture(root: Path) -> Path:
    (root / "d" / "sub").mkdir(parents=True)
    (root / "d" / "a.txt").write_text("alpha")
    (root / "d" / "sub" / "b.txt").write_text("beta\n")
    return root / "d"


def _project(tmp_path: Path) -> Path:
    """A DVC project root holding the fixture directory ``d``."""
    (tmp_path / ".dvc").mkdir()
    _fixture(tmp_path)
    return tmp_path


def _lock(root: Path, md5: str, path: str = "d") -> None:
    (root / "dvc.lock").write_text(
        f"schema: '2.0'\nstages:\n  build:\n    outs:\n    - path: {path}\n      md5: {md5}\n"
    )


def test_dir_md5_matches_what_dvc_add_records(tmp_path: Path):
    assert dvc_dir_md5(_fixture(tmp_path)) == PINNED


def test_dir_md5_follows_names_and_contents(tmp_path: Path):
    d = _fixture(tmp_path)
    (d / "a.txt").write_text("alpha!")
    changed = dvc_dir_md5(d)
    assert changed != PINNED
    (d / "a.txt").write_text("alpha")
    (d / "a.txt").rename(d / "c.txt")
    assert dvc_dir_md5(d) not in (PINNED, changed)


def test_version_is_committed_when_dvc_lock_agrees(tmp_path: Path):
    root = _project(tmp_path)
    _lock(root, PINNED)
    assert dvc_version(root / "d") == DvcVersion(md5=PINNED, committed=True)


def test_version_reports_the_files_on_disk_when_they_changed(tmp_path: Path):
    root = _project(tmp_path)
    _lock(root, PINNED)
    (root / "d" / "a.txt").write_text("edited")
    version = dvc_version(root / "d")
    assert version is not None and not version.committed
    assert version.md5 == dvc_dir_md5(root / "d") != PINNED


def test_version_reads_a_dot_dvc_file(tmp_path: Path):
    root = _project(tmp_path)
    (root / "d.dvc").write_text(f"outs:\n- md5: {PINNED}\n  hash: md5\n  path: d\n")
    assert dvc_version(root / "d") == DvcVersion(md5=PINNED, committed=True)


def test_version_finds_the_project_from_a_subdirectory_and_a_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = _project(tmp_path)
    (root / "idx").mkdir()
    (root / "idx" / "x.bin").write_bytes(b"x")
    md5 = dvc_dir_md5(root / "idx")
    _lock(root, md5, path="idx")
    monkeypatch.chdir(root / "d")
    assert dvc_version(Path("../idx")) == DvcVersion(md5=md5, committed=True)


def test_version_is_none_when_dvc_does_not_track_the_directory(tmp_path: Path):
    root = _project(tmp_path)
    assert dvc_version(root / "d") is None  # no dvc.lock or .dvc file
    _lock(root, PINNED, path="other")
    assert dvc_version(root / "d") is None  # the lock records a different path


def test_version_is_none_outside_a_dvc_project_or_for_a_file(tmp_path: Path):
    d = _fixture(tmp_path)
    assert dvc_version(d) is None
    (tmp_path / "p").mkdir()
    root = _project(tmp_path / "p")
    _lock(root, PINNED)
    assert dvc_version(root / "d" / "a.txt") is None


def test_version_rejects_a_broken_dvc_lock(tmp_path: Path):
    root = _project(tmp_path)
    (root / "dvc.lock").write_text("stages: [unclosed")
    with pytest.raises(ValueError, match="dvc.lock"):
        dvc_version(root / "d")


def test_data_config_records_the_dvc_version_of_a_subset(tmp_path: Path):
    root = _project(tmp_path)
    (root / "d" / "questions.jsonl").write_text("{}\n")
    _lock(root, dvc_dir_md5(root / "d"))
    data = data_config(root / "d")
    assert data.dataset_dvc is not None and data.dataset_dvc.committed
    assert data.dataset_hash  # the existing content hash is still recorded


def test_data_config_has_no_dvc_version_for_an_untracked_subset(tmp_path: Path):
    (tmp_path / "questions.jsonl").write_text("{}\n")
    assert data_config(tmp_path).dataset_dvc is None
