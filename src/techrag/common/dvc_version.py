"""The DVC version of a data or index directory, for the run manifest.

``dvc add`` and ``dvc repro`` record a directory as an MD5 of its file list (``abc….dir``) in a
``.dvc`` file or in ``dvc.lock``. :func:`dvc_version` recomputes that hash from the files on disk
and compares it with the recorded one, so a run can say which DVC version of the directory it read
and whether that is the committed one. A run's ``dvc_version`` plus ``dvc pull`` restores exactly
the files it used, provided the version was pushed.

This reads the DVC files itself rather than importing DVC, so ``dvc`` is not needed to run an
evaluation. It looks for ``dvc.lock`` and ``<directory>.dvc`` at the root of the DVC project.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from techrag.common.config import DvcVersion


def dvc_dir_md5(directory: Path) -> str:
    """Return the hash DVC records for ``directory``, as ``dvc add`` would compute it.

    DVC hashes the sorted list of ``{"md5": <file hash>, "relpath": <path in directory>}``
    entries, one per file, as JSON with sorted keys.

    Args:
        directory: Directory whose files to hash.

    Returns:
        The hash, ending in ``.dir``.

    Raises:
        OSError: If a file cannot be read.
    """
    entries = []
    for file in directory.rglob("*"):
        if file.is_file():
            with file.open("rb") as f:
                digest = hashlib.file_digest(f, "md5").hexdigest()
            entries.append({"md5": digest, "relpath": file.relative_to(directory).as_posix()})
    entries.sort(key=lambda e: e["relpath"])
    listing = json.dumps(entries, sort_keys=True).encode()
    return hashlib.md5(listing).hexdigest() + ".dir"  # DVC's format, not used for security


def _project_root(path: Path) -> Path | None:
    """Return the closest directory above ``path`` that holds a ``.dvc`` directory, if any."""
    return next((p for p in path.parents if (p / ".dvc").is_dir()), None)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Return the mapping in the YAML file ``path``, or an empty one if it does not exist."""
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ValueError(f"cannot parse {path}: {e}") from e
    return data if isinstance(data, dict) else {}


def _recorded_md5(root: Path, directory: Path) -> str | None:
    """Return the hash ``dvc.lock`` or a ``.dvc`` file records for ``directory``, if either does."""
    rel = directory.relative_to(root).as_posix()
    for stage in (_load_yaml(root / "dvc.lock").get("stages") or {}).values():
        for out in stage.get("outs") or []:
            if out.get("path") == rel and out.get("md5"):
                return str(out["md5"])
    dvc_file = directory.with_name(directory.name + ".dvc")
    for out in _load_yaml(dvc_file).get("outs") or []:
        if out.get("path") == directory.name and out.get("md5"):
            return str(out["md5"])
    return None


def dvc_version(directory: Path) -> DvcVersion | None:
    """Return the DVC version of the files in ``directory``, or ``None`` if DVC does not track it.

    Args:
        directory: A directory that is the output of a ``dvc.yaml`` stage (recorded in
            ``dvc.lock``) or of ``dvc add`` (recorded in ``<directory>.dvc``).

    Returns:
        The hash of the files now on disk, and whether it equals the recorded one. ``None`` if
        ``directory`` is outside a DVC project, is not a directory, or no DVC file records it.

    Raises:
        OSError: If a file in ``directory`` cannot be read.
        ValueError: If ``dvc.lock`` or the ``.dvc`` file is not valid YAML.
    """
    directory = directory.resolve()
    root = _project_root(directory)
    if root is None or not directory.is_dir():
        return None
    recorded = _recorded_md5(root, directory)
    if recorded is None:
        return None
    current = dvc_dir_md5(directory)
    return DvcVersion(md5=current, committed=current == recorded)
