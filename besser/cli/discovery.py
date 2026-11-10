"""Expand the model arguments of ``besser validate`` into the model files to check.

A file argument is checked as named. A directory is searched recursively, in
sorted order, for ``.json`` and ``.py`` files. Loading a ``.py`` model executes
it, so the ``.py`` files found in a directory are skipped unless the caller
allows executing them. The ``.json`` files found are sniffed first, so that the
JSON files a project folder holds besides its models (``package.json``,
``tsconfig.json``) do not fail the run.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from besser.cli.loading import STDIN

MODEL_SUFFIXES = (".json", ".py")

# Directories never searched: tools' environments, caches and installed
# packages, which hold third-party JSON and Python files, not the user's models.
# Hidden directories (".git", ".venv", ".tox", ...) are skipped too.
SKIPPED_DIRECTORIES = frozenset({"venv", "env", "node_modules", "__pycache__", "site-packages"})

# Why a file found in a directory is not checked.
SKIP_PY = "a .py model is executed when loaded; pass --exec-py to load it"
SKIP_NOT_A_MODEL = "not an editor export (no 'diagrams' or 'model' object)"
SKIP_NOT_JSON = "not a JSON file"
SKIP_OUTSIDE = "a symbolic link to a file outside the directory"
SKIP_BROKEN_LINK = "a symbolic link to nothing"
SKIP_LINKED_DIRECTORY = "a symbolic link to a directory: not followed"
SKIP_NO_MODEL = "holds no B-UML model"


@dataclass(frozen=True)
class ModelFile:
    """One model file to check."""

    # As the user can open it: the argument, or a directory argument joined with the path under it.
    path: str
    # Named on the command line rather than found in a directory.
    explicit: bool


@dataclass(frozen=True)
class SkippedFile:
    """A file found in a directory and not checked."""

    path: str
    reason: str


@dataclass
class ModelFiles:
    """What the model arguments expand to."""

    files: list[ModelFile] = field(default_factory=list)
    skipped: list[SkippedFile] = field(default_factory=list)
    # Some argument is a directory.
    searched_directory: bool = False


def is_editor_export(data: Any) -> bool:
    """Whether ``data``, a parsed JSON document, has the shape of an editor export.

    The same shapes ``besser.cli.loading`` reads: a project (an object with
    ``diagrams``), possibly wrapped in ``{"project": ...}``, or one diagram (an
    object with ``model``).
    """
    if isinstance(data, dict) and isinstance(data.get("project"), dict):
        data = data["project"]
    return isinstance(data, dict) and (isinstance(data.get("diagrams"), dict) or isinstance(data.get("model"), dict))


def _sniff_json(path: Path) -> str | None:
    """Why the JSON file at ``path`` is not a model, or ``None`` when it looks like an editor export."""
    try:
        data = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return SKIP_NOT_JSON
    return None if is_editor_export(data) else SKIP_NOT_A_MODEL


def _search(argument: str, exec_py: bool, found: ModelFiles) -> None:
    """Add the model files under the directory ``argument`` to ``found``.

    Symbolic links are not followed out of the directory: a linked file counts
    only when it resolves to a file inside it, and a linked directory is not
    entered (it could lead outside, or back up into a loop).
    """
    root = Path(argument)
    real_root = root.resolve()
    for directory, subdirectories, names in os.walk(root, followlinks=False):
        here = Path(directory)
        for name in sorted(subdirectories):
            if (here / name).is_symlink():
                found.skipped.append(SkippedFile(str(here / name), SKIP_LINKED_DIRECTORY))
        subdirectories[:] = sorted(
            name for name in subdirectories
            if not name.startswith(".") and name not in SKIPPED_DIRECTORIES and not (here / name).is_symlink()
        )
        for name in sorted(names):
            path = here / name
            suffix = path.suffix.lower()
            if suffix not in MODEL_SUFFIXES:
                continue
            if not path.resolve().is_relative_to(real_root):
                found.skipped.append(SkippedFile(str(path), SKIP_OUTSIDE))
            elif not path.is_file():
                found.skipped.append(SkippedFile(str(path), SKIP_BROKEN_LINK))
            elif suffix == ".py" and not exec_py:
                found.skipped.append(SkippedFile(str(path), SKIP_PY))
            elif suffix == ".json" and (reason := _sniff_json(path)):
                found.skipped.append(SkippedFile(str(path), reason))
            else:
                found.files.append(ModelFile(str(path), explicit=False))


def find_model_files(arguments: list[str] | tuple[str, ...], exec_py: bool = False) -> ModelFiles:
    """The model files ``arguments`` name, in order: each file as given, each directory searched.

    A file reached twice (named, and found in a directory) is checked once.
    Arguments that are neither a directory nor a file are kept as given, for
    the loader to report.
    """
    found = ModelFiles()
    for argument in arguments:
        if argument != STDIN and Path(argument).is_dir():
            found.searched_directory = True
            _search(argument, exec_py, found)
        else:
            found.files.append(ModelFile(argument, explicit=True))

    unique, seen = [], set()
    for model_file in found.files:
        key = model_file.path if model_file.path == STDIN else os.path.realpath(model_file.path)
        if key not in seen:
            seen.add(key)
            unique.append(model_file)
    found.files = unique
    return found
