"""Only a local module that could shadow the stdlib one for THIS file blocks ``import <name>``.

Second-wave review of round 10 (``imp_attack.py``): any ``logging.py`` or ``logging/``
anywhere in the workspace blocked ``import logging``, including a JS folder
``frontend/src/logging/`` and a deeper ``backend/app/logging.py`` that
``backend/main.py`` could never import as ``logging``.
"""
from __future__ import annotations

import pytest

from besser.spec_driven_agent.repair.import_repair import repair_missing_imports

_SOURCE = "log = logging.getLogger(__name__)\n"


def _workspace(tmp_path, *files: str) -> str:
    (tmp_path / "backend").mkdir()
    for rel in files:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x = 1\n", encoding="utf-8")
    return str(tmp_path)


@pytest.mark.parametrize("files", [
    (),
    ("frontend/src/logging/index.js",),       # not Python
    ("backend/app/logging.py",),              # below the file, not importable as `logging`
    ("other_service/logging.py",),            # another service's folder
])
def test_a_module_the_file_cannot_import_as_the_name_does_not_block(tmp_path, files):
    repaired, notes = repair_missing_imports("backend/main.py", _SOURCE, root=_workspace(tmp_path, *files))
    assert notes == ["added 'import logging'"], repaired


@pytest.mark.parametrize("files", [
    ("backend/logging.py",),                  # the file's own directory / its app folder
    ("logging.py",),                          # the workspace root
    ("backend/logging/__init__.py",),         # a package
    ("backend/logging/handlers.py",),         # a namespace package
])
def test_a_module_that_can_shadow_the_stdlib_one_blocks(tmp_path, files):
    source_root = _workspace(tmp_path, *files)
    assert repair_missing_imports("backend/main.py", _SOURCE, root=source_root) == (_SOURCE, [])


def test_the_files_own_directory_counts_below_the_app_folder(tmp_path):
    root = _workspace(tmp_path, "backend/app/logging.py")
    assert repair_missing_imports("backend/app/main.py", _SOURCE, root=root) == (_SOURCE, [])


def test_a_file_outside_the_workspace_gets_no_stdlib_import(tmp_path):
    root = _workspace(tmp_path)
    assert repair_missing_imports("../elsewhere/main.py", _SOURCE, root=root) == (_SOURCE, [])
