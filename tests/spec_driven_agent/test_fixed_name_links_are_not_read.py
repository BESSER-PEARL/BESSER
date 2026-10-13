"""Harness code that opens a workspace file BY NAME must not follow a link.

A walk now skips links, but many readers join a fixed name instead
(``.besser_recipe.json``, ``.besser_checkpoint.json``, ``sql_alchemy.py``,
``vite.config.js``...). ``ln -s /proc/self/environ .besser_recipe.json`` from
the sandboxed shell would then be opened by the unsandboxed worker. Every
``open()`` is recorded here, and none may reach the outside canary.
"""
import builtins
import json
import os
import types

import pytest

from besser.spec_driven_agent import run_report
from besser.spec_driven_agent.execution.workspace_fs import is_plain_file, read_plain_text
from besser.spec_driven_agent.pipeline.modify_run import ModifyRunMixin
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.repair import scaffold_repair
from besser.spec_driven_agent.state import checkpoint
from besser.spec_driven_agent.state.tracing import TraceWriter
from besser.spec_driven_agent.validation import python_imports, write_diagnostics


@pytest.fixture
def planted(tmp_path, monkeypatch):
    canary = tmp_path / "outside" / "environ"
    canary.parent.mkdir()
    canary.write_text(json.dumps({"OPENAI_API_KEY": "sk-canary"}), encoding="utf-8")
    ws = tmp_path / "ws"
    (ws / "backend").mkdir(parents=True)
    (ws / "frontend").mkdir()
    (ws / "backend" / "routers.py").write_text("from sql_alchemy import *\n", encoding="utf-8")
    names = [".besser_recipe.json", ".besser_checkpoint.json", ".besser_trace.jsonl",
             "backend/sql_alchemy.py", "frontend/vite.config.js"]
    try:
        for name in names:
            os.symlink(canary, ws / name)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable here: {exc}")

    opened = []
    real_open = builtins.open

    def recording_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            opened.append(os.path.realpath(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", recording_open)
    return ws, os.path.realpath(canary), opened


def _modify_seed(ws):
    host = types.SimpleNamespace(output_dir=str(ws), executor=types.SimpleNamespace(_generator_files=set()))
    ModifyRunMixin._seed_generator_files_from_recipe(host)


READERS = {
    "recipe_history": lambda ws: LLMOrchestrator._load_recipe_history(str(ws / ".besser_recipe.json")),
    "modify_recipe_seed": _modify_seed,
    "checkpoint": lambda ws: checkpoint.load_checkpoint(str(ws)),
    "trace_tail": lambda ws: TraceWriter(str(ws)).tail(),
    "run_report": lambda ws: run_report.build_report(str(ws)),
    "import_smoke_location": lambda ws: python_imports._import_smoke_location(
        str(ws), str(ws / "backend"), "backend", "", "name 'OPENAI_API_KEY' is not defined"),
    "star_import_resolution": lambda ws: write_diagnostics._star_import_scope(
        write_diagnostics.ast.parse("from sql_alchemy import *\n"), "backend/routers.py", str(ws)),
    "vite_config": lambda ws: scaffold_repair._vite_out_dir(str(ws / "frontend")),
}


@pytest.mark.parametrize("reader", sorted(READERS))
def test_a_fixed_name_link_is_never_opened(planted, reader):
    ws, canary, opened = planted

    try:
        READERS[reader](ws)
    except Exception:  # a refusal may surface as an error; a read may not
        pass

    assert canary not in opened, f"{reader} opened the planted link's target"


def test_the_helpers_refuse_links_specials_and_escapes(tmp_path):
    (tmp_path / "ws" / "real").mkdir(parents=True)
    ws = tmp_path / "ws"
    (ws / "real" / "a.txt").write_text("fine", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")
    try:
        os.symlink(tmp_path / "outside.txt", ws / "link.txt")
        os.symlink(ws / "real", ws / "dirlink", target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable here: {exc}")

    assert read_plain_text(ws / "real" / "a.txt", root=ws) == "fine"
    assert read_plain_text(ws / "link.txt", root=ws) is None
    assert read_plain_text(ws / "dirlink" / "a.txt", root=ws) is None
    assert read_plain_text(tmp_path / "outside.txt", root=ws) is None
    assert read_plain_text(ws / "missing.txt", root=ws) is None
    assert not is_plain_file(ws / "real", root=ws)
