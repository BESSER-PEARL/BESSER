"""Phase 3 rollback must never leave the workspace worse than it found it.

``_restore_snapshot`` used to delete the entire output directory and *then*
copy the snapshot back, swallowing any failure with ``logger.warning``. A copy
that died partway — full disk, locked file, a path Windows refuses — left a
half-erased workspace with no way back, and the run carried on and packaged
that as the deliverable.

It also restored over ``.besser_trace.jsonl`` / ``.besser_checkpoint.json``,
rewinding the append-only trace to its Phase-1 state and resurrecting a stale
checkpoint that a later resume would replay from.
"""
import os
import shutil

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.pipeline.orchestrator import (
    LLMOrchestrator, _ROLLBACK_DISCARD_DIR, _SNAPSHOT_DIR,
)


def _model():
    user = Class(name="User")
    user.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True)}
    return DomainModel(name="TestModel", types={user})


class _MockClient:
    model = "mock-model"
    usage = UsageTracker("mock-model")

    def chat(self, system, messages, tools):
        return {"stop_reason": "end_turn", "content": []}


@pytest.fixture
def orch(tmp_path):
    return LLMOrchestrator(
        llm_client=_MockClient(), domain_model=_model(), output_dir=str(tmp_path),
    )


def _write(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def test_rollback_restores_phase1_content(orch, tmp_path):
    _write(str(tmp_path), "app/main.py", "phase1")
    orch._create_snapshot()
    _write(str(tmp_path), "app/main.py", "phase3-broken")
    _write(str(tmp_path), "app/extra.py", "added in phase 3")

    assert orch._restore_snapshot() is True
    assert (tmp_path / "app" / "main.py").read_text() == "phase1"
    assert not (tmp_path / "app" / "extra.py").exists()
    # The parked copy is cleaned up on success.
    assert not (tmp_path / _ROLLBACK_DISCARD_DIR).exists()


def test_rollback_preserves_the_trace_and_checkpoint(orch, tmp_path):
    _write(str(tmp_path), ".besser_trace.jsonl", '{"phase":"1"}\n')
    _write(str(tmp_path), "app.py", "phase1")
    orch._create_snapshot()
    # Phase 2/3 append to the trace and rewrite the checkpoint.
    _write(str(tmp_path), ".besser_trace.jsonl", '{"phase":"1"}\n{"phase":"3"}\n')
    _write(str(tmp_path), ".besser_checkpoint.json", '{"turn": 12}')
    _write(str(tmp_path), "app.py", "phase3-broken")

    assert orch._restore_snapshot() is True
    assert (tmp_path / "app.py").read_text() == "phase1"
    # The append-only log keeps the Phase 3 entries; the checkpoint survives.
    assert '{"phase":"3"}' in (tmp_path / ".besser_trace.jsonl").read_text()
    assert (tmp_path / ".besser_checkpoint.json").read_text() == '{"turn": 12}'
    # ...and neither was copied into the snapshot in the first place.
    assert not (tmp_path / _SNAPSHOT_DIR / ".besser_trace.jsonl").exists()
    assert not (tmp_path / _SNAPSHOT_DIR / ".besser_checkpoint.json").exists()


def test_a_failing_restore_does_not_destroy_the_workspace(orch, tmp_path, monkeypatch):
    """The regression this file exists for."""
    _write(str(tmp_path), "app/main.py", "phase1")
    orch._create_snapshot()
    _write(str(tmp_path), "app/main.py", "phase2-work")
    _write(str(tmp_path), "app/feature.py", "phase2-work")

    real_copytree = shutil.copytree

    def _explode(src, dst, *a, **kw):
        if os.path.basename(os.path.normpath(dst)) == "app":
            raise OSError(28, "No space left on device")
        return real_copytree(src, dst, *a, **kw)

    monkeypatch.setattr(shutil, "copytree", _explode)

    assert orch._restore_snapshot() is False
    # Everything the run had produced is still there, unchanged.
    assert (tmp_path / "app" / "main.py").read_text() == "phase2-work"
    assert (tmp_path / "app" / "feature.py").read_text() == "phase2-work"
    assert not (tmp_path / _ROLLBACK_DISCARD_DIR).exists()


def test_restore_without_a_snapshot_reports_failure(orch):
    assert orch._restore_snapshot() is False


def test_the_parked_tree_is_never_packaged():
    """A leftover discard dir is recovery state, not deliverable output."""
    from besser.spec_driven_agent.pipeline.orchestrator import _RECIPE_EXCLUDED_DIRS
    from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
        _EXCLUDED_OUTPUT_DIRS,
    )
    assert _ROLLBACK_DISCARD_DIR in _RECIPE_EXCLUDED_DIRS
    assert _ROLLBACK_DISCARD_DIR in _EXCLUDED_OUTPUT_DIRS


def test_rollback_keeps_installed_dependencies(orch, tmp_path):
    # The snapshot skips node_modules / .venv to stay small, so the restore
    # must carry them over from the parked tree: a live run (A_gpt1) lost its
    # node_modules this way and shipped two "requires installed project
    # dependencies" blockers the rolled-back tree did not have.
    root = str(tmp_path)
    _write(root, "web/frontend/package.json", '{"name": "app"}')
    _write(root, "web/frontend/node_modules/react/index.js", "react")
    _write(root, "web/backend/.venv/lib/site.py", "venv")
    _write(root, "web/frontend/src/App.tsx", "phase1")
    orch._create_snapshot()
    assert not os.path.exists(
        os.path.join(root, _SNAPSHOT_DIR, "web/frontend/node_modules"))
    _write(root, "web/frontend/src/App.tsx", "phase3-broken")

    assert orch._restore_snapshot() is True
    assert (tmp_path / "web/frontend/src/App.tsx").read_text() == "phase1"
    assert (tmp_path / "web/frontend/node_modules/react/index.js").read_text() == "react"
    assert (tmp_path / "web/backend/.venv/lib/site.py").read_text() == "venv"
    assert not (tmp_path / _ROLLBACK_DISCARD_DIR).exists()


def test_rollback_does_not_revive_dependencies_of_a_removed_directory(orch, tmp_path):
    # node_modules under a directory Phase 3 created (absent from the snapshot)
    # has nowhere to go back to and is discarded with the rest of that tree.
    root = str(tmp_path)
    _write(root, "app.py", "phase1")
    orch._create_snapshot()
    _write(root, "extra/node_modules/x/index.js", "x")

    assert orch._restore_snapshot() is True
    assert not (tmp_path / "extra").exists()
