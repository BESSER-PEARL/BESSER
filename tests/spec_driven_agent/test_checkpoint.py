"""Tests for ``besser.spec_driven_agent.state.checkpoint``.

Covers the public save / load / delete API plus the fingerprint function
used to validate resume requests.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from besser.spec_driven_agent.state.checkpoint import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_SCHEMA_VERSION,
    Checkpoint,
    compute_fingerprint,
    delete_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


class _MockDomain:
    def __init__(self, class_names: list[str]):
        self._class_names = class_names

    def get_classes(self):
        return [type("C", (), {"name": n}) for n in self._class_names]


class _MockSM:
    def __init__(self, name: str):
        self.name = name


def _make_checkpoint(**overrides) -> Checkpoint:
    base = dict(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        run_id="abc123",
        instructions="Build a blog backend",
        primary_kind="class",
        turn=5,
        total_turns=5,
        messages=[
            {"role": "user", "content": "Build it"},
            {"role": "assistant", "content": "Building..."},
        ],
        tool_calls_log=[{"turn": 1, "tool": "read_file", "success": True}],
        validation_issues=[],
        inventory="Generated 3 files",
        generator_used="generate_fastapi_backend",
        estimated_cost_usd=0.123,
        compaction_count=0,
        project_fingerprint="fp123",
        saved_at=time.time(),
    )
    base.update(overrides)
    return Checkpoint(**base)


def test_save_load_round_trip(tmp_path):
    original = _make_checkpoint(tasks=[
        {"id": 1, "text": "Build the frontend", "done": False},
        {"id": 2, "text": "Write the README", "done": True},
    ], api_scenarios=[{
        "scenario_id": "room-create",
        "scenario": {"backend": None, "requests": [{"method": "GET", "path": "/room/"}]},
        "correction_history": [{"reason": "corrected response field", "previous_status": "failed"}],
    }], phase="phase3", messages=[], source_revision="source-sha",
        phase2_stop_reason="validation_required", phase2_exited_cleanly=False,
        repair_progress={"attempts_run": 3, "no_progress_streak": 1,
                         "seen_states": [["sha", "obligations", ["syntax error"]]]})
    path = save_checkpoint(str(tmp_path), original)
    assert path is not None
    assert os.path.isfile(path)
    assert os.path.basename(path) == CHECKPOINT_FILENAME

    loaded = load_checkpoint(str(tmp_path))
    assert loaded is not None
    assert loaded.run_id == original.run_id
    assert loaded.turn == original.turn
    assert loaded.messages == original.messages
    assert loaded.project_fingerprint == original.project_fingerprint
    assert loaded.tasks == original.tasks
    assert loaded.api_scenarios == original.api_scenarios
    assert loaded.phase == "phase3"
    assert loaded.source_revision == "source-sha"
    assert loaded.phase2_stop_reason == "validation_required"
    assert loaded.phase2_exited_cleanly is False
    assert loaded.repair_progress == original.repair_progress


def test_real_provider_blocks_survive_checkpoint_round_trip(tmp_path):
    from besser.spec_driven_agent.providers.llm_client import _TextBlock, _ToolUseBlock

    arguments = {"path": "app.py", "old_text": "x = 1", "new_text": "y = 2\nx = 1"}
    original = _make_checkpoint(messages=[
        {"role": "assistant", "content": [
            _TextBlock("Updating app.py"),
            _ToolUseBlock("edit-31", "modify_file", arguments),
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "edit-31", "content": "modified"},
        ]},
    ])
    assert save_checkpoint(str(tmp_path), original)
    loaded = load_checkpoint(str(tmp_path))
    assert loaded.messages[0]["content"] == [
        {"type": "text", "text": "Updating app.py"},
        {"type": "tool_use", "id": "edit-31", "name": "modify_file", "input": arguments},
    ]
    assert loaded.messages[1] == original.messages[1]


def test_load_older_checkpoint_without_tasks_is_backward_compatible(tmp_path):
    original = _make_checkpoint().to_dict()
    original["schema_version"] = 1
    original.pop("tasks", None)
    original.pop("api_scenarios", None)
    for key in ("phase", "source_revision", "phase2_stop_reason", "phase2_exited_cleanly", "repair_progress"):
        original.pop(key, None)
    (tmp_path / CHECKPOINT_FILENAME).write_text(
        json.dumps(original), encoding="utf-8"
    )

    loaded = load_checkpoint(str(tmp_path))

    assert loaded is not None
    assert loaded.tasks == []
    assert loaded.api_scenarios == []
    assert loaded.phase == "phase2"
    assert loaded.repair_progress == {}


def test_load_missing_returns_none(tmp_path):
    assert load_checkpoint(str(tmp_path)) is None


def test_load_wrong_schema_version_rejected(tmp_path):
    path = tmp_path / CHECKPOINT_FILENAME
    path.write_text(json.dumps({
        "schema_version": 999,
        "run_id": "x",
        "instructions": "x",
    }), encoding="utf-8")

    # Refusing to parse an unknown version is the desired behaviour —
    # better to start fresh than to silently corrupt state.
    assert load_checkpoint(str(tmp_path)) is None

    for invalid in (
        {"phase": "unknown"}, {"schema_version": 1, "phase": "phase3"},
        {"repair_progress": {"attempts_run": -1}},
        {"repair_progress": {"seen_states": [["missing", "messages"]]}},
    ):
        data = _make_checkpoint().to_dict()
        data.update(invalid)
        path.write_text(json.dumps(data), encoding="utf-8")
        assert load_checkpoint(str(tmp_path)) is None


def test_delete_checkpoint_is_idempotent(tmp_path):
    # Deleting when there's nothing to delete is a no-op, not an error
    delete_checkpoint(str(tmp_path))   # should not raise
    # Write + delete works
    save_checkpoint(str(tmp_path), _make_checkpoint())
    delete_checkpoint(str(tmp_path))
    assert not os.path.isfile(tmp_path / CHECKPOINT_FILENAME)
    # Re-deleting still no-op
    delete_checkpoint(str(tmp_path))


def test_save_is_atomic_via_sidecar(tmp_path):
    """save_checkpoint writes to ``.tmp`` then ``os.replace`` — after a
    successful save the sidecar should not be left behind.
    """
    save_checkpoint(str(tmp_path), _make_checkpoint())
    files = os.listdir(tmp_path)
    assert CHECKPOINT_FILENAME in files
    assert CHECKPOINT_FILENAME + ".tmp" not in files


def test_fingerprint_stable_for_same_inputs():
    """Same instructions + same models → same fingerprint across calls."""
    model = _MockDomain(["User", "Post"])
    fp1 = compute_fingerprint(
        instructions="Build a blog", primary_kind="class", domain_model=model,
    )
    fp2 = compute_fingerprint(
        instructions="Build a blog", primary_kind="class", domain_model=model,
    )
    assert fp1 == fp2


def test_fingerprint_changes_with_instructions():
    model = _MockDomain(["User"])
    fp_a = compute_fingerprint(
        instructions="Build a blog", primary_kind="class", domain_model=model,
    )
    fp_b = compute_fingerprint(
        instructions="Build a store", primary_kind="class", domain_model=model,
    )
    assert fp_a != fp_b


def test_fingerprint_changes_with_class_swap():
    """Swapping in a completely different set of classes must shift the
    fingerprint — this is the main thing resume validation protects
    against (resuming a run against the wrong project).
    """
    fp_users = compute_fingerprint(
        instructions="X", primary_kind="class",
        domain_model=_MockDomain(["User", "Post"]),
    )
    fp_orders = compute_fingerprint(
        instructions="X", primary_kind="class",
        domain_model=_MockDomain(["Order", "LineItem"]),
    )
    assert fp_users != fp_orders


def test_fingerprint_includes_state_machines():
    fp_none = compute_fingerprint(
        instructions="X", primary_kind="state_machine", state_machines=[],
    )
    fp_one = compute_fingerprint(
        instructions="X", primary_kind="state_machine",
        state_machines=[_MockSM("OrderSM")],
    )
    assert fp_none != fp_one


@pytest.mark.parametrize("model_arg", ["bpmn_model", "nn_model"])
def test_fingerprint_includes_new_model_presence_and_identity(model_arg):
    base = compute_fingerprint(
        instructions="X", primary_kind=model_arg.removesuffix("_model"),
    )
    first = compute_fingerprint(
        instructions="X",
        primary_kind=model_arg.removesuffix("_model"),
        **{model_arg: type("Model", (), {"name": "First"})()},
    )
    second = compute_fingerprint(
        instructions="X",
        primary_kind=model_arg.removesuffix("_model"),
        **{model_arg: type("Model", (), {"name": "Second"})()},
    )

    assert base != first
    assert first != second
