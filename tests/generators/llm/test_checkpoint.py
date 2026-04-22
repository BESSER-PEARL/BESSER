"""Tests for ``besser.generators.llm.checkpoint``.

Covers the public save / load / delete API plus the fingerprint function
used to validate resume requests.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from besser.generators.llm.checkpoint import (
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
    original = _make_checkpoint()
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
