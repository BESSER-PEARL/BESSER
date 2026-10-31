"""A resumed modify run must continue as a modify run.

Two defects (review 2026-10-30, B2): ``modify()`` saved its checkpoint with the
constructor's fingerprint, hashed from an empty instruction, so ``resume()``
refused every modify checkpoint as a different project. And the checkpoint
held no modify state, so a resume would have run as a create run: no
edit-first guard, no fix-target gate, no model change, no target switch.
"""

from __future__ import annotations

import json

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning.model_change import model_summary
from besser.spec_driven_agent.planning.target import WEB_APP, Target
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.state.checkpoint import load_checkpoint

_INSTRUCTIONS = "Creating a Book returns 500 Internal Server Error"


class _NoLLM:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self) -> None:
        self.usage = UsageTracker("mock-model")

    def chat(self, *args, **kwargs):
        raise RuntimeError("no LLM in this test")


def _model(*attributes: str) -> DomainModel:
    book = Class(name="Book")
    book.attributes = {Property(name=a, type=PrimitiveDataType("str")) for a in attributes}
    return DomainModel(name="Library", types={book})


def _orchestrator(tmp_path) -> LLMOrchestrator:
    return LLMOrchestrator(
        llm_client=_NoLLM(),
        domain_model=_model("title", "isbn"),
        output_dir=str(tmp_path),
        enable_tracing=False,
        enable_checkpointing=True,
        enable_toolchain_validation=False,
    )


def _crash_a_modify_run(tmp_path, monkeypatch) -> None:
    (tmp_path / ".besser_recipe.json").write_text(json.dumps({
        "generator_used": "generate_fastapi_backend",
        "output_files": [{"path": "backend/main_api.py", "source": "generator"}],
        "target": {"kind": "backend", "description": "REST API"},
        "model": {"user_model": model_summary(_model("title"))},
    }), encoding="utf-8")
    crashed = _orchestrator(tmp_path)
    monkeypatch.setattr(crashed, "_modify_target", lambda seed: Target(WEB_APP, "web app", True))
    monkeypatch.setattr(crashed, "_validate_phase1_output", lambda: [])

    def crash_in_phase2(instructions, extra_issues=None):
        crashed._save_checkpoint_for_turn(1, [{"role": "user", "content": instructions}], instructions)
        raise RuntimeError("worker died")

    monkeypatch.setattr(crashed, "_run_phase2", crash_in_phase2)
    with pytest.raises(RuntimeError, match="worker died"):
        crashed.modify(_INSTRUCTIONS)
    assert crashed._is_fix_run and crashed._previous_target is not None
    assert crashed._model_change is not None and crashed._model_change.changes


def test_resumed_modify_run_stays_a_modify_run(tmp_path, monkeypatch):
    _crash_a_modify_run(tmp_path, monkeypatch)

    resumed = _orchestrator(tmp_path)
    seen = {}

    def phase2(instructions, extra_issues=None):
        seen.update(modify_mode=resumed._modify_mode, guard=resumed.executor._modify_guard)

    for name, stub in {
        "_run_phase2": phase2,
        "_create_snapshot": lambda: None,
        "_run_phase3_validation": lambda: None,
        "_save_recipe": lambda instructions, elapsed: None,
        "_remove_snapshot": lambda: None,
    }.items():
        monkeypatch.setattr(resumed, name, stub)
    resumed.resume(_INSTRUCTIONS)

    assert seen == {"modify_mode": True, "guard": True}
    assert resumed._seed_generator_used == "generate_fastapi_backend"
    assert "backend/main_api.py" in resumed.executor._generator_files
    assert resumed._is_fix_run and "Book" in resumed._fix_target.entities
    assert resumed._previous_target == Target("backend", "REST API", False)
    change = resumed._model_change
    assert change is not None and change.changes and change.base and change.new
    # The gate a fix run ends on ran on the resumed run too.
    assert resumed._fix_target_resolved is True


def test_create_run_checkpoint_carries_no_modify_state(tmp_path):
    orch = _orchestrator(tmp_path)
    orch._save_checkpoint_for_turn(1, [], "Build a library")
    assert load_checkpoint(str(tmp_path)).modify is None
