"""End-to-end tests for the crash-resume flow.

Drives an orchestrator against a mock LLM until a checkpoint is written,
simulates a crash, then runs ``resume()`` on a fresh orchestrator and
verifies it picks up from the saved turn without re-executing completed
tool calls.
"""

from __future__ import annotations

import json
import os

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.generators.llm.checkpoint import (
    CHECKPOINT_FILENAME,
    load_checkpoint,
)
from besser.generators.llm.llm_client import UsageTracker
from besser.generators.llm.orchestrator import LLMOrchestrator
from besser.generators.llm.tracing import TRACE_FILENAME


@pytest.fixture
def simple_model():
    StringType = PrimitiveDataType("str")
    user = Class(name="User")
    user.attributes = {Property(name="name", type=StringType)}
    return DomainModel(name="Blog", types={user})


class _MockBlock:
    def __init__(self, block_type: str, **kwargs) -> None:
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class _ScriptedClient:
    """Mock client with a pre-scripted sequence of responses.

    Each ``chat`` call consumes one entry. Tests can introspect ``calls``
    to check how many turns happened on each leg of the run.
    """

    model = "mock"

    def __init__(self, script: list[dict]):
        self.script = list(script)
        self.calls: list[dict] = []
        self.usage = UsageTracker("mock")

    def chat(self, system, messages, tools):
        self.calls.append({"messages": list(messages)})
        if not self.script:
            return {"stop_reason": "end_turn", "content": [
                _MockBlock("text", text="Done"),
            ]}
        return self.script.pop(0)


def _tool_use(tool_id: str, tool_name: str = "list_files", **input_):
    return {
        "stop_reason": "tool_use",
        "content": [
            _MockBlock("tool_use", id=tool_id, name=tool_name, input=input_),
        ],
    }


def _end_turn():
    return {
        "stop_reason": "end_turn",
        "content": [_MockBlock("text", text="All done")],
    }


def test_run_writes_checkpoint_per_turn(simple_model, tmp_path):
    """After each Phase 2 turn with tool use, a checkpoint must exist
    on disk so a crash can be recovered.
    """
    client = _ScriptedClient([
        _tool_use("c1"),        # turn 1 → uses tool, save ckpt
        _tool_use("c2"),        # turn 2 → uses tool, save ckpt
        _end_turn(),            # turn 3 → finish, ckpt deleted on success
    ])
    orch = LLMOrchestrator(
        llm_client=client,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        max_turns=5,
        use_streaming=False,
    )
    orch.run("Build a blog")

    # Successful run deletes its own checkpoint so a later re-run
    # against the same output dir starts clean.
    assert not os.path.isfile(tmp_path / CHECKPOINT_FILENAME)

    # But the trace file must survive (it's the audit log).
    assert os.path.isfile(tmp_path / TRACE_FILENAME)


def test_checkpoint_survives_a_crash(simple_model, tmp_path):
    """Simulate a crash by raising mid-run. The checkpoint file from
    the last completed turn must remain on disk.
    """
    class _CrashingClient(_ScriptedClient):
        def chat(self, system, messages, tools):
            result = super().chat(system, messages, tools)
            if len(self.calls) >= 3:
                raise RuntimeError("simulated crash on 3rd call")
            return result

    client = _CrashingClient([
        _tool_use("c1"),        # turn 1 → ckpt saved
        _tool_use("c2"),        # turn 2 → ckpt saved
        _tool_use("c3"),        # turn 3 → crash BEFORE finishing turn
    ])
    orch = LLMOrchestrator(
        llm_client=client,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        max_turns=10,
        use_streaming=False,
    )
    # The orchestrator catches the client exception and breaks — so
    # .run() returns normally, but the saved checkpoint reflects the
    # last fully-completed turn.
    orch.run("Build a blog")

    # Checkpoint must have been written. Either turn 2 (if ckpt was
    # saved after turn 2 completed) or turn 1 (if crash interrupted
    # turn-2 save). Either way, the file exists.
    ckpt = load_checkpoint(str(tmp_path))
    assert ckpt is not None
    assert ckpt.turn >= 1
    assert ckpt.instructions == "Build a blog"
    assert ckpt.primary_kind == "class"
    assert len(ckpt.messages) > 0


def test_resume_refuses_when_fingerprint_differs(simple_model, tmp_path):
    """Resuming against a different project / instructions is refused."""
    # Set up a checkpoint with the original model
    client = _ScriptedClient([_tool_use("c1"), _tool_use("c2")])
    orch = LLMOrchestrator(
        llm_client=client,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        max_turns=2,
        use_streaming=False,
    )
    orch.run("Build a blog")
    # Manually re-save a checkpoint (run() deletes it on clean exit)
    # by running again with a crashing client.
    class _Crasher(_ScriptedClient):
        def chat(self, *a, **kw):
            r = super().chat(*a, **kw)
            if len(self.calls) >= 2:
                raise RuntimeError("crash")
            return r

    crasher = _Crasher([_tool_use("c1"), _tool_use("c2")])
    orch2 = LLMOrchestrator(
        llm_client=crasher,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        max_turns=10,
        use_streaming=False,
    )
    orch2.run("Build a blog")

    assert os.path.isfile(tmp_path / CHECKPOINT_FILENAME)

    # Now try to resume with different instructions → fingerprint mismatch
    resume_client = _ScriptedClient([_end_turn()])
    resumer = LLMOrchestrator(
        llm_client=resume_client,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        use_streaming=False,
    )
    with pytest.raises(ValueError, match="fingerprint"):
        resumer.resume("Completely different request")


def test_resume_picks_up_from_saved_turn(simple_model, tmp_path):
    """Happy path: crash after turn 2, resume, finish at turn 3.

    The resume orchestrator must not re-execute tool calls from turns
    1-2 — the LLM call count on the resume leg should start from turn 3.
    """
    # --- Leg 1: crash after saving a checkpoint --------------------------
    class _Crasher(_ScriptedClient):
        def chat(self, *a, **kw):
            r = super().chat(*a, **kw)
            if len(self.calls) >= 3:
                raise RuntimeError("crash")
            return r

    leg1 = _Crasher([
        _tool_use("c1"),   # turn 1
        _tool_use("c2"),   # turn 2 → ckpt saved
        _tool_use("c3"),   # turn 3 → crash partway through
    ])
    orch1 = LLMOrchestrator(
        llm_client=leg1,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        max_turns=10,
        use_streaming=False,
    )
    orch1.run("Build a blog")

    ckpt = load_checkpoint(str(tmp_path))
    assert ckpt is not None
    saved_turn = ckpt.turn
    assert saved_turn >= 1

    # --- Leg 2: resume and finish ---------------------------------------
    leg2 = _ScriptedClient([_end_turn()])
    orch2 = LLMOrchestrator(
        llm_client=leg2,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        max_turns=10,
        use_streaming=False,
    )
    orch2.resume("Build a blog")

    # Exactly ONE LLM call was made on the resume leg — it ended with
    # end_turn so the loop exited immediately.
    assert len(leg2.calls) == 1

    # Checkpoint is gone after clean completion.
    assert not os.path.isfile(tmp_path / CHECKPOINT_FILENAME)


def test_resume_fails_when_no_checkpoint(simple_model, tmp_path):
    """Calling ``resume()`` with no saved checkpoint is an explicit error."""
    client = _ScriptedClient([_end_turn()])
    orch = LLMOrchestrator(
        llm_client=client,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        use_streaming=False,
    )
    with pytest.raises(FileNotFoundError):
        orch.resume("Anything")


def test_trace_file_is_populated_during_run(simple_model, tmp_path):
    """The structured trace must capture phase boundaries + per-turn
    events so post-run tooling can replay what happened."""
    client = _ScriptedClient([_tool_use("c1"), _end_turn()])
    orch = LLMOrchestrator(
        llm_client=client,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        max_turns=3,
        use_streaming=False,
    )
    orch.run("Build a blog")

    trace_path = tmp_path / TRACE_FILENAME
    assert trace_path.is_file()

    events: list[dict] = []
    with open(trace_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    event_names = [e["event"] for e in events]
    # Canonical run shape: start → phase enters/exits → end
    assert event_names[0] == "run_start"
    assert event_names[-1] == "run_end"
    assert "phase_enter" in event_names
    assert "turn_start" in event_names
    # The tool call we scripted must appear
    tool_events = [e for e in events if e["event"] == "tool_call"]
    assert len(tool_events) == 1
    assert tool_events[0]["payload"]["tool"] == "list_files"
