"""A Phase 3 fix attempt must end in an edit, and must say what it did.

Run 7f918e11 (2026-09-18, Qwen3-30B-A3B via Nebius): one blocker, two fix
attempts, ten LLM turns, $0.0032, and the file that needed a one-line change
was never touched. The trace showed zero tool calls in Phase 3, which is not
what happened: the loop ran ``executor.execute`` directly, so nothing it did
reached the trace, the recipe or the sidecar, and every one of the ten
``tool_use`` turns is unrecorded. What is certain is that none of them was a
successful edit, and that the attempt was scored "no progress" without any
attempt to make the model act.

Three things follow, each pinned here:

* an attempt that ends with no successful edit is re-prompted ONCE, with
  ``modify_file`` forced through ``tool_choice`` where the client supports it
  and a high-salience reminder either way (the gateway may ignore
  ``tool_choice``) - after that the attempt ends instead of spinning;
* a model that reads until the turn cap gets that same single extra turn;
* Phase 3 tool calls go through the same recording path as Phase 2, so the
  next run like this one can be read instead of guessed at.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import (
    _MAX_TOOLCHAIN_FIX_ITERATIONS,
    _PHASE3_FIX_TURNS,
    LLMOrchestrator,
    ValidationIssue,
)

BLOCKER = ValidationIssue(
    "blocker",
    "create contract: app: POST /x/ cannot create an X - every schema-valid "
    "request is rejected (409: NOT NULL constraint failed: x.total). "
    "Fix site: create_x in app/main.py line 1",
)
EDIT = {"path": "app/main.py", "old_text": "x = 1\n", "new_text": "x = 2\n"}


class _MockStateMachine:
    name = "DummySM"


class _Usage:
    estimated_cost = 0.0


class _Block:
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for key, value in kwargs.items():
            setattr(self, key, value)


def _reminders(messages) -> list[str]:
    out = []
    for message in messages:
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, list):
            out.extend(b["text"] for b in content
                       if isinstance(b, dict) and b.get("type") == "text")
    return out


class _StructuredClient:
    """Explains instead of editing. Edits only when modify_file is forced -
    the shape of a model that answers a fix prompt conversationally."""
    model = "mock-model"

    def __init__(self, unforced_reply):
        self.usage = _Usage()
        self.calls = []          # (force_tool, reminders seen in that request)
        self._unforced_reply = unforced_reply

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        self.calls.append((force_tool, _reminders(messages)))
        if force_tool == "modify_file":
            return {"stop_reason": "tool_use", "content": [
                _Block("tool_use", id=f"m{len(self.calls)}", name="modify_file", input=dict(EDIT)),
            ]}
        return self._unforced_reply(len(self.calls))


def _prose(_n):
    return {"stop_reason": "end_turn", "content": [
        _Block("text", text="I would set total=0.0 in create_x before the insert."),
    ]}


def _read(n):
    return {"stop_reason": "tool_use", "content": [
        _Block("tool_use", id=f"r{n}", name="read_file", input={"path": "app/main.py"}),
    ]}


class _PlainClient:
    """No ``force_tool`` parameter at all, and it never takes the hint."""
    model = "mock-model"

    def __init__(self):
        self.usage = _Usage()
        self.calls = []

    def chat(self, system, messages, tools):
        self.calls.append(_reminders(messages))
        return _prose(len(self.calls))


def _build(tmp_path, client, **kwargs) -> LLMOrchestrator:
    os.makedirs(tmp_path / "app", exist_ok=True)
    (tmp_path / "app" / "main.py").write_text("x = 1\n", encoding="utf-8")
    return LLMOrchestrator(
        llm_client=client,
        state_machines=[_MockStateMachine()],
        output_dir=str(tmp_path),
        max_cost_usd=10.0,
        enable_tracing=kwargs.pop("enable_tracing", False),
        enable_checkpointing=False,
        **kwargs,
    )


def _main_py(tmp_path) -> str:
    return (tmp_path / "app" / "main.py").read_text(encoding="utf-8")


# ------------------------------------------------------------ the attempt acts


@pytest.mark.parametrize("verification_only", [False, "requirement unverified:", "verification setup:"])
def test_a_prose_reply_is_re_prompted_with_modify_file_forced(tmp_path, verification_only):
    client = _StructuredClient(_prose)
    orch = _build(tmp_path, client)

    blocker = ValidationIssue("blocker", f"{verification_only} verification is required") if verification_only else BLOCKER
    orch._invoke_phase3_fix_loop([blocker], is_first_attempt=True)

    if verification_only:
        assert _main_py(tmp_path) == "x = 1\n", "evidence correction must not force a gratuitous source edit"
        assert [force for force, _ in client.calls] == [None]
        return

    assert _main_py(tmp_path) == "x = 2\n"
    # Prose, the forced edit, then the model's own next turn (prose again,
    # which ends the attempt because an edit is now on record).
    assert [force for force, _ in client.calls] == [None, "modify_file", None]
    assert any("modify_file" in text for text in client.calls[1][1]), (
        "the re-prompt must say in words what tool_choice asks for - the "
        "gateway may drop tool_choice")


def test_reading_until_the_turn_cap_gets_one_forced_edit_turn(tmp_path):
    client = _StructuredClient(_read)
    orch = _build(tmp_path, client)

    orch._invoke_phase3_fix_loop([BLOCKER], is_first_attempt=True)

    assert _main_py(tmp_path) == "x = 2\n"
    assert [force for force, _ in client.calls] == [None] * _PHASE3_FIX_TURNS + ["modify_file"]
    assert orch.total_turns == _PHASE3_FIX_TURNS + 1
    edits = [call for call in orch.tool_calls_log if call["tool"] == "modify_file"]
    assert len(edits) == 1 and edits[0]["success"]


def test_a_client_that_ignores_the_re_prompt_ends_the_attempt(tmp_path):
    """Without tool_choice the reminder is all we have; it is sent once."""
    client = _PlainClient()
    orch = _build(tmp_path, client)

    orch._invoke_phase3_fix_loop([BLOCKER], is_first_attempt=True)

    assert _main_py(tmp_path) == "x = 1\n"
    assert len(client.calls) == 2
    assert client.calls[0] == []
    assert any("modify_file" in text for text in client.calls[1])


def test_an_attempt_that_edited_is_not_re_prompted(tmp_path):
    def edit_then_stop(n):
        if n == 1:
            return {"stop_reason": "tool_use", "content": [
                _Block("tool_use", id="m1", name="modify_file", input=dict(EDIT)),
            ]}
        return {"stop_reason": "end_turn", "content": []}

    client = _StructuredClient(edit_then_stop)
    orch = _build(tmp_path, client)

    orch._invoke_phase3_fix_loop([BLOCKER], is_first_attempt=True)

    assert _main_py(tmp_path) == "x = 2\n"
    assert [force for force, _ in client.calls] == [None, None]

    # The last permitted edit turn must still be verified, not treated as an
    # in-flight cancellation merely because no further editing turns remain.
    boundary_client = _StructuredClient(edit_then_stop)
    boundary = _build(tmp_path, boundary_client, max_turns=1, auto_fix_issues=True)
    boundary._phase2_stop_reason = "validation_required"
    with patch.object(boundary, "_collect_validation_issues", side_effect=[[BLOCKER], []]) as checks:
        boundary._run_phase3_validation()
    assert _main_py(tmp_path) == "x = 2\n" and checks.call_count == 2
    assert [force for force, _ in boundary_client.calls] == [None]
    assert not boundary._phase3_interrupted and boundary._phase2_exited_cleanly
    assert boundary._validation_issues == []


def test_the_fix_prompt_asks_for_an_edit_up_front(tmp_path):
    from besser.BUML.metamodel.structural import Class, DomainModel

    client = _PlainClient()
    orch = _build(tmp_path, client)
    orch._instructions = "Bookings must enforce combined room capacity."
    orch.domain_model = DomainModel(name="Hotel", types={Class(name="Booking")})
    expression = "context Booking inv capacity: self.guests->size() <= 4"
    orch.domain_model.conversion_issues = [{
        "id": "ocl-capacity", "expression": expression,
        "reason": "Property 'guests' not found in context 'Booking'",
    }]
    orch._recent_tool_failures = [{"tool": "read_file", "path": "models/booking.py", "error": "File not found"}]
    with patch.object(client, "chat", wraps=client.chat) as spy, patch(
        "besser.spec_driven_agent.pipeline.orchestrator.build_mutation_manifest",
        return_value="Relationship mutation coverage: reverse create, update and unlink paths",
    ):
        orch._invoke_phase3_fix_loop([BLOCKER], is_first_attempt=True)
    first_prompt = spy.call_args_list[0].kwargs["messages"][0]["content"]
    assert "modify_file" in first_prompt
    assert "do not stop" in first_prompt.lower() or "not done" in first_prompt.lower()
    assert orch._instructions in first_prompt
    assert "app/main.py" in first_prompt and "Current files and symbols" in first_prompt
    assert "models/booking.py" in first_prompt and "Recent rejected operations" in first_prompt
    assert "Domain model and conversion losses" in first_prompt
    assert expression in first_prompt and "ocl-capacity" in first_prompt
    assert "Requirements to verify (including conversion recovery)" in first_prompt
    assert "Relationship mutation coverage: reverse create, update and unlink paths" in first_prompt


def test_repair_excerpts_accept_real_finding_formats_and_reject_escape(tmp_path):
    orch = _build(tmp_path, _PlainClient())
    for location in ("frontend contract: app/main.py line 1", "ruff: app/main.py:1:2", "tsc [.]: app/main.py(1,2)"):
        excerpts = orch._excerpts_for([ValidationIssue("blocker", location)])
        assert len(excerpts) == 1 and "x = 1" in excerpts[0]
    assert orch._excerpts_for([ValidationIssue("blocker", "syntax error in ../escape.py line 1")]) == []


# ------------------------------------------------------- the attempt is seen


def test_phase3_tool_calls_are_recorded_like_phase2_ones(tmp_path):
    client = _StructuredClient(_prose)
    orch = _build(tmp_path, client, enable_tracing=True, run_id="t")

    orch._invoke_phase3_fix_loop([BLOCKER], is_first_attempt=True)

    logged = [(e["tool"], e["success"]) for e in orch.tool_calls_log]
    assert ("modify_file", True) in logged, logged

    # A validation request in the same batch must observe the completed edit,
    # even when the model lists validation first; response IDs keep input order.
    orch.executor.app_validator = lambda: {"blocker_count": 0, "source": _main_py(tmp_path)}
    results = orch._execute_tool_blocks([
        _Block("tool_use", id="validate", name="validate_app", input={}),
        _Block("tool_use", id="edit-again", name="modify_file", input={
            "path": "app/main.py", "old_text": "x = 2\n", "new_text": "x = 3\n",
        }),
    ], turn=orch.total_turns)
    assert [result["tool_use_id"] for result in results] == ["validate", "edit-again"]
    assert json.loads(results[0]["content"])["source"] == "x = 3\n"
    trace = (tmp_path / ".besser_trace.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in trace.splitlines()]
    assert any(e["event"] == "tool_call" and e["payload"]["tool"] == "modify_file"
               for e in events), [e["event"] for e in events]
    assert any(e["event"] == "tool_call" and e["payload"]["tool"] == "validate_app"
               for e in events)


def test_an_attempt_without_an_edit_is_said_so_in_the_log(tmp_path, caplog):
    """One attempt, not two.

    This client answers in prose and calls no tool at all, so the attempt left
    the tree byte-identical AND never reached for the editor: nothing it did
    reaches the next prompt, and the next round would be a replay of this one.
    Across the 221 runs recorded before this stop existed, the round after a
    prose-only round wrote source 0 times in 4. (An attempt whose edits were
    REJECTED is the opposite case and does buy a second round - 57% of those
    wrote next round, n=30 - see test_phase3_stall_guards.py.) The summary
    still prints the real attempt count, never the cap: printing the cap is
    what got this misdiagnosed in the first place.
    """
    client = _PlainClient()
    orch = _build(tmp_path, client, auto_fix_issues=True)

    with patch.object(orch, "_collect_validation_issues", return_value=[BLOCKER]), \
         patch.object(orch, "_create_snapshot"), \
         patch.object(orch, "_restore_snapshot"), \
         caplog.at_level(logging.WARNING, logger="besser.spec_driven_agent.pipeline.orchestrator"):
        orch._run_phase3_validation()

    text = caplog.text
    assert "no successful edit" in text, text
    assert "remain after 1 attempt(s)" in text, text
    assert "writes=0, attempted_writes=0" in text, text
    assert f"after {_MAX_TOOLCHAIN_FIX_ITERATIONS} attempt(s)" not in text


# ------------------------------------------------ the live run, end to end

FIXTURE_7F = Path(__file__).parent / "fixtures" / "run_7f918e11"
_BOOKING_ROUTER = "web_app/backend/routers/booking.py"


def test_run_7f918e11_is_repaired_end_to_end(tmp_path):
    """The whole chain on that run's backend, verbatim: the probe names the
    site, the fix prompt carries its lines, the model explains instead of
    editing, the forced modify_file lands on both sites, and the re-probe
    finds nothing."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    shutil.copytree(FIXTURE_7F, tmp_path, dirs_exist_ok=True)
    prompts = []

    class _Client(_StructuredClient):
        def chat(self, system, messages, tools, force_tool=None, model_override=None):
            prompts.append(messages[0]["content"])
            self.calls.append((force_tool, _reminders(messages)))
            if force_tool != "modify_file":
                return _prose(len(self.calls))
            return {"stop_reason": "tool_use", "content": [
                _Block("tool_use", id="m1", name="modify_file", input={
                    "path": _BOOKING_ROUTER,
                    "old_text": "contact_id=booking_data.contact        )",
                    "new_text": "contact_id=booking_data.contact,        totalAmountDue=0.0        )",
                }),
                _Block("tool_use", id="m2", name="modify_file", input={
                    "path": _BOOKING_ROUTER,
                    "old_text": "contact_id=item_data.contact            )",
                    "new_text": "contact_id=item_data.contact,            totalAmountDue=0.0            )",
                }),
            ]}

    client = _Client(_prose)
    orch = LLMOrchestrator(
        llm_client=client, state_machines=[_MockStateMachine()], output_dir=str(tmp_path),
        auto_fix_issues=True, max_cost_usd=10.0, enable_tracing=False,
        enable_checkpointing=False, enable_toolchain_validation=False,
    )
    orch._run_phase3_validation()

    assert "Fix site: create_booking in web_app/backend/routers/booking.py line 165" in prompts[0]
    assert "  165|     db_booking = Booking(" in prompts[0]
    assert [force for force, _ in client.calls][:2] == [None, "modify_file"]
    router = (tmp_path / _BOOKING_ROUTER).read_text(encoding="utf-8")
    assert router.count("totalAmountDue=0.0") == 2
    assert [i.message for i in orch._validation_issues if i.severity == "blocker"] == []
