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

from besser.generators.llm.orchestrator import (
    _MAX_TOOLCHAIN_FIX_ITERATIONS,
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


def test_a_prose_reply_is_re_prompted_with_modify_file_forced(tmp_path):
    client = _StructuredClient(_prose)
    orch = _build(tmp_path, client)

    orch._invoke_phase3_fix_loop([BLOCKER], is_first_attempt=True)

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
    assert [force for force, _ in client.calls] == [None] * 5 + ["modify_file"]


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


def test_the_fix_prompt_asks_for_an_edit_up_front(tmp_path):
    client = _PlainClient()
    orch = _build(tmp_path, client)
    with patch.object(client, "chat", wraps=client.chat) as spy:
        orch._invoke_phase3_fix_loop([BLOCKER], is_first_attempt=True)
    first_prompt = spy.call_args_list[0].kwargs["messages"][0]["content"]
    assert "modify_file" in first_prompt
    assert "do not stop" in first_prompt.lower() or "not done" in first_prompt.lower()


# ------------------------------------------------------- the attempt is seen


def test_phase3_tool_calls_are_recorded_like_phase2_ones(tmp_path):
    client = _StructuredClient(_prose)
    orch = _build(tmp_path, client, enable_tracing=True, run_id="t")

    orch._invoke_phase3_fix_loop([BLOCKER], is_first_attempt=True)

    logged = [(e["tool"], e["success"]) for e in orch.tool_calls_log]
    assert ("modify_file", True) in logged, logged
    trace = (tmp_path / ".besser_trace.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in trace.splitlines()]
    assert any(e["event"] == "tool_call" and e["payload"]["tool"] == "modify_file"
               for e in events), [e["event"] for e in events]


def test_an_attempt_without_an_edit_is_said_so_in_the_log(tmp_path, caplog):
    client = _PlainClient()
    orch = _build(tmp_path, client, auto_fix_issues=True)

    with patch.object(orch, "_collect_validation_issues", return_value=[BLOCKER]), \
         patch.object(orch, "_create_snapshot"), \
         patch.object(orch, "_restore_snapshot"), \
         caplog.at_level(logging.WARNING, logger="besser.generators.llm.orchestrator"):
        orch._run_phase3_validation()

    text = caplog.text
    assert "no successful edit" in text, text
    # Two attempts ran (the second stalled round ends the loop); the summary
    # used to print the cap instead, and that number was what got diagnosed.
    assert "remain after 2 attempt(s)" in text, text
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
