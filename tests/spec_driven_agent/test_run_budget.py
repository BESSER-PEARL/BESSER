"""A run is bounded by its budgets (time, cost, executed tool calls), and Phase 3 keeps a share of each.

The real orchestrator runs the shelf app (FastAPI scaffold) with a scripted
model (``scripted_rig``), as in ``test_scripted_loop_structure``.
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import besser.spec_driven_agent.pipeline.orchestrator as orchestrator_module
import besser.spec_driven_agent.pipeline.phase3_repair as phase3_module
from besser.spec_driven_agent.pipeline.constants import _PHASE3_BUDGET_SHARE
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from tests.spec_driven_agent.run_transport import plain
from tests.spec_driven_agent.scripted_rig import FakeClock, ScriptedAgent, run_scripted
from tests.spec_driven_agent.test_scripted_loop_structure import MAIN, TASKS, _no_app_boot, shelf_script

BROKEN = "backend/routers/broken_stats.py"


class _SpendsTheCapAgent(ScriptedAgent):
    """The cost cap is spent by its third Phase 2 request."""

    def reply(self, kind):
        if kind == "phase2" and self.phase() == "phase2" and len(self.sent["phase2"]) == 2:
            self.orchestrator.max_cost_usd = 1e-9
        return super().reply(kind)


def test_a_spent_cost_cap_still_runs_the_free_checks(tmp_path, monkeypatch):
    """The checks cost nothing: a file Phase 2 left unparseable is a blocker, not an unchecked run."""
    _no_app_boot(monkeypatch)
    script = [
        ("read_file", {"path": MAIN}),
        ("write_file", {"path": BROKEN, "content": "def broken(:\n    return 1\n"}),
        ("read_file", {"path": "backend/sql_alchemy.py"}),
    ]
    agent = _SpendsTheCapAgent(str(tmp_path), script, ["Add a shelf statistics module"])
    orchestrator = run_scripted(tmp_path, agent)

    assert orchestrator._phase2_stop_reason == "cost_cap"
    assert not orchestrator._phase3_exit_reason.startswith("skipped")
    blockers = [i.message for i in orchestrator._validation_issues if i.severity == "blocker"]
    assert any("broken_stats.py" in message for message in blockers), blockers
    # The repair turns are LLM calls: none is made on a spent cap.
    assert not agent.sent["phase3"]
    assert orchestrator._phase3_exit_reason == "cost budget exhausted"


def test_phase2_stops_at_its_share_of_the_clock_and_phase3_validates(tmp_path, monkeypatch):
    """60 s a turn on a 600 s cap: Phase 2 ends with a quarter of the cap left, and Phase 3's
    checks run in it. Before, Phase 2 ran to the cap and Phase 3 was skipped outright."""
    _no_app_boot(monkeypatch)
    clock = FakeClock()
    monkeypatch.setattr(orchestrator_module, "time", clock)
    monkeypatch.setattr(phase3_module, "time", clock)
    rounds = []
    agent = ScriptedAgent(str(tmp_path), shelf_script(), TASKS, clock=clock, seconds_per_turn=60.0)
    orchestrator = run_scripted(tmp_path, agent, max_runtime_seconds=600,
                                on_findings=lambda attempt, issues, restored=False: rounds.append(clock.now))

    # Its share, not the cap: it hands off to validation, not as a run stopped by its time limit.
    assert orchestrator._phase2_stop_reason == "validation_required"
    assert orchestrator._phase2_stop_detail == "time share used (150 of 600 s reserved for validation and repair)"
    assert rounds, "Phase 3 ran no validation round"
    assert not orchestrator._phase3_exit_reason.startswith("skipped")
    # A scripted turn takes 60 s: Phase 2's last call started before Phase 3's share began.
    last_call_at = rounds[0] - 1_000.0 - 60.0
    assert last_call_at < 600 * (1 - _PHASE3_BUDGET_SHARE), f"Phase 2's last call started {last_call_at:.0f} s in"


def _timed_run(tmp_path, monkeypatch, seconds_per_turn, **kwargs):
    _no_app_boot(monkeypatch)
    clock = FakeClock()
    monkeypatch.setattr(orchestrator_module, "time", clock)
    monkeypatch.setattr(phase3_module, "time", clock)
    agent = ScriptedAgent(str(tmp_path), shelf_script(), TASKS, clock=clock, seconds_per_turn=seconds_per_turn)
    return agent, run_scripted(tmp_path, agent, max_runtime_seconds=600, **kwargs)


def test_a_phase2_cut_by_its_time_share_with_its_work_done_completes_when_validation_is_clean(
        tmp_path, monkeypatch):
    """28 s a turn on a 600 s cap: every scripted call is made, and the share ends Phase 2 before
    the model ends its turn. It was reported as a run stopped by its time limit, though no cap
    was reached and validation found nothing."""
    agent, orchestrator = _timed_run(tmp_path, monkeypatch, 28.0)

    assert sum(map(len, agent.sent["phase2"])) == len(shelf_script())
    assert orchestrator._phase2_stop_detail.startswith("time share used")
    assert orchestrator._phase3_exit_reason == "no blocker-class issues"
    assert orchestrator._phase2_stop_reason == "completed" and orchestrator._phase2_exited_cleanly


def test_the_formatter_runs_after_phase2_used_its_time_share(tmp_path, monkeypatch):
    """Phase 3's share was held until Phase 3 began, so the formatter between the phases saw
    no time left and did not run on the files the model wrote."""
    seen = {}
    original = orchestrator_module.LLMOrchestrator._format_python_output

    def spy(self):
        seen["remaining"] = self.deadline.remaining()
        original(self)
        seen["notes"] = list(self._format_notes)
    monkeypatch.setattr(orchestrator_module.LLMOrchestrator, "_format_python_output", spy)
    _, orchestrator = _timed_run(tmp_path, monkeypatch, 52.0)

    assert orchestrator._phase2_stop_detail.startswith("time share used")
    assert seen["remaining"] > 100, seen
    assert not any("did not run" in note for note in seen["notes"]), seen["notes"]


def test_a_failing_or_id_only_result_is_not_new_information():
    """Stagnation credits a new successful result. A failing command's new stderr, and a
    result that differs only by a generated id, counted as new and kept resetting it."""
    def outcome(name, payload):
        block = SimpleNamespace(id="t1", name=name)
        return LLMOrchestrator._record_novel_results([block], [{"tool_use_id": "t1", "content": json.dumps(payload)}],
                                                     seen)
    seen: set = set()
    assert not outcome("run_command", {"exit_code": 1, "success": False, "stderr": "npm ERR! code ERESOLVE"})
    assert not outcome("run_command", {"exit_code": 1, "success": False, "stderr": "npm ERR! peer dep"})
    assert not outcome("run_command", {"exit_code": 2, "stdout": "", "stderr": "build failed"})
    created = [outcome("test_api", {"status": 201, "body": {"id": str(uuid.uuid4()), "title": "x"}})
               for _ in range(3)]
    assert created == [True, False, False]
    assert outcome("run_command", {"exit_code": 0, "success": True, "stdout": "2 passed"})


def test_every_stagnation_stretch_gets_its_reminder_before_the_handoff(tmp_path, monkeypatch):
    """The reminder with the open checklist was sent once per run: after a source change, the
    next stretch of calls with no new result handed off without one."""
    _no_app_boot(monkeypatch)
    stall = [("read_file", {"path": MAIN})] * 24
    edit = ("write_file", {"path": "backend/notes_extra.py", "content": "VALUE = 1\n"})
    script = shelf_script()[:8] + stall[:21] + [edit] + stall
    agent = ScriptedAgent(str(tmp_path), script, TASKS)
    orchestrator = run_scripted(tmp_path, agent)

    assert orchestrator._phase2_stop_detail.startswith("inspection handoff")
    last = plain(agent.transport.of("phase2")[len(agent.sent["phase2"]) - 1]["messages"])
    reminders = [i for i, m in enumerate(last) if "tool calls without changing source" in json.dumps(m["content"])]
    [edit_at] = [i for i, m in enumerate(last) if "notes_extra.py" in json.dumps(m["content"])
                 and m["role"] == "assistant"]
    assert len(reminders) == 2 and reminders[0] < edit_at < reminders[1], (reminders, edit_at)


def test_verifying_one_call_per_turn_is_not_stagnation(tmp_path, monkeypatch):
    """After its last edit the model checks its work one call per turn, each call with a
    new result. The inspection handoff counted that as no progress and cut it after ten
    turns; only new reads counted, and an unclosed checklist reached Phase 3 unseen."""
    _no_app_boot(monkeypatch)
    script = shelf_script()
    checks = [("search_in_files", {"pattern": pattern}) for pattern in (
        "def book_stats", "def shelf_stats", "def shelf_totals", "include_router", "APIRouter",
        "get_db", "Session", "Depends", "response_model", "tags=", "sum(", "query(Book)",
        "query(Library)", "library.books")]
    script[13:13] = checks   # after the second validate_app, before the last checklist update
    agent = ScriptedAgent(str(tmp_path), script, TASKS)
    orchestrator = run_scripted(tmp_path, agent)

    assert orchestrator._phase2_stop_reason == "completed", orchestrator._phase2_stop_detail
    assert sum(map(len, agent.sent["phase2"])) == len(script)


def test_the_model_is_told_what_budget_is_left_after_tool_results(tmp_path, monkeypatch):
    """It was never told its calls, time or cost left. The line goes after the tool
    results; the system prompt, the cached prefix, stays byte-identical."""
    _no_app_boot(monkeypatch)
    agent = ScriptedAgent(str(tmp_path), shelf_script(), TASKS)
    run_scripted(tmp_path, agent, max_turns=6)   # 24 calls, 18 of them Phase 2's: 16 used

    requests = agent.transport.of("phase2")[:len(agent.sent["phase2"])]
    assert len({json.dumps(r["system"], sort_keys=True) for r in requests}) == 1
    reminders = []
    for request in requests:
        messages = plain(request["messages"])
        for before, message in zip(messages, messages[1:]):
            text = json.dumps(message["content"])
            if "of this phase's budget is used" in text:
                assert message["role"] == "user" and "tool_result" in json.dumps(before["content"])
                reminder = text.split("<system-reminder>")[1].split("</system-reminder>")[0]
                if reminder not in reminders:
                    reminders.append(reminder)
    assert [r[:3] for r in reminders] == ["50%", "75%"]
    assert all("blocked" in r and "finish now" not in r.lower() for r in reminders)
    assert "of this phase's budget is used" not in json.dumps(requests[-1]["system"])
