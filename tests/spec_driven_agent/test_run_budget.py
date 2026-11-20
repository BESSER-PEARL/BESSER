"""A run is bounded by its budgets (time, cost, executed tool calls), and Phase 3 keeps a share of each.

The real orchestrator runs the shelf app (FastAPI scaffold) with a scripted
model (``scripted_rig``), as in ``test_scripted_loop_structure``.
"""

from __future__ import annotations

import besser.spec_driven_agent.pipeline.orchestrator as orchestrator_module
import besser.spec_driven_agent.pipeline.phase3_repair as phase3_module
from besser.spec_driven_agent.pipeline.constants import _PHASE3_BUDGET_SHARE
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

    assert orchestrator._phase2_stop_reason == "timeout"
    assert rounds, "Phase 3 ran no validation round"
    assert not orchestrator._phase3_exit_reason.startswith("skipped")
    # A scripted turn takes 60 s: Phase 2's last call started before Phase 3's share began.
    last_call_at = rounds[0] - 1_000.0 - 60.0
    assert last_call_at < 600 * (1 - _PHASE3_BUDGET_SHARE), f"Phase 2's last call started {last_call_at:.0f} s in"


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
