"""A run is bounded by its budgets (time, cost, executed tool calls), and Phase 3 keeps a share of each.

The real orchestrator runs the shelf app (FastAPI scaffold) with a scripted
model (``scripted_rig``), as in ``test_scripted_loop_structure``.
"""

from __future__ import annotations

from tests.spec_driven_agent.scripted_rig import ScriptedAgent, run_scripted
from tests.spec_driven_agent.test_scripted_loop_structure import MAIN, _no_app_boot

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
