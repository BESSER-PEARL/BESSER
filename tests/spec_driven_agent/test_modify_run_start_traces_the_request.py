"""A modify run's run_start trace holds the user's request, not the assistant's notes.

The orchestrator traces ``original_request(instructions)[:500]``; the modify run
traced the first 500 characters of the whole instructions, which with the
modeling agent's notes in front are the notes, never the request.
"""

from besser.spec_driven_agent.planning.requirements_ledger import VERBATIM_MARKER
from tests.spec_driven_agent.test_model_change import HOTEL, _domain, _orch, _summary

import json


def test_modify_run_start_traces_the_verbatim_request(tmp_path, monkeypatch):
    (tmp_path / ".besser_recipe.json").write_text(
        json.dumps({"model": {"user_model": _summary(HOTEL)}}), encoding="utf-8")
    orch = _orch(tmp_path, _domain(HOTEL))
    seen = {}
    for name in ("_validate_phase1_output", "_run_phase3_validation", "_create_snapshot", "_remove_snapshot"):
        monkeypatch.setattr(orch, name, lambda *a, **k: [])
    monkeypatch.setattr(orch, "_run_phase2", lambda instr, extra_issues=None: None)
    monkeypatch.setattr(orch, "_save_recipe", lambda instr, elapsed: None)
    monkeypatch.setattr(orch._trace, "write", lambda event, **kw: seen.setdefault(event, kw))

    notes = "## Assistant's notes (may be inaccurate)\n\n" + "Use these only for confirmed choices. " * 20
    orch.modify(f"{notes}\n\n{VERBATIM_MARKER}\n\nAdd a search box to the booking list")

    assert seen["run_start"]["instructions"] == "Add a search box to the booking list"
