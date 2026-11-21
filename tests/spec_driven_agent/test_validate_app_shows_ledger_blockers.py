"""validate_app does not report "verified" while the run holds a requirement blocker.

Live evidence: r11-p81 turn 24, r9-p21 turn 15 and r9-p81 turn 21. In the Phase 3
fix attempt the model called validate_app and got ``verified: true,
blocker_count: 0`` while the harness held a requirements-ledger blocker, and the
run then ended blocked on it.
"""

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker

R2 = {"id": 2, "text": "Show the prices of the last 7 days", "kind": "action", "status": "missing",
      "evidence": "", "note": ""}
# The judge said implemented; the citation was not located.
R3 = {"id": 3, "text": "List the prices", "kind": "action", "status": "unverified",
      "evidence": [{"path": "backend/routers/analytics.py", "line": "@router.get('/prices/')"}], "note": ""}


class _Client:
    model = "mock"
    usage = UsageTracker("mock")

    def chat(self, **kwargs):
        return {"stop_reason": "end_turn", "content": []}


def _orch(tmp_path, monkeypatch):
    orch = LLMOrchestrator(llm_client=_Client(), domain_model=DomainModel(name="M", types={Class(name="Company")}),
                           output_dir=str(tmp_path))
    monkeypatch.setattr(orch, "_collect_execution_issues", lambda: [])
    return orch


def test_a_held_requirement_blocker_is_reported(tmp_path, monkeypatch):
    orch = _orch(tmp_path, monkeypatch)
    assert orch._validate_app()["verified"] is True  # nothing judged yet (Phase 2)

    orch._requirement_verdicts = [R2]  # Phase 3's judgment, same source revision
    orch._requirement_verdicts_key = orch._requirement_cache_key(orch._requirements_to_judge())
    result = orch._validate_app()
    assert result["verified"] is False and result["blocker_count"] == 1
    assert any("R2" in issue for issue in result["issues"])


def test_a_requirement_not_confirmed_is_listed_without_a_blocker(tmp_path, monkeypatch):
    orch = _orch(tmp_path, monkeypatch)
    orch._requirement_verdicts = [R3]
    orch._requirement_verdicts_key = orch._requirement_cache_key(orch._requirements_to_judge())
    result = orch._validate_app()
    assert result["verified"] is False and result["blocker_count"] == 0
    assert any("R3" in issue for issue in result["issues"])
    assert "not confirmed" in result["next_step"]


def test_an_implemented_requirement_adds_nothing(tmp_path, monkeypatch):
    orch = _orch(tmp_path, monkeypatch)
    orch._requirement_verdicts = [{**R2, "status": "implemented"}]
    assert orch._validate_app()["verified"] is True


def test_a_judgment_of_an_older_revision_is_labelled_and_does_not_block(tmp_path, monkeypatch):
    """Review: after an edit to the cited file the old verdict was shown as a current
    blocker with "Repair concrete failures". It is re-judged when the turn ends."""
    orch = _orch(tmp_path, monkeypatch)
    service = tmp_path / "service.py"
    service.write_text("def detect(messages):\n    return []\n", encoding="utf-8")
    orch._requirement_verdicts = [R2]
    orch._requirement_verdicts_key = orch._requirement_cache_key(orch._requirements_to_judge())
    assert orch._validate_app()["blocker_count"] == 1

    service.write_text("def detect(messages):\n    return [m for m in messages if m.isUnwanted]\n",
                       encoding="utf-8")
    result = orch._validate_app()
    assert result["verified"] is True and result["blocker_count"] == 0
    (stale,) = [issue for issue in result["issues"] if "R2" in issue]
    assert "judged before your latest edits; it is re-judged when you finish" in stale


def test_an_edit_the_judge_does_not_read_keeps_the_blocker(tmp_path, monkeypatch):
    """Review 4: a README-only edit changed the workspace revision, so a current blocker
    was shown as advisory with "verified: true", though the judgment would not change."""
    orch = _orch(tmp_path, monkeypatch)
    (tmp_path / "service.py").write_text("def detect(messages):\n    return []\n", encoding="utf-8")
    orch._requirement_verdicts = [R2]
    orch._requirement_verdicts_key = orch._requirement_cache_key(orch._requirements_to_judge())
    (tmp_path / "README.md").write_text("# Capacity rule documented here\n", encoding="utf-8")
    result = orch._validate_app()
    assert result["verified"] is False and result["blocker_count"] == 1
