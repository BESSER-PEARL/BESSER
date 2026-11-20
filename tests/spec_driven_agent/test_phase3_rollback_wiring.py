"""Phase 3 must not ship an app it made worse.

``_restore_snapshot`` was implemented, unit-tested in
``test_rollback_safety.py``, and **never called from production code**. Every
Phase 3 test that touches it does ``patch.object(orch, "_restore_snapshot")``,
so the missing call site was invisible.

A recorded run (Qwen) entered Phase 3 with 11 blockers. Attempt
2 added an association table using ``Table`` without importing it; the count
went to 45 and the run shipped that tree, with ``sql_alchemy.py`` no longer
importable and every router that star-imports it dead.

Each fix attempt is now judged on its own (``_attempt_not_kept``): one that
retires none of the blockers it faced, adds an observed failure or leaves more
blockers is rolled back to the tree from before it (``_revert_attempt``), and
the findings only the discarded tree showed are still reported.
"""

import pytest

from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, ValidationIssue


@pytest.fixture
def orchestrator(simple_library_book_model, tmp_path):
    class Client:
        model = "mock-model"
        usage = UsageTracker("mock-model")

        def chat(self, **kwargs):
            raise AssertionError("no LLM call expected")

    return LLMOrchestrator(
        llm_client=Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False,
    )


def _blockers(n, prefix="python contract", start=0):
    return [ValidationIssue("blocker", f"{prefix}: defect {i}") for i in range(start, start + n)]


def _not_kept(orchestrator, before, after):
    hidden = orchestrator._hidden_before(before, None, "", orchestrator._phase3_tree_score(before),
                                         orchestrator._phase3_tree_score(after))
    return orchestrator._attempt_not_kept(before, after, hidden, orchestrator._booted(before, ""),
                                          orchestrator._booted(after, ""))


def _revert(orchestrator, after):
    return orchestrator._revert_attempt(1, "it retired none of the 11 blockers it faced", after, (None, None))


# ---------------------------------------------------------------- the decision


def test_an_attempt_that_only_adds_blockers_is_not_kept(orchestrator):
    assert _not_kept(orchestrator, _blockers(11), _blockers(45)) == "it retired none of the 11 blockers it faced"


def test_an_attempt_that_retires_blockers_is_kept(orchestrator):
    assert _not_kept(orchestrator, _blockers(11), _blockers(3)) is None


def test_an_attempt_that_leaves_the_same_blockers_is_not_kept(orchestrator):
    """A neutral pass retired nothing: there is no reason to keep its edits."""
    assert _not_kept(orchestrator, _blockers(11), _blockers(11)).startswith("it retired none")


def _startup(message="mapper config: sql_alchemy.py line 107: Mapper has no property 'bill'"):
    return ValidationIssue("blocker", message)


def test_an_attempt_that_trades_a_blocker_for_a_dead_app_is_not_kept(orchestrator):
    """A count cannot see a trade.

    Run mbzbzhq9 held 13 blockers flat across six Phase 3 attempts while
    swapping one hard blocker for a broken ORM mapper, and the delivered app
    returned 500 on every endpoint - 0/4 on the live workflow probe.
    """
    assert _not_kept(orchestrator, _blockers(1), [_startup()]).startswith("it added an observed failure: mapper config:")


def test_failing_to_fix_a_startup_blocker_retires_nothing(orchestrator):
    assert _not_kept(orchestrator, [_startup()], [_startup()]).startswith("it retired none")


@pytest.mark.parametrize("message", [
    "mapper config: x", "application startup: x", "python contract: x",
    "missing module: x",
])
def test_every_startup_class_counts_as_an_observed_failure(orchestrator, message):
    reason = _not_kept(orchestrator, _blockers(1, prefix="action contract"),
                       [ValidationIssue("blocker", message)])
    assert reason.startswith("it added an observed failure")


def test_a_traded_defect_is_an_added_observed_failure(orchestrator):
    """Three defects fixed and three different ones made is not progress."""
    reason = _not_kept(orchestrator, _blockers(3, prefix="data contract"), _blockers(3, prefix="action contract"))
    assert reason == "it added an observed failure: action contract: defect 0"


def test_a_new_judgment_or_unknown_is_not_an_observed_failure(orchestrator):
    after = [ValidationIssue("blocker", "requirement: R2 — Books list their author is not implemented")]
    assert _not_kept(orchestrator, _blockers(1), after) is None


# ---------------------------------------------------------------- the revert


def test_a_reverted_attempt_restores_the_tree_and_says_so(orchestrator, monkeypatch):
    restored = {}
    monkeypatch.setattr(orchestrator, "_restore_snapshot",
                        lambda: restored.setdefault("called", True) or True)
    monkeypatch.setattr(orchestrator, "_collect_validation_issues", lambda: _blockers(11))

    assert _revert(orchestrator, _blockers(45)) is True
    assert restored.get("called") is True
    assert orchestrator._phase3_rolled_back is True


def test_the_discarded_findings_are_still_reported(orchestrator, monkeypatch):
    """And ONLY the discarded ones.

    The note is the text a human reads to decide whether the rollback hid a
    real defect, so it has to name what the discarded tree added. It used to
    print the whole final blocker list: on one run on disk its three
    "discarded" findings all described the tree that SHIPPED. Here defects 0-10
    are in both trees and 11-44 only in the discarded one.
    """
    monkeypatch.setattr(orchestrator, "_restore_snapshot", lambda: True)
    monkeypatch.setattr(orchestrator, "_collect_validation_issues", lambda: _blockers(11))

    _revert(orchestrator, _blockers(45))

    notes = [i.message for i in orchestrator._validation_issues if "rolled back" in i.message]
    assert len(notes) == 1
    assert "fix attempt 1 was rolled back - it retired none of the 11 blockers it faced" in notes[0]
    assert "defect 11" in notes[0], "the discarded findings must survive the rollback"
    assert "defect 0;" not in notes[0], "a finding the shipped tree still has is not discarded"
    assert "more not listed" in notes[0], "a truncated list must say that it is truncated"


def test_a_failed_restore_reports_it(orchestrator, monkeypatch):
    monkeypatch.setattr(orchestrator, "_restore_snapshot", lambda: False)

    assert _revert(orchestrator, _blockers(45)) is False
    assert orchestrator._phase3_rolled_back is False


def test_the_recipe_records_the_rollback(orchestrator, monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator, "_restore_snapshot", lambda: True)
    monkeypatch.setattr(orchestrator, "_collect_validation_issues", lambda: _blockers(11))
    _revert(orchestrator, _blockers(45))

    import json
    orchestrator._save_recipe("do the thing", 1.0)
    recipe = json.loads((tmp_path / ".besser_recipe.json").read_text(encoding="utf-8"))

    assert recipe["phase3_rolled_back"] is True


def _run_cycle(orchestrator, after, edits=1, obligations_move=False):
    """The real cycle: 11 blockers on entry, one attempt, then ``after``."""
    from unittest.mock import patch

    orchestrator.auto_fix_issues = True
    state = {"collects": 0, "attempts": 0}

    def stub_collect():
        state["collects"] += 1
        return _blockers(11) if state["collects"] != 2 else after

    def stub_fix(blockers, is_first_attempt):
        state["attempts"] += 1
        return edits

    with patch.object(orchestrator, "_collect_validation_issues", side_effect=stub_collect), \
            patch.object(orchestrator, "_create_snapshot", return_value=True), \
            patch.object(orchestrator, "_restore_snapshot", return_value=True) as rollback, \
            patch.object(orchestrator, "_invoke_phase3_fix_loop", side_effect=stub_fix), \
            patch.object(orchestrator, "_repair_obligations_revision",
                         side_effect=lambda: f"obl-{state['attempts'] if obligations_move else 0}"), \
            patch.object(orchestrator, "_workspace_revision",
                         side_effect=lambda: f"rev-{state['attempts'] * edits}"):
        orchestrator._run_phase3_validation()
    return rollback, state


def test_run_phase3_validation_actually_calls_the_rollback(orchestrator):
    """The call site, not the method - that is what was missing. 11 blockers on
    entry, the attempt writes, the count goes to 45: trilraak's exact shape."""
    rollback, state = _run_cycle(orchestrator, _blockers(45))

    rollback.assert_called_once()
    assert state["attempts"] == 1
    assert orchestrator._phase3_rolled_back is True
    assert any("rolled back" in i.message for i in orchestrator._validation_issues)


def test_a_repair_that_wrote_nothing_is_never_rolled_back(orchestrator):
    """A blocker appearing with no edit is exposed truth or judge variance.

    Two judge passes on one app returned 12 then 22 missing requirements, and
    a syntax error that surfaces while nothing was written was always there.
    There is also nothing to undo: the tree is already the pre-repair tree.
    """
    rollback, state = _run_cycle(orchestrator, _blockers(45), edits=0, obligations_move=True)

    rollback.assert_not_called()
    assert state["attempts"] == 1
    assert orchestrator._phase3_rolled_back is False


def test_a_rollback_reopens_work_it_undid(orchestrator, monkeypatch):
    """The tree goes back; the checklist must go back with it.

    Run uvobkl4u shipped all six action endpoints as done/verified while every
    one was an HTTP 501 stub again: Phase 3 implemented them, the rollback put
    the stubs back, and nothing re-checked. The recipe claimed six endpoints
    the delivered app did not have.
    """
    implemented = {"yes": True}
    orchestrator.executor.set_tasks([
        {"text": "Implement POST /booking/{id}/methods/cancel/",
         "verify": lambda: implemented["yes"]},
        {"text": "Something with no verifier"},
    ])
    orchestrator.executor._task_list({"action": "done", "id": 1})
    assert orchestrator.executor.task_snapshot()[0]["done"] is True

    def restore():
        implemented["yes"] = False  # the stub is back on disk
        return True

    monkeypatch.setattr(orchestrator, "_restore_snapshot", restore)
    monkeypatch.setattr(orchestrator, "_collect_validation_issues", lambda: _blockers(11))

    _revert(orchestrator, _blockers(45))

    snapshot = {t["id"]: t for t in orchestrator.executor.task_snapshot()}
    assert snapshot[1]["done"] is False, "a discarded implementation still reported done"
    assert snapshot[1]["verification"] == "unverified"
    note = next(i.message for i in orchestrator._validation_issues if "rolled back" in i.message)
    assert "1 checklist item(s) verified during the attempt are open again" in note


def test_a_rollback_keeps_work_it_did_not_undo(orchestrator, monkeypatch):
    """Only items whose verifier now fails are reopened."""
    orchestrator.executor.set_tasks([
        {"text": "Still true after the restore", "verify": lambda: True},
    ])
    orchestrator.executor._task_list({"action": "done", "id": 1})

    monkeypatch.setattr(orchestrator, "_restore_snapshot", lambda: True)
    monkeypatch.setattr(orchestrator, "_collect_validation_issues", lambda: _blockers(11))

    _revert(orchestrator, _blockers(45))

    assert orchestrator.executor.task_snapshot()[0]["done"] is True


def _edit(orchestrator, path):
    orchestrator.tool_calls_log.append(
        {"turn": 1, "tool": "modify_file", "input": {"path": path}, "success": True, "status": "ok"})


def test_a_restore_marks_the_edits_it_discarded(orchestrator, tmp_path):
    (tmp_path / "app.py").write_text("v1\n", encoding="utf-8")
    _edit(orchestrator, "app.py")
    assert orchestrator._create_snapshot()
    _edit(orchestrator, "routes.py")
    _edit(orchestrator, "app.py")

    assert orchestrator._restore_snapshot()

    assert [call.get("rolled_back") for call in orchestrator.tool_calls_log] == [None, True, True]


def test_a_later_snapshot_keeps_the_edits_it_holds(orchestrator, tmp_path):
    (tmp_path / "app.py").write_text("v1\n", encoding="utf-8")
    assert orchestrator._create_snapshot()
    _edit(orchestrator, "app.py")
    assert orchestrator._create_snapshot()
    _edit(orchestrator, "routes.py")

    assert orchestrator._restore_snapshot()

    assert [call.get("rolled_back") for call in orchestrator.tool_calls_log] == [None, True]


def test_a_failed_restore_marks_nothing(orchestrator, monkeypatch):
    _edit(orchestrator, "app.py")
    assert not orchestrator._restore_snapshot()  # no snapshot was taken
    assert "rolled_back" not in orchestrator.tool_calls_log[0]
