"""Phase 3 must not ship an app it made worse.

``_restore_snapshot`` was implemented, unit-tested in
``test_rollback_safety.py``, and **never called from production code**. Every
Phase 3 test that touches it does ``patch.object(orch, "_restore_snapshot")``,
so the missing call site was invisible.

Live run trilraak (Qwen, 2026-09-19) entered Phase 3 with 11 blockers. Attempt
2 added an association table using ``Table`` without importing it; the count
went to 45 and the run shipped that tree, with ``sql_alchemy.py`` no longer
importable and every router that star-imports it dead. The class docstring and
CLAUDE.md both promise "snapshot/rollback if fixes make things worse".

A rising count *during* the loop is legitimate — fixing an import exposes the
errors behind it — so only the final state against the Phase 3 entry state
decides, and the discarded findings are still reported.
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


def _blockers(n, prefix="python contract"):
    return [ValidationIssue("blocker", f"{prefix}: defect {i}") for i in range(n)]


def test_a_worse_phase3_is_rolled_back(orchestrator, monkeypatch):
    restored = {}
    monkeypatch.setattr(orchestrator, "_restore_snapshot",
                        lambda: restored.setdefault("called", True) or True)
    monkeypatch.setattr(orchestrator, "_collect_validation_issues",
                        lambda: _blockers(11))

    rolled = orchestrator._rollback_phase3_if_worse(_blockers(11), _blockers(45), True)

    assert rolled is True
    assert restored.get("called") is True
    assert orchestrator._phase3_rolled_back is True


def test_the_discarded_findings_are_still_reported(orchestrator, monkeypatch):
    """And ONLY the discarded ones.

    The note is the text a human reads to decide whether the rollback hid a
    real defect, so it has to name what the discarded tree added. It used to
    print the whole final blocker list: on one run on disk its three
    "discarded" findings - no frontend directory at all, ``undefined name
    'ids'``, a 409 on ``POST /orderline/`` - all described the tree that
    SHIPPED. Here defects 0-10 are in both trees and 11-44 only in the
    discarded one.
    """
    monkeypatch.setattr(orchestrator, "_restore_snapshot", lambda: True)
    monkeypatch.setattr(orchestrator, "_collect_validation_issues",
                        lambda: _blockers(11))

    orchestrator._rollback_phase3_if_worse(_blockers(11), _blockers(45), True)

    notes = [i.message for i in orchestrator._validation_issues
             if "rolled back" in i.message]
    assert len(notes) == 1
    assert "45 hard blockers against 11" in notes[0]
    assert "defect 11" in notes[0], "the discarded findings must survive the rollback"
    assert "defect 0;" not in notes[0], "a finding the shipped tree still has is not discarded"
    assert "more not listed" in notes[0], "a truncated list must say that it is truncated"


def test_an_improved_phase3_is_kept(orchestrator, monkeypatch):
    monkeypatch.setattr(orchestrator, "_restore_snapshot",
                        lambda: pytest.fail("must not roll back an improvement"))

    assert orchestrator._rollback_phase3_if_worse(_blockers(11), _blockers(3), True) is False
    assert orchestrator._phase3_rolled_back is False


def test_an_unchanged_count_is_kept(orchestrator, monkeypatch):
    monkeypatch.setattr(orchestrator, "_restore_snapshot",
                        lambda: pytest.fail("must not roll back a neutral pass"))

    assert orchestrator._rollback_phase3_if_worse(_blockers(11), _blockers(11), True) is False


def test_a_failed_restore_keeps_the_repaired_tree_and_reports_it(orchestrator, monkeypatch):
    """Losing the snapshot must not also lose the findings."""
    monkeypatch.setattr(orchestrator, "_restore_snapshot", lambda: False)

    rolled = orchestrator._rollback_phase3_if_worse(_blockers(11), _blockers(45), True)

    assert rolled is False
    assert orchestrator._phase3_rolled_back is False
    assert len(orchestrator._validation_issues) == 45


def test_the_recipe_records_the_rollback(orchestrator, monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator, "_restore_snapshot", lambda: True)
    monkeypatch.setattr(orchestrator, "_collect_validation_issues",
                        lambda: _blockers(11))
    orchestrator._rollback_phase3_if_worse(_blockers(11), _blockers(45), True)

    import json
    orchestrator._save_recipe("do the thing", 1.0)
    recipe = json.loads((tmp_path / ".besser_recipe.json").read_text(encoding="utf-8"))

    assert recipe["phase3_rolled_back"] is True


def test_run_phase3_validation_actually_calls_the_rollback(orchestrator, monkeypatch):
    """The call site, not the method — that is what was missing.

    Every existing Phase 3 test patches ``_restore_snapshot`` out, so a method
    that was never invoked still looked covered. This drives the real
    ``_run_phase3_validation`` loop: it enters with 11 blockers, the repair
    writes, the count goes to 45 and stays there, and the tree stops changing
    so the loop stalls out — trilraak's exact shape.
    """
    from unittest.mock import patch

    orchestrator.auto_fix_issues = True
    state = {"collects": 0, "writes": 0}

    def stub_collect():
        state["collects"] += 1
        return _blockers(11) if state["collects"] == 1 else _blockers(45)

    def stub_fix(blockers, is_first_attempt):
        state["writes"] += 1
        return 1

    # Changes while the repair is still writing, then settles so the loop ends.
    def stub_revision():
        return f"rev-{min(state['writes'], 2)}"

    with patch.object(orchestrator, "_collect_validation_issues", side_effect=stub_collect),          patch.object(orchestrator, "_create_snapshot"),          patch.object(orchestrator, "_restore_snapshot", return_value=True) as rollback,          patch.object(orchestrator, "_invoke_phase3_fix_loop", side_effect=stub_fix),          patch.object(orchestrator, "_workspace_revision", side_effect=stub_revision):
        orchestrator._run_phase3_validation()

    rollback.assert_called_once()
    assert orchestrator._phase3_rolled_back is True
    assert any("rolled back" in i.message for i in orchestrator._validation_issues)


def test_a_repair_that_wrote_nothing_is_never_rolled_back(orchestrator, monkeypatch):
    """A blocker appearing with no edit is exposed truth or judge variance.

    Two judge passes on one app returned 12 then 22 missing requirements, and
    a syntax error that surfaces while nothing was written was always there.
    There is also nothing to undo: the tree is already the pre-repair tree.
    """
    monkeypatch.setattr(orchestrator, "_restore_snapshot",
                        lambda: pytest.fail("nothing was written; nothing to roll back"))

    assert orchestrator._rollback_phase3_if_worse(
        _blockers(11), _blockers(45), False) is False
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

    orchestrator._rollback_phase3_if_worse(_blockers(11), _blockers(45), True)

    snapshot = {t["id"]: t for t in orchestrator.executor.task_snapshot()}
    assert snapshot[1]["done"] is False, "a discarded implementation still reported done"
    assert snapshot[1]["verification"] == "unverified"
    note = next(i.message for i in orchestrator._validation_issues if "rolled back" in i.message)
    assert "1 checklist item(s) verified during the repair are open again" in note


def test_a_rollback_keeps_work_it_did_not_undo(orchestrator, monkeypatch):
    """Only items whose verifier now fails are reopened."""
    orchestrator.executor.set_tasks([
        {"text": "Still true after the restore", "verify": lambda: True},
    ])
    orchestrator.executor._task_list({"action": "done", "id": 1})

    monkeypatch.setattr(orchestrator, "_restore_snapshot", lambda: True)
    monkeypatch.setattr(orchestrator, "_collect_validation_issues", lambda: _blockers(11))

    orchestrator._rollback_phase3_if_worse(_blockers(11), _blockers(45), True)

    assert orchestrator.executor.task_snapshot()[0]["done"] is True


def _startup(message="mapper config: sql_alchemy.py line 107: Mapper has no property 'bill'"):
    return ValidationIssue("blocker", message)


def test_a_repair_that_trades_a_blocker_for_a_dead_app_is_rolled_back(orchestrator, monkeypatch):
    """A count cannot see a trade.

    Run mbzbzhq9 held 13 blockers flat across six Phase 3 attempts while
    swapping one hard blocker for a broken ORM mapper. "Not more than we
    started with" was true, the rollback stayed silent, and the delivered app
    returned 500 on every endpoint - 0/4 on the live workflow probe.
    """
    monkeypatch.setattr(orchestrator, "_restore_snapshot", lambda: True)
    monkeypatch.setattr(orchestrator, "_collect_validation_issues", lambda: _blockers(1))

    # Same hard count on both sides: one in, one out.
    rolled = orchestrator._rollback_phase3_if_worse(_blockers(1), [_startup()], True)

    assert rolled is True
    assert orchestrator._phase3_rolled_back is True


def test_a_startup_blocker_already_present_on_entry_is_not_a_new_trade(orchestrator, monkeypatch):
    """Failing to FIX a broken mapper is not the same as breaking one."""
    monkeypatch.setattr(orchestrator, "_restore_snapshot",
                        lambda: pytest.fail("the repair did not introduce this"))

    assert orchestrator._rollback_phase3_if_worse(
        [_startup()], [_startup()], True) is False


@pytest.mark.parametrize("message", [
    "mapper config: x", "application startup: x", "python contract: x",
    "missing module: x", "undefined name: x",
])
def test_every_startup_class_counts(orchestrator, monkeypatch, message):
    monkeypatch.setattr(orchestrator, "_restore_snapshot", lambda: True)
    monkeypatch.setattr(orchestrator, "_collect_validation_issues", lambda: [])

    assert orchestrator._rollback_phase3_if_worse(
        [], [ValidationIssue("blocker", message)], True) is True


def test_a_missing_feature_is_still_not_worth_a_rollback(orchestrator, monkeypatch):
    """Only startup-class blockers get the count-independent treatment."""
    monkeypatch.setattr(orchestrator, "_restore_snapshot",
                        lambda: pytest.fail("an ordinary blocker traded flat is not a regression"))

    assert orchestrator._rollback_phase3_if_worse(
        _blockers(3), _blockers(3, prefix="action contract"), True) is False
