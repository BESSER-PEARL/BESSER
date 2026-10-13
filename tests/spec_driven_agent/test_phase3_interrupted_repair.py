"""An interrupted Phase 3 repair must not ship its unvalidated attempt.

Run qwen1 (Qwen3-30B): attempts 1-5 each reached a strictly better tree and
re-snapshotted it (attempt 5: score [0, 3, 0, 15]). Attempt 6 edited files and
then its provider call failed. The loop broke straight out on the stop, never
validated attempt 6, and compared the best tree against attempt 5's STALE
findings - equal, so no rollback. The delivered app was attempt 6's tree, with
a page that did not compile, while the attempt-5 snapshot sat unused.
"""
import time
from unittest.mock import patch

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, ValidationIssue
from besser.spec_driven_agent.providers.llm_client import UsageTracker


@pytest.fixture
def orchestrator(simple_library_book_model, tmp_path):
    class Client:
        model = "mock-model"
        usage = UsageTracker("mock-model")

        def chat(self, **kwargs):
            raise AssertionError("no LLM call expected")

    orch = LLMOrchestrator(
        llm_client=Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False,
    )
    orch.auto_fix_issues = True
    return orch


def _blockers(n, prefix="python contract"):
    return [ValidationIssue("blocker", f"{prefix}: defect {i}") for i in range(n)]


def _provider_error(orch):
    orch._phase3_interrupted = True
    orch._phase3_interrupt_detail = "provider call failed: APIError"


def _runtime_cap(orch):
    orch._start_time = time.monotonic() - 10
    orch.max_runtime_seconds = 1


def _cost_cap(orch):
    orch.max_cost_usd = 0.0


def _drive(orch, second_attempt, state=None):
    """Entry 39 blockers; attempt 1 validates at 20 (the best tree); attempt 2
    writes and is then interrupted by ``second_attempt``."""
    state = {} if state is None else state
    state.update(collects=0, attempts=0, revision="entry")

    def collect():
        state["collects"] += 1
        return _blockers(39) if state["collects"] == 1 else _blockers(20)

    def fix(blockers, is_first_attempt):
        state["attempts"] += 1
        state["revision"] = f"attempt-{state['attempts']}"
        if state["attempts"] == 2:
            second_attempt(orch)
        return 1

    patches = (
        patch.object(orch, "_collect_validation_issues", side_effect=collect),
        patch.object(orch, "_create_snapshot"),
        patch.object(orch, "_restore_snapshot", return_value=True),
        patch.object(orch, "_invoke_phase3_fix_loop", side_effect=fix),
        patch.object(orch, "_workspace_revision", side_effect=lambda: state["revision"]),
    )
    with patches[0], patches[1], patches[2] as restore, patches[3], patches[4]:
        try:
            orch._run_phase3_validation()
        finally:
            state["restore"] = restore
    return state


@pytest.mark.parametrize("interruption", [_provider_error, _runtime_cap, _cost_cap],
                         ids=["provider-error", "runtime-cap", "cost-cap"])
def test_an_interrupted_attempt_is_replaced_by_the_best_validated_tree(orchestrator, interruption):
    state = _drive(orchestrator, interruption)

    state["restore"].assert_called_once()
    assert orchestrator._phase3_rolled_back is True
    # Validated twice (entry, attempt 1). A stop must not buy another round.
    assert state["collects"] == 2
    blockers = [i.message for i in orchestrator._validation_issues if i.severity == "blocker"]
    assert sorted(blockers) == sorted(i.message for i in _blockers(20)), \
        "the findings must describe the tree that ships - the best validated one"
    note = [i.message for i in orchestrator._validation_issues if "interrupted" in i.message]
    assert len(note) == 1 and "attempt 2" in note[0] and "discarded" in note[0]


def test_an_interruption_that_wrote_nothing_keeps_the_tree(orchestrator):
    """The tree on disk IS the last validated one; there is nothing to discard."""
    state = {"collects": 0, "attempts": 0}

    def collect():
        state["collects"] += 1
        return _blockers(39) if state["collects"] == 1 else _blockers(20)

    def fix(blockers, is_first_attempt):
        state["attempts"] += 1
        if state["attempts"] == 2:
            _provider_error(orchestrator)
        return 1 if state["attempts"] == 1 else 0

    with patch.object(orchestrator, "_collect_validation_issues", side_effect=collect), \
            patch.object(orchestrator, "_create_snapshot"), \
            patch.object(orchestrator, "_restore_snapshot") as restore, \
            patch.object(orchestrator, "_invoke_phase3_fix_loop", side_effect=fix), \
            patch.object(orchestrator, "_workspace_revision",
                         side_effect=lambda: "attempt-1" if state["attempts"] else "entry"):
        orchestrator._run_phase3_validation()

    restore.assert_not_called()
    assert orchestrator._phase3_rolled_back is False


def test_an_attempt_that_raises_is_discarded_and_the_error_still_propagates(orchestrator):
    """An auth failure must still reach the runner as INVALID_KEY."""
    from besser.spec_driven_agent.errors import InvalidApiKeyError

    def raise_after_writing(orch):
        raise InvalidApiKeyError("bad key")

    state = {}
    with pytest.raises(InvalidApiKeyError):
        _drive(orchestrator, raise_after_writing, state)

    state["restore"].assert_called_once()
    assert orchestrator._phase3_rolled_back is True
    blockers = [i.message for i in orchestrator._validation_issues if i.severity == "blocker"]
    assert sorted(blockers) == sorted(i.message for i in _blockers(20))
