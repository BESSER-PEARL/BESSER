"""Phase 3 reports every validation round, so the run card can show findings and repairs (#649).

Before, the stream carried one "N blockers / M total" line for the first check
and nothing after it: which checks failed, and what each fix attempt left, was
only in the server log and the recipe.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, ValidationIssue
from besser.spec_driven_agent.providers.llm_client import UsageTracker

IMPORT_ERROR = "import [backend/main.py]: No module named 'routers.book'"
SYNTAX_ERROR = "Syntax error in backend/models.py line 3: invalid syntax"


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self) -> None:
        self.usage = UsageTracker("mock-model")

    def chat(self, **kwargs):  # pragma: no cover - the fix loop is patched out
        raise AssertionError("no LLM call expected")


@pytest.fixture
def orch(simple_library_book_model, tmp_path):
    reported: list[tuple[int, list[str], bool]] = []
    orchestrator = LLMOrchestrator(
        llm_client=_Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False,
        enable_tracing=False, auto_fix_issues=True,
        on_findings=lambda attempt, issues, restored=False: reported.append(
            (attempt, [i.message for i in issues], restored)),
    )
    orchestrator.reported = reported
    return orchestrator


def _cycle(orch, rounds, *, fix_edits=1, rollback=lambda *a, **k: False):
    """Phase 3 where validation round k returns ``rounds[k]`` and each fix attempt edits the tree."""
    revisions = iter(f"rev{n}" for n in range(100))
    with patch.object(orch, "_collect_validation_issues", side_effect=rounds), \
            patch.object(orch, "_create_snapshot", return_value=True), \
            patch.object(orch, "_restore_snapshot", return_value=True), \
            patch.object(orch, "_workspace_revision", side_effect=lambda: next(revisions)), \
            patch.object(orch, "_repair_obligations_revision", return_value="obl"), \
            patch.object(orch, "_rollback_phase3_if_worse", side_effect=rollback), \
            patch.object(orch, "_invoke_phase3_fix_loop", return_value=fix_edits):
        orch._run_phase3_repair_cycle()


def test_first_check_and_each_recheck_are_reported(orch):
    _cycle(orch, [
        [ValidationIssue("blocker", IMPORT_ERROR), ValidationIssue("blocker", SYNTAX_ERROR),
         ValidationIssue("warning", "ruff: unused import")],
        [ValidationIssue("blocker", SYNTAX_ERROR)],
        [],
    ])
    assert orch.reported == [
        (0, [IMPORT_ERROR, SYNTAX_ERROR, "ruff: unused import"], False),
        (1, [SYNTAX_ERROR], False),
        (2, [], False),
    ]


def test_a_restored_tree_is_reported_as_what_ships(orch):
    """A rollback delivers an earlier tree: its findings, not the last attempt's, describe the app."""
    first = [ValidationIssue("blocker", IMPORT_ERROR)]
    worse = [ValidationIssue("blocker", IMPORT_ERROR), ValidationIssue("blocker", SYNTAX_ERROR)]

    def restore(*_args, **_kwargs):
        orch._validation_issues = list(first)
        return True

    # Rounds that make it worse end the loop on its plateau guard.
    _cycle(orch, [first, worse, worse, worse, worse], rollback=restore)
    assert orch.reported[0] == (0, [IMPORT_ERROR], False)
    assert orch.reported[-1][1:] == ([IMPORT_ERROR], True)
    assert all(not restored for _, _, restored in orch.reported[:-1])


def test_a_failing_listener_does_not_stop_the_repair(orch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("listener down")

    orch.on_findings = boom
    _cycle(orch, [[ValidationIssue("blocker", IMPORT_ERROR)], []])
    assert orch._phase3_exit_reason == "all blockers fixed"


def test_validation_is_announced_before_the_first_check(orch):
    """So the first check's time (npm, tsc: minutes) counts as validation, not customization."""
    order: list[str] = []
    orch.on_progress = lambda turn, tool, status, *rest: order.append(tool)

    def collect():
        order.append("check")
        return []

    with patch.object(orch, "_collect_validation_issues", side_effect=collect),             patch.object(orch, "_apply_runtime_gate"):
        orch._run_phase3_validation()
    assert order[:2] == ["__validating__", "check"]


def test_no_listener_is_fine(simple_library_book_model, tmp_path):
    orchestrator = LLMOrchestrator(
        llm_client=_Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False,
        enable_tracing=False, auto_fix_issues=True,
    )
    _cycle(orchestrator, [[ValidationIssue("blocker", IMPORT_ERROR)], []])
    assert orchestrator._phase3_exit_reason == "all blockers fixed"
