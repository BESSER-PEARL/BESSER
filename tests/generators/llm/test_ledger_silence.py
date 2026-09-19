"""An enabled requirements ledger that extracted nothing must say so.

``extract_requirements`` has five paths that return ``None`` and only two of
them log. Live run 7aybctis (Qwen, 2026-09-19) finished with
``requirements: []`` in the recipe and no other signal — indistinguishable
from "the user asked for nothing". On gpt-5.6 the same spec yields 38-51
items, so the run silently skipped every requirement check.
"""

import pytest

from besser.generators.llm.llm_client import UsageTracker
from besser.generators.llm.orchestrator import LLMOrchestrator


@pytest.fixture
def orchestrator(simple_library_book_model, tmp_path):
    class Client:
        model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        usage = UsageTracker("mock-model")

        def chat(self, **kwargs):
            raise AssertionError("no LLM call expected")

    return LLMOrchestrator(
        llm_client=Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False,
        enable_requirements_ledger=True,
    )


def _ledger_notes(issues):
    return [str(getattr(i, "message", i)) for i in issues
            if "requirements ledger did not run" in str(getattr(i, "message", i))]


def test_a_failed_extraction_is_reported_as_a_skipped_check(orchestrator):
    orchestrator._requirements = None
    orchestrator._requirement_extraction_attempts = 2

    issues = orchestrator._collect_validation_issues()

    notes = _ledger_notes(issues)
    assert notes, "an empty ledger was reported as if it had verified the run"
    assert "2 extraction attempt(s)" in notes[0]
    assert "Qwen/Qwen3-30B-A3B-Instruct-2507" in notes[0]


def test_a_successful_extraction_reports_nothing(orchestrator):
    orchestrator._requirements = [{"id": 1, "text": "Room numbers are unique", "kind": "uniqueness"}]
    orchestrator._requirement_extraction_attempts = 1

    assert not _ledger_notes(orchestrator._collect_validation_issues())


def test_an_extraction_never_attempted_reports_nothing(orchestrator):
    """Before Phase 3 the ledger simply has not run yet; that is not a failure."""
    orchestrator._requirements = None
    orchestrator._requirement_extraction_attempts = 0

    assert not _ledger_notes(orchestrator._collect_validation_issues())


def test_a_disabled_ledger_reports_nothing(simple_library_book_model, tmp_path):
    class Client:
        model = "mock-model"
        usage = UsageTracker("mock-model")

        def chat(self, **kwargs):
            raise AssertionError("no LLM call expected")

    orchestrator = LLMOrchestrator(
        llm_client=Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False,
        enable_requirements_ledger=False,
    )
    orchestrator._requirements = None
    orchestrator._requirement_extraction_attempts = 2

    assert not _ledger_notes(orchestrator._collect_validation_issues())
