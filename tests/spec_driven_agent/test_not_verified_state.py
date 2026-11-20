"""A run where a whole class of checks left no evidence is "not verified", never plain "completed".

A finding that is not confirmed (one citation not located, one guessed create refused) is no
blocker. But when NO requirement verdict exists for the delivered app (the extraction or the
judge failed, or the run's deadline stopped the judge), when nothing ever started the backend,
when it started and not one record could be created, or when its frontend was never built, the
run has nothing to show for that class. Before, such a run ended ``completed`` and the card read
"Application ready"; an app the probe could not start at all ended "incomplete" instead, so two
halves of one state gave opposite headlines.
"""

from __future__ import annotations

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, ValidationIssue
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.validation.issues import (
    _classify_issue,
    is_completion_issue,
    is_not_confirmed,
    required_dependency_setup,
    runtime_check_unverified,
    unresolved_defects,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven import run_record


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self) -> None:
        self.usage = UsageTracker("mock-model")

    def chat(self, **kwargs):  # pragma: no cover - no test here may call the LLM
        raise AssertionError("no LLM call expected")


@pytest.fixture
def orch(simple_library_book_model, tmp_path):
    orchestrator = LLMOrchestrator(
        llm_client=_Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False, enable_tracing=False,
    )
    orchestrator._validation_issues = []
    return orchestrator


def _booted(orch, entities: dict) -> None:
    """The probe started the backend on the delivered tree, with these per-entity verdicts."""
    orch._runtime_probe_facts = (orch._workspace_revision(), [
        {"boot": "ok", "entities": {name: {"verdict": verdict} for name, verdict in entities.items()}}])


def _stop(gaps, *, incomplete=False, blockers=0, not_confirmed=0) -> run_record.RunFacts:
    facts = run_record.RunFacts()
    facts.stop(stop_reason="completed", cancelled=False, cap_breach=False, timed_out=False,
               incomplete=incomplete, blocker_count=blockers, cost=0.1, not_confirmed=not_confirmed, gaps=gaps)
    return facts


# ---------------------------------------------------------------------------
# Requirements: no verdict exists for the delivered tree
# ---------------------------------------------------------------------------

def test_a_run_out_of_time_before_the_judge_does_not_end_plain_completed(orch):
    """With the merged deadline, LLM calls stop before the cap: the judge is skipped routinely."""
    orch._requirements = [{"id": 1, "text": "A book's title is required.", "kind": "validation"}]
    orch.deadline._time_left = lambda: 5.0  # far below what one LLM call needs

    findings = orch._collect_requirement_issues()
    orch._validation_issues = [_classify_issue(message) for message in findings]

    assert len(findings) == 1 and findings[0].startswith("requirement unverified: not checked, too little")
    assert not any(is_completion_issue(issue) for issue in orch._validation_issues)
    gaps = orch.verification_gaps()
    assert gaps == [("requirements", "too little time was left", False)]
    facts = _stop(gaps, not_confirmed=1)
    assert facts.outcome == "completed_unverified"
    assert facts.stop_message == ("The app was generated, but it was not checked against your request "
                                  "(too little time was left).")


@pytest.mark.parametrize("message, reason", [
    ("requirement unverified: requirement extraction failed; the original specification has not been checked.",
     "your request could not be turned into requirements"),
    ("validation unverified: the requirements judge did not run completely (it returned no verdicts; coverage "
     "of the original specification has not been verified) - required verification was SKIPPED",
     "the requirements check returned no verdict"),
    ("requirement unverified: Phase 3 validation did not run: cost budget exhausted. The current application "
     "has not completed final verification.", "the cost limit was reached"),
    ("requirement unverified: not checked, the run was cancelled; original-specification coverage is not "
     "verified.", "the run was cancelled"),
])
def test_no_requirement_verdict_is_a_requirements_gap(orch, message, reason):
    orch._validation_issues = [_classify_issue(message)]
    assert ("requirements", reason, False) in orch.verification_gaps()


def test_one_unlocated_citation_is_no_gap(orch):
    orch._validation_issues = [_classify_issue(
        "requirement unverified: R2 — Books have an ISBN — the cited evidence 'models.py: isbn' does not verify it")]
    assert orch.verification_gaps() == []


# ---------------------------------------------------------------------------
# Runtime: the backend was never started
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("finding, reason", [
    (runtime_check_unverified("backend startup and data entry", "the probe timed out after 90s"),
     "its startup timed out"),
    (runtime_check_unverified("the constructibility probe", "ModuleNotFoundError: No module named 'jose'"),
     "a dependency it needs is missing"),
    (None, "the delivered version was not started"),
])
def test_an_app_nothing_started_is_not_verified_and_not_incomplete(orch, monkeypatch, finding, reason):
    """The probe timing out and no probe at all are one state now, with one headline."""
    monkeypatch.setattr(orch, "_probeable_backends", lambda: ["web_app/backend"])
    orch._runtime_probe_verdict = 1 if finding else None
    orch._validation_issues = [_classify_issue(finding)] if finding else []

    orch._apply_runtime_gate()

    # Listed once: the probe's own note already says why; with no note, the gate says it.
    gate = [i for i in orch._validation_issues if "no runtime evidence" in i.message]
    assert len(gate) == (0 if finding else 1)
    assert len(orch._validation_issues) == 1
    assert all(is_not_confirmed(i) and not is_completion_issue(i) for i in orch._validation_issues)
    assert unresolved_defects([i for i in orch._validation_issues if is_completion_issue(i)]) == []
    assert ("runtime", reason, False) in orch.verification_gaps()
    assert _stop(orch.verification_gaps(), not_confirmed=1).stop_message.startswith(
        "The app was generated, but it was never started, so nothing shows it runs (")


def test_a_workflow_that_booted_the_app_is_runtime_evidence(orch, monkeypatch):
    monkeypatch.setattr(orch, "_probeable_backends", lambda: ["web_app/backend"])
    orch._api_scenarios = {"named:books": {"revision": orch._workspace_revision(), "scenario": {},
                                           "report": {"status": "failed", "boot": "ok", "responses": []}}}
    orch._runtime_probe_verdict = None
    orch._apply_runtime_gate()
    assert orch._validation_issues == []
    assert orch.verification_gaps() == []


def test_an_observed_startup_failure_still_makes_the_run_incomplete(orch, monkeypatch):
    monkeypatch.setattr(orch, "_probeable_backends", lambda: ["web_app/backend"])
    orch._runtime_probe_verdict = 2  # _RUNTIME_FAILED
    orch._validation_issues = [ValidationIssue("blocker", "application startup: main_api.py crashed on import")]
    orch._apply_runtime_gate()
    gate = [i for i in orch._validation_issues if i.message.startswith("runtime gate:")]
    assert gate and is_completion_issue(gate[0]) and unresolved_defects(gate)


# ---------------------------------------------------------------------------
# Records: started, but nothing could be created through the API (R4)
# ---------------------------------------------------------------------------

def test_no_entity_created_and_no_workflow_record_is_a_records_gap(orch, monkeypatch):
    monkeypatch.setattr(orch, "_probeable_backends", lambda: ["web_app/backend"])
    _booted(orch, {"Book": "rejected", "Library": "rejected"})
    orch._validation_issues = [_classify_issue(
        "runtime unverified: create unverified: web_app/backend: POST /book/ - autogenerated input was refused "
        "(422: title required)")]
    assert orch.verification_gaps() == [("records", "every create the check tried was refused", False)]
    assert "no record could be created through its API" in run_record.not_verified_message(orch.verification_gaps())


def test_one_created_entity_or_a_workflow_record_is_evidence(orch, monkeypatch):
    monkeypatch.setattr(orch, "_probeable_backends", lambda: ["web_app/backend"])
    _booted(orch, {"Book": "rejected", "Library": "created"})
    assert orch.verification_gaps() == []

    _booted(orch, {"Book": "rejected"})
    orch._api_scenarios = {"named:books": {"revision": orch._workspace_revision(), "scenario": {}, "report": {
        "status": "passed", "boot": "ok", "responses": [
            {"method": "POST", "path": "/book/", "status": 201, "json": {"id": 1, "title": "Dune"}},
            {"method": "GET", "path": "/book/1", "status": 200, "json": {"id": 1, "title": "Dune"}}]}}}
    assert orch.verification_gaps() == []


# ---------------------------------------------------------------------------
# Build: a frontend that was never built
# ---------------------------------------------------------------------------

def test_a_frontend_never_built_is_a_build_gap_and_setup_is_not_a_blocker_at_the_end(orch, tmp_path):
    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "package.json").write_text('{"name": "app"}', encoding="utf-8")
    (tmp_path / "frontend" / "App.tsx").write_text("export default () => null;\n", encoding="utf-8")
    orch.enable_toolchain_validation = orch.allow_shell_tools = True
    setup = _classify_issue(required_dependency_setup("tsc", "frontend"))
    orch._validation_issues = [setup]

    assert setup.severity == "blocker", "still repair-eligible: the agent may run the install"
    assert orch.verification_gaps() == [("build", "its dependencies were not installed", False)]

    # Still open when Phase 3 ends: rewritten once, so every surface reads "did not run".
    orch._settle_open_setup()
    settled, = orch._validation_issues
    assert settled.message.startswith("validation unverified: tsc did not run")
    assert is_not_confirmed(settled) and not is_completion_issue(settled)
    assert orch.verification_gaps() == [("build", "its dependencies were not installed", False)]


def test_a_backend_only_app_has_no_build_gap(orch):
    orch._validation_issues = []
    assert orch.verification_gaps() == []


# ---------------------------------------------------------------------------
# The run record
# ---------------------------------------------------------------------------

def test_the_outcome_and_message_of_each_state():
    gap = [("runtime", "its startup timed out")]
    assert _stop([]).outcome == "completed"
    assert _stop(gap).outcome == "completed_unverified"
    # A known defect, or a blocker the run counts, keeps its own outcome and message.
    assert _stop(gap, incomplete=True, blockers=1).outcome == "completed_with_blockers"
    assert _stop(gap, blockers=1).outcome == "completed_with_blockers"
    assert _stop(gap, blockers=1).stop_message == run_record.not_verified_message(gap)
    assert run_record.not_verified_message([("requirements", "too little time was left"), *gap]) == (
        "The app was generated, but it was not checked against your request (too little time was left); and "
        "it was never started, so nothing shows it runs (its startup timed out).")


def test_the_run_section_and_end_line_carry_the_classes():
    facts = _stop([("build", "the build did not run")], not_confirmed=1)
    section = run_record.run_section(facts, object(), provider="free", models=[], request_chars=0)
    assert section["outcome"] == "completed_unverified"
    assert section["not_verified"] == [{"check": "build", "reason": "the build did not run",
                                        "configured_off": False}]
    assert " not_verified=build " in run_record.end_line("r1", facts)


# ---------------------------------------------------------------------------
# A class the server has switched off is not an event of the run
# ---------------------------------------------------------------------------

def test_a_check_switched_off_on_the_server_is_reported_apart(orch, tmp_path):
    """A self-hosted server without the toolchain must not headline every web app "not verified"."""
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "package.json").write_text('{"name": "app"}', encoding="utf-8")
    orch.enable_toolchain_validation, orch.allow_shell_tools = False, True
    orch.enable_requirements_ledger = False
    orch._instructions = "Build a library app where every book has a title."

    gaps = orch.verification_gaps()
    assert ("build", "the build check is off on this server", True) in gaps
    assert ("requirements", "the requirements check is off on this server", True) in gaps
    facts = _stop(gaps)
    assert facts.outcome == "completed" and facts.stop_message is None
    section = run_record.run_section(facts, object(), provider="free", models=[], request_chars=0)
    assert {gap["check"] for gap in section["not_verified"] if gap["configured_off"]} == {"build", "requirements"}
    line = run_record.end_line("r1", facts)
    assert " not_verified=- " in line and " checks_off=" in line


def test_the_startup_check_switched_off_is_a_configuration_gap(orch, monkeypatch):
    monkeypatch.setattr(orch, "_probeable_backends", lambda: ["web_app/backend"])
    orch.enable_import_smoke_check = False
    assert orch.verification_gaps() == [("runtime", "the startup check is off on this server", True)]


def test_a_provider_quota_is_not_called_a_cost_limit(orch):
    orch._validation_issues = [_classify_issue(
        "requirement unverified: Phase 3 validation did not run: provider quota exhausted in Phase 2. "
        "The current application has not completed final verification.")]
    assert ("requirements", "the model provider's quota ran out", False) in orch.verification_gaps()


def test_verdicts_of_an_earlier_version_are_not_shown_for_an_unjudged_one(orch, tmp_path):
    """The deadline skipped the judge on the final tree: the last verdicts answer an earlier one,
    so the report must not say the code was found. Before: "found in the code" for all of them."""
    from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
        build_verification_report,
    )

    orch._requirements = [{"id": 1, "text": "A book's title is required.", "kind": "validation"}]
    orch._requirement_verdicts = [{"id": 1, "text": "A book's title is required.", "kind": "validation",
                                   "status": "implemented", "evidence": "models.py: title = Column(nullable=False)",
                                   "note": ""}]
    orch._requirement_verdicts_key = "the key of an earlier version"
    orch.deadline._time_left = lambda: 5.0
    orch._validation_issues = [_classify_issue(m) for m in orch._collect_requirement_issues()]

    orch._save_recipe("Build a library app where every book has a title.", 1.0)
    import json
    recipe = json.loads((tmp_path / ".besser_recipe.json").read_text(encoding="utf-8"))
    verdict, = recipe["requirements"]
    assert verdict["status"] == "unverified" and verdict["note"] == orch._NOT_JUDGED_ON_DELIVERED
    report = build_verification_report(recipe)
    assert report.foundInCode == [] and report.verified == []
    assert [item.id for item in report.notVerified if item.kind == "requirement"] == ["R1"]


def test_a_modify_run_says_the_app_was_updated():
    gap = [("runtime", "its startup timed out", False)]
    assert run_record.not_verified_message(gap, updated=True).startswith("The app was updated, but it was never")
    facts = run_record.RunFacts(mode="modify")
    facts.stop(stop_reason="completed", cancelled=False, cap_breach=False, timed_out=False,
               incomplete=False, blocker_count=0, cost=0.0, gaps=gap)
    assert facts.stop_message.startswith("The app was updated, but")
