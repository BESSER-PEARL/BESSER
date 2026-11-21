"""A modify run against its previous version: the defects it already had, and what is new.

The Phase 2 start's validate_app result and the seed recipe's findings are the previous
version's (``_seed_baseline``). At the end a defect found there again is "open since the
previous version", not this run's blocker; a hard blocker it did not have is a regression
to confirm. Findings are matched by what they test (Phase 3's identity keys), not by text.
"""

from __future__ import annotations

import json

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.planning.fix_target import parse_reported_target
from besser.spec_driven_agent.planning.requirements_ledger import source_hashes
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, ValidationIssue
from besser.spec_driven_agent.validation.issues import is_completion_issue, is_not_confirmed


class _Usage:
    estimated_cost = 0.0

    def summary(self) -> dict:
        return {"api_calls": 0, "cost_usd": 0.0}


class _Client:
    model = "test-model"
    usage = _Usage()
    max_tokens = 4096

    def chat(self, **kwargs):
        raise AssertionError("no LLM call expected")


def _orch(tmp_path, *classes: str) -> LLMOrchestrator:
    types = set()
    for name in classes:
        cls = Class(name=name)
        cls.attributes = {Property(name="title", type=PrimitiveDataType("str"), is_id=True)}
        types.add(cls)
    orch = LLMOrchestrator(
        llm_client=_Client(), domain_model=DomainModel(name="App", types=types), output_dir=str(tmp_path),
        enable_tracing=False, enable_checkpointing=False, enable_toolchain_validation=False,
    )
    orch._modify_mode = True
    return orch


_CRASH = "create contract: backend: POST /order/ - 500 Internal Server Error: IntegrityError: NOT NULL failed"
_UNDEFINED = "python contract: backend/main_api.py line {}: name 'helper' is not defined"


def test_a_defect_the_previous_version_had_is_open_since_then_not_a_blocker(tmp_path):
    """Before: re-detected as this run's blocker, repaired by Phase 3 and counted in blockerCount."""
    orch = _orch(tmp_path, "Order")
    orch._seed_baseline = [_CRASH, _UNDEFINED.format(12)]
    # The edit moved the undefined name; it is the same finding.
    found = [ValidationIssue("blocker", _CRASH), ValidationIssue("blocker", _UNDEFINED.format(40))]

    labelled = orch._promote_fix_target_findings(found)

    assert [i.message for i in labelled] == [
        "open since the previous version: " + _CRASH,
        "open since the previous version: " + _UNDEFINED.format(40)]
    assert all(i.severity == "warning" and not is_completion_issue(i) and is_not_confirmed(i) for i in labelled)


def test_a_new_error_on_the_same_route_is_this_runs_blocker(tmp_path):
    """Matched by what it tests AND the error seen there: the same route failing otherwise is new."""
    orch = _orch(tmp_path, "Order")
    orch._seed_baseline = [_CRASH]
    other = ValidationIssue("blocker", "create contract: backend: POST /order/ - 500 Internal Server Error: "
                                       "AttributeError: 'Order' object has no attribute 'total'")

    assert orch._promote_fix_target_findings([other]) == [other]


def test_a_long_traceback_whose_lines_moved_still_matches(tmp_path):
    def crash(line):
        return f"{_CRASH} at line {line}. Traceback: " + "frame " * 400 + "end"

    orch = _orch(tmp_path, "Order")
    orch._seed_baseline = [crash(7)]

    [labelled] = orch._promote_fix_target_findings([ValidationIssue("blocker", crash(1234))])

    assert labelled.severity == "warning"


def test_in_a_fix_run_the_reported_failure_is_never_labelled_old(tmp_path):
    orch = _orch(tmp_path, "Order")
    orch._is_fix_run = True
    orch._fix_target = parse_reported_target("Creating an Order fails with a 500.", orch.domain_model, "Order")
    orch._seed_baseline = [_CRASH]

    [kept] = orch._promote_fix_target_findings([ValidationIssue("blocker", _CRASH)])

    assert kept.severity == "blocker" and not kept.message.startswith("open since")


_STARTUP = "application startup: backend: ImportError: cannot import name 'get_session' from 'database'"


@pytest.mark.parametrize("request_text, entity", [
    ("The backend does not start any more, it crashes with ImportError. Please fix it.", ""),
    ("Creating an Order fails with a 500.", "Order"),
])
def test_in_a_fix_run_a_defect_the_target_cannot_be_checked_past_stays_a_blocker(tmp_path, request_text, entity):
    """Before: a reported failure that names no entity, or a startup failure that stops the reported one
    from being checked, was labelled open since the previous version and never repaired."""
    orch = _orch(tmp_path, "Order")
    orch._is_fix_run = True
    orch._fix_target = parse_reported_target(request_text, orch.domain_model, entity)
    unrelated = _UNDEFINED.format(12)
    orch._seed_baseline = [_STARTUP, unrelated]

    startup, other = orch._promote_fix_target_findings(
        [ValidationIssue("blocker", _STARTUP), ValidationIssue("blocker", unrelated)])

    assert startup == ValidationIssue("blocker", _STARTUP)
    # A target with entities still leaves the previous version's unrelated defects to the user.
    assert other.severity == ("warning" if entity else "blocker")


def _phase2_start(orch, monkeypatch, opening):
    orch._seed_source_hashes = source_hashes(orch.output_dir)
    _opening(orch, monkeypatch, opening)


@pytest.mark.parametrize("change", ["tree", "model"])
def test_a_defect_the_run_made_before_phase2_is_not_the_previous_versions(tmp_path, monkeypatch, change):
    """Before: regeneration after a model change broke startup before Phase 2, and that finding was kept
    as the previous version's: labelled old, not repaired, and no regression."""
    orch = _orch(tmp_path, "Order")
    _phase2_start(orch, monkeypatch, [_STARTUP])
    if change == "tree":
        (tmp_path / "main_api.py").write_text("from sql_alchemy import Order\n", encoding="utf-8")
    else:
        orch._model_sync_classes = ["Invoice"]

    orch._modify_opening_issues([])
    [issue] = orch._promote_fix_target_findings([ValidationIssue("blocker", _STARTUP)])
    orch._validation_issues = [issue]
    orch._modify_end_checks()

    assert issue == ValidationIssue("blocker", _STARTUP)
    assert orch._seed_regressions == [_STARTUP]


def test_a_hard_blocker_the_previous_version_did_not_have_is_a_regression(tmp_path):
    """Before: nothing told a broken start apart from a defect the app already had."""
    orch = _orch(tmp_path, "Order")
    orch._seed_baseline = ["ruff: F401 'os' imported but unused"]
    startup = "application startup: backend: ModuleNotFoundError: No module named 'routers.order'"
    orch._validation_issues = [ValidationIssue("blocker", startup),
                               ValidationIssue("blocker", "requirement: R2 — orders can be cancelled")]

    orch._modify_end_checks()

    assert orch._seed_regressions == [startup]
    orch._save_recipe("add a discount", elapsed=1.0)
    recipe = json.loads((tmp_path / ".besser_recipe.json").read_text(encoding="utf-8"))
    assert recipe["seed_regressions"] == [startup]


def test_a_failure_its_check_could_not_reach_before_is_no_regression(tmp_path):
    """The previous version did not start, so its creates were never tried: a create failing now is new
    work this run made visible, not a regression."""
    orch = _orch(tmp_path, "Order")
    orch._seed_baseline = ["application startup: backend: SyntaxError in main_api.py"]
    orch._validation_issues = [ValidationIssue("blocker", _CRASH)]

    orch._modify_end_checks()

    assert orch._seed_regressions == []


def _opening(orch, monkeypatch, execution, frontend=()):
    """What validate_app's checks find on the tree Phase 2 starts from."""
    monkeypatch.setattr(orch, "_collect_execution_issues", lambda: list(execution))
    monkeypatch.setattr(orch, "_static_frontend_issues", lambda: list(frontend))


def test_the_opening_issues_show_a_forwarded_defect_once(tmp_path, monkeypatch):
    """Before: the seed recipe's copy and validate_app's copy were both in "Issues to fix"."""
    orch = _orch(tmp_path, "Order")
    orch._seed_recorded_issues = [_UNDEFINED.format(12)]
    forwarded = "Unresolved from the previous run: " + _UNDEFINED.format(12)
    still_open = "Unresolved from the previous run: tsc [frontend]: src/App.tsx(3,1): error TS2304"
    _opening(orch, monkeypatch, [_UNDEFINED.format(14)], ["acceptance: entity Order has no page"])
    orch._seed_source_hashes = source_hashes(str(tmp_path))

    issues = orch._modify_opening_issues([forwarded, still_open, "Fix x", "Fix x"])

    assert issues == [still_open, "Fix x"]
    assert orch._seed_baseline == [_UNDEFINED.format(14), "acceptance: entity Order has no page",
                                   _UNDEFINED.format(12)]


def test_a_resume_keeps_the_previous_versions_baseline(tmp_path, monkeypatch):
    orch = _orch(tmp_path, "Order")
    orch._seed_baseline = [_CRASH]
    state = orch._modify_checkpoint_state()

    resumed = _orch(tmp_path, "Order")
    resumed._restore_modify_state(state)
    resumed._resume_messages = []
    _opening(resumed, monkeypatch, ["ruff: F401 'os' imported but unused"])
    resumed._modify_opening_issues([])

    assert resumed._seed_baseline == [_CRASH]


def test_a_defect_labelled_open_by_an_earlier_run_is_still_forwarded(tmp_path):
    (tmp_path / ".besser_recipe.json").write_text(json.dumps({"validation_issues": [
        {"severity": "warning", "message": "open since the previous version: " + _CRASH},
        {"severity": "warning", "message": "ruff: F401 'os' imported but unused"},
    ]}), encoding="utf-8")
    orch = _orch(tmp_path, "Order")

    orch._seed_generator_files_from_recipe()

    assert orch._seed_unresolved_issues == ["Unresolved from the previous run: " + _CRASH]
    assert orch._seed_recorded_issues == [_CRASH, "ruff: F401 'os' imported but unused"]
