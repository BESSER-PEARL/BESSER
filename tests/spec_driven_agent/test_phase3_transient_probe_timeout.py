"""One probe timeout on a repaired tree is re-measured before it counts as a loss.

Review (2026-11, PR 788): a round whose runtime probe timed out once was taken
as "the probe lost its measurement" and rolled back to the entry tree, even
when the repair was good and the probe was only slow. A did-not-run
measurement is now taken again on the same tree; a probe that still does not
run (a hung startup) rolls back as before (see test_phase3_stall_guards).
"""

from unittest.mock import patch

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.validation.constructibility import _issues_from_report
from besser.spec_driven_agent.validation.issues import _classify_issue, runtime_note_unverified


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self) -> None:
        self.usage = UsageTracker("mock-model")

    def chat(self, **kwargs):  # pragma: no cover
        raise AssertionError("no LLM call expected")


@pytest.fixture
def orch(simple_library_book_model, tmp_path):
    return LLMOrchestrator(llm_client=_Client(), domain_model=simple_library_book_model,
                           output_dir=str(tmp_path), enable_checkpointing=False,
                           enable_tracing=False, auto_fix_issues=True)


ONE_CREATE_REFUSED = {"boot": "ok", "action_calls": [], "entities": {
    **{f"E{i}": {"verdict": "created", "path": f"/e{i}/"} for i in range(4)},
    "E4": {"verdict": "rejected", "path": "/e4/",
           "attempts": [{"outcome": "rejected", "status": 422, "body": "bad", "payload": {}}]},
}}
ALL_CREATED = {"boot": "ok", "action_calls": [],
               "entities": {f"E{i}": {"verdict": "created", "path": f"/e{i}/"} for i in range(5)}}
TIMED_OUT = {"boot": "probe_error", "error": "timed out after 90s"}


def _issues(orch, report):
    raw = [runtime_note_unverified(i) if i.startswith("validation:") else i
           for i in _issues_from_report(report, "backend")]
    return list(map(_classify_issue, orch._apply_scenario_evidence(raw)))


def _run(orch, probe_reports):
    """Entry tree, then one repair round; each validation pass takes the next probe report."""
    state = {"rev": 0}
    reports = iter(probe_reports)

    def collect():
        report = next(reports)
        orch._runtime_probe_facts = (f"rev-{state['rev']}", [{"backend": "backend", **report}])
        return _issues(orch, report)

    def fix(blockers, is_first_attempt):
        state["rev"] = 1
        return 1

    with patch.object(orch, "_collect_validation_issues", side_effect=collect) as validate, \
            patch.object(orch, "_create_snapshot", return_value=True), \
            patch.object(orch, "_restore_snapshot", return_value=True) as restore, \
            patch.object(orch, "_invoke_phase3_fix_loop", side_effect=fix), \
            patch.object(orch, "_workspace_revision", side_effect=lambda: f"rev-{state['rev']}"):
        orch._run_phase3_validation()
    return validate, restore


def test_a_single_probe_timeout_does_not_roll_back_a_good_repair(orch):
    validate, restore = _run(orch, [ONE_CREATE_REFUSED, TIMED_OUT, ALL_CREATED])

    restore.assert_not_called()
    assert orch._phase3_rolled_back is False
    assert orch._phase3_exit_reason == "all blockers fixed"
    assert validate.call_count == 3


def test_a_probe_that_times_out_twice_still_rolls_back(orch):
    # The fourth pass validates the restored entry tree.
    validate, restore = _run(orch, [ONE_CREATE_REFUSED, TIMED_OUT, TIMED_OUT, ONE_CREATE_REFUSED])

    restore.assert_called_once()
    assert orch._phase3_rolled_back is True
    assert validate.call_count == 4


def test_a_failed_entry_snapshot_is_not_traced_as_taken(orch):
    """``_create_snapshot`` keeps the previous target on failure; the trace must not claim a new one."""
    from besser.spec_driven_agent.state.tracing import EVENT_SNAPSHOT

    events = []
    orch._trace = type("Trace", (), {"write": lambda self, event, **fields: events.append((event, fields))})()
    state = {"rev": 0}
    with patch.object(orch, "_collect_validation_issues", side_effect=lambda: _issues(orch, ONE_CREATE_REFUSED)), \
            patch.object(orch, "_create_snapshot", return_value=False), \
            patch.object(orch, "_restore_snapshot", return_value=True), \
            patch.object(orch, "_invoke_phase3_fix_loop", return_value=0), \
            patch.object(orch, "_workspace_revision", side_effect=lambda: f"rev-{state['rev']}"):
        orch._run_phase3_validation()

    assert ("phase3_fix_loop" not in
            [fields.get("before_phase") for event, fields in events if event == EVENT_SNAPSHOT])



def test_no_re_measurement_after_a_stop(orch):
    """Review (round 6): a cancel that lands while the repaired tree is validated
    must not buy a second full validation of that tree."""
    reports = iter([ONE_CREATE_REFUSED, TIMED_OUT, ALL_CREATED, ONE_CREATE_REFUSED])
    state = {"rev": 0}
    validated = []
    orch._should_continue = lambda: len(validated) < 2  # cancelled during the second pass

    def collect():
        validated.append(state["rev"])
        report = next(reports)
        orch._runtime_probe_facts = (f"rev-{state['rev']}", [{"backend": "backend", **report}])
        return _issues(orch, report)

    def fix(blockers, is_first_attempt):
        state["rev"] = 1
        return 1

    def restore():
        state["rev"] = 0
        return True

    with patch.object(orch, "_collect_validation_issues", side_effect=collect),             patch.object(orch, "_create_snapshot", return_value=True),             patch.object(orch, "_restore_snapshot", side_effect=restore),             patch.object(orch, "_invoke_phase3_fix_loop", side_effect=fix),             patch.object(orch, "_workspace_revision", side_effect=lambda: f"rev-{state['rev']}"):
        orch._run_phase3_validation()

    assert validated.count(1) == 1, f"the repaired tree was validated again after the stop: {validated}"
