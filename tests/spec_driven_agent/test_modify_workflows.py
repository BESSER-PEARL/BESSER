"""The API workflows a run verified travel with the app, and a modify run replays them.

Before: a modify run started with no workflow, and the recipe, once over its size budget,
kept a workflow's method and path only. Now ``.besser_api_workflows.json`` keeps each
definition whole within its own bound, and a workflow that passed on the previous version
and fails on the delivered tree is a regression to confirm, never a blocker.
"""

from __future__ import annotations

import json
import logging

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, ValidationIssue
from besser.spec_driven_agent.state import api_workflows
from besser.spec_driven_agent.state.api_workflows import WORKFLOWS_FILENAME, load_workflows, save_workflows
from besser.spec_driven_agent.validation import api_probe
from besser.spec_driven_agent.validation.issues import OPEN_SINCE_PREVIOUS, is_completion_issue, is_not_confirmed


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


def _orch(tmp_path) -> LLMOrchestrator:
    cls = Class(name="Book")
    cls.attributes = {Property(name="title", type=PrimitiveDataType("str"), is_id=True)}
    return LLMOrchestrator(
        llm_client=_Client(), domain_model=DomainModel(name="Library", types={cls}), output_dir=str(tmp_path),
        enable_tracing=False, enable_checkpointing=False, enable_toolchain_validation=False,
    )


def _requests(title="Dune"):
    return [{"method": "POST", "path": "/book/", "json": {"title": title, "pages": 412}, "expected_status": 201},
            {"method": "GET", "path": "/book/{{0.id}}/", "expected_fields": {"title": title}}]


def _workflow(scenario_id, status="passed", title="Dune", backend=None):
    return {"scenario_id": scenario_id, "scenario": {"requests": _requests(title), "backend": backend},
            "status": status}


def _record(scenario_id, title="Dune", revision="r1"):
    """A ``test_api`` record as the run retains it."""
    return {"scenario_id": scenario_id, "scenario": {"requests": _requests(title), "backend": None},
            "revision": revision, "report": {"status": "passed", "boot": "ok", "responses": [
                {"method": "POST", "path": "/book/", "status": 201, "json": {"id": 1, "title": title}}]}}


def test_the_workflows_file_keeps_request_bodies_and_reads_back(tmp_path):
    save_workflows(str(tmp_path), [_workflow("add-book"), _workflow("old", status="failed")])

    assert load_workflows(str(tmp_path)) == [_workflow("add-book"), _workflow("old", status="failed")]


def test_the_workflows_file_is_bounded_whole_workflow_by_whole_workflow(tmp_path, monkeypatch):
    monkeypatch.setattr(api_workflows, "MAX_WORKFLOWS_BYTES", 900)
    big = _workflow("big", title="x" * 1000)

    save_workflows(str(tmp_path), [_workflow("first"), big, _workflow("third")])

    assert [w["scenario_id"] for w in load_workflows(str(tmp_path))] == ["first", "third"]


@pytest.mark.parametrize("content", [
    "not json", json.dumps({"workflows": [{"scenario_id": "x", "scenario": {"requests": "rm -rf"}}]}),
    json.dumps({"workflows": [_workflow("a")]}) + " " * (api_workflows.MAX_WORKFLOWS_BYTES + 1),
], ids=["not-json", "invalid-definition", "oversized"])
def test_a_planted_or_oversized_workflows_file_is_not_read(tmp_path, content):
    (tmp_path / WORKFLOWS_FILENAME).write_text(content, encoding="utf-8")

    assert load_workflows(str(tmp_path)) == []


def test_a_recipe_over_its_budget_still_leaves_the_workflow_bodies(tmp_path):
    """Before: the size cuts left {'method': 'POST', 'path': '/book/'} and nothing to replay."""
    orch = _orch(tmp_path)
    orch._api_scenarios = {"named:add-book": _record("add-book", revision=orch._workspace_revision())}
    # Kept by the first cut (a path), so the recipe is still over its budget at the second.
    orch.tool_calls_log = [{"tool": "read_file", "input": {"path": f"{'d/' * 150}f{i}.py"}, "success": True}
                           for i in range(600)]

    orch._save_recipe("add books", elapsed=1.0)

    recipe = json.loads((tmp_path / ".besser_recipe.json").read_text(encoding="utf-8"))
    assert recipe["recipe_shortened"]["api_bodies_cut"] > 0
    assert "json" not in recipe["api_scenarios"][0]["scenario"]["requests"][0]
    [kept] = load_workflows(str(tmp_path))
    assert kept["scenario"]["requests"][0]["json"] == {"title": "Dune", "pages": 412}
    assert kept["status"] == "passed"


def _replayed(monkeypatch, report):
    calls = []

    def probe(output_dir, requests, backend=None):
        calls.append(requests)
        return report

    monkeypatch.setattr(api_probe, "probe_api_scenario", probe)
    return calls


_FAILED = {"status": "failed", "boot": "ok", "assertion_failures": [
    {"request": 1, "field": "title", "expected": "Dune", "actual": None}], "responses": []}


def test_a_workflow_the_change_broke_is_a_regression_to_confirm_not_a_blocker(tmp_path, monkeypatch):
    orch = _orch(tmp_path)
    orch._modify_mode = True
    orch._seed_workflows = [_workflow("add-book"), _workflow("never-passed", status="failed")]
    calls = _replayed(monkeypatch, _FAILED)

    orch._modify_end_checks()

    assert calls == [_requests()]
    [issue] = orch._validation_issues
    assert issue.message.startswith("regression to confirm: api scenario add-book passed on the previous version "
                                    "and fails now: ")
    assert not is_completion_issue(issue) and is_not_confirmed(issue)
    assert orch._seed_regressions == []
    assert orch._api_workflows_to_keep() == []


@pytest.mark.parametrize("report", [{"status": "passed", "boot": "ok", "responses": []},
                                    {"status": "error", "boot": "timeout", "responses": []},
                                    {"status": "error", "boot": "not_started",
                                     "error": "Select exactly one generated FastAPI backend"}])
def test_a_workflow_that_passes_or_could_not_run_is_kept_for_the_next_run(tmp_path, monkeypatch, report):
    orch = _orch(tmp_path)
    orch._modify_mode = True
    orch._seed_workflows = [_workflow("add-book")]
    _replayed(monkeypatch, report)

    orch._modify_end_checks()

    assert orch._validation_issues == []
    assert orch._api_workflows_to_keep() == [_workflow("add-book")]


_STARTUP = "application startup: backend: ImportError: cannot import name 'Book'"


@pytest.mark.parametrize("startup", [ValidationIssue("blocker", _STARTUP),
                                     ValidationIssue("warning", OPEN_SINCE_PREVIOUS + _STARTUP)],
                         ids=["new", "open-since-previous"])
def test_a_workflow_is_not_replayed_on_a_tree_whose_backend_does_not_start(tmp_path, monkeypatch, startup):
    """Before: each replay booted again and repeated the startup failure as its own regression;
    and a startup failure the previous version already had, once labelled, did not stop it."""
    orch = _orch(tmp_path)
    orch._modify_mode = True
    orch._seed_workflows = [_workflow("add-book")]
    orch._validation_issues = [startup]
    calls = _replayed(monkeypatch, {"status": "error", "boot": "failed", "error": "ImportError", "responses": []})

    orch._modify_end_checks()

    assert calls == [] and orch._validation_issues == [startup]
    assert orch._api_workflows_to_keep() == [_workflow("add-book")]


@pytest.mark.parametrize("present", [True, False], ids=["backend-present", "backend-removed"])
def test_a_workflow_whose_backend_folder_is_gone_is_dropped(tmp_path, monkeypatch, caplog, present):
    """Before: the probe found no such service, so the workflow counted as not run and was carried forever."""
    if present:
        (tmp_path / "services" / "api").mkdir(parents=True)
    orch = _orch(tmp_path)
    orch._modify_mode = True
    orch._seed_workflows = [_workflow("add-book", backend="services/api")]
    calls = _replayed(monkeypatch, {"status": "passed", "boot": "ok", "responses": []})

    with caplog.at_level(logging.INFO):
        orch._modify_end_checks()

    assert orch._validation_issues == []
    if present:
        assert len(calls) == 1 and orch._api_workflows_to_keep() == [_workflow("add-book", backend="services/api")]
    else:
        assert calls == [] and orch._api_workflows_to_keep() == []
        assert "add-book" in caplog.text and "services/api" in caplog.text


def test_a_pass_on_an_earlier_tree_is_saved_without_the_passed_claim(tmp_path):
    """Before: the last report's status was saved whatever tree it ran on, and the next run replayed it."""
    orch = _orch(tmp_path)
    current = orch._workspace_revision()
    orch._api_scenarios = {"named:add-book": _record("add-book", revision=current),
                           "named:old": _record("old", revision="an-earlier-tree")}

    assert [(w["scenario_id"], w["status"]) for w in orch._api_workflows_to_keep()] == [
        ("add-book", "passed"), ("old", None)]


def test_a_workflow_this_run_kept_under_the_same_id_is_not_replayed(tmp_path, monkeypatch):
    orch = _orch(tmp_path)
    orch._modify_mode = True
    orch._seed_workflows = [_workflow("add-book")]
    orch._api_scenarios = {"named:add-book": _record("add-book", title="Emma")}
    calls = _replayed(monkeypatch, _FAILED)

    orch._modify_end_checks()

    assert calls == [] and orch._validation_issues == []
    assert [w["scenario"]["requests"][0]["json"]["title"] for w in orch._api_workflows_to_keep()] == ["Emma"]


def test_a_modify_run_replays_the_seeds_workflows_on_the_delivered_tree(tmp_path, monkeypatch):
    save_workflows(str(tmp_path), [_workflow("add-book")])
    orch = _orch(tmp_path)
    for name, stub in {"_validate_phase1_output": lambda: [], "_run_phase2": lambda instr, extra_issues=None: None,
                       "_create_snapshot": lambda: None, "_remove_snapshot": lambda: None,
                       "_run_phase3_validation": lambda: None}.items():
        monkeypatch.setattr(orch, name, stub)
    _replayed(monkeypatch, _FAILED)

    orch.modify("Rename the book title field")

    assert any(i.message.startswith("regression to confirm: api scenario add-book") for i in orch._validation_issues)
    # Not carried: the next run does not replay a workflow the change broke.
    assert load_workflows(str(tmp_path)) == []
