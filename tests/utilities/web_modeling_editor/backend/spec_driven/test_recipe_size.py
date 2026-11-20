"""A run's record (``.besser_recipe.json``) stays readable however much the run did.

The backend refuses a recipe over 256 KiB when it packages a run. A run's
record grows with its tool calls and API workflows, so the orchestrator bounds
it when writing it; packaging never replaces a record it cannot read, and the
verification it reports comes from the run itself.
"""
import hashlib
import json
import zipfile

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.validation.issues import ValidationIssue, _check_did_not_run
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import SmartGenerationRunner
from tests.utilities.web_modeling_editor.backend._runtime import make_runtime
from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import _build_request

READER_LIMIT = 256 * 1024
WRITE_BUDGET = READER_LIMIT // 2
TOOL_CALLS = 240
WORKFLOWS = 4
RUFF_SKIPPED = _check_did_not_run("ruff", "timed out after 30s")
TASK_UNVERIFIED = "task unverified: Invoice totals: no test exercised the total"


class _Client:
    model = "mock-model"
    usage = UsageTracker("mock-model")

    def chat(self, **kwargs):
        raise AssertionError("no LLM call expected")


def _model() -> DomainModel:
    order = Class(name="Order")
    order.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True)}
    customer = Class(name="Customer")
    customer.attributes = {Property(name="name", type=PrimitiveDataType("str"))}
    return DomainModel(name="Shop", types={order, customer})


def _workspace(tmp_path):
    result = tmp_path / "besser_llm_example" / "result"
    (result / "backend").mkdir(parents=True)
    for name in ("main_api.py", "sql_alchemy.py"):
        (result / "backend" / name).write_text("app = None\n", encoding="utf-8")
    return result


def _busy_run(result) -> LLMOrchestrator:
    """A run whose recipe, written whole, is over 400 KB: many edits and API workflows."""
    orchestrator = LLMOrchestrator(llm_client=_Client(), domain_model=_model(), output_dir=str(result),
                                   enable_checkpointing=False)
    for turn in range(TOOL_CALLS):
        if turn % 3 == 0:
            call = {"tool": "read_file", "input": {"path": "backend/main_api.py"}}
        else:
            call = {"tool": "modify_file", "input": {"path": f"backend/routes_{turn % 7}.py",
                                                       "old_text": "o" * 500, "new_text": "n" * 500}}
        orchestrator.tool_calls_log.append({"turn": turn, **call, "success": True, "status": "ok"})
    revision = orchestrator._workspace_revision()
    for index in range(WORKFLOWS):
        requests = [{"method": "POST", "path": "/orders", "json": {"note": "x" * 1500}, "expected_status": 201}
                    for _ in range(20)]
        responses = [{"index": i, "method": "POST", "path": "/orders", "status": 201, "failures": [],
                      "json": {"note": "x" * 1900}} for i in range(20)]
        orchestrator._api_scenarios[f"named:order-flow-{index}"] = {
            "scenario_id": f"order-flow-{index}", "scenario": {"requests": requests}, "revision": revision,
            "report": {"status": "failed" if index == 0 else "passed", "responses": responses},
        }
    orchestrator._validation_issues = [ValidationIssue("warning", RUFF_SKIPPED),
                                       ValidationIssue("warning", TASK_UNVERIFIED)]
    return orchestrator


def _runner(result, orchestrator) -> SmartGenerationRunner:
    runner = SmartGenerationRunner(_build_request(), runtime=make_runtime())
    runner.temp_dir = str(result.parent)
    runner._t_orchestrator = orchestrator
    return runner


def _summary(entry) -> str:
    with zipfile.ZipFile(entry.file_path) as archive:
        return archive.read("BESSER_GENERATION.md").decode("utf-8")


def test_a_busy_run_keeps_its_record_and_reports_what_its_checks_found(tmp_path):
    result = _workspace(tmp_path)
    orchestrator = _busy_run(result)

    orchestrator._save_recipe("Track orders and their customers", 30.0)
    done, entry = _runner(result, orchestrator)._package_result(str(result))

    on_disk = (result / ".besser_recipe.json").read_bytes()
    assert len(on_disk) <= READER_LIMIT
    recipe = json.loads(on_disk)
    shortened = recipe["recipe_shortened"]
    assert shortened["bytes_before"] > 400_000 and shortened["budget_bytes"] == WRITE_BUDGET
    assert shortened["tool_call_inputs_cut"] == TOOL_CALLS * 2 // 3
    assert shortened["api_bodies_cut"] == WORKFLOWS * 40
    # What the readers use is all there; the true totals are not cut.
    assert recipe["tool_calls_count"] == TOOL_CALLS
    assert recipe["model"]["classes"] and recipe["model"]["user_model"]
    assert {f["path"] for f in recipe["output_files"]} >= {"backend/main_api.py", "backend/sql_alchemy.py"}
    assert [i["message"] for i in recipe["validation_issues"]] == [RUFF_SKIPPED, TASK_UNVERIFIED]
    assert recipe["history"][-1]["instructions"] == "Track orders and their customers"
    assert {"tasks", "requirements", "acceptance_matrix", "run", "verification"} <= set(recipe)
    assert [(s["scenario_id"], s["report"]["status"]) for s in recipe["api_scenarios"]][:2] == [
        ("order-flow-0", "failed"), ("order-flow-1", "passed")]
    assert done.changedFiles == [f"backend/routes_{n}.py" for n in range(7)]

    # The report is what Phase 3 found, not "nothing could be read".
    verification = done.verification
    assert verification.counts.verified == WORKFLOWS - 1
    # The failed workflow is a test the agent wrote, not a rule the user stated: not checked, not missing.
    assert verification.counts.shippedUnenforced == 0
    assert not any("could not be read" in (item.why or "") for item in verification.notVerified)
    assert verification.counts.notVerified == 3
    assert recipe["verification"] == verification.model_dump(mode="json")
    assert ("3 confirmed by running the app's API, 0 found in the code but not run, 3 not checked, "
            "0 checked and missing in the code") in _summary(entry)


def test_a_record_too_large_for_the_writer_budget_alone_still_stays_under_the_reader_limit(tmp_path):
    """Every cut runs, in order, when the first ones are not enough; each edited file stays listed."""
    result = _workspace(tmp_path)
    orchestrator = _busy_run(result)
    orchestrator.tool_calls_log *= 6

    orchestrator._save_recipe("Track orders and their customers", 30.0)

    recipe = json.loads((result / ".besser_recipe.json").read_bytes())
    shortened = recipe["recipe_shortened"]
    assert shortened["tool_calls_dropped"] == TOOL_CALLS * 6 - 7
    assert sorted(call["input"]["path"] for call in recipe["tool_calls"]) == [
        f"backend/routes_{n}.py" for n in range(7)]
    assert recipe["tool_calls_count"] == TOOL_CALLS * 6
    assert len((result / ".besser_recipe.json").read_bytes()) <= WRITE_BUDGET


@pytest.mark.parametrize("content", [
    json.dumps({"history": [{"instructions": "Track orders"}], "padding": "x" * (1024 * 1024)}).encode(),
    b'{"history": [{"instructions": "Track orders"}], ',
], ids=["oversized", "not-json"])
def test_a_record_that_cannot_be_read_is_kept_and_reported_as_unreadable(tmp_path, monkeypatch, content):
    result = _workspace(tmp_path)
    (result / ".besser_recipe.json").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    orchestrator = LLMOrchestrator(llm_client=_Client(), domain_model=_model(), output_dir=str(result),
                                   enable_checkpointing=False)  # failed before writing its own recipe
    if len(content) > READER_LIMIT:
        def refuse(*args, **kwargs):
            raise AssertionError("an oversized recipe is read")
        monkeypatch.setattr(
            "besser.utilities.web_modeling_editor.backend.services.spec_driven.runner.read_run_json", refuse)

    done, entry = _runner(result, orchestrator)._package_result(str(result))

    assert hashlib.sha256((result / ".besser_recipe.json").read_bytes()).hexdigest() == digest
    [item] = done.verification.notVerified
    assert item.what == "the results of this run's checks"
    assert "record could not be read" in item.why and "kept unchanged" in item.why
    assert "record could not be read" in _summary(entry)


def test_the_report_comes_from_the_run_even_when_its_record_file_was_replaced(tmp_path):
    result = _workspace(tmp_path)
    orchestrator = _busy_run(result)
    orchestrator._save_recipe("Track orders and their customers", 30.0)
    (result / ".besser_recipe.json").write_text(json.dumps({"padding": "x" * READER_LIMIT}), encoding="utf-8")

    done, _entry = _runner(result, orchestrator)._package_result(str(result))

    assert done.verification.counts.verified == WORKFLOWS - 1
    assert done.verification.counts.notVerified == 3
    recipe = json.loads((result / ".besser_recipe.json").read_bytes())
    assert recipe["tool_calls_count"] == TOOL_CALLS and "padding" not in recipe


def test_cutting_the_calls_keeps_the_last_successful_change_of_each_file():
    from besser.spec_driven_agent.state.recipe_size import _cut_tool_calls

    recipe = {"tool_calls": [
        {"tool": "modify_file", "input": {"path": "a.py"}, "success": True},
        {"tool": "modify_file", "input": {"path": "a.py"}, "success": True},
        {"tool": "modify_file", "input": {"path": "a.py"}, "success": False},
    ]}
    _cut_tool_calls(recipe)

    assert [call["success"] for call in recipe["tool_calls"]] == [True, False]

