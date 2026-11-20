"""A checklist item closes on a passing check or on the model's own word, never on bookkeeping.

An item nothing closed is "not confirmed": a warning in the verification
report, never a blocker or part of the blocker count. Its warning says whether
the files it names were edited, which is a fact about the run, not proof that
the item is done. Only an item whose deterministic check runs and fails stays a
blocker. A harness item that no other check covers says so when the model
closes it on its word.
"""

import pytest

import besser.spec_driven_agent.pipeline.orchestrator as orchestrator_module
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.validation.issues import (
    ValidationIssue, _classify_issue, _hard_blockers, is_completion_issue, unresolved_defects,
)
from besser.spec_driven_agent.validation.user_wording import user_sentence
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    build_verification_report,
)

ROUTER = "backend/routers/order.py"
SERVICE = "backend/services/totals.py"
ROUTER_SOURCE = (
    "from fastapi import APIRouter\n\nrouter = APIRouter()\n\n\n"
    '@router.get("/order/")\ndef list_orders():\n    return []\n'
)


class _Block:
    def __init__(self, name, tool_input):
        self.type = "tool_use"
        self.name = name
        self.input = tool_input
        self.id = "b1"


@pytest.fixture
def orchestrator(simple_library_book_model, tmp_path):
    class Client:
        model = "mock-model"
        usage = UsageTracker("mock-model")

        def chat(self, **kwargs):
            raise AssertionError("no LLM call expected")

    (tmp_path / "backend" / "routers").mkdir(parents=True)
    (tmp_path / ROUTER).write_text(ROUTER_SOURCE, encoding="utf-8")
    return LLMOrchestrator(
        llm_client=Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False, enable_tracing=False,
    )


def _run(orchestrator, tool, **tool_input):
    return orchestrator._execute_single_tool(_Block(tool, tool_input), turn=1)


def _read_and_edit(orchestrator, path, old, new):
    _run(orchestrator, "read_file", path=path)
    return _run(orchestrator, "modify_file", path=path, old_text=old, new_text=new)


def _task(orchestrator, text):
    return next(t for t in orchestrator.executor.task_snapshot() if t["text"] == text)


def _task_findings(orchestrator):
    return [_classify_issue(message) for message in orchestrator._collect_task_issues()]


TWO_FILES = f"Add an order total to {ROUTER} computed by {SERVICE}"


def test_an_edit_to_the_named_file_does_not_close_the_item(orchestrator):
    """One comment in a shared router file closed every item naming that file."""
    other = f"Paginate the order list in {ROUTER}"
    orchestrator.executor.set_tasks([TWO_FILES, other])

    result = _read_and_edit(orchestrator, ROUTER, "    return []", "    return []  # reviewed")
    _run(orchestrator, "write_file", path=SERVICE, content="def total(order):\n    return 0\n")

    assert "checklist_closed" not in result["content"]
    assert len(orchestrator.executor.open_tasks()) == 2
    findings = _task_findings(orchestrator)
    assert [f.severity for f in findings] == ["warning", "warning"]
    assert "Every file it names was edited" in findings[0].message


def test_the_warning_says_which_named_files_were_not_touched(orchestrator, tmp_path):
    (tmp_path / "backend" / "services").mkdir()
    (tmp_path / SERVICE).write_text("def total(order):\n    return 0\n", encoding="utf-8")
    orchestrator.executor.set_tasks([TWO_FILES, "Make the order list easier to scan"])
    _read_and_edit(orchestrator, ROUTER, "    return []", "    return [{'total': 0}]")

    partial, no_files = _task_findings(orchestrator)

    assert f"Edited: {ROUTER}; not touched: {SERVICE}." in partial.message
    assert "edited" not in no_files.message.lower() and "touched" not in no_files.message
    assert all(f.severity == "warning" and not is_completion_issue(f) for f in (partial, no_files))
    assert user_sentence(partial.message).endswith("(some of its files were edited)")


def test_a_named_path_that_is_no_workspace_file_is_said_to_be_absent(orchestrator):
    """The modeled-rule item names ``routers/<class>.py``; the scaffold's file is under ``backend/``."""
    orchestrator.executor.set_tasks(["Enforce the rule in the create endpoint of routers/order.py"])

    [finding] = _task_findings(orchestrator)

    assert "No workspace file is at routers/order.py." in finding.message
    assert "touched" not in finding.message


def test_an_identical_rewrite_is_not_an_edit(orchestrator):
    orchestrator.executor.set_tasks([f"Paginate {ROUTER}"])
    _run(orchestrator, "read_file", path=ROUTER)
    _run(orchestrator, "write_file", path=ROUTER, content=ROUTER_SOURCE)

    [finding] = _task_findings(orchestrator)
    assert "written" not in _task(orchestrator, f"Paginate {ROUTER}")
    assert "The files it names were not touched" in finding.message


def test_a_refused_or_failed_write_is_not_an_edit(orchestrator):
    orchestrator.executor.set_tasks([f"Add an order total to {ROUTER}"])

    _run(orchestrator, "read_file", path=ROUTER)
    refused = _run(orchestrator, "modify_file", path=ROUTER, old_text="text that is not there",
                   new_text="x")
    failed = _run(orchestrator, "write_file", path="../outside.py", content="x = 1\n")

    assert '"error"' in refused["content"] and '"error"' in failed["content"]
    assert "written" not in _task(orchestrator, f"Add an order total to {ROUTER}")


def test_an_edit_before_the_item_existed_is_not_counted(orchestrator):
    _read_and_edit(orchestrator, ROUTER, "    return []", "    return [{'total': 0}]")
    orchestrator.executor.execute_typed("task_list", {"action": "add", "text": f"Paginate {ROUTER}"})
    assert "written" not in _task(orchestrator, f"Paginate {ROUTER}")

    _read_and_edit(orchestrator, ROUTER, "[{'total': 0}]", "[{'total': 0, 'page': 1}]")
    assert _task(orchestrator, f"Paginate {ROUTER}")["written"] == [ROUTER]
    assert not _task(orchestrator, f"Paginate {ROUTER}")["done"]


def test_an_edit_a_rollback_discarded_is_forgotten(orchestrator):
    orchestrator.executor.set_tasks([TWO_FILES])
    _read_and_edit(orchestrator, ROUTER, "    return []", "    return [{'total': 0}]")
    assert orchestrator._create_snapshot()
    _run(orchestrator, "write_file", path=SERVICE, content="def total(order):\n    return 0\n")
    assert _task(orchestrator, TWO_FILES)["written"] == [ROUTER, SERVICE]

    assert orchestrator._restore_snapshot()

    assert _task(orchestrator, TWO_FILES)["written"] == [ROUTER]


def test_a_model_close_needs_no_evidence_quotes(orchestrator):
    orchestrator.executor.set_tasks(["Make the order list easier to scan"])
    task_id = _task(orchestrator, "Make the order list easier to scan")["id"]

    result = orchestrator.executor.execute_typed("task_list", {"action": "done", "id": task_id})

    assert result.status == "ok" and result.payload["done_ids"] == [task_id]
    assert orchestrator.executor.open_tasks() == []
    assert _task_findings(orchestrator) == []


def test_a_word_close_of_an_item_nothing_else_checks_is_reported(orchestrator, monkeypatch):
    """A modeled rule the generators could not enforce: only this item holds it."""
    rule = "Enforce the modeled rule 'positivePages' of Book"
    monkeypatch.setattr(orchestrator, "_unchecked_task_texts", lambda: {rule})
    orchestrator.executor.set_tasks([rule, "Make the order list easier to scan"])
    orchestrator.executor.execute_typed("task_list", {"action": "done", "ids": [1, 2]})

    [finding] = _task_findings(orchestrator)

    assert finding.severity == "warning"
    assert finding.message.startswith("task unverified: task 1 was closed by the agent, not checked:")
    assert user_sentence(finding.message) == ("A planned change could not be confirmed as done: "
                                              f"{rule} (the agent marked it done)")


def test_the_unchecked_items_are_the_model_rule_and_derived_attribute_items(orchestrator, monkeypatch):
    monkeypatch.setattr(orchestrator, "_unenforced_rule_tasks", lambda: ["rule item"])
    orchestrator._generator_used = "generate_fastapi_backend"
    monkeypatch.setattr(
        "besser.spec_driven_agent.pipeline.orchestrator.derived_attribute_tasks", lambda model: ["derived item"])

    assert orchestrator._unchecked_task_texts() == {"rule item", "derived item"}
    orchestrator._modify_mode = True
    assert orchestrator._unchecked_task_texts() == {"rule item"}


def test_an_item_whose_check_fails_stays_a_blocker(orchestrator):
    orchestrator.executor.set_tasks([{"text": "Implement cancel", "verify": lambda: False},
                                     {"text": "Implement close", "verify": lambda: True}])

    findings = _task_findings(orchestrator)

    assert [f.message.split(":")[0] for f in findings] == ["task incomplete"]
    assert findings[0].severity == "blocker"
    assert _task(orchestrator, "Implement close")["done"] is True


ACTION = ("backend/routers/order.py", "execute_order_cancel", "POST", "/order/{order_id}/methods/cancel/")
ACTION_CONTRACT = (f"action contract: {ACTION[0]} line 12: {ACTION[2]} {ACTION[3]} ({ACTION[1]}): "
                   "HTTP 501 placeholder. Implement this route and prove it with test_api.")


def test_an_action_stub_is_reported_once(orchestrator, monkeypatch):
    orchestrator.executor.set_tasks([{"text": "Implement POST cancel", "verify": lambda: False,
                                      "_action_contract": ACTION}])

    monkeypatch.setattr(orchestrator_module, "action_implementation_issues", lambda *a: [ACTION_CONTRACT])
    assert _task_findings(orchestrator) == []
    monkeypatch.setattr(orchestrator_module, "action_implementation_issues", lambda *a: [])
    assert [f.message.split(":")[0] for f in _task_findings(orchestrator)] == ["task incomplete"]


def test_a_failing_check_is_a_defect_but_not_a_hard_blocker():
    incomplete = ValidationIssue("blocker", "task incomplete: task 3 is not done: Add GET /stats. Its check fails.")
    assert unresolved_defects([incomplete]) == [incomplete]
    assert _hard_blockers([incomplete]) == []


def test_a_blocked_item_is_reported_not_repaired(orchestrator):
    orchestrator.executor.set_tasks(["Send a reminder email"])
    orchestrator.executor.execute_typed("task_list", {"action": "blocked", "id": 1, "reason": "no mail server"})

    [finding] = _task_findings(orchestrator)

    assert finding.severity == "warning" and "no mail server" in finding.message


def test_the_user_reads_both_states_and_only_the_known_gap_is_unenforced():
    not_confirmed = "task unverified: task 2 was not confirmed: Show open orders. Nothing settled it."
    not_done = "task incomplete: task 3 is not done: Add the requested endpoint GET /stats. Its check fails."
    report = build_verification_report({"validation_issues": [
        {"severity": "warning", "message": not_confirmed},
        {"severity": "blocker", "message": not_done},
    ]})

    assert user_sentence(not_confirmed) == "A planned change could not be confirmed as done: Show open orders"
    assert user_sentence(not_done) == ("A planned change is not done: Add the requested endpoint GET /stats")
    assert [i.what for i in report.notVerified if "planned change" in i.what] == [user_sentence(not_confirmed)]
    assert [i.what for i in report.shippedUnenforced] == [user_sentence(not_done)]
