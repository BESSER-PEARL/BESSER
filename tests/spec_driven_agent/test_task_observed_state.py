"""A checklist item's state comes from what the run did, not from bookkeeping.

An item that names files is done once a successful write touched each of them
after the item was created (and, when it names an HTTP route, once the app
declares it). The model may still close an item itself, without quoting code.
An item nothing settled is "not confirmed": a warning in the verification
report, never a blocker, a repair attempt or part of the blocker count. Only an
item whose deterministic check runs and fails stays a blocker.
"""

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.validation.issues import _classify_issue, is_completion_issue
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


def test_writes_to_every_named_file_close_the_item_without_task_list(orchestrator):
    orchestrator.executor.set_tasks([TWO_FILES])

    _read_and_edit(orchestrator, ROUTER, "    return []", "    return [{'total': 0}]")
    assert not _task(orchestrator, TWO_FILES)["done"], "one of two named files is not enough"
    _run(orchestrator, "write_file", path=SERVICE, content="def total(order):\n    return 0\n")

    task = _task(orchestrator, TWO_FILES)
    assert task["done"] is True and task["verification"] == "written"
    assert orchestrator.executor.open_tasks() == []
    assert _task_findings(orchestrator) == []


def test_an_item_nothing_settled_is_a_not_confirmed_warning(orchestrator):
    orchestrator.executor.set_tasks([TWO_FILES, "Make the order list easier to scan"])
    _read_and_edit(orchestrator, ROUTER, "    return []", "    return [{'total': 0}]")

    findings = _task_findings(orchestrator)

    assert len(findings) == 2
    assert all(f.message.startswith("task unverified:") and "was not confirmed" in f.message
               for f in findings)
    assert all(f.severity == "warning" and not is_completion_issue(f) for f in findings)


def test_a_model_close_needs_no_evidence_quotes(orchestrator):
    orchestrator.executor.set_tasks(["Make the order list easier to scan"])
    task_id = _task(orchestrator, "Make the order list easier to scan")["id"]

    result = orchestrator.executor.execute_typed("task_list", {"action": "done", "id": task_id})

    assert result.status == "ok" and result.payload["done_ids"] == [task_id]
    assert orchestrator.executor.open_tasks() == []
    assert _task_findings(orchestrator) == []


def test_a_refused_or_failed_write_closes_nothing(orchestrator):
    orchestrator.executor.set_tasks([f"Add an order total to {ROUTER}"])

    _run(orchestrator, "read_file", path=ROUTER)
    refused = _run(orchestrator, "modify_file", path=ROUTER, old_text="text that is not there",
                   new_text="x")
    failed = _run(orchestrator, "write_file", path="../outside.py", content="x = 1\n")

    assert '"error"' in refused["content"] and '"error"' in failed["content"]
    assert not _task(orchestrator, f"Add an order total to {ROUTER}")["done"]


def test_a_write_before_the_item_existed_does_not_count(orchestrator):
    _read_and_edit(orchestrator, ROUTER, "    return []", "    return [{'total': 0}]")
    orchestrator.executor.execute_typed("task_list", {"action": "add", "text": f"Paginate {ROUTER}"})

    assert not _task(orchestrator, f"Paginate {ROUTER}")["done"]
    _read_and_edit(orchestrator, ROUTER, "[{'total': 0}]", "[{'total': 0, 'page': 1}]")
    assert _task(orchestrator, f"Paginate {ROUTER}")["done"] is True


def test_a_named_route_must_also_be_declared(orchestrator):
    text = f"Add GET /order/count/ to {ROUTER}"
    orchestrator.executor.set_tasks([text])

    _read_and_edit(orchestrator, ROUTER, "    return []", "    return [{'total': 0}]")
    assert not _task(orchestrator, text)["done"], "the file was touched but the route is missing"
    _read_and_edit(orchestrator, ROUTER, "def list_orders():",
                   'def list_orders():\n    return []\n\n\n@router.get("/order/count/")\ndef count_orders():')
    assert _task(orchestrator, text)["done"] is True


def test_an_item_whose_check_fails_stays_a_blocker(orchestrator):
    orchestrator.executor.set_tasks([{"text": "Implement cancel", "verify": lambda: False},
                                     {"text": "Implement close", "verify": lambda: True}])

    findings = _task_findings(orchestrator)

    assert [f.message.split(":")[0] for f in findings] == ["task incomplete"]
    assert findings[0].severity == "blocker"
    assert _task(orchestrator, "Implement close")["done"] is True


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
