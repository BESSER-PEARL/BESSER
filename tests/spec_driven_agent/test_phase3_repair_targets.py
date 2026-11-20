"""A Phase 3 fix attempt has targets, no forced tool, and a stop rule.

Production, one day of runs: 22 repair loops made 52 attempts; 24 attempts made
no successful edit and 15 of 21 loops ended with zero writes; blockers fell only
21%; one loop went from 21 to 40 blockers; in a one-bug edit run the loop edited
six files no blocker named. Each attempt was a fresh conversation told to "fix
every one", free to edit any file, and forced to call ``modify_file`` when it had
not edited.

Now: an attempt edits the files its blockers name (plus their imports, their
importers and new files) unless it says which blocker another file is needed
for; it is never forced to call a tool; it ends the loop when it edits nothing;
and an attempt that edits is kept only if it retires a blocker without adding an
observed failure or blockers - otherwise the tree from before it is restored.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, ValidationIssue
from besser.spec_driven_agent.pipeline.repair_scope import edit_scope, named_locations

FILES = {
    "backend/main_api.py": "from routers import order\nfrom routers import customer\n",
    "backend/routers/order.py": "from sql_alchemy import Order\n\n\ndef create_order():\n    return Order()\n",
    "backend/routers/customer.py": "def list_customers():\n    return []\n",
    "backend/sql_alchemy.py": "class Order:\n    pass\n",
    "backend/pydantic_classes.py": "class OrderCreate:\n    pass\n",
}
# An observed failure that names its file by a partial path (ruff runs in the service folder).
IN_ORDER = ValidationIssue("blocker", "ruff: routers/order.py:5:12: F821 Undefined name `Customer`")
# An observed failure that names a route, not a file.
ROUTE_500 = ValidationIssue("blocker", "create contract: backend: POST /order/ - observed a server failure "
                                       "creating Order (500: Internal Server Error)")


class _MockStateMachine:
    name = "DummySM"


class _Usage:
    estimated_cost = 0.0


class _Block:
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for key, value in kwargs.items():
            setattr(self, key, value)


def _tools(*calls):
    return {"stop_reason": "tool_use", "content": [
        _Block("tool_use", id=f"c{index}", name=name, input=dict(arguments))
        for index, (name, arguments) in enumerate(calls)]}


def _done():
    return {"stop_reason": "end_turn", "content": [_Block("text", text="Done.")]}


class _Client:
    """Replies from a script, then ends its turn. Records every request."""
    model = "mock-model"

    def __init__(self, *replies):
        self.usage = _Usage()
        self.replies = list(replies)
        self.requests = []

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        self.requests.append({"force_tool": force_tool, "messages": list(messages)})
        return self.replies.pop(0) if self.replies else _done()


def _results(request) -> list[str]:
    """The tool results a request answers."""
    content = request["messages"][-1]["content"]
    return [block["content"] for block in content if isinstance(block, dict) and block.get("type") == "tool_result"]


def _build(tmp_path, client, **kwargs) -> LLMOrchestrator:
    for rel, text in FILES.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    return LLMOrchestrator(
        llm_client=client, state_machines=[_MockStateMachine()], output_dir=str(tmp_path),
        max_cost_usd=10.0, enable_tracing=True, run_id="t", enable_checkpointing=False,
        auto_fix_issues=True, **kwargs)


def _read(tmp_path, rel):
    return (tmp_path / rel).read_text(encoding="utf-8")


def _trace(tmp_path, event):
    lines = (tmp_path / ".besser_trace.jsonl").read_text(encoding="utf-8").splitlines()
    return [record["payload"] for record in map(json.loads, lines) if record["event"] == event]


def _edit(rel, old, new, **extra):
    return ("modify_file", {"path": rel, "old_text": old, "new_text": new, **extra})


# ---------------------------------------------------------------- targets


def test_an_edit_outside_the_files_the_blockers_name_is_refused_and_the_refusal_names_them(tmp_path):
    client = _Client(_tools(_edit("backend/routers/customer.py", "return []", "return [1]")))
    orch = _build(tmp_path, client)

    orch._invoke_phase3_fix_loop([IN_ORDER], is_first_attempt=True)

    assert _read(tmp_path, "backend/routers/customer.py") == FILES["backend/routers/customer.py"]
    [refusal] = _results(client.requests[1])
    assert "outside this fix attempt" in refusal and "needed_for" in refusal
    for allowed in ("backend/routers/order.py", "backend/sql_alchemy.py", "backend/main_api.py"):
        assert allowed in refusal, "the refusal names the files the attempt may edit"
    assert "backend/pydantic_classes.py" not in refusal
    prompt = client.requests[0]["messages"][0]["content"]
    assert "[in: backend/routers/order.py line 5]" in prompt
    assert "This attempt may edit" in prompt and "backend/sql_alchemy.py" in prompt
    assert _trace(tmp_path, "edit_scope")[0]["allowed"] == [
        "backend/main_api.py", "backend/routers/order.py", "backend/sql_alchemy.py"]


def test_naming_the_blocker_a_file_is_needed_for_widens_the_scope_and_is_traced(tmp_path):
    client = _Client(_tools(_edit("backend/routers/customer.py", "return []", "return [1]",
                                  needed_for=IN_ORDER.message)))
    orch = _build(tmp_path, client)

    assert orch._invoke_phase3_fix_loop([IN_ORDER], is_first_attempt=True) == 1

    assert _read(tmp_path, "backend/routers/customer.py") == "def list_customers():\n    return [1]\n"
    [widened] = [event for event in _trace(tmp_path, "edit_scope") if "widened" in event]
    assert widened["widened"] == "backend/routers/customer.py"
    assert widened["needed_for"] == IN_ORDER.message


def test_a_new_file_and_the_named_files_neighbours_need_no_widening(tmp_path):
    client = _Client(_tools(
        ("write_file", {"path": "backend/routers/order_helpers.py", "content": "VALUE = 1\n"}),
        _edit("backend/sql_alchemy.py", "    pass\n", "    total = 0\n"),
    ))
    orch = _build(tmp_path, client)

    assert orch._invoke_phase3_fix_loop([IN_ORDER], is_first_attempt=True) == 2
    assert _read(tmp_path, "backend/routers/order_helpers.py") == "VALUE = 1\n"


def test_a_blocker_that_names_no_file_leaves_the_scope_open(tmp_path):
    client = _Client(_tools(_edit("backend/routers/customer.py", "return []", "return [1]")))
    orch = _build(tmp_path, client)

    assert orch._invoke_phase3_fix_loop([IN_ORDER, ROUTE_500], is_first_attempt=True) == 1

    assert _read(tmp_path, "backend/routers/customer.py").endswith("return [1]\n")
    assert _trace(tmp_path, "edit_scope")[0]["allowed"] is None
    assert "This attempt may edit" not in client.requests[0]["messages"][0]["content"]


def test_the_scope_ends_with_the_attempt(tmp_path):
    orch = _build(tmp_path, _Client())
    orch._invoke_phase3_fix_loop([IN_ORDER], is_first_attempt=True)

    result = orch.executor.execute_typed("modify_file", {
        "path": "backend/routers/customer.py", "old_text": "return []", "new_text": "return [2]"})
    assert result.succeeded


@pytest.mark.parametrize("message, expected", [
    ("tsc [frontend]: src/pages/OrderList.tsx(12,5): error TS2304: Cannot find name 'x'.",
     [("frontend/src/pages/OrderList.tsx", 12)]),
    ("ruff: backend/routers/order.py:3:1: F821 Undefined name `Customer`", [("backend/routers/order.py", 3)]),
    ("Syntax error in backend/sql_alchemy.py line 2: invalid syntax", [("backend/sql_alchemy.py", 2)]),
    ("missing module: routers/order.py imports 'services'", [("backend/routers/order.py", None)]),
    # Two files end in index.ts: a suffix that is not unique names no file.
    ("frontend contract: index.ts has no default export", []),
    (ROUTE_500.message, []),
])
def test_a_blocker_names_the_workspace_files_it_locates(message, expected):
    files = ["frontend/src/pages/OrderList.tsx", "frontend/src/index.ts", "frontend/src/api/index.ts",
             *FILES]
    assert named_locations(message, files) == expected


def test_a_frontend_file_brings_its_folder_and_the_apps_routing_and_entry_files(tmp_path):
    files = {
        "frontend/package.json": "{}",
        "frontend/src/main.tsx": "import App from './App'\n",
        "frontend/src/App.tsx": "import OrderList from './pages/OrderList'\n",
        "frontend/src/routes.tsx": "export const routes = <Route path='/' />\n",
        "frontend/src/pages/OrderList.tsx": "export default function OrderList() { return null }\n",
        "frontend/src/pages/CustomerList.tsx": "export default function CustomerList() { return null }\n",
        "frontend/src/components/Table.tsx": "export const Table = 1\n",
    }
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")

    allowed = edit_scope({"frontend/src/pages/OrderList.tsx"}, str(tmp_path), list(files))

    assert allowed == {"frontend/src/pages/OrderList.tsx", "frontend/src/pages/CustomerList.tsx",
                       "frontend/src/App.tsx", "frontend/src/main.tsx", "frontend/src/routes.tsx"}


# ---------------------------------------------------------------- no forced tool


def test_no_request_in_a_repair_attempt_carries_a_forced_tool(tmp_path):
    client = _Client(_tools(("read_file", {"path": "backend/routers/order.py"})))
    orch = _build(tmp_path, client)
    orch._force_tool_next = "replace_file_lines"  # left by the executor's recovery ladder

    assert orch._invoke_phase3_fix_loop([IN_ORDER], is_first_attempt=True) == 0

    assert [request["force_tool"] for request in client.requests] == [None, None]


def test_an_attempt_that_ends_its_turn_without_an_edit_ends_the_loop(tmp_path):
    client = _Client()
    orch = _build(tmp_path, client)

    with patch.object(orch, "_collect_validation_issues", return_value=[IN_ORDER]):
        orch._run_phase3_validation()

    assert len(client.requests) == 1, "one request: no re-prompt, no second attempt"
    assert orch._phase3_exit_reason == "replay (attempt 1 called no tool)"


# ---------------------------------------------------------------- keep or revert


def _six_unrelated_edits():
    return _tools(*[("write_file", {"path": rel, "content": text + "# touched\n"})
                    for rel, text in FILES.items()],
                  ("write_file", {"path": "backend/routers/order_helpers.py", "content": "VALUE = 1\n"}))


def test_an_attempt_that_edits_six_files_and_retires_nothing_is_reverted_and_ends_the_loop(tmp_path):
    client = _Client(_six_unrelated_edits())
    orch = _build(tmp_path, client)
    for rel in FILES:  # write_file rewrites only a file read in full this run
        orch.executor.execute_typed("read_file", {"path": rel})

    with patch.object(orch, "_collect_validation_issues", return_value=[ROUTE_500]):
        orch._run_phase3_validation()

    assert len(client.requests) == 2, "one attempt: its edits and its end of turn"
    assert orch._phase3_exit_reason.startswith("attempt 1 reverted: it retired none of the 1 blocker")
    for rel, text in FILES.items():
        assert _read(tmp_path, rel) == text
    assert not (tmp_path / "backend/routers/order_helpers.py").exists()
    writes = [call for call in orch.tool_calls_log if call["tool"] == "write_file"]
    assert len(writes) == 6 and all(call["success"] and call["rolled_back"] for call in writes)
    assert orch._phase3_rolled_back
    assert any("fix attempt 1 was rolled back" in issue.message for issue in orch._validation_issues)
    [rollback] = _trace(tmp_path, "rollback")
    assert rollback["attempt"] == 1


def _cycle(tmp_path, after_each_attempt, *replies, entry=(IN_ORDER, ROUTE_500)):
    """The real cycle; validation returns ``entry``, then each attempt's findings in turn."""
    client = _Client(*replies)
    orch = _build(tmp_path, client)
    findings = [list(entry), *after_each_attempt]
    with patch.object(orch, "_collect_validation_issues",
                      side_effect=lambda: findings.pop(0) if len(findings) > 1 else findings[0]):
        orch._run_phase3_validation()
    return orch, client


FIX_ORDER = _edit("backend/routers/order.py", "    return Order()\n", "    return Order()  # fixed\n")
MAPPER = ValidationIssue("blocker", "mapper config: backend/sql_alchemy.py line 2: Mapper has no property 'customer'")


def test_an_attempt_that_retires_a_blocker_but_adds_an_observed_failure_is_reverted(tmp_path):
    orch, client = _cycle(tmp_path, [[ROUTE_500, MAPPER], [IN_ORDER, ROUTE_500]], _tools(FIX_ORDER))

    assert orch._phase3_exit_reason.startswith("attempt 1 reverted: it added an observed failure: mapper config:")
    assert _read(tmp_path, "backend/routers/order.py") == FILES["backend/routers/order.py"]
    assert len(client.requests) == 2


def test_an_attempt_that_retires_a_blocker_and_adds_no_failure_is_kept_and_the_next_one_knows(tmp_path):
    orch, client = _cycle(tmp_path, [[ROUTE_500]], _tools(FIX_ORDER), _done())

    assert _read(tmp_path, "backend/routers/order.py").endswith("# fixed\n")
    assert not orch._phase3_rolled_back
    # Attempt 2 made no edit and ended the loop; its prompt carried attempt 1's outcome.
    assert orch._phase3_exit_reason == "replay (attempt 2 called no tool)"
    second = client.requests[2]["messages"][0]["content"]
    assert "## Files this run has edited" in second and "backend/routers/order.py" in second
    assert "Fix attempt 1 was kept" in second
    assert "Fix every one" not in second
    assert "retired 1 blocker: ruff:" in second


def test_a_loop_never_ends_with_more_blockers_than_it_started_with(tmp_path):
    more = [ROUTE_500, ValidationIssue("blocker", "requirement: R2 — Orders list their customer is not implemented"),
            ValidationIssue("blocker", "requirement: R3 — Customers can be searched is not implemented")]
    orch, _ = _cycle(tmp_path, [more, [IN_ORDER, ROUTE_500]], _tools(FIX_ORDER))

    assert orch._phase3_exit_reason == "attempt 1 reverted: it left more blockers (2 -> 3)"
    assert [i.message for i in orch._validation_issues
            if i.severity == "blocker" and not i.message.startswith("runtime gate:")] == [
        IN_ORDER.message, ROUTE_500.message]


def test_an_attempt_that_makes_the_app_start_is_kept_whatever_the_probe_now_sees(tmp_path):
    """Boot dominates the tree ranking: the failures a started app shows were hidden, not added."""
    startup = ValidationIssue("blocker", "application startup: backend/main_api.py line 1: ImportError")
    creates = [ValidationIssue("blocker", f"create contract: backend: POST /{name}/ - observed a server "
                                          "failure (500)") for name in ("order", "customer")]
    orch, _ = _cycle(tmp_path, [creates], _tools(FIX_ORDER), entry=(startup,))

    assert not orch._phase3_rolled_back
    assert _read(tmp_path, "backend/routers/order.py").endswith("# fixed\n")


def test_a_moved_line_number_is_not_a_new_failure(tmp_path):
    """An edit above an error it leaves moves that error's line: not retired, not added."""
    other = ValidationIssue("blocker", "python contract: backend/routers/order.py line 9: undefined name 'Total'")
    moved = ValidationIssue("blocker", "python contract: backend/routers/order.py line 11: undefined name 'Total'")
    orch, _ = _cycle(tmp_path, [[moved]], _tools(FIX_ORDER), entry=(IN_ORDER, other))

    assert not orch._phase3_rolled_back


# ---------------------------------------------------------------- partition


def test_findings_that_are_only_could_not_check_start_no_attempt(tmp_path):
    client = _Client()
    orch = _build(tmp_path, client)
    unknown = ValidationIssue("warning", "validation unverified: tsc [frontend] did not run completely (timed out)")

    with patch.object(orch, "_collect_validation_issues", return_value=[unknown]):
        orch._run_phase3_validation()

    assert client.requests == []
    assert orch._phase3_exit_reason == "no blocker-class issues"
