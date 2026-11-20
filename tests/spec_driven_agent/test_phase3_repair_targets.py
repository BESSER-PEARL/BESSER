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
import pathlib
from unittest.mock import patch

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, ValidationIssue
from besser.spec_driven_agent.pipeline.repair_scope import edit_scope, named_locations

FILES = {
    "backend/main_api.py": "from routers import order\nfrom services import customer\n",
    "backend/routers/order.py": "from sql_alchemy import Order\n\n\ndef create_order():\n    return Order()\n",
    "backend/services/customer.py": "def list_customers():\n    return []\n",
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
        self.requests.append({"force_tool": force_tool, "messages": list(messages), "tools": tools})
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
    client = _Client(_tools(_edit("backend/services/customer.py", "return []", "return [1]")))
    orch = _build(tmp_path, client)

    orch._invoke_phase3_fix_loop([IN_ORDER], is_first_attempt=True)

    assert _read(tmp_path, "backend/services/customer.py") == FILES["backend/services/customer.py"]
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
    client = _Client(_tools(_edit("backend/services/customer.py", "return []", "return [1]",
                                  needed_for=IN_ORDER.message)))
    orch = _build(tmp_path, client)

    assert orch._invoke_phase3_fix_loop([IN_ORDER], is_first_attempt=True) == 1

    assert _read(tmp_path, "backend/services/customer.py") == "def list_customers():\n    return [1]\n"
    [widened] = [event for event in _trace(tmp_path, "edit_scope") if "widened" in event]
    assert widened["widened"] == "backend/services/customer.py"
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
    client = _Client(_tools(_edit("backend/services/customer.py", "return []", "return [1]")))
    orch = _build(tmp_path, client)

    assert orch._invoke_phase3_fix_loop([IN_ORDER, ROUTE_500], is_first_attempt=True) == 1

    assert _read(tmp_path, "backend/services/customer.py").endswith("return [1]\n")
    assert _trace(tmp_path, "edit_scope")[0]["allowed"] is None
    assert "This attempt may edit" not in client.requests[0]["messages"][0]["content"]


def test_the_scope_ends_with_the_attempt(tmp_path):
    orch = _build(tmp_path, _Client())
    orch._invoke_phase3_fix_loop([IN_ORDER], is_first_attempt=True)

    result = orch.executor.execute_typed("modify_file", {
        "path": "backend/services/customer.py", "old_text": "return []", "new_text": "return [2]"})
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


def _cycle(tmp_path, after_each_attempt, *replies, entry=(IN_ORDER, ROUTE_500), before=None, on_check=None):
    """The real cycle; validation returns ``entry``, then each attempt's findings in turn.
    ``before(orch)`` runs first; ``on_check(orch)`` with every validation."""
    client = _Client(*replies)
    orch = _build(tmp_path, client)
    if before:
        before(orch)
    findings = [list(entry), *after_each_attempt]

    def collect():
        if on_check:
            on_check(orch)
        return findings.pop(0) if len(findings) > 1 else findings[0]

    with patch.object(orch, "_collect_validation_issues", side_effect=collect):
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


def test_a_kept_attempt_never_leaves_more_blockers_than_it_faced(tmp_path):
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


# ---------------------------------------------------------------- layered failures
# Fixing the front failure of a layer shows the failure behind it: newly visible, not newly broken.

B = ValidationIssue
MISSING = B("blocker", "missing module: backend/routers/order.py imports 'dateutil', which the app does not ship")
STARTUP = B("blocker", "application startup: backend/main_api.py line 4: RuntimeError: DATABASE_URL is not set")
CREATE = B("blocker", "create contract: backend: POST /order/ - observed a server failure creating Order (500)")
ACTION = B("blocker", "action call: backend: POST /order/{id}/methods/submit/ - observed a server failure (500)")
DEP_CREATE = B("blocker", "create contract: backend: POST /orderline/ - observed a server failure creating "
                          "OrderLine (500)")
DEFERRED = B("warning", "validation: business-requirement judgment deferred until startup/data-entry "
                        "blockers are fixed")
REQ1 = B("blocker", "requirement: R1 — Orders can be submitted is not implemented")
REQ2 = B("blocker", "requirement: R2 — Orders show a total is not implemented")
SETUP = B("blocker", "verification setup: tsc [frontend] requires installed project dependencies.")
TSC = B("blocker", "tsc [frontend]: src/pages/OrderList.tsx(5,3): error TS2304: Cannot find name 'x'.")
BUILD = B("blocker", "frontend build [frontend]: Could not resolve './missing' from src/pages/OrderList.tsx")
SYNTAX = B("blocker", "Syntax error in backend/routers/order.py line 2: invalid syntax")
# The probe could not build an OrderLine without an Order: an unconfirmed create, a warning.
DEP_UNCONFIRMED = B("warning", "runtime unverified: create unverified: backend: POST /orderline/ - no request "
                               "could be constructed; unresolved related IDs: [[\"order\", \"Order\"]]")


@pytest.mark.parametrize("entry, after", [
    pytest.param((MISSING,), [STARTUP], id="missing module hides boot"),
    pytest.param((CREATE, DEFERRED), [REQ1], id="a hard blocker defers the judge"),
    pytest.param((SETUP,), [TSC], id="missing dependencies hide tsc"),
    pytest.param((TSC,), [BUILD], id="tsc errors hide the build"),
    pytest.param((SYNTAX,), [IN_ORDER], id="a syntax error hides lint of its file"),
])
def test_a_fix_that_shows_the_failure_it_hid_is_kept(tmp_path, entry, after):
    orch, _ = _cycle(tmp_path, [after], _tools(FIX_ORDER), entry=entry)

    assert not orch._phase3_rolled_back, orch._phase3_exit_reason
    assert _read(tmp_path, "backend/routers/order.py").endswith("# fixed\n")


def _backend(entities: dict, actions=()):
    """One backend as the probe records it: entity -> (create path, verdict); actions as (entity, route, verdict)."""
    return {"boot": "ok",
            "entities": {name: {"path": path, "verdict": verdict} for name, (path, verdict) in entities.items()},
            "action_calls": [{"entity": e, "route": r, "verdict": v} for e, r, v in actions]}


def _probe(before: dict, after: dict):
    """The probe's records: ``before`` for the tree Phase 3 starts from, ``after`` once the fix is on disk."""
    def on_check(orch):
        fixed = _read(pathlib.Path(orch.output_dir), "backend/routers/order.py").endswith("# fixed\n")
        orch._runtime_probe_facts = (orch._workspace_revision(), [after if fixed else before])
    return on_check


ORDER = ("/order/", "crashed")
SUBMIT = "/order/{id}/methods/submit/"


def test_a_fixed_create_shows_the_failing_action_of_the_entity_it_creates(tmp_path):
    orch, _ = _cycle(tmp_path, [[ACTION], [CREATE]], _tools(FIX_ORDER), entry=(CREATE,), on_check=_probe(
        _backend({"Order": ORDER}),
        _backend({"Order": ("/order/", "created")}, [("Order", SUBMIT, "crashed")])))

    assert not orch._phase3_rolled_back, orch._phase3_exit_reason


def test_a_fixed_create_shows_the_failing_create_that_needed_its_record(tmp_path):
    orch, _ = _cycle(tmp_path, [[DEP_CREATE], [CREATE, DEP_UNCONFIRMED]], _tools(FIX_ORDER),
                     entry=(CREATE, DEP_UNCONFIRMED), on_check=_probe(
        _backend({"Order": ORDER, "OrderLine": ("/orderline/", "unresolved")}),
        _backend({"Order": ("/order/", "created"), "OrderLine": ("/orderline/", "crashed")})))

    assert not orch._phase3_rolled_back, orch._phase3_exit_reason


def test_a_fixed_create_does_not_excuse_new_failures_of_entities_created_before(tmp_path):
    """Fixing one create made five actions of OTHER entities, created before, fail."""
    others = ("bravo", "charlie", "delta", "echo", "foxtrot")
    actions = [B("blocker", f"action call: backend: POST /{n}/{{id}}/methods/run/ - observed a server failure (500)")
               for n in others]
    alpha = B("blocker", "create contract: backend: POST /alpha/ - observed a server failure creating Alpha (500)")
    created = {n: (f"/{n}/", "created") for n in others}
    orch, _ = _cycle(tmp_path, [actions, [alpha]], _tools(FIX_ORDER), entry=(alpha,), on_check=_probe(
        _backend({"Alpha": ("/alpha/", "crashed"), **created}),
        _backend({"Alpha": ("/alpha/", "created"), **created},
                 [(n, f"/{n}/{{id}}/methods/run/", "crashed") for n in others])))

    assert orch._phase3_exit_reason.startswith("attempt 1 reverted: it added an observed failure: action call: "
                                               "backend: POST /bravo/")


def test_without_the_probe_records_a_new_action_failure_is_not_excused(tmp_path):
    orch, _ = _cycle(tmp_path, [[ACTION], [CREATE]], _tools(FIX_ORDER), entry=(CREATE,))

    assert orch._phase3_exit_reason.startswith("attempt 1 reverted: it added an observed failure: action call:")


def test_a_deferred_judge_brings_its_verdicts_without_counting_against_the_fix(tmp_path):
    orch, _ = _cycle(tmp_path, [[REQ1, REQ2]], _tools(FIX_ORDER), entry=(CREATE, DEFERRED))

    assert not orch._phase3_rolled_back, orch._phase3_exit_reason


def test_more_blockers_from_a_judge_that_ran_on_both_trees_is_still_reverted(tmp_path):
    """21 -> 40 with the judge run before and after: no family was hidden."""
    before = [CREATE] + [B("blocker", f"requirement: R{n} — rule {n} is not implemented") for n in range(1, 21)]
    after = [B("blocker", f"requirement: R{n} — rule {n} is not implemented") for n in range(1, 41)]
    orch, _ = _cycle(tmp_path, [after, before], _tools(FIX_ORDER), entry=before)

    assert orch._phase3_exit_reason == "attempt 1 reverted: it left more blockers (21 -> 40)"


def test_a_new_mapper_failure_nothing_hid_is_still_reverted(tmp_path):
    orch, _ = _cycle(tmp_path, [[MAPPER], [CREATE, IN_ORDER]], _tools(FIX_ORDER), entry=(CREATE, IN_ORDER))

    assert orch._phase3_exit_reason.startswith("attempt 1 reverted: it added an observed failure: mapper config:")


def _booted_before(orch):
    """The probe measured the tree Phase 3 starts from: it boots."""
    orch._runtime_probe_facts = (orch._workspace_revision(), [{"boot": "ok", "entities": {}, "action_calls": []}])


def test_a_python_contract_on_an_app_that_boots_does_not_excuse_runtime_failures(tmp_path):
    contract = B("blocker", "python contract: backend/routers/order.py line 5: undefined name 'Total'")
    creates = [B("blocker", f"create contract: backend: POST /{n}/ - observed a server failure (500)") for n in "ab"]
    orch, _ = _cycle(tmp_path, [creates, [contract]], _tools(FIX_ORDER), entry=(contract,), before=_booted_before)

    assert orch._phase3_exit_reason.startswith("attempt 1 reverted: it added an observed failure: create contract:")


def test_an_app_that_no_longer_starts_is_reverted(tmp_path):
    def probe(orch):
        if _read(tmp_path, "backend/routers/order.py").endswith("# fixed\n"):
            orch._runtime_probe_facts = (orch._workspace_revision(), [{"boot": "probe_error"}])

    # A backend the probe no longer measures is measured once more before the attempt is ranked.
    orch, _ = _cycle(tmp_path, [[ROUTE_500], [ROUTE_500], [IN_ORDER, ROUTE_500]], _tools(FIX_ORDER),
                     before=_booted_before, on_check=probe)

    assert orch._phase3_exit_reason == "attempt 1 reverted: the application no longer starts"


# ---------------------------------------------------------------- requirement status


PARTIAL5 = B("blocker", "requirement partial: R5 — Orders show a total; inspect and complete the missing behaviour")
MISSING5 = B("blocker", "requirement: R5 — Orders show a total is not implemented")


def test_a_requirement_whose_status_falls_is_not_retired(tmp_path):
    orch, _ = _cycle(tmp_path, [[MISSING5], [PARTIAL5]], _tools(FIX_ORDER), entry=(PARTIAL5,))

    assert orch._phase3_exit_reason == "attempt 1 reverted: it retired none of the 1 blocker it faced"


def test_a_requirement_whose_status_rises_is_progress(tmp_path):
    orch, _ = _cycle(tmp_path, [[PARTIAL5]], _tools(FIX_ORDER), entry=(MISSING5,))

    assert not orch._phase3_rolled_back, orch._phase3_exit_reason


def test_a_requirement_whose_status_falls_counts_as_one_more_blocker(tmp_path):
    r6 = B("blocker", "requirement: R6 — Orders can be cancelled is not implemented")
    orch, _ = _cycle(tmp_path, [[MISSING5, r6], [IN_ORDER, PARTIAL5]], _tools(FIX_ORDER), entry=(IN_ORDER, PARTIAL5))

    assert orch._phase3_exit_reason == "attempt 1 reverted: it left more blockers (2 -> 3)"


# ---------------------------------------------------------------- attempts without an edit


def test_after_every_write_was_refused_one_more_attempt_is_told_why(tmp_path):
    outside = _edit("backend/services/customer.py", "return []", "return [1]")
    client = _Client(_tools(outside), _done(), _tools(outside), _done(), _tools(outside))
    orch = _build(tmp_path, client)

    with patch.object(orch, "_collect_validation_issues", return_value=[IN_ORDER]):
        orch._run_phase3_validation()

    assert len(client.requests) == 4, "two attempts, each a refused write and an end of turn"
    second = client.requests[2]["messages"][0]["content"]
    assert "every write it tried was refused" in second and "outside this fix attempt" in second
    assert orch._phase3_exit_reason == (
        "no successful edit (attempt 2 made 1 tool call, 1 failed: modify_file x1)")


def test_a_read_only_attempt_still_ends_the_loop(tmp_path):
    client = _Client(_tools(("read_file", {"path": "backend/routers/order.py"})))
    orch = _build(tmp_path, client)

    with patch.object(orch, "_collect_validation_issues", return_value=[IN_ORDER]):
        orch._run_phase3_validation()

    assert len(client.requests) == 2
    assert orch._phase3_exit_reason.startswith("no edit attempted (attempt 1")


# ---------------------------------------------------------------- scope extras


def _write(root, files):
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return sorted(files)


def test_a_missing_module_brings_the_dependency_manifest(tmp_path):
    files = _write(tmp_path, {**FILES, "backend/requirements.txt": "fastapi\n"})

    assert "backend/requirements.txt" in edit_scope(
        {"backend/routers/order.py"}, str(tmp_path), files, frozenset({"backend/routers/order.py"}))
    assert "backend/requirements.txt" not in edit_scope({"backend/routers/order.py"}, str(tmp_path), files)


def test_a_model_file_brings_its_sibling_modules(tmp_path):
    files = _write(tmp_path, FILES)

    allowed = edit_scope({"backend/sql_alchemy.py"}, str(tmp_path), files)

    assert {"backend/main_api.py", "backend/pydantic_classes.py"} <= allowed


def test_a_frontend_file_brings_what_it_imports_through_a_path_alias(tmp_path):
    files = _write(tmp_path, {
        "frontend/package.json": "{}",
        "frontend/tsconfig.json": '{"compilerOptions": {"baseUrl": ".", // aliases\n "paths": {"@/*": ["src/*"]}}}',
        "frontend/src/pages/OrderList.tsx": "import { api } from '@/lib/api'\nexport default function OrderList() {}\n",
        "frontend/src/lib/api.ts": "export const api = 1\n",
        "frontend/src/lib/unrelated.ts": "export const x = 1\n",
    })

    allowed = edit_scope({"frontend/src/pages/OrderList.tsx"}, str(tmp_path), files)
    assert "frontend/src/lib/api.ts" in allowed
    assert "frontend/src/lib/unrelated.ts" not in allowed
    assert "frontend/src/pages/OrderList.tsx" in edit_scope({"frontend/src/lib/api.ts"}, str(tmp_path), files)


# ---------------------------------------------------------------- one failing create hides nothing else


def _creates(*names):
    return [B("blocker", f"create contract: backend: POST /{n}/ - observed a server failure creating {n} (500)")
            for n in names]


def test_retiring_one_create_while_breaking_five_others_is_reverted(tmp_path):
    """Starting with one failing create, an attempt that fixes it and breaks five others."""
    orch, _ = _cycle(tmp_path, [_creates("b", "c", "d", "e", "f"), _creates("a")], _tools(FIX_ORDER),
                     entry=tuple(_creates("a")))

    assert orch._phase3_exit_reason.startswith("attempt 1 reverted: it added an observed failure: create contract:")
    assert _read(tmp_path, "backend/routers/order.py") == FILES["backend/routers/order.py"]


def test_a_traded_create_is_not_hidden_by_a_number_in_its_path(tmp_path):
    other = B("blocker", "ruff: backend/routers/order.py:5:1: F821 Undefined name `Total`")
    orch, _ = _cycle(tmp_path, [_creates("e3"), [other, *_creates("e2")]], _tools(FIX_ORDER),
                     entry=(other, *_creates("e2")))

    assert orch._phase3_exit_reason.startswith("attempt 1 reverted: it added an observed failure: create contract: "
                                               "backend: POST /e3/")


def test_more_entities_failing_to_be_created_reverts_where_runtime_checks_ran(tmp_path):
    """Rule 5 on the counted findings: an unconfirmed create (a warning) that grows."""
    unconfirmed = [B("warning", f"runtime unverified: create unverified: backend: POST /{n}/ - autogenerated "
                                "input was refused (422)") for n in ("x", "y")]
    orch, _ = _cycle(tmp_path, [unconfirmed, [IN_ORDER]], _tools(FIX_ORDER), entry=(IN_ORDER,))

    assert orch._phase3_exit_reason.startswith("attempt 1 reverted: more entities or actions fail at runtime")


# ---------------------------------------------------------------- a judge that did not run on the re-check

JUDGE_FAILED = B("warning", "validation unverified: the requirements judge did not run completely (it returned "
                            "no verdicts; coverage of the original specification has not been verified) - "
                            "required verification was SKIPPED; generated output is not verified complete")
JUDGE_STOPPED = B("blocker", "requirement unverified: not checked, the run's time budget is spent; "
                             "original-specification coverage is not verified.")


@pytest.mark.parametrize("note", [JUDGE_FAILED, JUDGE_STOPPED], ids=["judge failed", "judge stopped"])
def test_a_requirement_the_judge_did_not_look_at_again_is_not_retired(tmp_path, note):
    orch, _ = _cycle(tmp_path, [[note], [REQ1, REQ2]], _tools(FIX_ORDER), entry=(REQ1, REQ2))

    assert orch._phase3_exit_reason == "attempt 1 reverted: it retired none of the 2 blockers it faced"


NOT_EXTRACTED = B("warning", "requirement unverified: requirement extraction failed; the original specification "
                             "has not been checked. Do not treat this as an empty requirement list or verified "
                             "completion.")
PHASE3_SKIPPED = B("warning", "requirement unverified: Phase 3 validation did not run: the run was cancelled. "
                              "The current application has not completed final verification.")


@pytest.mark.parametrize("note", [JUDGE_FAILED, NOT_EXTRACTED, PHASE3_SKIPPED],
                         ids=["judge failed", "extraction failed", "phase 3 skipped"])
def test_verdicts_a_returning_judge_gives_are_not_new_blockers(tmp_path, note):
    """The tree before was not judged, so its verdicts are newly visible (extraction is retried once)."""
    orch, _ = _cycle(tmp_path, [[REQ1, REQ2]], _tools(FIX_ORDER), entry=(IN_ORDER, note))

    assert not orch._phase3_rolled_back, orch._phase3_exit_reason


# ---------------------------------------------------------------- an attempt that only installed


def test_an_attempt_that_only_ran_an_install_is_checked_again(tmp_path):
    """``verification setup:`` tells the model to install, not to edit: the install moves no source."""
    setup = B("blocker", "verification setup: tsc [frontend] requires installed project dependencies.")
    client = _Client(_tools(("run_command", {"command": "echo installed"})))
    orch = _build(tmp_path, client, allow_shell_tools=True)
    checks = iter([[setup], []])

    with patch.object(orch, "_collect_validation_issues", side_effect=lambda: next(checks)):
        orch._run_phase3_validation()

    assert orch._phase3_exit_reason == "all blockers fixed"


def test_a_change_in_installed_dependencies_is_checked_again(tmp_path):
    orch = _build(tmp_path, _Client())
    before = orch._install_state()
    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "package.json").write_text("{}", encoding="utf-8")
    assert orch._install_state() != before
    middle = orch._install_state()
    (tmp_path / "frontend" / "node_modules").mkdir()
    assert orch._install_state() != middle


# ---------------------------------------------------------------- requirement ranks

UNSUPPORTED5 = B("blocker", "requirement unsupported: R5 — Orders show a total; the cited code is a stub")
UNVERIFIED5 = B("blocker", "requirement unverified: R5 — Orders show a total — the cited evidence does not verify it")


@pytest.mark.parametrize("before, after", [
    pytest.param(MISSING5, UNSUPPORTED5, id="missing to unsupported"),
    pytest.param(UNSUPPORTED5, MISSING5, id="unsupported to missing"),
])
def test_a_requirement_moving_between_equal_ranks_is_not_retired(tmp_path, before, after):
    orch, _ = _cycle(tmp_path, [[after], [before]], _tools(FIX_ORDER), entry=(before,))

    assert orch._phase3_exit_reason == "attempt 1 reverted: it retired none of the 1 blocker it faced"


def test_partial_to_unsupported_is_a_fall(tmp_path):
    r6 = B("blocker", "requirement: R6 — Orders can be cancelled is not implemented")
    orch, _ = _cycle(tmp_path, [[UNSUPPORTED5, r6], [IN_ORDER, PARTIAL5]], _tools(FIX_ORDER),
                     entry=(IN_ORDER, PARTIAL5))

    assert orch._phase3_exit_reason == "attempt 1 reverted: it left more blockers (2 -> 3)"


def test_a_requirement_judged_implemented_but_not_located_is_retired(tmp_path):
    orch, _ = _cycle(tmp_path, [[UNVERIFIED5]], _tools(FIX_ORDER), entry=(PARTIAL5,))

    assert not orch._phase3_rolled_back, orch._phase3_exit_reason


# ---------------------------------------------------------------- an evidence-only attempt is held to the rules


def test_evidence_that_adds_an_observed_failure_ends_the_loop(tmp_path):
    """No edit to revert, so the attempt cannot be kept or undone: the loop stops there."""
    scenario = ("test_api", {"scenario_id": "orders", "requests": [{"method": "GET", "path": "/order/"}]})
    orch, client = _cycle(tmp_path, [[ROUTE_500, MAPPER]], _tools(scenario), _done(), entry=(IN_ORDER, ROUTE_500),
                          before=lambda o: setattr(o, "_repair_obligations_revision",
                                                   lambda: str(len(o.tool_calls_log))))

    assert orch._phase3_exit_reason.endswith("it added an observed failure: " + MAPPER.message)
    assert len(client.requests) == 2


# ---------------------------------------------------------------- needed_for only where it means something

WRITE_TOOLS = {"write_file", "modify_file", "replace_file_lines", "delete_file"}


def _offers_needed_for(tools) -> set[str]:
    return {tool["name"] for tool in tools if "needed_for" in tool["input_schema"].get("properties", {})}


def test_needed_for_is_offered_only_while_an_attempts_edits_are_limited(tmp_path):
    client = _Client()
    orch = _build(tmp_path, client)
    assert _offers_needed_for(orch.tools) == set(), "every Phase 2 turn re-sends the schemas"

    orch._invoke_phase3_fix_loop([IN_ORDER], is_first_attempt=True)
    orch._invoke_phase3_fix_loop([ROUTE_500], is_first_attempt=True)  # names no file: scope open

    assert _offers_needed_for(client.requests[0]["tools"]) == WRITE_TOOLS
    assert _offers_needed_for(client.requests[1]["tools"]) == set()


def test_needed_for_where_it_is_not_offered_is_ignored(tmp_path):
    orch = _build(tmp_path, _Client())

    result = orch.executor.execute_typed("modify_file", {
        "path": "backend/services/customer.py", "old_text": "return []", "new_text": "return [2]",
        "needed_for": "a blocker"})

    assert result.succeeded


# ---------------------------------------------------------------- judged by what was tested

CRASH_TOTAL = B("blocker", "create contract: backend: POST /order/ - observed a server failure creating Order "
                           "(500: IntegrityError NOT NULL constraint failed: order.total)")


def test_a_create_that_still_crashes_with_another_error_is_still_failing_not_retired(tmp_path):
    orch, client = _cycle(tmp_path, [[CRASH_TOTAL]], _tools(FIX_ORDER), _done(), entry=(CREATE,), on_check=_probe(
        _backend({"Order": ORDER}), _backend({"Order": ORDER})))

    assert not orch._phase3_rolled_back, orch._phase3_exit_reason
    second = client.requests[2]["messages"][0]["content"]
    assert "retired 0 blockers. Still failing, now: " + CRASH_TOTAL.message in second


def _crash(placed_at: str) -> ValidationIssue:
    return B("blocker", "create contract: backend: POST /order/ - observed a server/persistence failure creating "
                        f'Order (500: Internal Server Error); attempted input: {{"placed_at": "{placed_at}"}}. '
                        "This is one observed failure, not proof that every valid request fails.")


def test_a_create_whose_only_change_is_the_probe_input_is_not_retired(tmp_path):
    before, after = _crash("2030-01-01T10:00:00.120000"), _crash("2030-01-01T10:00:03.450000")
    orch, client = _cycle(tmp_path, [[after], [before]], _tools(FIX_ORDER), entry=(before,), on_check=_probe(
        _backend({"Order": ORDER}), _backend({"Order": ORDER})))

    assert orch._phase3_exit_reason == "attempt 1 reverted: it retired none of the 1 blocker it faced"
    assert len(client.requests) == 2, "no progress, so no second attempt"

