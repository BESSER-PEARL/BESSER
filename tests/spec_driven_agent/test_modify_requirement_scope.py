"""A follow-up run answers for its own request and for what it broke, not for the whole app again.

Live evidence (2026-11-19, free tier, a follow-up asking to fix one ``TypeError``):
the fix landed by turn 7, then Phase 3 extracted the whole original request
again, judged all 35 requirements, reported 23 of them as blockers and spent 20
minutes editing six files the request did not concern, until the user cancelled.
The same happened in every follow-up run of that day.

A modify run now keeps the previous run's requirements and its verdicts on the
tree it delivered, and extracts only the messages the previous run did not have,
with the earlier conversation as context. Those messages may replace or remove a
previous requirement, or only ask to retry or finish earlier work. A verdict whose
files the run left unchanged is not judged again; a requirement the previous run
had left unmet and that is still unmet is reported as still open from that run (a
warning) unless the messages ask to finish earlier work; one that was implemented
and no longer is stays a blocker.
"""

import json
from types import SimpleNamespace

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.planning.specification import MAX_REQUIREMENTS
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.validation.issues import _classify_issue, is_completion_issue

ORDERS_PY = (
    "from fastapi import APIRouter, HTTPException\n\n"
    "router = APIRouter()\n\n"
    "def create_order(order):\n"
    "    if order.quantity < 1:\n"
    "        raise HTTPException(422)\n"
    "    return order\n"
)
CUSTOMERS_PY = "from fastapi import APIRouter\n\nrouter = APIRouter()\n\ndef list_customers():\n    return []\n"
REPORTS_PY = "def monthly_total(orders):\n    return 0\n"
MAIN_API_PY = (
    "from fastapi import FastAPI\n"
    "from routers import orders, customers\n\n"
    "app = FastAPI()\n"
    "app.include_router(orders.router)\n"
    "app.include_router(customers.router)\n"
)
ORIGINAL = "An order needs a quantity of at least one. Customers can be listed. A report shows the monthly total."
EXPORT = "Add a CSV export of orders."
LATEST = "Creating an order without a quantity raises TypeError. Fix this error."
RETRY = "Can you retry the previous step? It got stuck."


def _instructions(*messages):
    text = ORIGINAL
    for message in messages:
        text += (f"\n\n{ledger.LATER_REQUEST_MARKER}\n\n"
                 "This later request takes precedence where it changes earlier requirements.\n\n" + message)
    return text


INSTRUCTIONS = _instructions(LATEST)

PREVIOUS = [
    {"id": 1, "text": "An order needs a quantity of at least one", "kind": "validation", "status": "implemented",
     "evidence": [{"path": "backend/routers/orders.py", "line": "if order.quantity < 1:"}], "note": "",
     "verification": "unverified", "verification_note": "Found in the source"},
    {"id": 2, "text": "Customers can be listed", "kind": "action", "status": "implemented",
     "evidence": [{"path": "backend/routers/customers.py", "line": "def list_customers():"}], "note": ""},
    {"id": 3, "text": "A report shows the monthly total", "kind": "computed", "status": "missing",
     "evidence": "", "note": "monthly_total returns 0", "inspection_paths": ["backend/services/reports.py"]},
]
NEW = {"text": "Creating an order without a quantity is refused, not a server error", "kind": "validation"}
FIXED_LINE = [{"path": "backend/routers/orders.py", "line": "if order.quantity is None:"}]


class _Client:
    """A real provider whose extraction answers with ``reply`` through the follow-up tool."""

    model = "mock-model"
    planning_model = None

    def __init__(self, reply=None):
        self._client = object()
        self.usage = UsageTracker("mock-model")
        self.reply = reply if reply is not None else {}
        self.prompts = []

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        self.prompts.append(messages[0]["content"])
        block = SimpleNamespace(type="tool_use", name=tools[0]["name"], input=self.reply)
        return {"stop_reason": "tool_use", "content": [block]}


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _model():
    order = Class(name="Order")
    order.attributes = {Property(name="quantity", type=PrimitiveDataType("int"))}
    return DomainModel(name="Shop", types={order})


def _seed(root, previous=PREVIOUS, instructions=ORIGINAL):
    _write(root, "backend/main_api.py", MAIN_API_PY)
    _write(root, "backend/routers/orders.py", ORDERS_PY)
    _write(root, "backend/routers/customers.py", CUSTOMERS_PY)
    _write(root, "backend/services/reports.py", REPORTS_PY)
    (root / ".besser_recipe.json").write_text(
        json.dumps({"instructions": instructions, "requirements": previous}), encoding="utf-8")


def _orchestrator(root, client, instructions=INSTRUCTIONS, **kwargs):
    orch = LLMOrchestrator(llm_client=client, domain_model=_model(), output_dir=str(root),
                           enable_tracing=False, enable_checkpointing=False, **kwargs)
    orch._instructions = instructions
    orch._generator_used = "generate_fastapi_backend"
    orch._modify_mode = True
    orch._seed_source_hashes = ledger.source_hashes(str(root))
    orch._previous_verdicts = orch._load_previous_verdicts()
    return orch


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A modify run seeded with a previous run's app and recipe; the judge's answers and calls."""
    _seed(tmp_path)
    client = _Client({"requirements": [NEW], "replaces": [], "continues_earlier_work": False})
    orch = _orchestrator(tmp_path, client)
    answers = {4: {"status": "implemented", "evidence": FIXED_LINE}}
    calls = []

    def judge(requirements, digest, client, **kwargs):
        calls.append([item["id"] for item in requirements])
        return [{**item, "note": "", "evidence": "", "status": "missing", **answers.get(item["id"], {})}
                for item in requirements]

    monkeypatch.setattr(ledger, "judge_coverage", judge)
    return SimpleNamespace(orch=orch, root=tmp_path, answers=answers, calls=calls, client=client)


def _fix_the_reported_error(root):
    _write(root, "backend/routers/orders.py", ORDERS_PY.replace(
        "    if order.quantity < 1:", "    if order.quantity is None:\n        raise HTTPException(422)\n"
                                     "    if order.quantity < 1:"))


def _blockers(issues):
    return [issue for issue in issues if _classify_issue(issue).severity == "blocker"]


def _about(issues, rid):
    return [issue for issue in issues if f"R{rid} " in issue]


# ------------------------------------------------------------------ the ledger


def test_the_ledger_is_the_previous_runs_plus_the_new_messages(run):
    run.orch._extract_requirements(run.orch._instructions)

    assert [(item["id"], item["text"]) for item in run.orch._requirements] == [
        (1, PREVIOUS[0]["text"]), (2, PREVIOUS[1]["text"]), (3, PREVIOUS[2]["text"]), (4, NEW["text"])]
    [prompt] = run.client.prompts
    new_part = prompt.split("## The new user messages", 1)[1]
    assert LATEST in new_part and ORIGINAL not in new_part


def test_every_message_the_previous_run_did_not_have_is_extracted(tmp_path, monkeypatch):
    """A cancelled run's message is not in the base's request: the retry after it brings both."""
    _seed(tmp_path, instructions=_instructions(EXPORT))
    client = _Client({"requirements": [NEW], "replaces": [], "continues_earlier_work": False})
    orch = _orchestrator(tmp_path, client, instructions=_instructions(EXPORT, LATEST, RETRY))

    orch._extract_requirements(orch._instructions)

    earlier, new = client.prompts[0].split("## The new user messages", 1)
    assert LATEST in new and RETRY in new and EXPORT not in new
    assert EXPORT in earlier and ORIGINAL in earlier


def test_a_retry_puts_the_open_items_in_scope(run):
    """Nothing new is stated, the messages ask to finish earlier work: R3 is judged and a blocker."""
    run.client.reply = {"requirements": [], "replaces": [], "continues_earlier_work": True}
    run.orch._instructions = _instructions("Can you retry fixing the requirements?")

    issues = run.orch._collect_requirement_issues()

    assert run.calls == [[3]]
    [r3] = _about(issues, 3)
    assert r3.startswith("requirement: R3") and _classify_issue(r3).severity == "blocker"
    assert not any("previous run" in issue for issue in issues)


@pytest.mark.parametrize("change", ["remove", "change"])
def test_a_requirement_the_new_message_replaces_is_not_a_regression(run, change):
    if change == "remove":
        run.orch._instructions = _instructions("Remove the customer list, nobody uses it.")
        run.client.reply = {"requirements": [{"text": "The app has no customer list", "kind": "absence"}],
                            "replaces": [2], "continues_earlier_work": False}
        _write(run.root, "backend/routers/customers.py", CUSTOMERS_PY.replace("def list_customers():", "def _x():"))
        run.answers[4] = {"status": "implemented",
                          "evidence": [{"path": "backend/routers/customers.py", "line": "def _x():"}]}
        replaced = 2
    else:
        run.orch._instructions = _instructions("An order needs a quantity of at least two.")
        run.client.reply = {"requirements": [{"text": "An order needs a quantity of at least two",
                                              "kind": "validation"}],
                            "replaces": [1], "continues_earlier_work": False}
        _write(run.root, "backend/routers/orders.py", ORDERS_PY.replace("< 1", "< 2"))
        run.answers[4] = {"status": "implemented",
                          "evidence": [{"path": "backend/routers/orders.py", "line": "if order.quantity < 2:"}]}
        replaced = 1

    issues = run.orch._collect_requirement_issues()

    assert replaced not in [item["id"] for item in run.orch._requirements]
    assert not _about(issues, replaced)
    assert _blockers(issues) == []


def test_a_new_requirement_with_the_text_of_an_open_one_is_judged_as_this_runs(run):
    run.client.reply = {"requirements": [{"text": PREVIOUS[2]["text"], "kind": "computed"}],
                        "replaces": [], "continues_earlier_work": False}
    run.answers.pop(4)

    issues = run.orch._collect_requirement_issues()

    assert 4 in sum(run.calls, [])
    [r4] = _about(issues, 4)
    assert r4.startswith("requirement: R4") and _classify_issue(r4).severity == "blocker"


# ------------------------------------------------------------------ what is judged


def test_only_the_new_request_and_the_requirements_on_changed_files_are_judged(run):
    """orders.py changed: R1 (cited there) and the new R4 are judged; R2 and R3 are not."""
    _fix_the_reported_error(run.root)

    run.orch._collect_requirement_issues()

    assert run.calls == [[1, 4]]


def test_a_requirement_the_previous_run_left_open_is_reported_not_repaired(run):
    _fix_the_reported_error(run.root)
    run.answers[1] = {"status": "implemented",
                      "evidence": [{"path": "backend/routers/orders.py", "line": "if order.quantity < 1:"}]}

    issues = run.orch._collect_requirement_issues()

    [open_item] = _about(issues, 3)
    assert open_item.startswith("requirement still open from the previous run: R3")
    assert _classify_issue(open_item).severity == "warning"
    assert not is_completion_issue(_classify_issue(open_item))
    assert _blockers(issues) == []
    [r3] = [item for item in run.orch._requirement_verdicts if item["id"] == 3]
    assert r3["status"] == "missing" and r3[ledger.INHERITED]


def test_a_requirement_left_open_and_still_open_after_its_files_changed_is_still_inherited(run):
    _write(run.root, "backend/services/reports.py", REPORTS_PY.replace("return 0", "return -1"))

    issues = run.orch._collect_requirement_issues()

    assert run.calls[0] == [3, 4]
    [open_item] = _about(issues, 3)
    assert open_item.startswith("requirement still open from the previous run: R3")


def test_a_requirement_the_run_broke_is_a_blocker(run):
    _write(run.root, "backend/routers/customers.py", CUSTOMERS_PY.replace("def list_customers():", "def _gone():"))

    issues = run.orch._collect_requirement_issues()

    [broken] = _about(issues, 2)
    assert broken.startswith("requirement: R2") and _classify_issue(broken).severity == "blocker"
    assert "implemented when the previous run ended" in broken


def test_the_new_request_unmet_is_a_blocker(run):
    run.answers.pop(4)

    issues = run.orch._collect_requirement_issues()

    [new] = _about(issues, 4)
    assert _classify_issue(new).severity == "blocker"


def test_a_recorded_implemented_whose_citation_no_longer_matches_is_judged_and_can_regress(tmp_path, monkeypatch):
    previous = json.loads(json.dumps(PREVIOUS))
    previous[1]["evidence"] = [{"path": "backend/routers/customers.py", "line": "def all_customers():"}]
    _seed(tmp_path, previous=previous)
    orch = _orchestrator(tmp_path, _Client({"requirements": [NEW], "replaces": [], "continues_earlier_work": False}))
    calls = []
    monkeypatch.setattr(ledger, "judge_coverage", lambda reqs, digest, client, **kw: calls.append(
        [r["id"] for r in reqs]) or [{**r, "note": "", "evidence": "", "status": "missing"} for r in reqs])
    _write(tmp_path, "backend/routers/customers.py", CUSTOMERS_PY.replace("def list_customers():", "def _gone():"))

    issues = orch._collect_requirement_issues()

    assert 2 in calls[0]
    [r2] = _about(issues, 2)
    assert r2.startswith("requirement: R2") and "implemented when the previous run ended" in r2


def test_the_incomplete_extraction_marker_is_never_inherited(tmp_path, monkeypatch):
    marker = {"id": 4, "text": ledger._EXTRACTION_INCOMPLETE, "kind": "verification", "status": "unverified",
              "evidence": "", "note": ""}
    _seed(tmp_path, previous=[*PREVIOUS, marker])
    orch = _orchestrator(tmp_path, _Client({"requirements": [], "replaces": [], "continues_earlier_work": False}))
    monkeypatch.setattr(ledger, "judge_coverage", lambda reqs, digest, client, **kw: [
        {**r, "note": "", "evidence": "", "status": "unverified"} for r in reqs])

    issues = orch._collect_requirement_issues()

    [gate] = _about(issues, 4)
    assert gate.startswith("requirement unverified: R4") and _classify_issue(gate).severity == "blocker"


# ------------------------------------------------------------------ the recipe it reads


def test_malformed_recorded_verdicts_are_skipped_not_fatal(tmp_path):
    previous = [*json.loads(json.dumps(PREVIOUS)),
                {"id": 4, "text": "Bad shape", "kind": 7, "status": "missing", "evidence": {"path": 1},
                 "inspection_paths": 5, "note": None},
                {"id": "5", "text": "no int id", "status": "missing"}, "not a verdict", None]
    _seed(tmp_path, previous=previous)

    orch = _orchestrator(tmp_path, _Client())

    assert [item["id"] for item in orch._previous_verdicts] == [1, 2, 3, 4]
    assert orch._previous_verdicts[3]["evidence"] == "" and orch._previous_verdicts[3]["inspection_paths"] == []


def test_a_recipe_that_cannot_be_used_falls_back_to_the_whole_request(tmp_path, monkeypatch):
    _seed(tmp_path)

    def broken(recorded, output_dir):
        raise ValueError("unexpected recipe")

    monkeypatch.setattr("besser.spec_driven_agent.pipeline.modify_run.previous_run_verdicts", broken)
    assert _orchestrator(tmp_path, _Client())._previous_verdicts is None


def test_the_carried_requirements_are_capped(tmp_path):
    previous = [{"id": n, "text": f"Requirement {n}", "kind": "rule", "status": "missing", "evidence": "",
                 "note": ""} for n in range(1, MAX_REQUIREMENTS + 50)]
    _seed(tmp_path, previous=previous)

    assert len(_orchestrator(tmp_path, _Client())._previous_verdicts) == MAX_REQUIREMENTS


def test_the_editor_ledger_takes_previous_verdicts_by_text_not_by_number(tmp_path, monkeypatch):
    _seed(tmp_path)
    stated = [{"id": "a", "text": "Orders can be exported as CSV", "kind": "action"},
              {"id": "b", "text": PREVIOUS[2]["text"], "kind": "computed"}]
    orch = _orchestrator(tmp_path, _Client(), instructions=ORIGINAL, stated_requirements=stated)
    monkeypatch.setattr(ledger, "judge_coverage", lambda reqs, digest, client, **kw: [
        {**r, "note": "", "evidence": "", "status": "missing"} for r in reqs])

    issues = orch._collect_requirement_issues()

    [export] = [issue for issue in issues if "exported as CSV" in issue]
    assert export.startswith("requirement: R1") and _classify_issue(export).severity == "blocker"
    [report] = [issue for issue in issues if PREVIOUS[2]["text"] in issue]
    assert report.startswith("requirement still open from the previous run: R2")


# ------------------------------------------------------------------ create runs, resume, report


def test_a_create_run_extracts_and_judges_the_whole_request(tmp_path, monkeypatch):
    _write(tmp_path, "backend/routers/orders.py", ORDERS_PY)
    orch = LLMOrchestrator(llm_client=_Client(), domain_model=_model(), output_dir=str(tmp_path),
                           enable_tracing=False, enable_checkpointing=False)
    orch._instructions = INSTRUCTIONS
    extracted = []
    monkeypatch.setattr(ledger, "extract_requirements",
                        lambda instructions, client, **kw: extracted.append(instructions) or [{"id": 1, **NEW}])

    orch._extract_requirements(orch._instructions)

    assert extracted == [INSTRUCTIONS]


def test_a_resumed_modify_run_keeps_what_it_carried(run):
    run.orch._extract_requirements(run.orch._instructions)
    state = json.loads(json.dumps(run.orch._modify_checkpoint_state()))
    resumed = LLMOrchestrator(llm_client=_Client(), domain_model=_model(), output_dir=str(run.root),
                              enable_tracing=False, enable_checkpointing=False)
    resumed._instructions = run.orch._instructions
    resumed._generator_used = run.orch._generator_used
    resumed._restore_modify_state(state)
    resumed._requirements = json.loads(json.dumps(run.orch._requirements))

    resumed._collect_requirement_issues()

    assert 3 not in sum(run.calls, []), "a requirement carried before the resume was judged again"


def test_the_report_says_the_open_item_is_from_the_previous_run(run):
    from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import build_verification_report

    run.orch._collect_requirement_issues()
    recipe = {"requirements": ledger.without_fingerprints(run.orch._requirement_verdicts)}

    report = build_verification_report(recipe)

    [r3] = [item for item in report.shippedUnenforced if item.id == "R3"]
    assert r3.why.startswith("open since the previous run")


def test_a_modify_run_judges_its_request_and_its_changes_only(tmp_path, monkeypatch):
    """Through ``modify()``: the fix edits orders.py; the open report requirement is not a blocker."""
    _seed(tmp_path)
    client = _Client({"requirements": [NEW], "replaces": [], "continues_earlier_work": False})
    orch = LLMOrchestrator(llm_client=client, domain_model=_model(), output_dir=str(tmp_path),
                           enable_tracing=False, enable_checkpointing=False, enable_toolchain_validation=False)
    calls, found = [], {}

    def judge(requirements, digest, client, **kwargs):
        calls.append([item["id"] for item in requirements])
        cited = {1: "if order.quantity < 1:", 2: "def list_customers():", 4: "if order.quantity is None:"}
        return [{**item, "note": "", "status": "implemented" if item["id"] in cited else "missing",
                 "evidence": [{"path": "backend/routers/orders.py" if item["id"] != 2 else
                               "backend/routers/customers.py", "line": cited[item["id"]]}]
                 if item["id"] in cited else ""} for item in requirements]

    monkeypatch.setattr(ledger, "judge_coverage", judge)
    for name in ("_route_modify_request", "_derive_and_apply_model_deltas", "_regenerate_generated_files",
                 "_create_snapshot", "_save_recipe", "_remove_snapshot"):
        monkeypatch.setattr(orch, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(orch, "_validate_phase1_output", lambda: [])
    monkeypatch.setattr(orch, "_run_phase2", lambda instructions, extra_issues=None: _fix_the_reported_error(tmp_path))
    monkeypatch.setattr(orch, "_run_phase3_validation",
                        lambda: found.update(issues=orch._collect_requirement_issues()))

    orch.modify(INSTRUCTIONS)

    assert calls == [[1, 4]]
    assert _blockers(found["issues"]) == []
    assert any(issue.startswith("requirement still open from the previous run: R3") for issue in found["issues"])
