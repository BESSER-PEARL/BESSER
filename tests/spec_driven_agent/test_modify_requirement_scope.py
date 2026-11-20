"""A follow-up run answers for its own request and for what it broke, not for the whole app again.

Live evidence (2026-11-19, free tier, a follow-up asking to fix one ``TypeError``):
the fix landed by turn 7, then Phase 3 extracted the whole original request
again, judged all 35 requirements, reported 23 of them as blockers and spent 20
minutes editing six files the request did not concern, until the user cancelled.
The same happened in every follow-up run of that day.

A modify run now keeps the previous run's requirements and its verdicts on the
tree it delivered, and extracts only the newest message. A verdict whose files
the run left unchanged is not judged again; a requirement the previous run had
left unmet and that is still unmet is reported as still open from that run (a
warning); one that was implemented and no longer is stays a blocker.
"""

import json

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning import requirements_ledger as ledger
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
LATEST = "Creating an order without a quantity raises TypeError. Fix this error."
INSTRUCTIONS = (f"{ORIGINAL}\n\n{ledger.LATER_REQUEST_MARKER}\n\n"
                "This later request takes precedence where it changes earlier requirements.\n\n" + LATEST)

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


class _Client:
    model = "mock-model"
    planning_model = None

    def __init__(self):
        self._client = object()
        self.usage = UsageTracker("mock-model")


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _model():
    order = Class(name="Order")
    order.attributes = {Property(name="quantity", type=PrimitiveDataType("int"))}
    return DomainModel(name="Shop", types={order})


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A modify run seeded with a previous run's app and recipe; the judge's answers and calls."""
    _write(tmp_path, "backend/main_api.py", MAIN_API_PY)
    _write(tmp_path, "backend/routers/orders.py", ORDERS_PY)
    _write(tmp_path, "backend/routers/customers.py", CUSTOMERS_PY)
    _write(tmp_path, "backend/services/reports.py", REPORTS_PY)
    (tmp_path / ".besser_recipe.json").write_text(json.dumps({"requirements": PREVIOUS}), encoding="utf-8")
    orch = LLMOrchestrator(llm_client=_Client(), domain_model=_model(), output_dir=str(tmp_path),
                           enable_tracing=False, enable_checkpointing=False)
    orch._instructions = INSTRUCTIONS
    orch._generator_used = "generate_fastapi_backend"
    orch._modify_mode = True
    orch._seed_source_hashes = ledger.source_hashes(str(tmp_path))
    orch._previous_verdicts = orch._load_previous_verdicts()

    extracted = []

    def extract(instructions, client, **kwargs):
        extracted.append(instructions)
        return [{"id": 1, **NEW}]

    answers = {4: {"status": "implemented",
                   "evidence": [{"path": "backend/routers/orders.py", "line": "if order.quantity is None:"}]}}
    calls = []

    def judge(requirements, digest, client, **kwargs):
        calls.append([item["id"] for item in requirements])
        return [{**item, "note": "", "evidence": "", "status": "missing", **answers.get(item["id"], {})}
                for item in requirements]

    monkeypatch.setattr(ledger, "extract_requirements", extract)
    monkeypatch.setattr(ledger, "judge_coverage", judge)
    return orch, tmp_path, answers, calls, extracted


def _fix_the_reported_error(root):
    _write(root, "backend/routers/orders.py", ORDERS_PY.replace(
        "    if order.quantity < 1:", "    if order.quantity is None:\n        raise HTTPException(422)\n"
                                     "    if order.quantity < 1:"))


def test_the_ledger_is_the_previous_runs_plus_the_newest_message(run):
    orch, root, _, _, extracted = run
    orch._extract_requirements(orch._instructions)

    assert extracted == [LATEST]
    assert [(item["id"], item["text"]) for item in orch._requirements] == [
        (1, PREVIOUS[0]["text"]), (2, PREVIOUS[1]["text"]), (3, PREVIOUS[2]["text"]), (4, NEW["text"])]


def test_only_the_new_request_and_the_requirements_on_changed_files_are_judged(run):
    """orders.py changed: R1 (cited there) and the new R4 are judged; R2 and R3 are not."""
    orch, root, _, calls, _ = run
    _fix_the_reported_error(root)

    orch._collect_requirement_issues()

    assert calls == [[1, 4]]


def test_a_requirement_the_previous_run_left_open_is_reported_not_repaired(run):
    orch, root, answers, _, _ = run
    _fix_the_reported_error(root)
    answers[1] = {"status": "implemented",
                  "evidence": [{"path": "backend/routers/orders.py", "line": "if order.quantity < 1:"}]}

    issues = orch._collect_requirement_issues()

    [open_item] = [issue for issue in issues if "R3" in issue]
    assert open_item.startswith("requirement still open from the previous run: R3")
    assert _classify_issue(open_item).severity == "warning"
    assert not is_completion_issue(_classify_issue(open_item))
    assert not [issue for issue in issues if _classify_issue(issue).severity == "blocker"]
    [r3] = [item for item in orch._requirement_verdicts if item["id"] == 3]
    assert r3["status"] == "missing" and r3[ledger.INHERITED]


def test_a_requirement_left_open_and_still_open_after_its_files_changed_is_still_inherited(run):
    orch, root, answers, calls, _ = run
    _write(root, "backend/services/reports.py", REPORTS_PY.replace("return 0", "return -1"))
    answers[4] = {"status": "implemented",
                  "evidence": [{"path": "backend/routers/orders.py", "line": "if order.quantity < 1:"}]}

    issues = orch._collect_requirement_issues()

    assert calls == [[3, 4]]
    [open_item] = [issue for issue in issues if "R3" in issue]
    assert open_item.startswith("requirement still open from the previous run: R3")


def test_a_requirement_the_run_broke_is_a_blocker(run):
    orch, root, _, _, _ = run
    _write(root, "backend/routers/customers.py", CUSTOMERS_PY.replace("def list_customers():", "def _gone():"))

    issues = orch._collect_requirement_issues()

    [broken] = [issue for issue in issues if "R2" in issue]
    assert broken.startswith("requirement: R2") and _classify_issue(broken).severity == "blocker"
    assert "implemented when the previous run ended" in broken


def test_the_new_request_unmet_is_a_blocker(run):
    orch, root, answers, _, _ = run
    answers.pop(4)

    issues = orch._collect_requirement_issues()

    [new] = [issue for issue in issues if "R4" in issue]
    assert _classify_issue(new).severity == "blocker"


def test_a_previous_verdict_whose_citation_no_longer_matches_is_judged(run):
    """The previous run's "implemented" is re-checked against the tree it delivered."""
    orch, root, _, calls, _ = run
    recipe = json.loads((root / ".besser_recipe.json").read_text(encoding="utf-8"))
    recipe["requirements"][1]["evidence"] = [{"path": "backend/routers/customers.py", "line": "def all_customers():"}]
    (root / ".besser_recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    orch._previous_verdicts = orch._load_previous_verdicts()

    orch._collect_requirement_issues()

    assert 2 in calls[0]


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


def test_a_resumed_modify_run_keeps_the_previous_verdicts(run):
    orch, root, _, _, _ = run
    state = json.loads(json.dumps(orch._modify_checkpoint_state()))
    resumed = LLMOrchestrator(llm_client=_Client(), domain_model=_model(), output_dir=str(root),
                              enable_tracing=False, enable_checkpointing=False)
    resumed._restore_modify_state(state)

    assert resumed._previous_verdicts == orch._previous_verdicts


def test_the_report_says_the_open_item_is_from_the_previous_run(run):
    from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import build_verification_report

    orch, root, _, _, _ = run
    orch._collect_requirement_issues()
    recipe = {"requirements": ledger.without_fingerprints(orch._requirement_verdicts)}

    report = build_verification_report(recipe)

    [r3] = [item for item in report.shippedUnenforced if item.id == "R3"]
    assert r3.why.startswith("open since the previous run")


def test_a_modify_run_judges_its_request_and_its_changes_only(tmp_path, monkeypatch):
    """Through ``modify()``: the fix edits orders.py; the open report requirement is not a blocker."""
    _write(tmp_path, "backend/main_api.py", MAIN_API_PY)
    _write(tmp_path, "backend/routers/orders.py", ORDERS_PY)
    _write(tmp_path, "backend/routers/customers.py", CUSTOMERS_PY)
    _write(tmp_path, "backend/services/reports.py", REPORTS_PY)
    (tmp_path / ".besser_recipe.json").write_text(json.dumps({"requirements": PREVIOUS}), encoding="utf-8")
    orch = LLMOrchestrator(llm_client=_Client(), domain_model=_model(), output_dir=str(tmp_path),
                           enable_tracing=False, enable_checkpointing=False, enable_toolchain_validation=False)
    extracted, calls, found = [], [], {}
    monkeypatch.setattr(ledger, "extract_requirements",
                        lambda instructions, client, **kw: extracted.append(instructions) or [{"id": 1, **NEW}])

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

    assert extracted == [LATEST]
    assert calls == [[1, 4]]
    blockers = [issue for issue in found["issues"] if _classify_issue(issue).severity == "blocker"]
    assert blockers == []
    assert any(issue.startswith("requirement still open from the previous run: R3") for issue in found["issues"])
