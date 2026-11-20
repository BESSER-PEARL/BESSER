"""An implemented requirement says how far it was checked, and claims no more.

Live evidence, r9-p61 (wastewater dashboard): all four requirements were
recorded as a bare "implemented" from citations alone. Three are screens and the
fourth is a map click, all cited in frontend code, and the frontend build was
skipped. The checklist items carried ``verification: unverified``; the
requirements carried nothing.

Review: one passing workflow that never touches the cited file is no evidence for
it, and a frontend that compiles has had no screen run. r13-p21: a 15-request workflow
with value assertions passed, yet all 5 requirements read "unverified" because each
also cites a page, and the call named was the alphabetically first of the cited FILE.
The note now gives the API half and the screens half apart, ``api_verified`` marks a
backend half that a workflow exercised, and the call is matched to the cited route.
"""

import json
import os

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.providers.llm_client import UsageTracker

DASHBOARD = "frontend/src/pages/Dashboard.jsx"
ROUTER = "backend/routers/dashboard.py"
SERVICE = "backend/metrics.py"
ROUTER_SOURCE = '''from fastapi import APIRouter

router = APIRouter(prefix="/dashboard")


@router.get("/metrics/{plant_id}/")
def metrics(plant_id: int):
    return compute(plant_id)


@router.get("/alerts/")
def alerts():
    return []
'''


def _verdict(rid, kind, *paths, status="implemented", line="x"):
    return {"id": rid, "text": f"R{rid}", "kind": kind, "status": status, "note": "",
            "evidence": [{"path": path, "line": line} for path in paths] if status == "implemented" else ""}


def _app(tmp_path):
    for rel, text in ((ROUTER, ROUTER_SOURCE), (SERVICE, "def compute(p):\n    return p\n"),
                      (DASHBOARD, "export default function D() { return null; }\n")):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    return str(tmp_path)


def _levels(tmp_path, verdicts, *, build="not_run", calls=(), passed=None):
    return ledger.with_verification_levels(
        verdicts, _app(tmp_path), frontend_build=build, exercised=set(calls),
        workflows_passed=len(calls) if passed is None else passed)


# r9-p61's four requirements, by kind and where they were cited.
P61 = [_verdict(1, "ui", DASHBOARD), _verdict(2, "ui", DASHBOARD), _verdict(3, "ui", DASHBOARD),
       _verdict(4, "action", DASHBOARD)]
CALL = ("GET", "/dashboard/metrics/3/")
METRICS_LINE = "return compute(plant_id)"


def test_frontend_citations_are_never_verified_and_the_note_says_why(tmp_path):
    for build, reason in (("not_run", "the frontend build check did not run"), ("failed", "the frontend build failed"),
                          ("passed", "the frontend builds"), ("none", "no frontend")):
        leveled = _levels(tmp_path / build, P61, build=build, calls=[CALL])
        assert [item["verification"] for item in leveled] == ["unverified"] * 4
        assert all(f"screens: not run ({reason})" in item["verification_note"] for item in leveled)


def test_a_backend_citation_is_verified_by_a_call_to_the_cited_route(tmp_path):
    called = _levels(tmp_path / "a", [_verdict(5, "action", ROUTER, line=METRICS_LINE)],
                     calls=[("GET", "/dashboard/alerts/"), CALL])[0]
    assert called["verification"] == "verified"
    assert "API: a passing workflow called GET /dashboard/metrics/3/, the cited route" in called["verification_note"]

    elsewhere = _levels(tmp_path / "b", [_verdict(5, "action", ROUTER, line=METRICS_LINE)],
                        calls=[("GET", "/dashboard/alerts/")])[0]
    assert elsewhere["verification"] == "unverified"
    assert "API: a workflow passed but called no cited route" in elsewhere["verification_note"]

    service = _levels(tmp_path / "c", [_verdict(6, "rule", SERVICE, line="return p")], calls=[CALL])[0]
    assert service["verification"] == "unverified"  # the module declares no route

    none = _levels(tmp_path / "d", [_verdict(5, "action", ROUTER, line=METRICS_LINE)])[0]
    assert "API: no workflow passed" in none["verification_note"]


ORDERS = "backend/routers/orders.py"
ORDERS_SOURCE = '''from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/orders")


def check_quantity(quantity: int):
    if quantity <= 0:
        raise HTTPException(status_code=422, detail="quantity must be positive")


def require_clerk(role: str = "clerk"):
    if role != "clerk":
        raise HTTPException(status_code=403, detail="clerks only")


@router.get("/")
def list_orders():
    result = []
    return result


@router.post("/", dependencies=[Depends(require_clerk)])
def create_order(quantity: int):
    if quantity > 100:
        raise HTTPException(status_code=422, detail="at most 100 items")
    check_quantity(quantity)
    result = {"quantity": quantity}
    return result


def cancel_order(order_id: int):
    if order_id < 1:
        raise HTTPException(status_code=404, detail="no such order")
    return {}


router.add_api_route("/{order_id}/cancel", cancel_order, methods=["POST"])
'''
GET_ONLY = [("GET", "/orders/")]
WITH_POST = [("GET", "/orders/"), ("POST", "/orders/")]


def _orders_level(tmp_path, line, calls, *, rel=ORDERS, source=ORDERS_SOURCE):
    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / rel).write_text(source, encoding="utf-8")
    return ledger.with_verification_levels([_verdict(1, "rule", rel, line=line)], str(tmp_path),
                                           frontend_build="none", exercised=set(calls),
                                           workflows_passed=len(calls))[0]


# Code POST /orders/ runs, cited as the evidence check accepts it: verbatim, with a comment the
# source does not have, wrapped differently, abridged with an ellipsis around a line GET has too,
# a helper the handler calls, a dependency of the route, and its decorator alone.
POST_QUOTES = ("if quantity > 100:",
               'if quantity > 100:  # cap\n        raise HTTPException(status_code=422, detail="at most 100 items")',
               'raise HTTPException(\n  status_code=422,\n  detail="at most 100 items")',
               "if quantity > 100: ... return result",
               "if quantity <= 0:",
               'if role != "clerk":',
               '@router.post("/", dependencies=[Depends(require_clerk)])')


def test_code_a_route_runs_is_found_in_code_until_that_route_ran(tmp_path):
    for index, line in enumerate(POST_QUOTES):
        item = _orders_level(tmp_path / str(index), line, GET_ONLY)
        assert item["verification"] == "unverified", line
        assert item["verification_note"].startswith("Found in the source; "), line
        assert "API: a workflow passed but called no cited route" in item["verification_note"], line


def test_code_a_route_runs_is_verified_when_that_route_ran(tmp_path):
    for index, line in enumerate(POST_QUOTES):
        item = _orders_level(tmp_path / str(index), line, WITH_POST)
        assert item["verification"] == "verified", line
        assert "API: a passing workflow called POST /orders/, the cited route" in item["verification_note"], line


def test_the_handler_registered_with_add_api_route_is_its_route(tmp_path):
    item = _orders_level(tmp_path / "a", "if order_id < 1:", [("POST", "/orders/3/cancel")])
    assert item["verification"] == "verified"
    assert _orders_level(tmp_path / "b", "if order_id < 1:", WITH_POST)["verification"] == "unverified"


def test_a_citation_no_route_runs_is_found_in_code_whatever_ran(tmp_path):
    calls = [*WITH_POST, ("POST", "/orders/3/cancel")]
    for line in ("from fastapi import APIRouter, Depends, HTTPException", 'router = APIRouter(prefix="/orders")'):
        item = _orders_level(tmp_path, line, calls)
        assert item["verification"] == "unverified", line
        assert "API: not run (no cited line is in a route handler)" in item["verification_note"], line


def test_backend_code_whose_routes_cannot_be_read_says_so(tmp_path):
    item = _orders_level(tmp_path, "if (quantity > 100) {", WITH_POST, rel="backend/src/OrderController.java",
                         source="class OrderController {\n  void create(int quantity) {\n"
                                "    if (quantity > 100) {\n      throw new IllegalArgumentException();\n"
                                "    }\n  }\n}\n")
    assert item["verification"] == "unverified"
    assert "API: not run (the check cannot read the routes of the cited code)" in item["verification_note"]


def test_a_full_stack_requirement_whose_api_half_ran_is_api_verified(tmp_path):
    both = {**_verdict(7, "action", ROUTER, line=METRICS_LINE),
            "evidence": [{"path": ROUTER, "line": METRICS_LINE}, {"path": DASHBOARD, "line": "x"}]}
    leveled = _levels(tmp_path, [both], build="passed", calls=[CALL])[0]
    assert leveled["verification"] == "api_verified"
    assert "API: a passing workflow called GET /dashboard/metrics/3/" in leveled["verification_note"]
    assert "screens: not run (the frontend builds)" in leveled["verification_note"]


def test_only_implemented_verdicts_get_a_level(tmp_path):
    missing = _verdict(5, "rule", ROUTER, status="missing")
    assert _levels(tmp_path, [missing], build="passed", calls=[CALL]) == [missing]


class _Client:
    model = "mock"
    usage = UsageTracker("mock")

    def chat(self, **kwargs):
        return {"stop_reason": "end_turn", "content": []}


def _recipe(orch):
    orch._save_recipe("Build the dashboard", 1.0)
    with open(os.path.join(orch.output_dir, ".besser_recipe.json"), encoding="utf-8") as handle:
        return json.load(handle)


def test_the_recipe_records_the_level_from_the_calls_that_passed(tmp_path):
    _app(tmp_path)
    orch = LLMOrchestrator(llm_client=_Client(), domain_model=DomainModel(name="M", types={Class(name="Plant")}),
                           output_dir=str(tmp_path))
    orch._requirement_verdicts = [*P61, _verdict(5, "action", ROUTER, line=METRICS_LINE)]

    first = _recipe(orch)["requirements"]
    assert [item["verification"] for item in first] == ["unverified"] * 5
    assert "the frontend build check did not run" in first[0]["verification_note"]

    revision = orch._workspace_revision()
    orch._api_scenarios["named:flow"] = {
        "scenario": {"requests": [{"method": "GET", "path": "/dashboard/metrics/3/"}]}, "scenario_id": "flow",
        "revision": revision, "report": {"status": "passed", "responses": [
            {"index": 0, "method": "GET", "path": "/dashboard/metrics/3/", "status": 200}]}}
    second = _recipe(orch)["requirements"]
    assert [item["verification"] for item in second] == ["unverified"] * 4 + ["verified"]


def test_a_router_prefix_in_the_cited_file_is_part_of_its_routes(tmp_path):
    """Review 4: GET /supplier/3/orders 'verified' customer.py, whose router has the
    prefix /customer; a prefix that is not a literal claims nothing."""
    customer = "backend/routers/customer.py"
    for rel, prefix in ((customer, '"/customer"'), ("backend/routers/dynamic.py", "PREFIX")):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(f"router = APIRouter(prefix={prefix})\n\n\n"
                                    '@router.get("/{customer_id}/orders")\n'
                                    "def orders(customer_id: int):\n    return []\n", encoding="utf-8")

    def level(rel, call):
        return ledger.with_verification_levels([_verdict(1, "action", rel, line="return []")], str(tmp_path),
                                               frontend_build="none", exercised={call},
                                               workflows_passed=1)[0]["verification"]

    assert level(customer, ("GET", "/supplier/3/orders")) == "unverified"
    assert level(customer, ("GET", "/customer/3/orders")) == "verified"
    assert level("backend/routers/dynamic.py", ("GET", "/customer/3/orders")) == "unverified"

    # Review 5: an annotated assignment declares the router too.
    (tmp_path / customer).write_text('router: APIRouter = APIRouter(prefix="/customer")\n\n\n'
                                     '@router.get("/{customer_id}/orders")\n'
                                     "def orders(customer_id: int):\n    return []\n", encoding="utf-8")
    assert level(customer, ("GET", "/supplier/3/orders")) == "unverified"
    assert level(customer, ("GET", "/customer/3/orders")) == "verified"
