"""An implemented requirement says how far it was checked, and claims no more.

Live evidence, r9-p61 (wastewater dashboard): all four requirements were
recorded as a bare "implemented" from citations alone. Three are screens and the
fourth is a map click, all cited in frontend code, and the frontend build was
skipped. The checklist items carried ``verification: unverified``; the
requirements carried nothing.

Review: one passing workflow that never touches the cited file is no evidence for
it, and a frontend that compiles has had no screen run. ``verified`` now means a
passing workflow called a route the cited backend file declares.
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
'''


def _verdict(rid, kind, *paths, status="implemented"):
    return {"id": rid, "text": f"R{rid}", "kind": kind, "status": status, "note": "",
            "evidence": [{"path": path, "line": "x"} for path in paths] if status == "implemented" else ""}


def _app(tmp_path):
    for rel, text in ((ROUTER, ROUTER_SOURCE), (SERVICE, "def compute(p):\n    return p\n"),
                      (DASHBOARD, "export default function D() { return null; }\n")):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    return str(tmp_path)


def _levels(tmp_path, verdicts, *, built=False, calls=(), passed=None):
    return ledger.with_verification_levels(
        verdicts, _app(tmp_path), frontend_built=built, exercised=set(calls),
        workflows_passed=len(calls) if passed is None else passed)


# r9-p61's four requirements, by kind and where they were cited.
P61 = [_verdict(1, "ui", DASHBOARD), _verdict(2, "ui", DASHBOARD), _verdict(3, "ui", DASHBOARD),
       _verdict(4, "action", DASHBOARD)]
CALL = ("GET", "/dashboard/metrics/3/")


def test_frontend_citations_are_never_verified_by_a_build(tmp_path):
    unbuilt = _levels(tmp_path / "a", P61, calls=[CALL])
    assert [item["verification"] for item in unbuilt] == ["unverified"] * 4
    assert all("the frontend was not built" in item["verification_note"] for item in unbuilt)
    built = _levels(tmp_path / "b", P61, built=True, calls=[CALL])
    assert [item["verification"] for item in built] == ["unverified"] * 4
    assert all("the frontend builds; its screens were not run" in item["verification_note"] for item in built)


def test_a_backend_citation_is_verified_only_by_a_call_to_a_route_it_declares(tmp_path):
    called = _levels(tmp_path / "a", [_verdict(5, "action", ROUTER)], calls=[CALL])[0]
    assert called["verification"] == "verified"
    assert "GET /dashboard/metrics/3/" in called["verification_note"]

    elsewhere = _levels(tmp_path / "b", [_verdict(5, "action", ROUTER)], calls=[("GET", "/plant/")])[0]
    assert elsewhere["verification"] == "unverified"
    assert "an API workflow passed but called no route in the cited files" in elsewhere["verification_note"]

    service = _levels(tmp_path / "c", [_verdict(6, "rule", SERVICE)], calls=[CALL])[0]
    assert service["verification"] == "unverified"  # the module declares no route

    none = _levels(tmp_path / "d", [_verdict(5, "action", ROUTER)])[0]
    assert "no API workflow passed against the app" in none["verification_note"]


def test_only_implemented_verdicts_get_a_level(tmp_path):
    missing = _verdict(5, "rule", ROUTER, status="missing")
    assert _levels(tmp_path, [missing], built=True, calls=[CALL]) == [missing]


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
    orch._requirement_verdicts = [*P61, _verdict(5, "action", ROUTER)]

    first = _recipe(orch)["requirements"]
    assert [item["verification"] for item in first] == ["unverified"] * 5

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
        return ledger.with_verification_levels([_verdict(1, "action", rel)], str(tmp_path), frontend_built=False,
                                               exercised={call}, workflows_passed=1)[0]["verification"]

    assert level(customer, ("GET", "/supplier/3/orders")) == "unverified"
    assert level(customer, ("GET", "/customer/3/orders")) == "verified"
    assert level("backend/routers/dynamic.py", ("GET", "/customer/3/orders")) == "unverified"

    # Review 5: an annotated assignment declares the router too.
    (tmp_path / customer).write_text('router: APIRouter = APIRouter(prefix="/customer")\n\n\n'
                                     '@router.get("/{customer_id}/orders")\n'
                                     "def orders(customer_id: int):\n    return []\n", encoding="utf-8")
    assert level(customer, ("GET", "/supplier/3/orders")) == "unverified"
    assert level(customer, ("GET", "/customer/3/orders")) == "verified"
