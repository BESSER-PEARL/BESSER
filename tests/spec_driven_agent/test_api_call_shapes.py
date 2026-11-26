"""LLM-written frontend calls are checked against what their route reads and returns.

A page that guesses the generated API's shapes type-checks and builds, and the
backend's own tests pass with the right shapes, so nothing else notices that:
- a relation route answers ``{<parent>_id, <end>_count, <end>: [...]}`` and the
  page treats that object as the array (always empty, or a TypeError);
- a ``/methods/<name>/`` route reads ``params: dict = Body(embed=True)``, i.e.
  ``{"params": {...}}``, and the page posts the values flat (every value lost);
- a create body leaves out a required field (every call a 422).
"""
import json

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.validation.api_call_shapes import (
    collect_api_call_diagnostics,
    collect_api_call_issues,
)
from besser.spec_driven_agent.validation.issues import _classify_issue
from besser.spec_driven_agent.validation.user_wording import user_sentence

SCHEMAS = '''from typing import Optional
from pydantic import BaseModel


class OrderCreate(BaseModel):
    reference: str
    total: float
    note: Optional[str] = None
    customer: str
'''

ROUTER = '''from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session
from pydantic_classes import OrderCreate

router = APIRouter()


@router.get("/order/", response_model=None, tags=["Order"])
def get_all_order(database: Session = Depends(get_db)) -> list:
    return database.query(Order).all()


@router.get("/order/{order_id}/", response_model=None, tags=["Order"])
async def get_order(order_id: str, database: Session = Depends(get_db)):
    db_order = database.query(Order).filter(Order.id == order_id).first()
    response_data = {
        "order": db_order,
        "item_ids": [],
    }
    return response_data


@router.post("/order/", response_model=None, tags=["Order"])
async def create_order(order_data: OrderCreate, database: Session = Depends(get_db)):
    return Order(**order_data.model_dump())


@router.get("/customer/{customer_id}/orders/", response_model=None, tags=["Customer Relationships"])
async def get_orders_of_customer(customer_id: str, database: Session = Depends(get_db)):
    orders_list = database.query(Order).filter(Order.customer_id == customer_id).all()
    return {
        "customer_id": customer_id,
        "orders_count": len(orders_list),
        "orders": orders_list
    }


@router.post("/order/{order_id}/methods/ship/", response_model=None, tags=["Order Methods"])
async def execute_order_ship(
    order_id: str,
    params: dict = Body(default=None, embed=True),
    database: Session = Depends(get_db)
):
    params = params or {}
    carrier = params.get('carrier')
    return {"order": order_id, "result": carrier}
'''

HEADER = 'import axios from "axios";\nconst API = import.meta.env.VITE_API_URL;\n'


def _workspace(tmp_path, page: str, name: str = "Orders.tsx"):
    files = {
        "backend/pydantic_classes.py": SCHEMAS,
        "backend/routers/order.py": ROUTER,
        f"frontend/src/pages/{name}": HEADER + page,
    }
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return str(tmp_path)


def test_relation_object_counted_as_an_array_is_a_blocker(tmp_path):
    root = _workspace(tmp_path, (
        "export function load(customerId, setCount) {\n"
        "  axios\n"
        "    .get(`${API}/customer/${customerId}/orders/`)\n"
        "    .then((response) => {\n"
        "      setCount(Array.isArray(response.data) ? response.data.length : 0);\n"
        "    });\n"
        "}\n"))

    [issue] = collect_api_call_issues(root)

    assert issue.startswith("api call: frontend/src/pages/Orders.tsx line 4: GET /customer/{customer_id}/orders/")
    assert "{customer_id, orders, orders_count}" in issue
    assert "response.data as an array (Array.isArray, .length)" in issue
    assert "backend/routers/order.py line" in issue
    assert _classify_issue(issue).severity == "blocker"
    assert "shape the backend does not use" in user_sentence(issue)


def test_reading_the_list_from_its_key_is_not_reported(tmp_path):
    root = _workspace(tmp_path, (
        "export async function load(customerId) {\n"
        "  const response = await axios.get(`${API}/customer/${customerId}/orders/`);\n"
        "  return Array.isArray(response.data) ? response.data : response.data.orders;\n"
        "}\n"))

    assert collect_api_call_issues(root) == []


def test_fetched_relation_object_filtered_as_an_array_is_reported(tmp_path):
    root = _workspace(tmp_path, (
        "export async function recent(customerId) {\n"
        "  const response = await fetch(`${API}/customer/${customerId}/orders/`);\n"
        "  const orders = await response.json();\n"
        "  return orders.filter((order) => order.total > 10);\n"
        "}\n"))

    [issue] = collect_api_call_issues(root)
    assert "uses orders as an array (.filter())" in issue


def test_flat_list_route_used_as_an_array_is_not_reported(tmp_path):
    root = _workspace(tmp_path, (
        "export async function all() {\n"
        "  const response = await axios.get(`${API}/order/`);\n"
        "  return response.data.map((order) => order.reference);\n"
        "}\n"))

    assert collect_api_call_issues(root) == []


def test_reading_a_key_the_wrapped_record_does_not_have_is_reported(tmp_path):
    root = _workspace(tmp_path, (
        "export async function total(orderId) {\n"
        "  const response = await axios.get(`${API}/order/${orderId}/`);\n"
        "  return response.data.total;\n"
        "}\n"))

    [issue] = collect_api_call_issues(root)
    assert "{item_ids, order}" in issue and "reads response.data.total" in issue


def test_fetch_error_body_and_template_reads_are_not_misread(tmp_path):
    # ``data.detail`` is the error body of a non-2xx fetch; ``data.order`` is read inside ``${}``.
    root = _workspace(tmp_path, (
        "export async function show(orderId, say) {\n"
        "  const response = await fetch(`${API}/order/${orderId}/`);\n"
        "  const data = await response.json();\n"
        "  say(response.ok ? `Order ${data.order.reference}` : data.detail);\n"
        "}\n"))

    assert collect_api_call_issues(root) == []


def test_method_body_sent_without_its_params_wrapper_is_a_blocker(tmp_path):
    root = _workspace(tmp_path, (
        "export async function ship(orderId, carrier) {\n"
        "  await axios.post(\n"
        "    `${API}/order/${orderId}/methods/ship/`,\n"
        "    { carrier: carrier },\n"
        "  );\n"
        "}\n"))

    [issue] = collect_api_call_issues(root)
    assert "POST /order/{order_id}/methods/ship/ reads its JSON body as one top-level key per Body() parameter, "         "{params} (handler: backend/routers/order.py line" in issue
    assert "(carrier)" in issue
    assert 'Send {"params": {...}} with the values inside.' in issue


def test_wrapped_and_empty_method_bodies_pass(tmp_path):
    root = _workspace(tmp_path, (
        "export async function ship(orderId, carrier) {\n"
        "  await axios.post(`${API}/order/${orderId}/methods/ship/`, { params: { carrier } });\n"
        "  await axios.post(`${API}/order/${orderId}/methods/ship/`, {});\n"
        "}\n"))

    assert collect_api_call_issues(root) == []


def test_create_body_missing_a_required_field_is_a_blocker(tmp_path):
    root = _workspace(tmp_path, (
        "export async function add(form) {\n"
        "  await fetch(`${API}/order/`, {\n"
        "    method: 'POST',\n"
        "    headers: { 'Content-Type': 'application/json' },\n"
        "    body: JSON.stringify({ reference: form.reference, total: Number(form.total), customerId: form.customer }),\n"
        "  });\n"
        "}\n"))

    [issue] = collect_api_call_issues(root)
    assert "POST /order/ reads its JSON body as OrderCreate" in issue
    assert "leaves out the required customer" in issue and "422" in issue


def test_create_body_with_a_spread_or_every_required_field_passes(tmp_path):
    root = _workspace(tmp_path, (
        "export async function add(form) {\n"
        "  await axios.post(`${API}/order/`, { ...form });\n"
        "  await axios.post(`${API}/order/`, { reference: form.reference, total: 1, customer: form.c });\n"
        "}\n"))

    assert collect_api_call_issues(root) == []


def test_axios_interceptor_leaves_responses_alone(tmp_path):
    root = _workspace(tmp_path, (
        "axios.interceptors.response.use((response) => response.data);\n"
        "export async function load(customerId) {\n"
        "  const response = await axios.get(`${API}/customer/${customerId}/orders/`);\n"
        "  return response.data.map((order) => order.reference);\n"
        "}\n"))

    assert collect_api_call_issues(root) == []


def test_the_write_that_makes_the_mistake_is_told_at_once(tmp_path):
    root = _workspace(tmp_path, "export const ready = true;\n")
    page = HEADER + (
        "export async function ship(orderId) {\n"
        "  await axios.post(`${API}/order/${orderId}/methods/ship/`, { carrier: 'post' });\n"
        "}\n")

    result = json.loads(ToolExecutor(workspace=root).execute(
        "write_file", {"path": "frontend/src/pages/Shipping.tsx", "content": page}))

    [finding] = [item for item in result.get("diagnostics", []) if item["code"] == "api-call-shape"]
    assert finding["path"] == "frontend/src/pages/Shipping.tsx" and finding["line"] == 4
    assert 'Send {"params": {...}}' in finding["message"]
    assert collect_api_call_diagnostics(root) == [
        {key: finding[key] for key in ("source", "severity", "code", "path", "line", "message")}]


def test_validate_app_and_phase3_carry_the_finding(tmp_path):
    from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
    from tests.spec_driven_agent.shelf_app import shelf_model

    root = _workspace(tmp_path, (
        "export async function ship(orderId) {\n"
        "  await axios.post(`${API}/order/${orderId}/methods/ship/`, { carrier: 'post' });\n"
        "}\n"))
    orchestrator = LLMOrchestrator(
        llm_client=object(), domain_model=shelf_model(), output_dir=root, enable_tracing=False,
        enable_checkpointing=False, enable_import_smoke_check=False, enable_requirements_ledger=False,
        use_streaming=False)

    assert any(issue.startswith("api call: frontend/src/pages/Orders.tsx line 4: POST")
               for issue in orchestrator._collect_execution_issues())


def _shape(source: str):
    import ast

    from besser.spec_driven_agent.validation.route_returns import response_shape, route_handlers

    [(_, _, function, decorator)] = route_handlers(ast.parse(
        "from fastapi import APIRouter\nrouter = APIRouter()\n" + source))
    return response_shape(function, decorator)


def test_response_shape_is_only_claimed_when_every_return_agrees():
    assert _shape('@router.get("/x/")\ndef f():\n    out = {"a": 1}\n    out["b"] = 2\n    return out\n') == (
        "object", frozenset({"a", "b"}))
    assert _shape('@router.get("/x/")\ndef f(q):\n    return q.all() if False else []\n') is None
    assert _shape('@router.get("/x/")\ndef f(q):\n    if q:\n        return []\n    return q.all()\n') == ("list", None)
    # Mixed kinds, a declared response model, a spread, a loop-bound name, a mutated dict.
    assert _shape('@router.get("/x/")\ndef f(q):\n    if q:\n        return {"a": 1}\n    return []\n') is None
    assert _shape('@router.get("/x/", response_model=Out)\ndef f():\n    return {"a": 1}\n') is None
    assert _shape('@router.get("/x/")\ndef f(extra):\n    return {"a": 1, **extra}\n') is None
    assert _shape('@router.get("/x/")\ndef f(rows):\n    for out in rows:\n        pass\n    out = {"a": 1}\n'
                  '    return out\n') is None
    assert _shape('@router.get("/x/")\ndef f(more):\n    out = {"a": 1}\n    out.update(more)\n    return out\n') is None
