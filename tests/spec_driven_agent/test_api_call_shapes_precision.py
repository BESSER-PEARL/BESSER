"""The api call check stays silent on correct callers and on contracts it cannot read for certain.

Each backend variation here makes the request or response contract differ from what
the plain handler signature says; a finding on them would be a false blocker that
also advises the agent to send something the route refuses.
"""
import pytest

from besser.spec_driven_agent.validation.api_call_shapes import collect_api_call_issues
from tests.spec_driven_agent.test_api_call_shapes import HEADER, ROUTER, SCHEMAS


def _workspace(tmp_path, page, schemas=SCHEMAS, router=ROUTER, extra=None):
    files = {"backend/pydantic_classes.py": schemas, "backend/routers/order.py": router,
             "frontend/src/pages/Orders.tsx": HEADER + page, **(extra or {})}
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return str(tmp_path)


def _fn(body: str) -> str:
    return "export async function f(c, o, r, v, k, b, fd){ " + body + " }"


CORRECT = [
    ("optional_chain_key", "const x = await axios.get(`${API}/customer/${c}/orders/`); return x?.data?.orders ?? [];"),
    ("nullish_union", "const { data } = await axios.get(`${API}/customer/${c}/orders/`); "
                      "const list = data?.orders ?? data; return list.map(i => i);"),
    ("isarray_union", "const { data } = await axios.get(`${API}/customer/${c}/orders/`); "
                      "return Array.isArray(data) ? data : data.orders;"),
    ("destructure_from_data", "const { data } = await axios.get(`${API}/customer/${c}/orders/`); "
                              "const { orders } = data; return orders.map(i => i);"),
    ("data_alias", "const { data: payload } = await axios.get(`${API}/customer/${c}/orders/`); "
                   "return payload.orders.length;"),
    ("helper", "const x = await axios.get(`${API}/customer/${c}/orders/`); return unwrap(x.data);"),
    ("then_helper", "return axios.get(`${API}/customer/${c}/orders/`).then(x => x.data).then(d => d.orders);"),
    ("fetch_ok_union", "const res = await fetch(`${API}/customer/${c}/orders/`); if (!res.ok) throw new Error('x'); "
                       "const json = await res.json(); return Array.isArray(json) ? json : json.orders;"),
    ("string_member", "const x = await axios.get(`${API}/customer/${c}/orders/`); return x.data['orders'].map(i => i);"),
    ("ts_cast", "const x = await axios.get(`${API}/customer/${c}/orders/`); return (x.data as any).orders;"),
    ("template_read", "const x = await axios.get(`${API}/order/${o}/`); return `${x.data.order.reference}`;"),
    ("method_wrapped", "await axios.post(`${API}/order/${o}/methods/ship/`, { params: { carrier: 'x' } });"),
    ("method_wrapped_fetch", "await fetch(`${API}/order/${o}/methods/ship/`, { method: 'POST', "
                             "body: JSON.stringify({ params: { carrier: 'x' } }) });"),
    ("method_no_body", "await axios.post(`${API}/order/${o}/methods/ship/`);"),
    ("method_spread", "await axios.post(`${API}/order/${o}/methods/ship/`, { ...v });"),
    ("method_computed", "await axios.post(`${API}/order/${o}/methods/ship/`, { [k]: 1 });"),
    ("method_parameter_body", "await axios.post(`${API}/order/${o}/methods/ship/`, b);"),
    ("method_formdata", "await fetch(`${API}/order/${o}/methods/ship/`, { method: 'POST', body: fd });"),
    ("create_shorthand", "const reference = r, total = 1, customer = c; "
                         "await axios.post(`${API}/order/`, { reference, total, customer });"),
    ("create_comment_inside", "await axios.post(`${API}/order/`, { reference: r, // note\n total: 1, customer: c });"),
    ("create_with_query", "await axios.post(`${API}/order/?notify=1`, { reference: r, total: 1, customer: c });"),
    ("absolute_localhost", "const x = await axios.get(`http://localhost:8000/customer/${c}/orders/`); return x.data.orders;"),
]


@pytest.mark.parametrize("body", [body for _, body in CORRECT], ids=[name for name, _ in CORRECT])
def test_a_correct_caller_is_not_reported(tmp_path, body):
    assert collect_api_call_issues(_workspace(tmp_path, _fn(body))) == []


ANNOTATED_SCHEMAS = '''from typing import Annotated
from pydantic import BaseModel, Field


class OrderCreate(BaseModel):
    reference: Annotated[str, Field(alias="ref")]
    total: Annotated[float, Field(default=0)]
    customer: str
'''


def test_annotated_field_alias_and_default_are_read(tmp_path):
    page = _fn("await axios.post(`${API}/order/`, { ref: 'r', customer: 'c' });")
    assert collect_api_call_issues(_workspace(tmp_path, page, schemas=ANNOTATED_SCHEMAS)) == []


@pytest.mark.parametrize("validator", [
    "@root_validator(pre=True)\n    def fill(cls, values):",
    "@model_validator(mode=\"before\")\n    @classmethod\n    def fill(cls, values):",
    "@model_validator(mode=\"wrap\")\n    @classmethod\n    def fill(cls, values, handler):",
])
def test_a_model_validator_before_validation_can_fill_a_required_field(tmp_path, validator):
    schemas = SCHEMAS + f"\n    {validator}\n        return values\n"
    page = _fn("await axios.post(`${API}/order/`, { reference: 'r', total: 1 });")
    assert collect_api_call_issues(_workspace(tmp_path, page, schemas=schemas)) == []


@pytest.mark.parametrize("validator", [
    "@validator(\"customer\", pre=True)\n    def fill(cls, value):",
    "@field_validator(\"customer\", mode=\"before\")\n    @classmethod\n    def fill(cls, value):",
])
def test_a_field_validator_does_not_fill_a_missing_field(tmp_path, validator):
    from besser.spec_driven_agent.validation.frontend_schema import _model_resolver

    schemas = SCHEMAS + f"\n    {validator}\n        return value\n"
    page = _fn("await axios.post(`${API}/order/`, { reference: 'r', total: 1 });")
    [issue] = collect_api_call_issues(_workspace(tmp_path, page, schemas=schemas))
    assert "leaves out the required customer" in issue
    # The model stays known to the form checks as well.
    _, resolve = _model_resolver([("pydantic_classes.py", schemas)])
    assert resolve("OrderCreate").required == {"reference", "total", "customer"}


def test_a_handler_reading_the_raw_request_has_no_fixed_body(tmp_path):
    router = ROUTER.replace(
        "    order_id: str,\n    params: dict = Body(default=None, embed=True),",
        "    order_id: str,\n    request: Request,\n    params: dict = Body(default=None, embed=True),")
    router = router.replace('return {"order": order_id, "result": carrier}',
                            'return {"order": order_id, "result": carrier or await request.json()}')
    assert "request: Request" in router and "await request.json()" in router
    page = _fn("await axios.post(`${API}/order/${o}/methods/ship/`, { carrier: 'x' });")
    assert collect_api_call_issues(_workspace(tmp_path, page, router=router)) == []


def test_a_returned_dict_handed_to_a_helper_has_unknown_keys(tmp_path):
    router = ROUTER.replace('        "item_ids": [],\n    }\n    return response_data',
                            '    }\n    add_links(response_data, db_order)\n    return response_data')
    assert "add_links(response_data" in router
    page = _fn("const x = await axios.get(`${API}/order/${o}/`); return x.data.item_ids.map(i => i);")
    assert collect_api_call_issues(_workspace(tmp_path, page, router=router)) == []


def test_a_fetch_method_written_after_its_body_is_read(tmp_path):
    router = ROUTER + '''

@router.put("/order/{order_id}/", response_model=None)
async def update_order(order_id: str, order_data: OrderCreate, database: Session = Depends(get_db)):
    return [order_data]
'''
    page = _fn("const res = await fetch(`${API}/order/${o}/`, "
               "{ body: JSON.stringify({ reference: 'r', total: 1, customer: 'c' }), method: 'PUT' }); "
               "const rows = await res.json(); return rows.map(x => x);")
    assert collect_api_call_issues(_workspace(tmp_path, page, router=router)) == []


def test_a_method_named_inside_the_body_is_not_the_request_method(tmp_path):
    from besser.spec_driven_agent.validation.endpoint_coherence import _calls_in_file

    [call] = _calls_in_file("a.ts", "fetch('/order/', { body: JSON.stringify({ method: 'GET' }), method: 'POST' })")
    assert call.method == "POST"
    [call] = _calls_in_file("a.ts", "fetch('/order/', { body: JSON.stringify({ method: 'DELETE' }) })")
    assert call.method == "GET"


def test_a_key_beside_params_that_the_handler_takes_elsewhere_is_not_reported(tmp_path):
    page = _fn("await axios.post(`${API}/order/${o}/methods/ship/`, { params: { carrier: 'x' }, order_id: o });")
    assert collect_api_call_issues(_workspace(tmp_path, page)) == []


def test_a_key_the_handler_reads_from_params_sent_beside_it_is_reported(tmp_path):
    page = _fn("await axios.post(`${API}/order/${o}/methods/ship/`, { params: {}, carrier: 'x' });")
    [issue] = collect_api_call_issues(_workspace(tmp_path, page))
    assert "the keys it sends beside them (carrier) never reach the handler" in issue


def test_a_deleted_key_makes_a_body_variable_unknown(tmp_path):
    page = _fn("const body = { params: { carrier: 'x' }, carrier: 'x' }; delete body.carrier; "
               "await axios.post(`${API}/order/${o}/methods/ship/`, body);")
    assert collect_api_call_issues(_workspace(tmp_path, page)) == []


INSTANCE_PAGE = ('import api from "../client";\n'
                 + _fn("const x = await api.get(`/customer/${c}/orders/`); "
                       "await api.post(`/order/${o}/methods/ship/`, { carrier: 'x' }); return x.data.map(i => i);"))


@pytest.mark.parametrize("create", [
    "{ baseURL: import.meta.env.VITE_API_URL, transformRequest: [(d) => JSON.stringify({ params: d })] }",
    "{ baseURL: import.meta.env.VITE_API_URL, transformResponse: [(d) => JSON.parse(d).orders] }",
    "{ baseURL: BASE }",
], ids=["transform_request", "transform_response", "base_constant_with_a_path"])
def test_an_instance_that_reshapes_or_moves_its_calls_is_not_followed(tmp_path, create):
    client = ("import axios from 'axios';\nconst BASE = 'http://localhost:8000/v2';\n"
              f"const api = axios.create({create});\nexport default api;\n")
    assert collect_api_call_issues(_workspace(tmp_path, INSTANCE_PAGE, extra={"frontend/src/client.js": client})) == []


@pytest.mark.parametrize("setup", [
    "api.interceptors.request.use((cfg) => { cfg.data = { params: cfg.data }; return cfg; });\n",
    "axios.defaults.transformRequest = [(d) => JSON.stringify({ params: d })];\n",
], ids=["request_interceptor", "default_transform_request"])
def test_a_global_request_rewrite_leaves_bodies_alone(tmp_path, setup):
    client = ("import axios from 'axios';\nconst api = axios.create({ baseURL: import.meta.env.VITE_API_URL });\n"
              + setup + "export default api;\n")
    [issue] = collect_api_call_issues(_workspace(tmp_path, INSTANCE_PAGE, extra={"frontend/src/client.js": client}))
    assert "returns an object" in issue  # the response is still read as sent
