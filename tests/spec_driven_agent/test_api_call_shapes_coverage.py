"""The api call check also reads the call shapes generated frontends commonly use.

Most LLM-written pages call through an ``axios.create()`` instance, often from a
shared module, use the axios config form or a TypeScript generic, chain fetch's
``.then(r => r.json())``, keep the body in a variable, or put the data in array
state. Each case below is the same wrong caller as in ``test_api_call_shapes``.
"""
import pytest

from besser.spec_driven_agent.validation.api_call_shapes import collect_api_call_issues
from tests.spec_driven_agent.test_api_call_shapes import HEADER, ROUTER, SCHEMAS

ORDERS = "${API}/customer/${c}/orders/"
SHIP = "${API}/order/${o}/methods/ship/"


def _workspace(tmp_path, page, extra=None):
    files = {"backend/pydantic_classes.py": SCHEMAS, "backend/routers/order.py": ROUTER,
             "frontend/src/pages/Orders.tsx": HEADER + page, **(extra or {})}
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return str(tmp_path)


def _fn(body: str) -> str:
    return "export async function f(c, o){ " + body + " }\n"


WRONG = [
    ("isarray_else_empty", f"const r = await axios.get(`{ORDERS}`); return Array.isArray(r.data) ? r.data : [];"),
    ("fetch_then_chain", f"return fetch(`{ORDERS}`).then(r => r.json()).then(items => items.map(i => i));"),
    ("generic_axios", f"const r = await axios.get<any[]>(`{ORDERS}`); return r.data.map(i => i);"),
    ("axios_config_form", f"await axios({{ method: 'post', url: `{SHIP}`, data: {{ carrier: 'x' }} }});"),
    ("body_in_a_variable", f"const body = {{ carrier: 'x' }}; await axios.post(`{SHIP}`, body);"),
    ("fetch_body_in_a_variable", f"const payload = {{ carrier: 'x' }}; "
                                 f"await fetch(`{SHIP}`, {{ body: JSON.stringify(payload), method: 'POST' }});"),
    ("flat_beside_params", f"await axios.post(`{SHIP}`, {{ carrier: 'x', params: {{}} }});"),
    ("same_file_instance", "const api = axios.create({ baseURL: API });\n"
                           + _fn("await api.post(`/order/${o}/methods/ship/`, { carrier: 'x' });")),
]


@pytest.mark.parametrize("page", [page if page.startswith("const api") else _fn(page) for _, page in WRONG],
                         ids=[name for name, _ in WRONG])
def test_the_wrong_caller_is_reported(tmp_path, page):
    [issue] = collect_api_call_issues(_workspace(tmp_path, page))
    assert issue.startswith("api call: frontend/src/pages/Orders.tsx line ")


def test_an_instance_imported_from_a_shared_module_is_followed(tmp_path):
    client = 'import axios from "axios";\nexport const api = axios.create({ baseURL: import.meta.env.VITE_API_URL });\n'
    page = ('import { api } from "../client";\n'
            + _fn("const r = await api.get(`/customer/${c}/orders/`); return r.data.map(i => i);"))
    [issue] = collect_api_call_issues(_workspace(tmp_path, page, extra={"frontend/src/client.ts": client}))
    assert "uses r.data as an array (.map())" in issue


def test_a_default_exported_instance_is_followed(tmp_path):
    client = 'import axios from "axios";\nconst http = axios.create({ baseURL: "http://localhost:8000" });\nexport default http;\n'
    page = 'import client from "../client";\n' + _fn("await client.post(`/order/${o}/methods/ship/`, { carrier: 'x' });")
    [issue] = collect_api_call_issues(_workspace(tmp_path, page, extra={"frontend/src/client.ts": client}))
    assert 'Send {"params": {...}}' in issue


def test_array_state_set_from_an_object_response_is_reported(tmp_path):
    page = ("export function Orders({ c }) {\n"
            "  const [orders, setOrders] = useState<any[]>([]);\n"
            "  useEffect(() => { axios.get(`${API}/customer/${c}/orders/`).then((r) => setOrders(r.data)); }, [c]);\n"
            "  return <ul>{orders.map((o) => <li key={o.id}>{o.reference}</li>)}</ul>;\n"
            "}\n")
    [issue] = collect_api_call_issues(_workspace(tmp_path, page))
    assert "setOrders(r.data)" in issue and "orders.map()" in issue


CORRECT = [
    ("instance_with_a_path_in_its_base", "const api = axios.create({ baseURL: `${API}/v2` });\n"
                                         + _fn("await api.post(`/order/${o}/methods/ship/`, { carrier: 'x' });")),
    ("body_variable_changed_later", _fn(f"const body = {{ carrier: 'x' }}; body.params = {{ carrier: 'x' }}; "
                                        f"await axios.post(`{SHIP}`, body);")),
    ("body_variable_handed_to_a_helper", _fn(f"const body = {{ carrier: 'x' }}; wrap(body); "
                                             f"await axios.post(`{SHIP}`, body);")),
    ("config_method_inside_data", _fn(f"await axios({{ url: `{SHIP}`, method: 'post', "
                                      f"data: {{ params: {{ method: 'get' }} }} }});")),
    ("state_read_by_its_key", "export function Orders({ c }) {\n"
                              "  const [orders, setOrders] = useState([]);\n"
                              "  useEffect(() => { axios.get(`${API}/customer/${c}/orders/`).then((r) => setOrders(r.data)); }, [c]);\n"
                              "  return <ul>{orders.orders.map((o) => <li key={o.id}>{o.reference}</li>)}</ul>;\n"
                              "}\n"),
    ("state_not_an_array", "export function Orders({ c }) {\n"
                           "  const [orders, setOrders] = useState(null);\n"
                           "  useEffect(() => { axios.get(`${API}/customer/${c}/orders/`).then((r) => setOrders(r.data)); }, [c]);\n"
                           "  return <ul>{orders && orders.map((o) => <li key={o.id}>{o.reference}</li>)}</ul>;\n"
                           "}\n"),
]


@pytest.mark.parametrize("page", [page for _, page in CORRECT], ids=[name for name, _ in CORRECT])
def test_a_caller_that_may_be_right_is_left_alone(tmp_path, page):
    assert collect_api_call_issues(_workspace(tmp_path, page)) == []


WRAPPER = ("const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';\n"
           "export async function api(path, options = {}) {\n"
           "  const r = await fetch(API + path, { headers: { 'Content-Type': 'application/json' }, ...options });\n"
           "  if (!r.ok) throw new Error(await r.text());\n"
           "  return r.status === 204 ? null : r.json();\n"
           "}\n")
WRAPPER_CALLS = ('import { api } from "../client";\n'
                 + _fn("await api(`/order/${o}/methods/ship/`, { method: 'POST', body: JSON.stringify({ carrier: 'x' }) });\n"
                       "  const rows = await api(`/customer/${c}/orders/`); return rows.map(i => i);"))


def test_calls_through_a_fetch_wrapper_are_read_as_fetch_calls(tmp_path):
    issues = collect_api_call_issues(_workspace(tmp_path, WRAPPER_CALLS, extra={"frontend/src/client.js": WRAPPER}))

    assert len(issues) == 2
    assert 'Send {"params": {...}}' in issues[0]
    assert "uses rows as an array (.map())" in issues[1]


@pytest.mark.parametrize("wrapper", [
    WRAPPER.replace("...options }", "...options, method: 'POST' }"),
    WRAPPER.replace("...options }", "...options, body: JSON.stringify(options.body) }"),
    WRAPPER.replace("import.meta.env.VITE_API_URL || 'http://localhost:8000'", "'http://localhost:8000/api'"),
    WRAPPER.replace("return r.status === 204 ? null : r.json();", "return r;"),
], ids=["sets_its_own_method", "stringifies_the_body_itself", "base_adds_a_path", "returns_the_response"])
def test_a_wrapper_that_changes_the_request_or_answer_is_not_followed(tmp_path, wrapper):
    assert collect_api_call_issues(_workspace(tmp_path, WRAPPER_CALLS, extra={"frontend/src/client.js": wrapper})) == []
