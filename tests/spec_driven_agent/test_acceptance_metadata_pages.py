"""One metadata-driven page for every entity, and the creates that are not an entity's own.

Each list page renders one shared component with its metadata entry
(``<RecordPage record={RECORDS.order} />``) and the component POSTs to
``record.path``. The create is credited to the entity whose entry a page
passes; a message naming an entity does not make the shared component that
entity's own; a POST to another route that carries a record id is not the
create of every entity the page lists; an abstract class needs no create form.
"""
from __future__ import annotations

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.validation.acceptance import build_acceptance_matrix, matrix_issues

_STR = PrimitiveDataType("str")


def _model(*names: str, abstract: tuple[str, ...] = ()) -> DomainModel:
    classes = set()
    for name in names:
        cls = Class(name=name, is_abstract=name in abstract)
        cls.attributes = {Property(name="label", type=_STR)}
        classes.add(cls)
    return DomainModel(name="Shop", types=classes)


def _matrix(tmp_path, files: dict[str, str], model: DomainModel, manifest: str = "x") -> dict:
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return build_acceptance_matrix(str(tmp_path), model, endpoint_manifest=manifest)


def _creates(tmp_path, files: dict[str, str], *names: str) -> dict[str, bool | None]:
    matrix = _matrix(tmp_path, files, _model(*names))
    return {name: matrix[name]["create"] for name in names}


API = """\
async function request(method, path, body) {
  const res = await fetch(`${BASE}${path}`, { method, body: JSON.stringify(body) });
  return res.json();
}
export const api = {
  get: (path) => request('GET', path),
  post: (path, body) => request('POST', path, body ?? {}),
  put: (path, body) => request('PUT', path, body ?? {}),
};
"""
RECORDS = """\
export const RECORDS = {
  order: { path: '/order/', label: 'Orders', fields: [{ name: 'customer', type: 'ref', ref: 'customer' }] },
  customer: { path: '/customer/', label: 'Customers', fields: [{ name: 'name', type: 'text' }] },
  'invoice': { path: '/invoice/', label: 'Invoices', fields: [{ name: 'total', type: 'number' }] },
};
"""
# Holds a `customer` in its code, so it is not read as a generic page.
RECORD_PAGE = """\
import { useState } from 'react';
import { api } from '../api.js';
export default function RecordPage({ record, onSaved }) {
  const [values, setValues] = useState({});
  const [customer, setCustomer] = useState(null);
  const submit = async (event) => {
    event.preventDefault();
    await api.post(record.path, { ...values, customer });
  };
  return <form onSubmit={submit}><h1>{record.label}</h1><button>Save</button></form>;
}
"""


def _list_page(entry: str) -> str:
    return ("import RecordPage from '../components/RecordPage.jsx';\n"
            "import { RECORDS } from '../records.js';\n"
            "export default function ListPage() {\n"
            f"  return <RecordPage onSaved={{() => {{}}}} record={{{entry}}} />;\n}}\n")


def _record_app(order_entry: str = "RECORDS.order", page: str = RECORD_PAGE) -> dict[str, str]:
    return {
        "frontend/src/api.js": API,
        "frontend/src/records.js": RECORDS,
        "frontend/src/components/RecordPage.jsx": page,
        "frontend/src/pages/OrderList.jsx": _list_page(order_entry),
        "frontend/src/pages/CustomerList.jsx": _list_page("RECORDS.customer"),
    }


@pytest.mark.parametrize("entry", ["RECORDS.order", "RECORDS['order']", 'RECORDS?.["order"]'])
def test_a_path_read_off_a_prop_is_the_entry_each_page_passes(tmp_path, entry):
    """Invoice has an entry but no page passes it: it is still reported."""
    assert _creates(tmp_path, _record_app(entry), "Order", "Customer", "Invoice") == {
        "Order": True, "Customer": True, "Invoice": False}


def test_a_quoted_entry_key_is_traced(tmp_path):
    files = {**_record_app(), "frontend/src/pages/InvoiceList.jsx": _list_page("RECORDS.invoice")}
    assert _creates(tmp_path, files, "Order", "Customer", "Invoice") == {
        "Order": True, "Customer": True, "Invoice": True}


def test_a_page_passing_a_computed_entry_creates_every_entry(tmp_path):
    files = _record_app()
    del files["frontend/src/pages/OrderList.jsx"]
    files["frontend/src/App.jsx"] = (
        "import RecordPage from './components/RecordPage.jsx';\nimport { RECORDS } from './records.js';\n"
        "export default function App({ keys }) {\n"
        "  return <>{keys.map((key) => <RecordPage key={key} record={RECORDS[key]} />)}</>;\n}\n")
    assert _creates(tmp_path, files, "Order", "Customer", "Invoice") == {
        "Order": True, "Customer": True, "Invoice": True}


@pytest.mark.parametrize("entry", ["current", "pick(RECORDS, 'order')", "{ path }"])
def test_a_prop_that_is_not_a_metadata_entry_traces_nothing(tmp_path, entry):
    assert _creates(tmp_path, _record_app(entry), "Order", "Customer", "Invoice") == {
        "Order": False, "Customer": True, "Invoice": False}


def test_a_property_the_passed_entry_does_not_hold_traces_nothing(tmp_path):
    page = RECORD_PAGE.replace("api.post(record.path,", "api.post(record.endpoint,")
    assert _creates(tmp_path, _record_app(page=page), "Order", "Customer", "Invoice") == {
        "Order": False, "Customer": True, "Invoice": False}


def test_a_shared_page_that_never_posts_creates_nothing_through_its_entries(tmp_path):
    page = RECORD_PAGE.replace("api.post(record.path,", "api.put(record.path,")
    assert _creates(tmp_path, _record_app(page=page), "Order", "Customer", "Invoice") == {
        "Order": False, "Customer": False, "Invoice": False}


def test_a_name_that_is_not_a_prop_is_not_traced_through_render_sites(tmp_path):
    """`record` is fetched inside the component; the attribute of the same name feeds nothing."""
    page = RECORD_PAGE.replace("({ record, onSaved })", "({ onSaved })").replace(
        "  const [values,", "  const record = useRecord();\n  const [values,")
    assert _creates(tmp_path, _record_app(page=page), "Order", "Customer", "Invoice") == {
        "Order": False, "Customer": True, "Invoice": False}


@pytest.mark.parametrize("post", [
    "api.post(`${record.path}?owner=${ownerId}`, values)",
    "api.post(record.path + '?owner=' + ownerId, values)",
    "api.post(`${record.path}bulk/`, [values])",
    "api.post(`${record.path}${ownerId}/`, values)",
])
def test_a_path_that_extends_the_entry_path_is_still_that_entrys_create(tmp_path, post):
    """A query or a suffix after the entry's path does not turn it into a literal path."""
    page = RECORD_PAGE.replace("api.post(record.path, { ...values, customer })", post)
    assert _creates(tmp_path, _record_app(page=page), "Order", "Customer", "Invoice") == {
        "Order": True, "Customer": True, "Invoice": False}


@pytest.mark.parametrize("flag", ["allowCreate: false", "readOnly: true", "creatable: false"])
def test_an_entry_that_offers_no_create_form_is_not_credited(tmp_path, flag):
    records = RECORDS.replace("order: { path: '/order/',", f"order: {{ path: '/order/', {flag},")
    files = {**_record_app(), "frontend/src/records.js": records}
    assert _creates(tmp_path, files, "Order", "Customer", "Invoice") == {
        "Order": False, "Customer": True, "Invoice": False}


def test_a_computed_entry_credits_only_the_entries_that_offer_a_create(tmp_path):
    files = _record_app()
    del files["frontend/src/pages/OrderList.jsx"]
    files["frontend/src/records.js"] = RECORDS.replace("order: { path: '/order/',",
                                                       "order: { path: '/order/', allowCreate: false,")
    files["frontend/src/App.jsx"] = (
        "import RecordPage from './components/RecordPage.jsx';\nimport { RECORDS } from './records.js';\n"
        "export default function App({ keys }) {\n"
        "  return <>{keys.map((key) => <RecordPage key={key} record={RECORDS[key]} />)}</>;\n}\n")
    assert _creates(tmp_path, files, "Order", "Customer", "Invoice") == {
        "Order": False, "Customer": True, "Invoice": True}


def test_a_form_posting_to_a_path_its_parent_hands_over_is_not_credited_through_its_config(tmp_path):
    """The config is imported for its labels; the parent, not the config, decides the path."""
    form = ("import { api } from '../api.js';\nimport { RECORDS } from '../records.js';\n"
            "export default function Checkout({ endpoint, cart }) {\n"
            "  const pay = () => api.post(endpoint, cart);\n"
            "  return <button onClick={pay}>{Object.keys(RECORDS).length} sections</button>;\n}\n")
    files = {"frontend/src/api.js": API, "frontend/src/records.js": RECORDS,
             "frontend/src/components/Checkout.jsx": form}
    assert _creates(tmp_path, files, "Order", "Customer", "Invoice") == {
        "Order": False, "Customer": False, "Invoice": False}


# One form for every entry of the config; its only entity words are in a message.
CONFIG = """\
export const RECORDS = {
  order: { resource: 'order', slug: 'orders', label: 'Order',
           fields: [{ key: 'products', type: 'multiref', ref: 'product' }] },
  customer: { resource: 'customer', slug: 'customers', label: 'Customer', fields: [{ key: 'name' }] },
  product: { resource: 'product', slug: 'products', label: 'Product', fields: [{ key: 'name' }] },
};
"""
CONFIG_CLIENT = """\
async function request(method, path, body) {
  const res = await fetch(path, { method, body: JSON.stringify(body) });
  return res.json();
}
export const api = { create: (resource, payload) => request('POST', `/${resource}/`, payload) };
"""
CONFIG_FORM = """\
import { useState } from 'react';
import { api } from '../api.js';
import { RECORDS } from '../records.js';
export default function RecordForm({ config }) {
  const [values, setValues] = useState({});
  const [error, setError] = useState('');
  const onSubmit = async (e) => {
    e.preventDefault();
    if (config.needsItems && !values.items) { setError("An order needs at least one product."); return; }
    await api.create(config.resource, values);
  };
  return <form onSubmit={onSubmit}>{config.label}{error}</form>;
}
"""
CONFIG_APP = """\
import { Route, Routes } from 'react-router-dom';
import { RECORDS } from './records.js';
import RecordForm from './components/RecordForm.jsx';
export default function App() {
  return <Routes>{Object.values(RECORDS).map((cfg) =>
    <Route key={cfg.slug} path={`/${cfg.slug}/new`} element={<RecordForm config={cfg} />} />)}</Routes>;
}
"""


def _config_creates(tmp_path, form: str = CONFIG_FORM, config: str = CONFIG) -> dict:
    files = {
        "frontend/src/api.js": CONFIG_CLIENT,
        "frontend/src/records.js": config,
        "frontend/src/components/RecordForm.jsx": form,
        "frontend/src/App.jsx": CONFIG_APP,
    }
    return _creates(tmp_path, files, "Order", "Customer", "Product")


def test_a_shared_form_naming_an_entity_only_in_a_message_creates_them_all(tmp_path):
    assert _config_creates(tmp_path) == {"Order": True, "Customer": True, "Product": True}


def test_a_shared_form_with_a_message_still_credits_only_configured_entities(tmp_path):
    config = CONFIG.replace(
        "  customer: { resource: 'customer', slug: 'customers', label: 'Customer', fields: [{ key: 'name' }] },\n", "")
    assert _config_creates(tmp_path, config=config)["Customer"] is False


def test_a_shared_form_holding_an_entity_name_as_a_value_is_still_that_entitys(tmp_path):
    form = CONFIG_FORM.replace('setError("An order needs at least one product.")', "setKind('product')")
    assert _config_creates(tmp_path, form=form) == {"Order": False, "Customer": False, "Product": True}


def test_a_form_whose_message_names_its_entity_is_that_entitys_without_a_config(tmp_path):
    """No config declares the entities: the message is what ties the form to Product."""
    form = ("import { api } from '../api.js';\n"
            "export default function QuickForm({ target, onError }) {\n"
            "  const save = (body) => api.post(target, body).catch(() => onError('The product was not saved.'));\n"
            "  return <button onClick={save}>Save</button>;\n}\n")
    page = ("import QuickForm from '../components/QuickForm.jsx';\n"
            "export default function CustomerPage({ target }) {\n"
            "  return <section><h1>Customers</h1><QuickForm target={target} /></section>;\n}\n")
    files = {"frontend/src/api.js": API, "frontend/src/components/QuickForm.jsx": form,
             "frontend/src/pages/CustomerPage.jsx": page}
    assert _creates(tmp_path, files, "Customer", "Product") == {"Customer": False, "Product": True}


def _checkout(post: str) -> dict[str, str]:
    page = ("import axios from 'axios';\nimport { useEffect, useState } from 'react';\n"
            "import { api } from '../api.js';\n"
            "const API = '/api';\n"
            "export default function Checkout({ cart, cartId }) {\n"
            "  const [customers, setCustomers] = useState([]);\n"
            "  const [products, setProducts] = useState([]);\n"
            "  useEffect(() => { api.get('/customer/').then(setCustomers); }, []);\n"
            "  useEffect(() => { api.get('/product/').then(setProducts); }, []);\n"
            f"  const submit = (body) => {post};\n"
            "  return <button onClick={submit}>{customers.length} customers, {products.length} products</button>;\n}\n")
    return {"frontend/src/api.js": API, "frontend/src/pages/Checkout.jsx": page}


@pytest.mark.parametrize("post", [
    "api.post(`/checkout/${cartId}/`, body)",
    "axios.post(`${API}/checkout/${cartId}/`, body)",
    "axios.post(API + '/checkout/' + cartId + '/', body)",
    "fetch(`/checkout/${cart.id}/`, { method: 'POST', body })",
    "fetch(`/checkout/${cart?.id}/confirm/`, { method: 'POST', body })",
])
def test_a_post_to_another_route_with_a_record_id_is_not_a_listed_entitys_create(tmp_path, post):
    assert _creates(tmp_path, _checkout(post), "Customer", "Product") == {"Customer": False, "Product": False}


def test_a_post_whose_path_names_an_entity_before_a_record_id_is_that_entitys(tmp_path):
    files = _checkout("api.post(`/customer/${customerId}/orders/`, body)")
    assert _creates(tmp_path, files, "Customer", "Order", "Product") == {
        "Customer": True, "Order": True, "Product": False}


def test_a_slot_that_is_not_a_record_id_still_names_no_resource(tmp_path):
    files = _checkout("api.post(`/${kind}/${cartId}/`, body)")
    assert _creates(tmp_path, files, "Customer", "Product") == {"Customer": True, "Product": True}


_PERSON_PAGES = {
    "frontend/src/api.js": API,
    "frontend/src/pages/PersonList.jsx": (
        "import { api } from '../api.js';\n"
        "export default function PersonList() { api.get('/person/'); return <h1>People</h1>; }\n"),
    "frontend/src/pages/EmployeeList.jsx": (
        "import { api } from '../api.js';\n"
        "export default function EmployeeList() {\n"
        "  const save = (body) => api.post('/employee/', body);\n  return <button onClick={save}>New</button>;\n}\n"),
    "frontend/src/pages/ContractorList.jsx": (
        "import { api } from '../api.js';\n"
        "export default function ContractorList() { api.get('/contractor/'); return <h1>Contractors</h1>; }\n"),
}


def test_an_abstract_class_needs_no_create_path(tmp_path):
    model = _model("Person", "Employee", "Contractor", abstract=("Person",))
    matrix = _matrix(tmp_path, _PERSON_PAGES, model, manifest="GET /person/ POST /employee/ GET /contractor/")
    assert matrix["Person"] == {"route": True, "page": True, "create": None}
    assert matrix["Employee"]["create"] is True
    issues = matrix_issues(matrix)
    assert not any("entity Person" in issue for issue in issues)
    assert any(issue.startswith("acceptance: entity Contractor") and "create path" in issue for issue in issues)


def test_a_create_naming_an_abstract_class_does_not_mark_it_creatable(tmp_path):
    files = {**_PERSON_PAGES, "frontend/src/pages/PersonList.jsx": (
        "import { api } from '../api.js';\n"
        "export default function PersonList() { const save = (body) => api.post('/person/', body); }\n")}
    matrix = _matrix(tmp_path, files, _model("Person", "Employee", abstract=("Person",)))
    assert matrix["Person"]["create"] is None
