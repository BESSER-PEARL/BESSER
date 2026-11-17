"""A create path read off a registry entry that is handed down through components.

Each page renders ``<EntityPage entity={ENTITIES.order} />``; the shared
``EntityPage`` hands its ``entity`` prop on to ``EntityForm``, which creates
with ``api.post(entity.path, payload)``. The POST names no literal path, so the
check must follow the prop from the page through ``EntityPage`` to the form to
know which entry, and so which entity, each page creates.
"""
from __future__ import annotations

from besser.spec_driven_agent.validation.acceptance import build_acceptance_matrix


class _Class:
    def __init__(self, name: str):
        self.name = name


class _Model:
    def __init__(self, *names: str):
        self._names = names

    def get_classes(self) -> list[_Class]:
        return [_Class(n) for n in self._names]


def _creates(tmp_path, files: dict[str, str], *names: str) -> dict[str, bool]:
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    matrix = build_acceptance_matrix(str(tmp_path), _Model(*names), endpoint_manifest="x")
    return {n: matrix[n]["create"] for n in names}


_API = """\
async function request(method, path, body) {
  const res = await fetch(`${BASE}${path}`, { method, body: JSON.stringify(body) });
  return res.json();
}

export const api = {
  get: (path) => request('GET', path),
  post: (path, body) => request('POST', path, body),
  put: (path, body) => request('PUT', path, body),
};
"""
_ENTITIES = """\
export const ENTITIES = {
  order: { key: 'order', route: 'orders', path: '/order/', fields: [{ name: 'total', type: 'float' }] },
  customer: { key: 'customer', route: 'customers', path: '/customer/', fields: [{ name: 'email', type: 'text' }] },
  invoice: { key: 'invoice', route: 'invoices', path: '/invoice/', fields: [{ name: 'due', type: 'date' }] },
};
"""
_ENTITY_FORM = """\
import { useState } from 'react';
import { api } from '../api';

export default function EntityForm({ entity, record, onSaved }) {
  const [values, setValues] = useState({});
  const submit = async (ev) => {
    ev.preventDefault();
    if (record) {
      await api.put(`${entity.path}${record.id}/`, values);
    } else {
      await api.post(entity.path, values);
    }
    onSaved();
  };
  return <form onSubmit={submit}>{entity.fields.map((f) => <input key={f.name} />)}</form>;
}
"""
_ENTITY_PAGE = """\
import { useEffect, useState } from 'react';
import { api } from '../api';
import EntityForm from './EntityForm';

export default function EntityPage({ entity, footer }) {
  const [rows, setRows] = useState([]);
  useEffect(() => { api.get(entity.path).then(setRows); }, [entity.key]);
  return (
    <div>
      <EntityForm entity={entity} record={null} onSaved={() => {}} />
      <table>{rows.map((r) => <tr key={r.id}><td>{r.id}</td></tr>)}</table>
      {footer}
    </div>
  );
}
"""
_ENTITY_TABLE = """\
import { useEffect, useState } from 'react';
import { api } from '../api';

export default function EntityTable({ entity }) {
  const [rows, setRows] = useState([]);
  useEffect(() => { api.get(entity.path).then(setRows); }, [entity.key]);
  return <table>{rows.map((r) => <tr key={r.id}><td>{r.id}</td></tr>)}</table>;
}
"""


def _page(component: str, key: str, name: str, attrs: str = "") -> str:
    return (f"import {component} from '../components/{component}';\n"
            "import { ENTITIES } from '../entities';\n\n"
            f"export default function {name}() {{\n"
            f"  return <{component} entity={{ENTITIES.{key}}}{attrs} />;\n"
            "}\n")


def _files(**pages: str) -> dict[str, str]:
    return {
        "frontend/src/api.js": _API,
        "frontend/src/entities.js": _ENTITIES,
        "frontend/src/components/EntityForm.jsx": _ENTITY_FORM,
        "frontend/src/components/EntityPage.jsx": _ENTITY_PAGE,
        "frontend/src/components/EntityTable.jsx": _ENTITY_TABLE,
        **{f"frontend/src/pages/{name}.jsx": text for name, text in pages.items()},
    }


_REGISTRY_PAGES = {
    "OrderList": _page("EntityPage", "order", "OrderList"),
    "CustomerList": _page("EntityPage", "customer", "CustomerList"),
    "InvoiceList": _page("EntityPage", "invoice", "InvoiceList"),
}


def test_a_registry_entry_handed_down_to_the_form_credits_each_page_entity(tmp_path):
    assert _creates(tmp_path, _files(**_REGISTRY_PAGES), "Customer", "Invoice", "Order") == {
        "Customer": True, "Invoice": True, "Order": True}


def test_a_longer_chain_of_forwarding_components_is_followed(tmp_path):
    shell = ("import EntityPage from './EntityPage';\n\n"
             "export default function Shell({ entity }) {\n"
             "  return <main><EntityPage entity={entity} /></main>;\n"
             "}\n")
    files = {**_files(OrderList=_page("Shell", "order", "OrderList")),
             "frontend/src/components/Shell.jsx": shell}
    assert _creates(tmp_path, files, "Order") == {"Order": True}


def test_a_page_rendering_a_component_without_the_form_is_still_reported(tmp_path):
    pages = {**_REGISTRY_PAGES, "InvoiceList": _page("EntityTable", "invoice", "InvoiceList")}
    assert _creates(tmp_path, _files(**pages), "Invoice", "Order") == {"Invoice": False, "Order": True}


def test_a_read_only_registry_entry_is_still_reported(tmp_path):
    files = _files(**_REGISTRY_PAGES)
    files["frontend/src/entities.js"] = _ENTITIES.replace("invoice: { key", "invoice: { readOnly: true, key")
    assert _creates(tmp_path, files, "Invoice", "Order") == {"Invoice": False, "Order": True}


def test_a_page_that_switches_the_create_form_off_is_still_reported(tmp_path):
    pages = {**_REGISTRY_PAGES,
             "InvoiceList": _page("EntityPage", "invoice", "InvoiceList", " readOnly"),
             "CustomerList": _page("EntityPage", "customer", "CustomerList", " allowCreate={false}")}
    assert _creates(tmp_path, _files(**pages), "Customer", "Invoice", "Order") == {
        "Customer": False, "Invoice": False, "Order": True}


def test_a_forwarded_value_that_is_no_prop_ties_the_form_to_no_entry(tmp_path):
    """`EntityPage` looks its entity up itself: nothing static says which entry the form gets."""
    page = (_ENTITY_PAGE.replace("({ entity, footer })", "({ name, footer })")
            .replace("  const [rows", "  const entity = lookup(name);\n  const [rows"))
    pages = {name: f"import EntityPage from '../components/EntityPage';\n\n"
                   f"export default function {name}() {{ return <EntityPage name={{KEY}} />; }}\n"
             for name in _REGISTRY_PAGES}
    files = {**_files(**pages), "frontend/src/components/EntityPage.jsx": page}
    assert _creates(tmp_path, files, "Customer", "Invoice", "Order") == {
        "Customer": False, "Invoice": False, "Order": False}


def test_a_literal_create_is_still_credited_to_the_entity_it_names(tmp_path):
    page = ("import { api } from '../api';\n\n"
            "export default function OrderList() {\n"
            "  const save = (body) => api.post('/order/', body);\n"
            "  return <form onSubmit={save} />;\n"
            "}\n")
    pages = {"OrderList": page, "InvoiceList": _page("EntityTable", "invoice", "InvoiceList")}
    assert _creates(tmp_path, _files(**pages), "Invoice", "Order") == {"Invoice": False, "Order": True}
