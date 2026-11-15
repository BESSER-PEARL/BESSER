"""A generic create whose path is a property of an imported metadata entry.

Live run r11-p51 (expense tracker): ``EntityPage`` calls
``api.create(config.path, payload)`` with ``const config = ENTITIES[entityKey]``
and ``ENTITIES`` imported from ``entities.js``, one entry per entity, each with
a literal ``path``. The page names Expense once in JSX text (a "related
expenses" column), so it was not taken for a generic page and the metadata was
never read: BankAccount, CashAccount, Category, Tag and User were reported with
"no frontend create path found". The same shape was reported at 1ef4ab0e.

The trace credits only the entities whose entry the value is read from: an
entity the metadata does not list is still reported.
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


# Trimmed from r11-p51.
_API = """\
async function request(method, path, body) {
  const options = { method, headers: {} };
  const res = await fetch(`${BASE_URL}${path}`, options);
  return res.json();
}

export const api = {
  list: (entity, detailed = false) => request('GET', `/${entity}/${detailed ? '?detailed=true' : ''}`),
  create: (entity, payload) => request('POST', `/${entity}/`, payload),
  update: (entity, id, payload) => request('PUT', `/${entity}/${id}/`, payload),
};
"""
_ENTITIES = """\
const userRef = (key, label) => ({ key, label, type: 'ref', ref: 'user', required: true });

export const ENTITIES = {
  user: {
    path: 'user',
    route: '/users',
    singular: 'User',
    fields: [{ key: 'name', label: 'Name', type: 'text', required: true }],
  },
  expense: { path: 'expense', route: '/expenses', singular: 'Expense', fields: [] },
  category: {
    path: 'category',
    route: '/categories',
    singular: 'Category',
    fields: [userRef('user', 'User')],
    related: { entity: 'expense', fk: 'category_id' },
  },
  bankaccount: {
    path: 'bankaccount',
    route: '/bank-accounts',
    singular: 'Bank account',
    fields: [userRef('owner', 'Owner')],
  },
  cashaccount: { path: 'cashaccount', route: '/cash-accounts', singular: 'Cash account', fields: [] },
  tag: { path: 'tag', route: '/tags', singular: 'Tag', fields: [] },
};

export const NAV_ORDER = ['expense', 'category', 'tag', 'bankaccount', 'cashaccount', 'user'];
"""
_ENTITY_PAGE = """\
import { useState } from 'react';
import { api } from '../api.js';
import { ENTITIES } from '../entities.js';

// Generic list + create/edit/delete screen driven by entities.js metadata.
export default function EntityPage({ entityKey }) {
  const config = ENTITIES[entityKey];
  const [editing, setEditing] = useState(null);
  const submit = async (e) => {
    e.preventDefault();
    const payload = {};
    if (editing === 'new') {
      await api.create(config.path, payload);
    } else {
      await api.update(config.path, editing.id, payload);
    }
  };
  return (
    <form onSubmit={submit}>
      <table><thead><tr>{config.related && <th className="num">Expenses</th>}</tr></thead></table>
    </form>
  );
}
"""
_EXPENSES_PAGE = """\
import { api } from '../api.js';
export default function ExpensesPage() {
  const save = async (payload) => { await api.create('expense', payload); };
  return <form onSubmit={save}><h1>Expenses</h1></form>;
}
"""
_APP = """\
import { Route, Routes } from 'react-router-dom';
import ExpensesPage from './pages/ExpensesPage.jsx';
import EntityPage from './pages/EntityPage.jsx';
import { ENTITIES, NAV_ORDER } from './entities.js';

export default function App() {
  return (
    <Routes>
      <Route path="/expenses" element={<ExpensesPage />} />
      {NAV_ORDER.filter((k) => k !== 'expense').map((k) => (
        <Route key={k} path={ENTITIES[k].route} element={<EntityPage entityKey={k} />} />
      ))}
    </Routes>
  );
}
"""
_FILES = {
    "frontend/src/api.js": _API,
    "frontend/src/entities.js": _ENTITIES,
    "frontend/src/pages/EntityPage.jsx": _ENTITY_PAGE,
    "frontend/src/pages/ExpensesPage.jsx": _EXPENSES_PAGE,
    "frontend/src/App.jsx": _APP,
}
_LISTED = ("BankAccount", "CashAccount", "Category", "Expense", "Tag", "User")


def test_a_create_path_read_from_imported_metadata_credits_the_listed_entities(tmp_path):
    assert _creates(tmp_path, _FILES, *_LISTED) == dict.fromkeys(_LISTED, True)


def test_an_entity_the_metadata_does_not_list_is_still_reported(tmp_path):
    """PaymentMethod has no entry in ENTITIES and no page creates it."""
    assert _creates(tmp_path, _FILES, "PaymentMethod", "Tag") == {"PaymentMethod": False, "Tag": True}


def test_a_path_indexed_straight_off_the_metadata_is_traced(tmp_path):
    files = {**_FILES, "frontend/src/pages/EntityPage.jsx": _ENTITY_PAGE.replace(
        "api.create(config.path, payload)", "api.create(ENTITIES[entityKey].path, payload)")}
    assert _creates(tmp_path, files, *_LISTED) == dict.fromkeys(_LISTED, True)


def test_a_property_the_entries_do_not_hold_traces_nothing(tmp_path):
    """`config.slug` names no literal in the entries: no metadata credit."""
    files = {**_FILES, "frontend/src/pages/EntityPage.jsx": _ENTITY_PAGE.replace(
        "api.create(config.path, payload)", "api.create(config.slug, payload)")}
    assert _creates(tmp_path, files, "BankAccount", "Expense") == {"BankAccount": False, "Expense": True}


def test_metadata_that_comes_from_props_cannot_be_traced(tmp_path):
    """The config arrives as a prop: nothing static names its path, so only the
    entities the page itself names are credited, as before."""
    page = _ENTITY_PAGE.replace("({ entityKey })", "({ config })").replace(
        "  const config = ENTITIES[entityKey];\n", "")
    files = {**_FILES, "frontend/src/pages/EntityPage.jsx": page}
    assert _creates(tmp_path, files, "BankAccount", "Expense") == {"BankAccount": False, "Expense": True}
