"""The acceptance matrix must see a create made through a generic entity page.

A live hotel run rendered every class through one ``EntityList`` page: App.jsx
held a config per entity and rendered ``<EntityList entity={key} />``, and the
page called ``api.create(entity, payload)``. The file naming the entity made no
create and the file making the create named no entity, so every class was
reported "no frontend create path" although each one was creatable.
"""

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.validation.acceptance import build_acceptance_matrix

_STR = PrimitiveDataType("str")

API = """\
export async function request(path, options = {}) {
  const response = await fetch(`http://localhost:8000${path}`, options);
  return response.json();
}
export const api = {
  list: (entity) => request(`/${entity}/`),
  create: (entity, payload) => request(`/${entity}/`, { method: 'POST', body: JSON.stringify(payload) }),
};
"""

ENTITY_LIST = """\
import { useState } from 'react';
import { api } from '../api';
export default function EntityList({ entity, config }) {
  const [form, setForm] = useState({});
  const submit = async (event) => { event.preventDefault(); await api.create(entity, form); };
  return <form onSubmit={submit}><h1>{config.title}</h1><button>Create</button></form>;
}
"""

APP = """\
import { Route, Routes } from 'react-router-dom';
import EntityList from './pages/EntityList';
const configs = {
  person: { title: 'People', fields: ['firstName'] },
  employee: { title: 'Employees', fields: ['firstName'] },
};
export default function App() {
  return <Routes>{Object.entries(configs).map(([key, config]) =>
    <Route key={key} path={`/${key}`} element={<EntityList entity={key} config={config} />} />)}</Routes>;
}
"""


def _model(*names):
    classes = set()
    for name in names:
        cls = Class(name=name)
        cls.attributes = {Property(name="firstName", type=_STR)}
        classes.add(cls)
    return DomainModel(name="Hotel", types=classes)


def _workspace(tmp_path, files):
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return str(tmp_path)


def _app(tmp_path, app=APP, entity_list=ENTITY_LIST):
    return _workspace(tmp_path, {
        "frontend/src/api.js": API,
        "frontend/src/pages/EntityList.jsx": entity_list,
        "frontend/src/App.jsx": app,
    })


def test_entities_created_through_a_generic_page_have_a_create_path(tmp_path):
    matrix = build_acceptance_matrix(_app(tmp_path), _model("Person", "Employee"))
    assert matrix["Person"]["create"] is True
    assert matrix["Employee"]["create"] is True


def test_an_entity_the_config_does_not_list_still_has_no_create_path(tmp_path):
    matrix = build_acceptance_matrix(_app(tmp_path), _model("Person", "Invoice"))
    assert matrix["Invoice"]["create"] is False


def test_a_generic_page_that_never_creates_proves_nothing(tmp_path):
    read_only = ENTITY_LIST.replace("await api.create(entity, form);", "await api.list(entity);")
    matrix = build_acceptance_matrix(_app(tmp_path, entity_list=read_only), _model("Person"))
    assert matrix["Person"]["create"] is False


# Live (2026-11-13, claude-sonnet-5-5, risk-awareness app): the shared EntityPage
# read `e.message` and `err.message`, the model had a Message class, so the form
# was taken for Message's own page and four entities with a "+ New" route each
# were reported as having no create path.
CLIENT = """\
export async function create(path, body) {
  const res = await fetch(path, { method: 'POST', body: JSON.stringify(body) });
  return res.json();
}
"""
ENTITIES = """\
export const ENTITIES = {
  message: { key: 'message', path: '/message/', fields: [{ name: 'body' }] },
  report: { key: 'report', path: '/report/', fields: [{ name: 'reason' }] },
};
"""
ENTITY_PAGE = """\
import { useState } from 'react';
import { create } from '../api.js';
export default function EntityPage({ entity }) {
  const [error, setError] = useState('');
  const [values, setValues] = useState({});
  const submit = async (e) => {
    e.preventDefault();
    try { await create(entity.path, values); } catch (err) { setError(err.message); }
  };
  return <form onSubmit={submit}>{error && <p>{error?.message ?? error['message']}</p>}<button>Save</button></form>;
}
"""


def _list_page(key):
    return ("import EntityPage from '../components/EntityPage.jsx';\n"
            "import { ENTITIES } from '../entities.js';\n"
            f"export default function Page() {{ return <EntityPage entity={{ENTITIES.{key}}} />; }}\n")


def test_a_shared_form_reading_err_message_is_still_generic(tmp_path):
    root = _workspace(tmp_path, {
        "frontend/src/api.js": CLIENT,
        "frontend/src/entities.js": ENTITIES,
        "frontend/src/components/EntityPage.jsx": ENTITY_PAGE,
        "frontend/src/pages/MessageList.jsx": _list_page("message"),
        "frontend/src/pages/ReportList.jsx": _list_page("report"),
    })
    matrix = build_acceptance_matrix(root, _model("Message", "Report"))
    assert matrix["Message"]["create"] is True
    assert matrix["Report"]["create"] is True


def test_a_form_that_names_an_entity_in_its_code_is_still_that_entitys(tmp_path):
    composer = ENTITY_PAGE.replace("await create(entity.path, values);",
                                   "await create(entity.path, { body: message });")
    root = _workspace(tmp_path, {
        "frontend/src/api.js": CLIENT,
        "frontend/src/entities.js": ENTITIES,
        "frontend/src/components/EntityPage.jsx": composer,
        "frontend/src/pages/MessageList.jsx": _list_page("message"),
        "frontend/src/pages/ReportList.jsx": _list_page("report"),
    })
    matrix = build_acceptance_matrix(root, _model("Message", "Report"))
    assert matrix["Message"]["create"] is True
    assert matrix["Report"]["create"] is False


_HTTP = "import axios from 'axios';\nexport const http = axios.create({ baseURL: '/api' });\n"


def _reserve_form(imports, target):
    return (f"{imports}export default function ReserveForm(props) {{\n"
            f"  return <form onSubmit={{async (e) => {{ e.preventDefault(); await http.post({target}, {{}}); }}}}>"
            "<button>Reserve</button></form>;\n}\n")


@pytest.mark.parametrize("files", [
    {"api.js": _HTTP + "export const endpoints = { booking: '/booking/', room: '/room/' };\n",
     "ReserveForm.jsx": _reserve_form("import { http, endpoints } from './api';\n", "endpoints.booking")},
    {"api.js": _HTTP,
     "config.js": "export const ENTITIES = { booking: { path: '/booking/' }, room: { path: '/room/' } };\n",
     "ReserveForm.jsx": _reserve_form("import { http } from './api';\nimport { ENTITIES } from './config';\n",
                                      "ENTITIES['booking'].path")},
    {"api.js": _HTTP, "ReserveForm.jsx": _reserve_form("import { http } from './api';\n", "props.urls.bookings")},
    {"api.js": _HTTP,
     "ReserveForm.jsx": _reserve_form("import { http } from './api';\n", "props.api?.bookingPath?.()")},
], ids=["endpoints-member", "config-bracket", "props-member", "optional-call"])
def test_a_form_naming_its_entity_through_a_property_is_not_generic(tmp_path, files):
    """Review probe: a booking-only form rendered on RoomPage posted to `endpoints.booking`;
    blanking every property read took it for a generic form and hid Room's missing create."""
    room_page = ("import ReserveForm from './ReserveForm';\n"
                 "export default function RoomPage() { return <div><h1>Rooms</h1><ReserveForm /></div>; }\n")
    files = {**files, "RoomPage.jsx": room_page}
    root = _workspace(tmp_path, {f"frontend/src/{rel}": text for rel, text in files.items()})
    matrix = build_acceptance_matrix(root, _model("Booking", "Room"))
    assert matrix["Booking"]["create"] is True
    assert matrix["Room"]["create"] is False



# Live (2026-11-13, claude-sonnet-5-5, expense tracker): the shared EntityForm, routed at
# `/<slug>/new` for every config entry, named its checkbox handler's argument `tagId`, so
# it was read as Tag's own form and five entities were reported "no frontend create path".
REF_CONFIG = """\
export const ENTITIES = {
  expense: { resource: 'expense', slug: 'expenses', label: 'Expense',
             fields: [{ key: 'category', type: 'ref', ref: 'category' }, { key: 'tags', type: 'multiref', ref: 'tag' }] },
  category: { resource: 'category', slug: 'categories', label: 'Category', fields: [{ key: 'name' }] },
  tag: { resource: 'tag', slug: 'tags', label: 'Tag', fields: [{ key: 'name' }] },
};
"""
REF_CLIENT = """\
async function request(method, path, body) {
  const res = await fetch(path, { method, body: JSON.stringify(body) });
  return res.json();
}
export const api = { create: (resource, payload) => request('POST', `/${resource}/`, payload) };
"""
REF_FORM = """\
import { useState } from 'react';
import { api } from '../api.js';
import { ENTITIES } from '../entities.js';
export default function EntityForm({ config }) {
  const [values, setValues] = useState({});
  const toggleMulti = (key, tagId) => setValues((v) => ({ ...v, [key]: [...(v[key] || []), tagId] }));
  const onSubmit = async (e) => { e.preventDefault(); await api.create(config.resource, values); };
  return <form onSubmit={onSubmit}>{config.label}<input onChange={() => toggleMulti(config.resource, 1)} /></form>;
}
"""
REF_APP = """\
import { Route, Routes } from 'react-router-dom';
import { ENTITIES } from './entities.js';
import EntityForm from './components/EntityForm.jsx';
export default function App() {
  return <Routes>{Object.values(ENTITIES).map((cfg) =>
    <Route key={cfg.slug} path={`/${cfg.slug}/new`} element={<EntityForm config={cfg} />} />)}</Routes>;
}
"""


def _ref_creates(tmp_path, form=REF_FORM, config=REF_CONFIG):
    root = _workspace(tmp_path, {
        "frontend/src/api.js": REF_CLIENT,
        "frontend/src/entities.js": config,
        "frontend/src/components/EntityForm.jsx": form,
        "frontend/src/App.jsx": REF_APP,
    })
    matrix = build_acceptance_matrix(root, _model("Expense", "Category", "Tag"))
    return {name: matrix[name]["create"] for name in ("Expense", "Category", "Tag")}


def test_a_shared_form_holding_a_record_id_named_after_an_entity_creates_them_all(tmp_path):
    assert _ref_creates(tmp_path) == {"Expense": True, "Category": True, "Tag": True}


def test_a_shared_form_with_a_variable_named_after_an_entity_is_still_that_entitys(tmp_path):
    form = REF_FORM.replace("tagId", "tag")
    assert _ref_creates(tmp_path, form=form) == {"Expense": False, "Category": False, "Tag": True}


def test_a_shared_form_holding_a_record_id_still_needs_its_create(tmp_path):
    form = REF_FORM.replace("await api.create(config.resource, values);", "")
    assert _ref_creates(tmp_path, form=form) == {"Expense": False, "Category": False, "Tag": False}


def test_an_entity_the_config_omits_has_no_create_through_a_form_holding_an_id(tmp_path):
    config = REF_CONFIG.replace(
        "  category: { resource: 'category', slug: 'categories', label: 'Category', fields: [{ key: 'name' }] },\n", "")
    assert _ref_creates(tmp_path, config=config) == {"Expense": True, "Category": False, "Tag": True}
