"""The acceptance matrix must see a create made through a generic entity page.

A live hotel run rendered every class through one ``EntityList`` page: App.jsx
held a config per entity and rendered ``<EntityList entity={key} />``, and the
page called ``api.create(entity, payload)``. The file naming the entity made no
create and the file making the create named no entity, so every class was
reported "no frontend create path" although each one was creatable.
"""

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
