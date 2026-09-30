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
