"""A create reached through a member chain (``api.tag.create(payload)``) is credited.

Live run r13-web1: the client is built by a factory,
``const crud = (name) => ({ create: (payload) => request('POST', `/${name}/`, payload) })``
and ``export const api = { tag: crud('tag'), ... }``; pages call ``api.tag.create(payload)``.
The call-site pattern allowed one member after the import binding, so every page read
"no frontend create path found" (three false warnings). The chain's segments now name the
resource when neither the helper's path nor the call's arguments do.
"""
from tests.spec_driven_agent.test_acceptance_real_run_shapes import _creates

FACTORY_API = """
async function request(method, path, body) {
  const res = await fetch(`${BASE_URL}${path}`, { method, body: JSON.stringify(body) });
  return res.json();
}
const crud = (name) => ({
  list: (detailed = false) => request('GET', `/${name}/${detailed ? '?detailed=true' : ''}`),
  create: (payload) => request('POST', `/${name}/`, payload),
  update: (id, payload) => request('PUT', `/${name}/${id}/`, payload),
  remove: (id) => request('DELETE', `/${name}/${id}/`),
});
export const api = {
  tag: crud('tag'),
  todoitem: crud('todoitem'),
  todolist: crud('todolist'),
};
"""


def _page(resource: str, title: str, create: bool = True, extra: str = "") -> str:
    save = f"    if (editId === null) await api.{resource}.create(payload);\n" if create else ""
    return ("import { api } from '../api.js';\n"
            f"export default function {title}() {{\n"
            f"  const load = async () => setRows(await api.{resource}.list());\n{extra}"
            "  const save = async (payload) => {\n"
            f"{save}"
            f"    else await api.{resource}.update(editId, payload);\n"
            "  };\n}\n")


def _files(**overrides):
    files = {
        "frontend/src/api.js": FACTORY_API,
        "frontend/src/pages/TagList.jsx": _page("tag", "TagList"),
        "frontend/src/pages/TodoItemList.jsx": _page("todoitem", "TodoItemList",
                                                     extra="  const tags = api.tag.list();\n"),
        "frontend/src/pages/TodoListList.jsx": _page("todolist", "TodoListList"),
    }
    files.update(overrides)
    return files


def test_the_r13_factory_client_credits_every_create(tmp_path):
    assert _creates(tmp_path, _files(), "Tag", "TodoItem", "TodoList") == {
        "Tag": True, "TodoItem": True, "TodoList": True}


def test_removing_only_the_tag_create_uncredits_tag(tmp_path):
    # TodoItemList still mentions tags (it lists them for a picker); that is not Tag's create.
    files = _files(**{"frontend/src/pages/TagList.jsx": _page("tag", "TagList", create=False)})
    assert _creates(tmp_path, files, "Tag", "TodoItem", "TodoList") == {
        "Tag": False, "TodoItem": True, "TodoList": True}


def test_a_chain_naming_another_entity_is_that_entitys_create(tmp_path):
    # The Tag page creates a todo item: Tag is not created anywhere.
    files = _files(**{"frontend/src/pages/TagList.jsx": _page("tag", "TagList", create=False,
                                                              extra="  api.todoitem.create(draft);\n")})
    assert _creates(tmp_path, files, "Tag", "TodoItem") == {"Tag": False, "TodoItem": True}
