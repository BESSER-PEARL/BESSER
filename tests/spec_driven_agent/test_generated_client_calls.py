"""The validators read calls made through the generated API client as they read axios calls.

The React generator writes ``src/api/generated/client.ts``: ``apiRequest(<METHOD>, <path>, {body})``
and one typed function per route (``bookApi.create(...)``). A wrong route, method, body key or
response use written that way must be reported like the same mistake written with axios; a cast
(``as any``) hides it from ``tsc``, so the validators are what still sees it.
"""
import json
import pathlib
import shutil

import pytest

from besser.BUML.metamodel.gui import GUIModel, Module, Screen, Text
from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Multiplicity, Property, StringType,
)
from besser.generators.web_app import WebAppGenerator
from besser.spec_driven_agent.validation.acceptance import build_acceptance_matrix
from besser.spec_driven_agent.validation.api_call_shapes import collect_api_call_diagnostics, collect_api_call_issues
from besser.spec_driven_agent.validation.endpoint_coherence import collect_endpoint_coherence_issues
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_class_diagram,
    process_gui_diagram,
)

TESTS = pathlib.Path(__file__).parents[1]
LIBRARY_CLASS = TESTS / "generators" / "fixtures" / "library_template_class_diagram.json"
LIBRARY_GUI = TESTS / "utilities" / "web_modeling_editor" / "backend" / "fixtures" / "library_template_gui_diagram.json"
PAGE = "frontend/src/pages/PlantedPage.tsx"
# `library_id` is the key a Book is READ under; its create body takes `library`.
BOOK_BODY = 'title: "x", genre: "Fiction", pages: 1, price: 1, release: "2020-01-01", stock: 1, library_id: 1, authors: []'

PLANTED = {
    "axios": f'''import axios from "axios";
const API = "http://localhost:8000";
export async function planted(id: number) {{
  await axios.get(`${{API}}/books/`);
  await axios.patch(`${{API}}/book/${{id}}/`, {{ title: "x" }});
  await axios.post(`${{API}}/book/`, {{ {BOOK_BODY} }});
  const res = await axios.get(`${{API}}/book/${{id}}/authors/`);
  return res.data.map((a: any) => a.name);
}}
''',
    "apiRequest": f'''import {{ apiRequest }} from "../api/generated/client";
export async function planted(id: number) {{
  await apiRequest("GET", "/books/");
  await apiRequest("PATCH", `/book/${{id}}/`, {{ body: {{ title: "x" }} }});
  await apiRequest("POST", "/book/", {{ body: {{ {BOOK_BODY} }} }});
  const authors = await apiRequest<any>("GET", `/book/${{id}}/authors/`);
  return authors.map((a: any) => a.name);
}}
''',
    # A typed client call cannot name a wrong route or method; a cast or an unknown function can.
    "typed": f'''import {{ bookApi }} from "../api/generated/client";
export async function planted() {{
  await (bookApi as any).patch(1, {{ title: "x" }});
  return bookApi.create({{ {BOOK_BODY} }} as any);
}}
''',
}


@pytest.fixture(scope="module")
def library_app(tmp_path_factory):
    class_json = json.loads(LIBRARY_CLASS.read_text(encoding="utf-8"))
    domain = process_class_diagram(class_json)
    gui = process_gui_diagram(json.loads(LIBRARY_GUI.read_text(encoding="utf-8"))["model"], class_json["model"], domain)
    app = tmp_path_factory.mktemp("library")
    WebAppGenerator(domain, gui, output_dir=str(app)).generate()
    return app


def _with_page(app, tmp_path, page):
    work = tmp_path / "app"
    shutil.copytree(app, work)
    if page is not None:
        (work / PAGE).parent.mkdir(parents=True, exist_ok=True)
        (work / PAGE).write_text(page, encoding="utf-8")
    return str(work)


def test_the_generated_client_and_blocks_raise_no_finding(library_app, tmp_path):
    work = _with_page(library_app, tmp_path, None)
    assert collect_endpoint_coherence_issues(work) == []
    assert collect_api_call_issues(work) == []


@pytest.mark.parametrize("style", ["axios", "apiRequest"])
def test_a_planted_defect_is_reported_whatever_the_request_style(library_app, tmp_path, style):
    work = _with_page(library_app, tmp_path, PLANTED[style])
    coherence = "\n".join(collect_endpoint_coherence_issues(work))
    shapes = "\n".join(collect_api_call_issues(work))
    assert "calls GET /books/, but no generated backend route matches it" in coherence
    assert "calls PATCH /book/{param}/, but the path exists only for" in coherence
    assert "POST /book/ reads its JSON body as BookCreate" in shapes and "required library" in shapes
    assert "GET /book/{book_id}/authors/ returns an object" in shapes and "as an array (.map())" in shapes


def test_a_typed_client_call_hidden_from_tsc_by_a_cast_is_reported(library_app, tmp_path):
    work = _with_page(library_app, tmp_path, PLANTED["typed"])
    coherence = "\n".join(collect_endpoint_coherence_issues(work))
    shapes = "\n".join(collect_api_call_issues(work))
    assert "calls bookApi.patch(...), which the generated API client does not have" in coherence
    assert "POST /book/ reads its JSON body as BookCreate" in shapes and "required library" in shapes
    # The same finding reaches the agent when it writes the file.
    written = collect_api_call_diagnostics(work, changed_path=PAGE, content=PLANTED["typed"])
    assert any("required library" in finding["message"] for finding in written)


def _tracker():
    """Project 1 - * Task, and a class named like the client module the blocks import."""
    def plain(name):
        return Class(name=name, attributes={Property(name="label", type=StringType)})
    project, task, client = plain("Project"), plain("Task"), plain("Client")
    tasks = BinaryAssociation(name="project_tasks", ends={
        Property(name="project", type=project, multiplicity=Multiplicity(1, 1)),
        Property(name="tasks", type=task, multiplicity=Multiplicity(0, "*"))})
    domain = DomainModel(name="Tracker", types={project, task, client}, associations={tasks})
    screen = Screen(name="Home", description="", view_elements={Text(name="t", content="x")}, is_main_page=True)
    gui = GUIModel(name="G", package="a", versionCode="1", versionName="1", description="",
                   modules={Module(name="M", screens={screen})})
    return domain, gui


def test_a_typed_create_is_credited_to_its_class_only(tmp_path):
    domain, gui = _tracker()
    WebAppGenerator(domain, gui, output_dir=str(tmp_path)).generate()
    (tmp_path / "frontend/src/pages/TaskPage.tsx").write_text('''import { taskApi, projectApi } from "../api/generated/client";
export async function saveTask(projectId: number) {
  const projects = await projectApi.list();
  await taskApi.create({ label: "x", project: projectId });
  return projects;
}
''', encoding="utf-8")
    matrix = build_acceptance_matrix(str(tmp_path), domain)
    assert {name: cells["create"] for name, cells in matrix.items()} == {
        "Client": False, "Project": False, "Task": True}


def _with_renamed_route(app, tmp_path, page):
    work = pathlib.Path(_with_page(app, tmp_path, page))
    router = work / "backend" / "routers" / "book.py"
    text = router.read_text(encoding="utf-8")
    assert '@router.post("/book/bulk/"' in text
    router.write_text(text.replace('@router.post("/book/bulk/"', '@router.post("/book/batch/"'), encoding="utf-8")
    return str(work)


def test_a_renamed_route_is_reported_where_code_calls_it_not_in_the_generated_client(library_app, tmp_path):
    """The generated client is rewritten by regeneration: a client function nothing calls is no finding."""
    assert collect_endpoint_coherence_issues(_with_renamed_route(library_app, tmp_path / "unused", None)) == []

    page = '''import { bookApi } from "../api/generated/client";
export const importBooks = () => bookApi.bulkCreate([]);
'''
    issues = collect_endpoint_coherence_issues(_with_renamed_route(library_app, tmp_path / "used", page))
    assert len(issues) == 1 and f"{PAGE} line 2 calls POST /book/bulk/" in issues[0]
