"""The generated frontend reaches the backend through one typed client.

``src/api/generated/types.ts`` and ``client.ts`` are rendered from the backend's own route list
(``besser/generators/backend/api_contract.py``): the client unwraps the backend's response
wrappers, and the types state the keys a body writes and a record reads, so ``tsc`` rejects a
call that guesses a shape.
"""
import json
import pathlib
import re
import shutil
import subprocess

import pytest

from besser.BUML.metamodel.gui import GUIModel, Module, Screen
from besser.BUML.metamodel.structural import (
    BinaryAssociation, BooleanType, Class, DomainModel, Enumeration, EnumerationLiteral, Method, Multiplicity,
    Parameter, Property, StringType,
)
from besser.generators.backend.api_contract import build_api_contract
from besser.generators.react import ReactGenerator
from besser.generators.web_app import WebAppGenerator
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_class_diagram,
    process_gui_diagram,
)

TESTS = pathlib.Path(__file__).parents[2]
LIBRARY_CLASS = TESTS / "generators" / "fixtures" / "library_template_class_diagram.json"
LIBRARY_GUI = TESTS / "utilities" / "web_modeling_editor" / "backend" / "fixtures" / "library_template_gui_diagram.json"


def _task_model():
    """Project 1 - * Task * - * Tag; Task.status is an enumeration, Task.close(reason) a method."""
    status = Enumeration(name="Status", literals={EnumerationLiteral(name="OPEN"), EnumerationLiteral(name="DONE")})
    project = Class(name="Project", attributes={Property(name="name", type=StringType)})
    task = Class(name="Task", attributes={
        Property(name="title", type=StringType), Property(name="status", type=status),
        Property(name="urgent", type=BooleanType, is_optional=True)})
    task.methods = {Method(name="close", parameters={Parameter(name="reason", type=StringType)})}
    tag = Class(name="Tag", attributes={Property(name="label", type=StringType)})
    associations = {
        BinaryAssociation(name="project_tasks", ends={
            Property(name="project", type=project, multiplicity=Multiplicity(1, 1)),
            Property(name="tasks", type=task, multiplicity=Multiplicity(0, "*"))}),
        BinaryAssociation(name="task_tags", ends={
            Property(name="tasks", type=task, multiplicity=Multiplicity(0, "*")),
            Property(name="tags", type=tag, multiplicity=Multiplicity(0, "*"))}),
    }
    return DomainModel(name="Tasks", types={status, project, task, tag}, associations=associations)


def _generate(tmp_path, model):
    screen = Screen(name="Home", description="", view_elements=set(), is_main_page=True)
    gui = GUIModel(name="GUI", package="app", versionCode="1", versionName="1.0", description="",
                   modules={Module(name="Main", screens={screen})})
    ReactGenerator(model=model, gui_model=gui, output_dir=str(tmp_path)).generate()
    return tmp_path / "src"


def _read(path):
    return pathlib.Path(path).read_text(encoding="utf-8")


def test_the_client_has_one_function_per_route(tmp_path):
    model = _task_model()
    client = _read(_generate(tmp_path, model) / "api" / "generated" / "client.ts")
    for route in build_api_contract(model)["routes"]:
        literal = re.sub(r"\{(\w+)\}", lambda m: "${seg(" + m.group(1) + ")}", route["path"])
        assert f'"{route["path"]}"' in client or f"`{literal}`" in client, route
    # The wrappers stay inside the client.
    assert "(await apiRequest<{ task: T.Task }>(\"GET\", `/task/${seg(task_id)}/`)).task" in client
    assert "(await apiRequest<{ tags: T.Tag[] }>(\"GET\", `/task/${seg(task_id)}/tags/`)).tags" in client
    assert "{ body: { params: args } }" in client


def test_the_library_template_gets_a_client_and_a_contract(tmp_path):
    class_json = json.loads(LIBRARY_CLASS.read_text(encoding="utf-8"))
    domain = process_class_diagram(class_json)
    gui = process_gui_diagram(json.loads(LIBRARY_GUI.read_text(encoding="utf-8"))["model"], class_json["model"], domain)
    WebAppGenerator(domain, gui, output_dir=str(tmp_path)).generate()

    contract = json.loads(_read(tmp_path / "backend" / "api_contract.json"))
    assert contract == build_api_contract(domain)
    assert "`POST /book/`" in _read(tmp_path / "backend" / "API.md")
    generated = tmp_path / "frontend" / "src" / "api" / "generated"
    assert "export interface BookCreate {" in _read(generated / "types.ts")
    assert "export const bookApi = {" in _read(generated / "client.ts")


def test_the_base_url_is_defined_once(tmp_path):
    class_json = json.loads(LIBRARY_CLASS.read_text(encoding="utf-8"))
    domain = process_class_diagram(class_json)
    gui = process_gui_diagram(json.loads(LIBRARY_GUI.read_text(encoding="utf-8"))["model"], class_json["model"], domain)
    WebAppGenerator(domain, gui, output_dir=str(tmp_path)).generate()

    definitions = {}
    for path in (tmp_path / "frontend" / "src").rglob("*.ts*"):
        count = sum(1 for line in _read(path).splitlines()
                    if re.search(r"import\.meta\.env\.VITE_API_URL|[\"'`]http://localhost:8000", line))
        if count:
            definitions[path.relative_to(tmp_path / "frontend" / "src").as_posix()] = count
    assert definitions == {"api/generated/client.ts": 1}


def _tsc():
    tsc = shutil.which("tsc")
    if not tsc or not shutil.which("node"):
        pytest.skip("needs Node.js and the TypeScript compiler (tsc) on PATH")
    return tsc


@pytest.fixture
def typecheck(tmp_path):
    """Type-check a snippet against the generated client; returns tsc's (exit code, output)."""
    tsc = _tsc()
    generated = _generate(tmp_path, _task_model()) / "api" / "generated"
    # The generated app gets ImportMeta.env from vite/client, which is not installed here.
    (generated / "env.d.ts").write_text("interface ImportMeta { readonly env: Record<string, string | undefined> }\n",
                                        encoding="utf-8")

    def check(code):
        (generated / "snippet.ts").write_text('import { taskApi, projectApi } from "./client";\n' + code,
                                              encoding="utf-8")
        result = subprocess.run(
            [tsc, "--noEmit", "--strict", "--target", "es2017", "--lib", "dom,esnext", "--module", "esnext",
             "--moduleResolution", "node", "--isolatedModules", "env.d.ts", "types.ts", "client.ts", "snippet.ts"],
            cwd=str(generated), capture_output=True, text=True,
        )
        return result.returncode, result.stdout + result.stderr

    return check


CORRECT = """
export async function demo(): Promise<string> {
  const project = await projectApi.create({ name: "Launch" });
  const task = await taskApi.create({ title: "Write", status: "OPEN", project: project.id, tags: [] });
  const tags = await taskApi.relations.tags.list(task.id);
  const outcome = await taskApi.methods.close(task.id, { reason: "done" });
  return `${task.project_id} ${tags.length} ${outcome.result}`;
}
"""


def test_the_documented_calls_type_check(typecheck):
    code, output = typecheck(CORRECT)
    assert code == 0, output


@pytest.mark.parametrize("snippet, error", [
    # A relation reader returns the related records, not the route's {task_id, tags_count, tags} wrapper.
    ("export const f = async () => (await taskApi.relations.tags.list(1)).tags_count;", "TS2339"),
    # Created with the key a record is READ under, instead of the end name the body WRITES.
    ('export const f = () => taskApi.create({ title: "x", status: "OPEN", project_id: 1 });', "TS2561"),
    # GET by id returns the record, not {"task": ...}.
    ("export const f = async () => (await taskApi.get(1)).task.title;", "TS2339"),
    # An enumeration takes its literals only.
    ('export const f = () => taskApi.create({ title: "x", status: "CLOSED", project: 1 });', "TS2322"),
    # A method takes its arguments flat; the client adds the {"params": ...} wrapper.
    ('export const f = () => taskApi.methods.close(1, { params: { reason: "x" } });', "TS2353"),
], ids=["relation-wrapper", "read-key-in-body", "get-wrapper", "enum-literal", "method-params-wrapper"])
def test_a_guessed_shape_fails_to_type_check(typecheck, snippet, error):
    code, output = typecheck(snippet)
    assert code != 0 and error in output, output
