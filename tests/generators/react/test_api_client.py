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
    BinaryAssociation, BooleanType, Class, DomainModel, Enumeration, EnumerationLiteral, Method,
    MethodImplementationType, Multiplicity, Parameter, Property, StringType,
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
        literal = re.sub(r"\{(\w+)\}", lambda m: "${apiSegment(" + m.group(1) + ")}", route["path"])
        assert f'"{route["path"]}"' in client or f"`{literal}`" in client, route
    # The wrappers stay inside the client.
    assert "(await apiRequest<{ task: T.Task }>(\"GET\", `/task/${apiSegment(task_id)}/`)).task" in client
    assert "(await apiRequest<{ tags: T.Tag[] }>(\"GET\", `/task/${apiSegment(task_id)}/tags/`)).tags" in client
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


def test_the_client_alone_is_the_one_the_whole_frontend_ships(tmp_path):
    """A frontend without a GUI model gets the same client files, byte for byte."""
    model = _task_model()
    generated = _generate(tmp_path / "full", model) / "api" / "generated"

    written = ReactGenerator(model, None, output_dir=str(tmp_path / "alone")).generate_api_client()

    assert written == ["src/api/generated/types.ts", "src/api/generated/client.ts"]
    alone = tmp_path / "alone" / "src" / "api" / "generated"
    assert sorted(p.name for p in alone.iterdir()) == ["client.ts", "types.ts"]
    for name in ("client.ts", "types.ts"):
        assert (alone / name).read_bytes() == (generated / name).read_bytes()


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


# What the generated app gets from vite/client and the axios package, neither installed here:
# ImportMeta.env and the two axios members client.ts uses, with axios's own signatures.
AMBIENT = """
interface ImportMeta { readonly env: Record<string, string | undefined> }
declare module "axios" {
  interface AxiosResponse<R> { data: R; status: number }
  interface AxiosError { response?: AxiosResponse<unknown> }
  const axios: {
    request<R>(config: { method?: string; url?: string; data?: unknown }): Promise<AxiosResponse<R>>;
    isAxiosError(payload: unknown): payload is AxiosError;
  };
  export default axios;
}
"""


@pytest.fixture
def typecheck(tmp_path):
    """Type-check a snippet against the client generated for a model; returns tsc's (exit code, output)."""
    tsc = _tsc()

    def check(code, model=None, name="app"):
        generated = _generate(tmp_path / name, model or _task_model()) / "api" / "generated"
        (generated / "ambient.d.ts").write_text(AMBIENT, encoding="utf-8")
        (generated / "snippet.ts").write_text(code, encoding="utf-8")
        result = subprocess.run(
            [tsc, "--noEmit", "--strict", "--target", "es2017", "--lib", "dom,esnext", "--module", "esnext",
             "--moduleResolution", "node", "--isolatedModules", "ambient.d.ts", "types.ts", "client.ts", "snippet.ts"],
            cwd=str(generated), capture_output=True, text=True,
        )
        return result.returncode, result.stdout + result.stderr

    return check


CORRECT = """import { taskApi, projectApi } from "./client";
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


# The codes tsc reports: TypeScript 5's first, then TypeScript 4.9's (the generated app's) where it differs.
@pytest.mark.parametrize("snippet, errors", [
    # A relation reader returns the related records, not the route's {task_id, tags_count, tags} wrapper.
    ("export const f = async () => (await taskApi.relations.tags.list(1)).tags_count;", ("TS2339",)),
    # Created with the key a record is READ under, instead of the end name the body WRITES.
    ('export const f = () => taskApi.create({ title: "x", status: "OPEN", project_id: 1 });', ("TS2561", "TS2345")),
    # GET by id returns the record, not {"task": ...}.
    ("export const f = async () => (await taskApi.get(1)).task.title;", ("TS2339",)),
    # An enumeration takes its literals only.
    ('export const f = () => taskApi.create({ title: "x", status: "CLOSED", project: 1 });', ("TS2322",)),
    # A method takes its arguments flat; the client adds the {"params": ...} wrapper.
    ('export const f = () => taskApi.methods.close(1, { params: { reason: "x" } });', ("TS2353", "TS2345")),
], ids=["relation-wrapper", "read-key-in-body", "get-wrapper", "enum-literal", "method-params-wrapper"])
def test_a_guessed_shape_fails_to_type_check(typecheck, snippet, errors):
    code, output = typecheck('import { taskApi } from "./client";\n' + snippet)
    assert code != 0 and any(error in output for error in errors), output


def _colliding_model():
    """Classes named like the client's fixed exports and like another class's derived types."""
    def plain(name):
        return Class(name=name, attributes={Property(name="label", type=StringType)})
    classes = {name: plain(name) for name in (
        "System", "ApiPage", "ApiMethodResult", "ApiError", "Task", "TaskCreate", "TaskDetailed", "TaskSearch")}
    owns = BinaryAssociation(name="task_items", ends={
        Property(name="task", type=classes["Task"], multiplicity=Multiplicity(0, 1)),
        Property(name="items", type=classes["ApiPage"], multiplicity=Multiplicity(0, "*"))})
    return DomainModel(name="Names", types=set(classes.values()), associations={owns})


def test_a_class_cannot_collide_with_a_name_the_client_fixes(typecheck):
    code, output = typecheck("""import { apiSystem, systemApi, apiPageApi, taskApi, taskCreateApi, ApiPage } from "./client";
import type { TaskCreate, TaskCreate2, TaskDetailed2 } from "./types";
export async function demo(): Promise<string> {
  const health = await apiSystem.health();
  const systems = await systemApi.list();
  const page: ApiPage<unknown> = await apiPageApi.page();
  const created = await taskApi.create({ label: "a" } as TaskCreate2);
  const record: TaskCreate = await taskCreateApi.get(1);
  const detailed: TaskDetailed2[] = await taskApi.listDetailed();
  return `${health.status} ${systems.length} ${page.total} ${created.id} ${record.label} ${detailed.length}`;
}
""", model=_colliding_model())
    assert code == 0, output


def test_derived_type_names_never_merge_with_a_class(tmp_path):
    types = _read(_generate(tmp_path, _colliding_model()) / "api" / "generated" / "types.ts")
    for name in ("TaskCreate", "TaskDetailed", "TaskSearch", "ApiPage"):
        assert types.count(f"export interface {name} ") == 1, name
    assert "export interface TaskCreate2 {" in types


def test_a_bal_body_that_does_not_parse_does_not_stop_the_frontend(tmp_path, caplog):
    counter = Class(name="Counter", attributes={Property(name="n", type=StringType)})
    counter.methods = {Method(name="inc", code="this is not ( valid BAL",
                              implementation_type=MethodImplementationType.BAL)}
    client = _read(_generate(tmp_path, DomainModel(name="B", types={counter})) / "api" / "generated" / "client.ts")
    # Routed like every BAL method that parses: per instance.
    assert "`/counter/${apiSegment(counter_id)}/methods/inc/`" in client
    assert any("does not parse" in record.message for record in caplog.records)


def _runnable(source):
    """A generated module as Node.js runs it: env read from globalThis, relative imports with .ts."""
    source = source.replace("import.meta.env.VITE_API_URL", "(globalThis as any).VITE_API_URL")
    return re.sub(r'(from "\.{1,2}/[^"]+)"', r'\1.ts"', source)




# The axios default instance as the generated app uses it: `defaults.headers.common`, request
# interceptors, then the adapter; a non-2xx answer rejects with an error carrying `response`.
# `next` sets what the backend answers; `sent` records each request with the headers it carried.
AXIOS_STUB = """
export const sent = [];
export let next = { status: 200, data: [] };
export const answer = (response) => { next = response; };
const interceptors = [];
const axios = {
  defaults: { headers: { common: {} } },
  interceptors: { request: { use: (fn) => interceptors.push(fn) } },
  isAxiosError: (value) => Boolean(value && value.isAxiosError),
  async request(config) {
    let merged = { ...config, headers: { ...axios.defaults.headers.common, ...(config.headers || {}) } };
    for (const fn of interceptors) merged = await fn(merged);
    sent.push({ method: merged.method, url: merged.url, headers: merged.headers, data: merged.data ?? null });
    const response = { status: next.status, data: next.data };
    if (next.status >= 200 && next.status < 300) return response;
    throw Object.assign(new Error(`Request failed with status code ${next.status}`), { isAxiosError: true, response });
  },
};
export default axios;
"""


@pytest.fixture
def client_harness(build_app, tmp_path):
    """The generated client.ts and bindingData.ts, runnable by Node.js against the axios stub,
    for an app with a Book and a class named like the `/api` base path; plus FormBlock's
    error message function. Returns the ``run_ts`` path prefix."""
    app = build_app({"Book": [("title", "str")], "Api": [("label", "str")]}, {"Home": []})
    harness = tmp_path / "h"
    (harness / "api").mkdir(parents=True)
    (harness / "api" / "client.ts").write_text(_runnable(app.file("api", "generated", "client.ts")), encoding="utf-8")
    (harness / "bindingData.ts").write_text(
        _runnable(app.file("components", "runtime", "bindingData.ts")).replace("../../api/generated/client",
                                                                             "./api/client"),
        encoding="utf-8")
    form = app.file("components", "runtime", "FormBlock.tsx")
    error_text = form[form.index("const errorText"):form.index("\n};\n", form.index("const errorText")) + 4]
    (harness / "formErrors.ts").write_text('import { ApiError } from "./api/client.ts";\nexport '
                                           + error_text, encoding="utf-8")
    axios = tmp_path / "node_modules" / "axios"
    axios.mkdir(parents=True)
    (axios / "package.json").write_text('{"name": "axios", "type": "module", "main": "index.js"}', encoding="utf-8")
    (axios / "index.js").write_text(AXIOS_STUB, encoding="utf-8")
    return app


def test_every_block_joins_a_relative_base_url_once(client_harness, run_ts):
    """Behind a same-origin proxy (VITE_API_URL=/api) each request goes to /api/<route>, not
    /api/api/; a route or endpoint that itself starts with /api keeps the prefix."""
    sent = run_ts("""
(globalThis as any).VITE_API_URL = "/api";
const { apiRequest, apiApi } = await import("./h/api/client.ts");
const { buildFilteredRequest } = await import("./h/bindingData.ts");
const { sent } = await import("axios");
const filter = { serverCompatible: true, queryParams: { title: "x" }, conditions: [], hasNestedFields: false } as any;
const chart = buildFilteredRequest("/book/", { detailed: true });
await apiRequest("GET", chart.path, { query: chart.query });
const searched = buildFilteredRequest("/book/", { filter, preferSearch: true });
await apiRequest("GET", searched.path, { query: searched.query });
const custom = buildFilteredRequest("/stats/revenue", { custom: true, filter });
await apiRequest("GET", custom.path, { query: custom.query });
await apiRequest("GET", "/book/");
await apiRequest("POST", "/book/", { body: { title: "x" } });
await apiApi.list();
const stats = buildFilteredRequest("/api/stats", { custom: true });
await apiRequest("GET", stats.path, { query: stats.query });
console.log(JSON.stringify(sent.map((r: any) => `${r.method} ${r.url}`)));
""", name="run.mts")
    assert sent == ["GET /api/book/?detailed=true", "GET /api/book/search/?title=x", "GET /api/stats/revenue",
                    "GET /api/book/", "POST /api/book/", "GET /api/api/", "GET /api/api/stats"]


def test_every_block_hands_the_client_a_path(build_app):
    app = build_app({"Book": [("title", "str")]}, {"Home": []})
    runtime = pathlib.Path(app.src) / "components" / "runtime"
    for block in sorted(runtime.glob("*.ts*")):
        for call in re.findall(r"apiRequest(?:<[^>]*>)?\(([^)]*)", block.read_text(encoding="utf-8")):
            assert "backendUrl" not in call and "buildFilteredUrl" not in call, (block.name, call)


def test_axios_defaults_and_interceptors_reach_every_client_request(client_harness, run_ts):
    """The app's axios setup (a default header, an Authorization interceptor) applies to the blocks'
    requests and to the typed functions alike."""
    sent = run_ts("""
const axios = (await import("axios")).default;
const { sent } = await import("axios");
axios.defaults.headers.common["X-Tenant"] = "acme";
axios.interceptors.request.use((config: any) => ({ ...config, headers: { ...config.headers, Authorization: "Bearer t" } }));
const { apiRequest, bookApi } = await import("./h/api/client.ts");
await apiRequest("GET", "/book/");
await apiRequest("POST", "/book/", { body: { title: "x" } });
await bookApi.list();
console.log(JSON.stringify(sent.map((r: any) => [r.method, r.headers.Authorization, r.headers["X-Tenant"], r.data])));
""", name="run.mts")
    assert sent == [["GET", "Bearer t", "acme", None], ["POST", "Bearer t", "acme", {"title": "x"}],
                    ["GET", "Bearer t", "acme", None]]


def test_failures_become_one_api_error_and_the_form_messages_stay(client_harness, run_ts):
    """Each error body the backend sends becomes an ApiError; FormBlock shows what it showed when
    it called axios itself."""
    shown = run_ts("""
const { answer } = await import("axios");
const { apiRequest, ApiError } = await import("./h/api/client.ts");
const { errorText } = await import("./h/formErrors.ts");
const failures: Record<string, any> = {
  invalid: { status: 422, data: { detail: [{ loc: ["body", "title"], msg: "Field required", type: "missing" }] } },
  missing: { status: 404, data: { error: "Book not found", message: "Book not found", detail: "Book not found" } },
  crashed: { status: 500, data: "Internal Server Error" },
};
const out: Record<string, any> = {};
for (const [name, response] of Object.entries(failures)) {
  answer(response);
  try {
    await apiRequest("POST", "/book/", { body: {} });
  } catch (err: any) {
    out[name] = [err instanceof ApiError, err.status, err.message, err.fields, errorText(err)];
  }
}
out.network = errorText(new TypeError("Failed to fetch"));
console.log(JSON.stringify(out));
""", name="run.mts")
    assert shown == {
        "invalid": [True, 422, "title: Field required", {"title": "Field required"}, "title: Field required"],
        "missing": [True, 404, "Book not found", {}, "Book not found"],
        "crashed": [True, 500, "Internal Server Error", {}, "Could not save. Please check your input."],
        "network": "Network error. Please try again.",
    }
