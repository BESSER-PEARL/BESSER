"""Every registered generator through the traced endpoints (``/besser_api/traceability/...``).

For each generator the editor offers, the traced endpoint must accept what the plain one accepts,
deliver the same download, and return a valid trace fragment with at least file-level links: each
diagram the run read is a root model artifact (its diagram id, its real metaclass) linked to every
text file of the download with the rule ``generator:<key>``. Generators the plain endpoint refuses
must be refused the same way.

The diagrams are the editor's own (its templates, as the editor sends them), so the trace is also
checked at element level: every generator that traces elements has, for each diagram listed in
``_ELEMENT_LEVEL``, element artifacts under that diagram's root, linked to line ranges, whose ids are
the editor ids of elements of the diagram sent (the ids the editor's own trace records them by).
"""

import base64
import io
import json
import pathlib
import re
import zipfile

import pytest

from besser.utilities.provenance import PROVENANCE_FILENAME
from besser.utilities.web_modeling_editor.backend.config import SUPPORTED_GENERATORS
from besser.utilities.web_modeling_editor.backend.services.converters import process_trace
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.nn_diagram_converter import (
    nn_model_to_json,
)
from tests.forecaster_networks import HORIZON, WINDOW, mlp

TRACED = "/besser_api/traceability/generate-output"
TRACED_PROJECT = "/besser_api/traceability/generate-output-from-project"
PLAIN = "/besser_api/generate-output"
PLAIN_PROJECT = "/besser_api/generate-output-from-project"

GENERATOR_FIXTURES = pathlib.Path(__file__).parents[3] / "generators" / "fixtures"
BACKEND_FIXTURES = pathlib.Path(__file__).parent / "fixtures"

# The editor's "Library (Full Stack)" template: its class, agent and GUI diagrams.
CLASS_DIAGRAM = {"id": "class-diagram",
                 **json.loads((GENERATOR_FIXTURES / "library_template_class_diagram.json").read_text("utf-8"))}
AGENT_DIAGRAM = {"id": "agent-diagram",
                 **json.loads((GENERATOR_FIXTURES / "library_template_agent_diagram.json").read_text("utf-8"))}
GUI_DIAGRAM = {"id": "gui-diagram",
               **json.loads((BACKEND_FIXTURES / "library_template_gui_diagram.json").read_text("utf-8"))}


def _template(name: str) -> dict:
    """An editor template (``packages/webapp/src/main/templates/pattern`` of the editor), as it sends it."""
    return json.loads((BACKEND_FIXTURES / name).read_text("utf-8"))


BPMN_DIAGRAM = {"id": "bpmn-diagram", "title": "Pizza store", "model": _template("template_bpmn_pizza_store.json")}


def _quantum_diagram() -> dict:
    """A Bell circuit as the editor's circuit editor (Quirk columns) sends it: H, then a controlled X."""
    return {"id": "quantum-diagram", "title": "Bell",
            "model": {"version": "1.0.0", "cols": [["H", 1], ["\u2022", "X"]], "gateMetadata": {}}}


NN_DIAGRAM = {"id": "nn-diagram", "title": "Tutorial", "model": _template("template_nn_tutorial_example.json")}
# The forecaster needs a network of the shape it forecasts with.
FORECAST_DIAGRAM = {"id": "forecast-diagram", "title": "Forecast", "model": nn_model_to_json(mlp())}


def _deployment_diagram(agent_diagram_id=None) -> dict:
    """A node running two artifacts, one of them the agent of ``agent_diagram_id`` when given."""
    def element(element_id, element_type, name, owner=None, **extra):
        return {"id": element_id, "type": element_type, "name": name, "owner": owner,
                "bounds": {"x": 0, "y": 0, "width": 200, "height": 100}, **extra}
    agent = {"agentModelRef": agent_diagram_id} if agent_diagram_id else {}
    return {"id": "deployment-diagram", "title": "Cloud", "model": {
        "version": "3.0.0", "type": "DeploymentDiagram", "relationships": {},
        "elements": {"node-1": element("node-1", "DeploymentNode", "server"),
                     "artifact-api": element("artifact-api", "DeploymentArtifact", "api", "node-1"),
                     "artifact-bot": element("artifact-bot", "DeploymentArtifact", "librarybot", "node-1", **agent)}}}


DEPLOYMENT_DIAGRAM = _deployment_diagram()


def _alloy_class_diagram() -> dict:
    """The library diagram without the attribute types the Alloy translation refuses (float, datetime, ...)."""
    diagram = json.loads(json.dumps(CLASS_DIAGRAM))
    elements = diagram["model"]["elements"]
    refused = {key for key, element in elements.items() if element.get("type") == "ClassAttribute"
               and str(element.get("attributeType", "")).lower() in {"float", "datetime", "time", "timedelta", "any"}}
    for key in refused:
        del elements[key]
    for element in elements.values():
        if isinstance(element.get("attributes"), list):
            element["attributes"] = [key for key in element["attributes"] if key not in refused]
    return diagram


OBJECT_DIAGRAM = {"id": "object-diagram", "title": "Books", "model": {
    "type": "ObjectDiagram", "relationships": {},
    "elements": {"o1": {"id": "o1", "type": "ObjectName", "name": "dune", "className": "Book", "attributes": []}},
    "referenceDiagramData": {"title": CLASS_DIAGRAM["title"], "model": CLASS_DIAGRAM["model"]}}}

# What each generator runs on: the diagram it reads at /generate-output, its config, and the
# metaclass of the model the trace records for it.
_ON_ONE_DIAGRAM = {
    "jsonobject": (OBJECT_DIAGRAM, None, "ObjectModel"),
    "agent": (AGENT_DIAGRAM, None, "Agent"),
    "qiskit": (_quantum_diagram(), None, "QuantumCircuit"),
    "bpmn": (BPMN_DIAGRAM, None, "BPMNModel"),
    "docker_compose": (DEPLOYMENT_DIAGRAM, None, "DeploymentModel"),
    "pytorch": (NN_DIAGRAM, None, "NN"),
    "tensorflow": (NN_DIAGRAM, None, "NN"),
    "forecaster": (FORECAST_DIAGRAM, {"window": WINDOW, "horizon": HORIZON}, "NN"),
    "django": (CLASS_DIAGRAM, {"project_name": "library_site", "app_name": "books"}, "DomainModel"),
    "alloy": (_alloy_class_diagram(), None, "DomainModel"),
}
# Generators /generate-output refuses (they need a second model); the traced endpoint must too.
_REFUSED_ON_ONE_DIAGRAM = {key for key, info in SUPPORTED_GENERATORS.items() if info.needs_beyond_one_diagram}


def _one_diagram_case(generator: str):
    diagram, config, metaclass = _ON_ONE_DIAGRAM.get(generator, (CLASS_DIAGRAM, None, "DomainModel"))
    return {**diagram, "generator": generator, "config": config}, {diagram["id"]: metaclass}


def _entries(download: bytes, is_zip: bool, file_name: str) -> dict:
    if not is_zip:
        return {file_name: download}
    with zipfile.ZipFile(io.BytesIO(download)) as archive:
        return {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}


# Two runs of the same request differ, traced or not, where a generator is not deterministic; the
# comparison ignores only that. Django's startproject writes a fresh random SECRET_KEY on every run.
_SECRET_KEY = re.compile(rb"SECRET_KEY = .*")
# The REST BAL translation names loop temporaries after node.__hash__() (an id()), so the methods
# routers of the backend (and of the web app's backend) differ between runs.
_BAL_TEMPORARY = re.compile(rb"\b([A-Za-z]\w*?)_\d{9,}\b")


def _comparable(generator: str, entries: dict) -> dict:
    if generator == "django":
        return {name: _SECRET_KEY.sub(b"SECRET_KEY = <random>", data) for name, data in entries.items()}
    if generator in ("backend", "web_app"):
        return {name: _BAL_TEMPORARY.sub(rb"\1_<hash>", data) for name, data in entries.items()}
    if generator == "supabase":
        # Supabase names its migration after the current time, and two runs order (and number, as
        # ``books_1``) enum literals and association tables in set order: only the name is comparable.
        return {re.sub(r"^\d+_", "<time>_", name): "<not deterministic>" for name in entries}
    return entries


def _assert_same_delivery_and_a_valid_trace(generator, traced_response, plain_response, roots: dict):
    assert plain_response.status_code == 200, plain_response.text
    assert traced_response.status_code == 200, traced_response.text
    body = traced_response.json()
    download = base64.b64decode(body["contentBase64"])
    is_zip = body["mediaType"] == "application/zip"
    assert body["mediaType"] == plain_response.headers["content-type"].split(";")[0]
    if generator != "supabase":  # its file name is the time of the run
        assert body["fileName"] in plain_response.headers["content-disposition"]
    if is_zip:  # the same entries with the same bytes; only the archive's timestamps may differ
        traced_entries = _entries(download, True, body["fileName"])
        assert _comparable(generator, traced_entries) == _comparable(generator, _entries(plain_response.content, True, ""))
    else:
        traced_entries = {body["fileName"]: download}
        assert (_comparable(generator, traced_entries)
                == _comparable(generator, {body["fileName"]: plain_response.content}))

    trace = body["trace"]
    process_trace(trace)  # a valid trace: TraceModel.validate() passes
    artifacts = {a["id"]: a for a in trace["artifacts"]}
    root_artifacts = {a["elementId"]: a for a in artifacts.values()
                      if a["type"] == "ModelArtifact" and "parentId" not in a}
    assert {element_id: a["elementType"] for element_id, a in root_artifacts.items()} == roots
    files = {a["path"] for a in artifacts.values() if a["type"] == "CodeArtifact" and "startLine" not in a}
    text_files = set()
    for name, data in traced_entries.items():
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        text_files.add(name)
    assert files == text_files - {PROVENANCE_FILENAME}
    assert files, f"{generator} delivered no text file"
    for root in root_artifacts.values():
        linked = {artifacts[link["targetId"]]["path"] for link in trace["traceLinks"]
                  if link["sourceId"] == root["id"] and link["evidence"][0]["rule"] == f"generator:{generator}"}
        assert linked == files
    return trace


def _editor_ids(diagram: dict) -> set:
    """The ids the editor knows the elements of ``diagram`` by: the keys and ``id`` values its JSON holds
    at any depth (Apollon elements and relationships, GUI pages and components' ``attributes.id``), and
    for a quantum circuit, whose cells have no ids, each gate's cell (``<diagram id>:gate:<column>:<wire>``,
    the editor's ``snapshotQuantumModel``)."""
    ids, pending = set(), [diagram["model"]]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            for key in ("elements", "relationships", "components"):
                if isinstance(value.get(key), dict):
                    ids.update(value[key])
            if isinstance(value.get("id"), str):
                ids.add(value["id"])
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    for column, cells in enumerate(diagram["model"].get("cols") or []):
        ids.update(f"{diagram['id']}:gate:{column}:{wire}" for wire, cell in enumerate(cells) if cell not in (1, None, ""))
    return ids


def _assert_element_level(generator: str, trace: dict, diagrams) -> None:
    """For each diagram, the trace has element artifacts under the diagram's root, each an element of the
    diagram sent (by its editor id), and some of them linked to line ranges of the delivery."""
    artifacts = {a["id"]: a for a in trace["artifacts"]}

    def root_of(artifact):
        while "parentId" in artifact:
            artifact = artifacts[artifact["parentId"]]
        return artifact

    fragments = {link["sourceId"] for link in trace["traceLinks"] if "startLine" in artifacts[link["targetId"]]}
    for diagram in diagrams:
        elements = [a for a in artifacts.values() if a["type"] == "ModelArtifact" and "parentId" in a
                    and root_of(a)["elementId"] == diagram["id"]]
        unknown = {a["elementId"] for a in elements} - _editor_ids(diagram)
        assert not unknown, f"{generator}: {diagram['id']} elements the editor does not know: {sorted(unknown)}"
        assert any(a["id"] in fragments for a in elements), \
            f"{generator}: no element of {diagram['id']} is linked to a line range"


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# The generators that trace elements of the diagram they read, and the one that stays at file level: the JSON
# object generator dumps the object model as one document (an ``Object`` has no id to refer to it by).
_FILE_LEVEL = {"jsonobject", "alloy"}


@pytest.mark.parametrize("generator", sorted(SUPPORTED_GENERATORS))
def test_every_generator_traces_its_one_diagram_delivery(client, generator):
    payload, roots = _one_diagram_case(generator)
    traced = client.post(TRACED, json={**payload, "actorName": "Ada"})
    plain = client.post(PLAIN, json=payload)
    if generator in _REFUSED_ON_ONE_DIAGRAM:
        assert traced.status_code == plain.status_code == 400
        assert traced.json()["detail"] == plain.json()["detail"]
        return
    trace = _assert_same_delivery_and_a_valid_trace(generator, traced, plain, roots)
    if generator in _FILE_LEVEL:
        assert not any(a["type"] == "ModelArtifact" and "parentId" in a for a in trace["artifacts"])
    else:
        _assert_element_level(generator, trace, [_ON_ONE_DIAGRAM.get(generator, (CLASS_DIAGRAM,))[0]])


def _project(generator: str, diagrams: dict, current: str, config=None) -> dict:
    return {"id": "project-1", "type": "Project", "name": "Library", "createdAt": "2026-01-01T00:00:00Z",
            "currentDiagramType": current, "diagrams": diagrams,
            "settings": {"generator": generator, "config": config or {}}}


# Project-scoped runs: the generators that read several diagrams, or a project's diagrams.
_PROJECT_CASES = {
    "web_app": (_project("web_app", {"ClassDiagram": [CLASS_DIAGRAM], "GUINoCodeDiagram": [GUI_DIAGRAM],
                                     "AgentDiagram": [AGENT_DIAGRAM]}, "GUINoCodeDiagram"),
                {"class-diagram": "DomainModel", "gui-diagram": "GUIModel", "agent-diagram": "Agent"}),
    "backend": (_project("backend", {"ClassDiagram": [CLASS_DIAGRAM], "NNDiagram": [NN_DIAGRAM]}, "ClassDiagram"),
                {"class-diagram": "DomainModel"}),
    "docker_compose": (_project("docker_compose", {"DeploymentDiagram": [_deployment_diagram(AGENT_DIAGRAM["id"])],
                                                   "AgentDiagram": [AGENT_DIAGRAM]}, "DeploymentDiagram"),
                       {"deployment-diagram": "DeploymentModel", "agent-diagram": "Agent"}),
    "qiskit": (_project("qiskit", {"QuantumCircuitDiagram": [_quantum_diagram()]}, "QuantumCircuitDiagram"),
               {"quantum-diagram": "QuantumCircuit"}),
    "sqlalchemy": (_project("sqlalchemy", {"ClassDiagram": [CLASS_DIAGRAM]}, "ClassDiagram", {"dbms": "postgresql"}),
                   {"class-diagram": "DomainModel"}),
}


@pytest.mark.parametrize("generator", sorted(_PROJECT_CASES))
def test_project_generations_trace_every_diagram_they_read(client, generator):
    project, roots = _PROJECT_CASES[generator]
    traced = client.post(TRACED_PROJECT, json={**project, "actorName": "Ada"})
    plain = client.post(PLAIN_PROJECT, json=project)
    trace = _assert_same_delivery_and_a_valid_trace(generator, traced, plain, roots)
    assert [a["name"] for a in trace["actors"]] == ["Ada"]
    # Every diagram the run read has elements traced to lines: the web app's GUI screens and components
    # (React) and its agent (BAF) as much as its classes; Docker Compose's services and the agent it bakes.
    read = [diagram for diagrams in project["diagrams"].values() for diagram in diagrams if diagram["id"] in roots]
    _assert_element_level(generator, trace, read)


def test_the_web_app_traces_the_lines_of_its_backend(client):
    """An instrumented part of a composite keeps its line ranges, under the part's folder."""
    project, _ = _PROJECT_CASES["web_app"]
    trace = client.post(TRACED_PROJECT, json=project).json()["trace"]
    fragments = {a["path"] for a in trace["artifacts"] if a["type"] == "CodeArtifact" and "startLine" in a}
    assert {"backend/sql_alchemy.py", "backend/pydantic_classes.py", "backend/main_api.py"} <= fragments
    book = next(e["id"] for e in CLASS_DIAGRAM["model"]["elements"].values() if e.get("name") == "Book")
    book_artifact = next(a for a in trace["artifacts"] if a.get("elementId") == book)
    class_root = next(a for a in trace["artifacts"] if a.get("elementId") == "class-diagram")
    assert book_artifact["parentId"] == class_root["id"]


def test_the_backend_project_run_links_the_networks_it_uses(client):
    """The project path links NN-implemented methods (link_method_neural_networks); traced or not."""
    project, _ = _PROJECT_CASES["backend"]
    traced = client.post(TRACED_PROJECT, json=project)
    plain = client.post(PLAIN_PROJECT, json=project)
    assert traced.status_code == plain.status_code == 200
    assert (_comparable("backend", _entries(base64.b64decode(traced.json()["contentBase64"]), True, ""))
            == _comparable("backend", _entries(plain.content, True, "")))


def test_a_project_diagram_without_an_id_is_refused(client):
    project, _ = _PROJECT_CASES["qiskit"]
    (diagram,) = project["diagrams"]["QuantumCircuitDiagram"]
    no_id = {**project, "diagrams": {"QuantumCircuitDiagram": [{**diagram, "id": None}]}}
    response = client.post(TRACED_PROJECT, json=no_id)
    assert response.status_code == 400
    assert "id of every diagram" in response.json()["detail"]
