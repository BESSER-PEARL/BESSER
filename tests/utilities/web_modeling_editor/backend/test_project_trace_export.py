"""A project exported as B-UML with its trace (``/export-project-as-buml``) and imported back
(``/get-project-json-model``)."""

import copy

import pytest

from besser.utilities.web_modeling_editor.backend.services.converters import class_buml_to_json

EXPORT_URL = "/besser_api/export-project-as-buml"
IMPORT_URL = "/besser_api/get-project-json-model"


@pytest.fixture
def class_diagram(library_book_author_model):
    return {"id": "diagram-1", "title": "Library", "model": class_buml_to_json(library_book_author_model)}


@pytest.fixture
def trace(client, class_diagram):
    """A real trace: the one a traced SQLAlchemy generation returns, given a project id."""
    response = client.post("/besser_api/traceability/generate-output", json={
        **class_diagram, "generator": "sqlalchemy", "config": {"dbms": "sqlite"}, "actorName": "Ada Lovelace",
    })
    assert response.status_code == 200, response.text
    return {**response.json()["trace"], "projectId": "project-1"}


@pytest.fixture
def project(class_diagram):
    return {
        "id": "project-1", "type": "Project", "name": "Library", "owner": "Ada Lovelace",
        "createdAt": "2026-10-29T09:00:00.000Z", "currentDiagramType": "ClassDiagram",
        "currentDiagramIndices": {"ClassDiagram": 0}, "diagrams": {"ClassDiagram": [class_diagram]},
    }


def _export(client, project) -> str:
    response = client.post(EXPORT_URL, json=project)
    assert response.status_code == 200, response.text
    return response.text


def _class_names(body) -> set:
    (diagram,) = body["project"]["diagrams"]["ClassDiagram"]
    return {e["name"] for e in diagram["model"]["elements"].values() if e["type"] == "Class"}


def _import(client, source: str):
    return client.post(IMPORT_URL, files={"buml_file": ("library.py", source.encode("utf-8"), "text/x-python")})


def test_trace_survives_export_and_import(client, project, trace):
    source = _export(client, {**project, "trace": trace})
    assert "# TRACEABILITY MODEL #" in source
    assert "trace_model" not in source.split("models=[")[1].split("]")[0]

    response = _import(client, source)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["trace"] == trace
    assert _class_names(body) == {"Library", "Book", "Author"}


def test_trace_section_leaves_the_models_alone(client, project, trace):
    without = _export(client, project)
    with_trace = _export(client, {**project, "trace": trace})
    models, definition = without.split("# PROJECT DEFINITION #")
    assert with_trace.startswith(models.rstrip("#\n"))
    assert with_trace.endswith(definition)


def test_project_without_trace_has_none(client, project):
    source = _export(client, project)
    assert "TRACEABILITY" not in source
    assert _import(client, source).json()["trace"] is None


def test_invalid_trace_is_rejected_on_export(client, project, trace):
    broken = copy.deepcopy(trace)
    broken["traceLinks"][0]["sourceId"] = "missing"
    response = client.post(EXPORT_URL, json={**project, "trace": broken})
    assert response.status_code == 400
    assert "unknown artifact" in response.json()["detail"]


def test_unsafe_trace_section_is_rejected_on_import(client, project, trace):
    source = _export(client, {**project, "trace": trace})
    tampered = source.replace("trace_model = TraceModel(", 'open("x", "w")\ntrace_model = TraceModel(', 1)
    response = _import(client, tampered)
    assert response.status_code == 400
    assert "traceability section" in response.json()["detail"]
