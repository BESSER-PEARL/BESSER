"""Legacy v3 diagram payloads are rejected with HTTP 400, not converted to nothing.

The JSON -> B-UML converters read only the v4 ``{nodes, edges}`` wire shape.
Before the guard, a v3 ``{version: '3.x', elements, relationships}`` payload
was silently accepted: ``/export-buml`` returned an empty model,
``/validate-diagram`` answered ``isValid: true`` and the generators produced
near-empty code. Every endpoint that takes diagram or project JSON must now
refuse it with an actionable message, while v4 payloads and the GUI
(GrapesJS) / quantum-circuit formats are unaffected.
"""

import asyncio
import copy
import json
import os
from typing import Any, Dict

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.models import DiagramInput, ProjectInput
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
    LegacyDiagramFormatError,
)
from besser.utilities.web_modeling_editor.backend.services.validators.legacy_format import (
    ensure_project_not_legacy,
    is_legacy_v3_model,
)

LEGACY_MARKER = "legacy v3 editor format"
FIXTURES_V4 = os.path.join(os.path.dirname(__file__), "..", "..", "..", "fixtures", "v4")


def _post(url: str, **kwargs) -> httpx.Response:
    async def _go():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
            return await ac.post(url, **kwargs)
    return asyncio.run(_go())


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------

def _v3_model(diagram_type: str = "ClassDiagram") -> Dict[str, Any]:
    """A v3 editor model: one element, keyed maps, version 3.0.0."""
    return {
        "version": "3.0.0",
        "type": diagram_type,
        "size": {"width": 800, "height": 600},
        "interactive": {"elements": {}, "relationships": {}},
        "elements": {
            "e1": {
                "id": "e1", "name": "Book", "type": "Class", "owner": None,
                "bounds": {"x": 0, "y": 0, "width": 160, "height": 100},
                "attributes": [], "methods": [],
            },
        },
        "relationships": {},
        "assessments": {},
    }


def _v4_class_model() -> Dict[str, Any]:
    with open(os.path.join(FIXTURES_V4, "class_diagram_basic.json"), encoding="utf-8") as f:
        return json.load(f)["model"]


def _gui_model() -> Dict[str, Any]:
    # GrapesJS project data: its own unrelated version, no ``type``.
    return {"pages": [], "styles": [], "assets": [], "symbols": [], "version": "0.21.13"}


def _quantum_model() -> Dict[str, Any]:
    return {"cols": [], "gates": [], "gateMetadata": {}, "initialStates": [], "version": "1.0.0"}


def _diagram(model: Dict[str, Any], title: str = "Legacy", **extra) -> Dict[str, Any]:
    return {"title": title, "model": model, **extra}


def _project(class_model: Dict[str, Any], **settings) -> Dict[str, Any]:
    return {
        "id": "proj-1",
        "type": "Project",
        "name": "LegacyProject",
        "createdAt": "2025-01-01T00:00:00Z",
        "currentDiagramType": "ClassDiagram",
        "currentDiagramIndices": {"ClassDiagram": 0},
        "diagrams": {
            "ClassDiagram": [{"id": "cd-1", "title": "Domain", "model": class_model}],
            "GUINoCodeDiagram": [{"id": "gui-1", "title": "GUI", "model": _gui_model()}],
            "QuantumCircuitDiagram": [{"id": "q-1", "title": "Circuit", "model": _quantum_model()}],
        },
        "settings": settings or None,
    }


def _assert_legacy_rejected(response: httpx.Response) -> None:
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert LEGACY_MARKER in detail
    assert "current BESSER web editor" in detail


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestDetection:
    @pytest.mark.parametrize("model", [
        _v3_model(),
        {"version": "3.0.0", "type": "AgentDiagram"},                   # version rule
        {"type": "ClassDiagram", "elements": {}, "relationships": {}},  # shape rule, no version
        {"elements": {}},                                               # untyped v3 shape
    ])
    def test_legacy_models_are_detected(self, model):
        assert is_legacy_v3_model(model)

    @pytest.mark.parametrize("model", [
        {"version": "4.0.0", "type": "ClassDiagram", "nodes": [], "edges": []},
        {},
        _gui_model(),
        _quantum_model(),
        # GUI / quantum are exempt by type even with a v3-looking body.
        {"type": "GUINoCodeDiagram", "version": "3.1.0", "elements": {}},
        {"type": "QuantumCircuitDiagram", "version": "3.0.0", "elements": {}},
        # A typeless model with a 3.x version is not UML (GrapesJS could get there).
        {"pages": [], "version": "3.0.0"},
        None,
        "not a dict",
    ])
    def test_non_legacy_models_pass(self, model):
        assert not is_legacy_v3_model(model)

    def test_exempt_by_project_key(self):
        assert not is_legacy_v3_model({"elements": {}}, "GUINoCodeDiagram")
        assert not is_legacy_v3_model({"elements": {}}, "QuantumCircuitDiagram")

    def test_error_is_a_conversion_error(self):
        assert issubclass(LegacyDiagramFormatError, ConversionError)

    def test_diagram_input_rejects_v3(self):
        with pytest.raises(LegacyDiagramFormatError, match="Diagram 'Legacy' \\(ClassDiagram\\)"):
            DiagramInput(**_diagram(_v3_model()))

    def test_diagram_input_rejects_v3_reference_data(self):
        with pytest.raises(LegacyDiagramFormatError):
            DiagramInput(**_diagram(
                {"version": "4.0.0", "type": "ObjectDiagram", "nodes": [], "edges": []},
                referenceDiagramData=_v3_model(),
            ))
        with pytest.raises(LegacyDiagramFormatError):
            DiagramInput(**_diagram({
                "version": "4.0.0", "type": "ObjectDiagram", "nodes": [], "edges": [],
                "referenceDiagramData": {"title": "Ref", "model": _v3_model()},
            }))

    def test_diagram_input_accepts_v4_gui_and_quantum(self):
        DiagramInput(**_diagram(_v4_class_model()))
        DiagramInput(**_diagram(_gui_model()))
        DiagramInput(**_diagram(_quantum_model()))

    def test_project_input_rejects_any_v3_uml_diagram(self):
        with pytest.raises(LegacyDiagramFormatError, match="'Domain'"):
            ProjectInput(**_project(_v3_model()))

    def test_project_input_accepts_v4_with_gui_and_quantum(self):
        ProjectInput(**_project(_v4_class_model()))

    def test_raw_project_diagrams_single_and_list_layouts(self):
        with pytest.raises(LegacyDiagramFormatError):
            ensure_project_not_legacy({"ClassDiagram": {"title": "D", "model": _v3_model()}})
        with pytest.raises(LegacyDiagramFormatError):
            ensure_project_not_legacy({"AgentDiagram": [{"title": "A", "model": _v3_model("AgentDiagram")}]})
        ensure_project_not_legacy(_project(_v4_class_model())["diagrams"])
        ensure_project_not_legacy(None)


# ---------------------------------------------------------------------------
# Single-diagram endpoints
# ---------------------------------------------------------------------------

UML_TYPES = [
    "ClassDiagram", "ObjectDiagram", "StateMachineDiagram", "AgentDiagram",
    "UserDiagram", "NNDiagram", "BPMNDiagram",
]


@pytest.mark.parametrize("diagram_type", UML_TYPES)
@pytest.mark.parametrize("endpoint", ["/besser_api/export-buml", "/besser_api/validate-diagram"])
def test_single_diagram_endpoints_reject_v3(endpoint, diagram_type):
    _assert_legacy_rejected(_post(endpoint, json=_diagram(_v3_model(diagram_type))))


@pytest.mark.parametrize("generator", ["python", "pydantic", "django", "sql"])
def test_generate_output_rejects_v3(generator):
    payload = _diagram(_v3_model(), generator=generator)
    _assert_legacy_rejected(_post("/besser_api/generate-output", json=payload))


def test_generate_output_rejects_v3_agent():
    payload = _diagram(_v3_model("AgentDiagram"), generator="agent")
    _assert_legacy_rejected(_post("/besser_api/generate-output", json=payload))


def test_deploy_app_rejects_v3():
    payload = _diagram(_v3_model(), generator="django")
    _assert_legacy_rejected(_post("/besser_api/deploy-app", json=payload))


def test_transform_agent_model_json_rejects_v3():
    _assert_legacy_rejected(
        _post("/besser_api/transform-agent-model-json", json=_diagram(_v3_model("AgentDiagram")))
    )


def test_object_diagram_with_v3_reference_rejected():
    payload = _diagram(
        {"version": "4.0.0", "type": "ObjectDiagram", "nodes": [], "edges": []},
        referenceDiagramData={"title": "Ref", "model": _v3_model()},
    )
    _assert_legacy_rejected(_post("/besser_api/export-buml", json=payload))


def test_v4_single_diagram_endpoints_unaffected():
    payload = _diagram(_v4_class_model(), title="Library")
    assert _post("/besser_api/export-buml", json=payload).status_code == 200
    validated = _post("/besser_api/validate-diagram", json=payload)
    assert validated.status_code == 200
    assert validated.json()["isValid"] is True
    generated = _post("/besser_api/generate-output", json={**payload, "generator": "python"})
    assert generated.status_code == 200
    assert "class " in generated.text


def test_quantum_generation_unaffected():
    payload = _diagram(_quantum_model(), title="Circuit", generator="qiskit")
    response = _post("/besser_api/generate-output", json=payload)
    assert LEGACY_MARKER not in response.text


# ---------------------------------------------------------------------------
# Project endpoints
# ---------------------------------------------------------------------------

def test_export_project_as_buml_rejects_v3():
    _assert_legacy_rejected(_post("/besser_api/export-project-as-buml", json=_project(_v3_model())))


@pytest.mark.parametrize("generator", ["python", "web_app", "backend"])
def test_generate_output_from_project_rejects_v3(generator):
    payload = _project(_v3_model(), generator=generator)
    _assert_legacy_rejected(_post("/besser_api/generate-output-from-project", json=payload))


def test_v4_project_with_gui_and_quantum_unaffected():
    exported = _post("/besser_api/export-project-as-buml", json=_project(_v4_class_model()))
    assert exported.status_code == 200, exported.text
    generated = _post(
        "/besser_api/generate-output-from-project",
        json=_project(_v4_class_model(), generator="python"),
    )
    assert generated.status_code == 200, generated.text


@pytest.mark.parametrize("endpoint", ["/besser_api/spec-driven/generate", "/besser_api/spec-driven/preview"])
def test_spec_driven_rejects_v3_project(endpoint):
    body = {
        "project": _project(_v3_model()),
        "instructions": "Build a library app.",
        "api_key": "sk-test",
    }
    _assert_legacy_rejected(_post(endpoint, json=body))


# ---------------------------------------------------------------------------
# Agent simulator, personalization, GitHub deploy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("endpoint", ["/besser_api/simulation/validate", "/besser_api/simulation/sessions"])
def test_simulation_rejects_v3_agent(endpoint):
    _assert_legacy_rejected(_post(endpoint, json={"title": "Agent", "model": _v3_model("AgentDiagram")}))


def test_personalize_gui_page_rejects_v3_user_profile():
    body = {"guiPage": {"components": [], "css": []}, "userProfileModel": _v3_model("UserDiagram")}
    _assert_legacy_rejected(_post("/besser_api/personalize-gui-page", json=body))


@pytest.mark.parametrize("endpoint", [
    "/besser_api/recommend-agent-config-mapping",
    "/besser_api/recommend-agent-config-llm",
])
def test_agent_config_recommendation_rejects_v3_user_profile(monkeypatch, endpoint):
    from besser.utilities.web_modeling_editor.backend.routers import generation_router as gr
    monkeypatch.setattr(gr, "get_user_token", lambda _session: "fake-token")
    _assert_legacy_rejected(_post(
        endpoint,
        json={"userProfileModel": _v3_model("UserDiagram")},
        headers={"X-GitHub-Session": "test-session"},
    ))


def test_github_deploy_webapp_rejects_v3_before_calling_github(monkeypatch):
    from besser.utilities.web_modeling_editor.backend.services.deployment import github_deploy_api as gda

    monkeypatch.setattr(gda, "get_user_token", lambda _session: "fake-token")

    def _no_github(_token):
        raise AssertionError("GitHub must not be contacted for a legacy payload")

    monkeypatch.setattr(gda, "create_github_service", _no_github)
    body = copy.deepcopy(_project(_v3_model()))
    body["deploy_config"] = {"repo_name": "legacy-app"}
    _assert_legacy_rejected(_post(
        "/besser_api/github/deploy-webapp", json=body, headers={"X-GitHub-Session": "test-session"},
    ))


def test_github_project_save_skips_buml_export_of_legacy_diagrams():
    """Saving to GitHub is storage: a v3 project is stored, but no empty B-UML is written for it."""
    from besser.utilities.web_modeling_editor.backend.services.deployment import github_deploy_api as gda

    pushed = []

    class _FakeGitHub:
        async def create_or_update_file(self, **kwargs):
            pushed.append(kwargs["file_path"])

    for class_model, expected in ((_v3_model(), []), (_v4_class_model(), ["buml/domain_model.py"])):
        pushed.clear()
        asyncio.run(gda._export_buml_files_to_repo(
            _FakeGitHub(), "owner", "repo", "main", "project.json",
            _project(class_model), "save",
        ))
        assert pushed == expected
