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


def _kg_model() -> Dict[str, Any]:
    # Cytoscape-backed: nodes/edges like v4 UML, but its own format.
    return {
        "type": "KnowledgeGraphDiagram",
        "version": "1.0.0",
        "nodes": [{"id": "n1", "nodeType": "class", "label": "Person"}],
        "edges": [],
    }


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
            "KnowledgeGraphDiagram": [{"id": "kg-1", "title": "Graph", "model": _kg_model()}],
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
        _kg_model(),
        # GUI / quantum / KG are exempt by type even with a v3-looking body.
        {"type": "GUINoCodeDiagram", "version": "3.1.0", "elements": {}},
        {"type": "QuantumCircuitDiagram", "version": "3.0.0", "elements": {}},
        {"type": "KnowledgeGraphDiagram", "version": "3.0.0", "elements": {}},
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
        assert not is_legacy_v3_model({"elements": {}}, "KnowledgeGraphDiagram")

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
        DiagramInput(**_diagram(_kg_model()))

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


# ---------------------------------------------------------------------------
# Models nested in an agent diagram's config
# ---------------------------------------------------------------------------

def _v4_agent_model() -> Dict[str, Any]:
    with open(os.path.join(FIXTURES_V4, "agent_diagram_basic.json"), encoding="utf-8") as f:
        return json.load(f)["model"]


def _v4_user_model() -> Dict[str, Any]:
    return {"version": "4.0.0", "type": "UserDiagram", "nodes": [], "edges": []}


@pytest.mark.parametrize("entry", [
    {"user_profile": _v3_model("UserDiagram")},
    {"user_profile": _v4_user_model(), "agent_model": _v3_model("AgentDiagram")},
], ids=["v3-user-profile", "v3-agent-model"])
def test_generate_output_rejects_v3_personalization_mapping(entry):
    payload = _diagram(
        _v4_agent_model(), generator="agent",
        config={"personalizationMapping": [{"name": "variant", **entry}]},
    )
    response = _post("/besser_api/generate-output", json=payload)
    _assert_legacy_rejected(response)
    assert "personalizationMapping[0]" in response.json()["detail"]


def test_normalize_personalization_mapping_rejects_v3_before_conversion():
    from besser.utilities.web_modeling_editor.backend.services.utils.agent_generation_utils import (
        normalize_personalization_mapping,
    )

    def _must_not_run(_payload):
        raise AssertionError("a v3 user profile must be rejected before conversion")

    config = {"personalizationMapping": [{"user_profile": _v3_model("UserDiagram")}]}
    with pytest.raises(LegacyDiagramFormatError):
        normalize_personalization_mapping(config, {"model": _v4_agent_model()}, _must_not_run)


def test_transform_agent_model_json_rejects_v3_user_profile_in_config():
    payload = _diagram(
        _v4_agent_model(), config={"userProfileModel": _v3_model("UserDiagram")},
    )
    response = _post("/besser_api/transform-agent-model-json", json=payload)
    _assert_legacy_rejected(response)
    assert "config.userProfileModel" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Malformed v4 payloads are 400s, and the converters refuse v3 themselves
# ---------------------------------------------------------------------------

def _v4(diagram_type: str, **parts) -> Dict[str, Any]:
    model = {"version": "4.0.0", "type": diagram_type, "nodes": [], "edges": []}
    model.update(parts)
    return model


_CLASS_NODE = {"id": "a", "type": "class", "data": {"name": "A"}}
_MALFORMED = {
    "nodes_dict": {"nodes": {"a": {}}},
    "nodes_strs": {"nodes": ["x", 1, None], "edges": ["y"]},
    "node_data_list": {"nodes": [{"id": "a", "type": "class", "data": [1]}]},
    "edge_data_list": {"nodes": [_CLASS_NODE], "edges": [
        {"id": "e", "type": "ClassBidirectional", "source": "a", "target": "a", "data": [1]}]},
    "edge_ids_unhashable": {"nodes": [{"id": ["a"], "type": "class", "data": {"name": "A"}}],
                            "edges": [{"id": {}, "type": "ClassInheritance", "source": ["a"],
                                       "target": {"x": 1}}]},
}


@pytest.mark.parametrize("diagram_type", ["ClassDiagram", "StateMachineDiagram", "AgentDiagram"])
@pytest.mark.parametrize("case", sorted(_MALFORMED))
def test_malformed_v4_payload_is_a_400(diagram_type, case):
    body = {"title": "T", "model": _v4(diagram_type, **_MALFORMED[case])}
    for url, extra in (("/besser_api/export-buml", {}),
                       ("/besser_api/generate-output", {"generator": "python"})):
        response = _post(url, json={**body, **extra})
        assert response.status_code == 400, (url, response.status_code, response.text[:200])


@pytest.mark.parametrize("process, diagram_type", [
    ("process_class_diagram", "ClassDiagram"),
    ("process_state_machine", "StateMachineDiagram"),
    ("process_agent_diagram", "AgentDiagram"),
    ("process_bpmn_diagram", "BPMNDiagram"),
    ("process_component_diagram", "ComponentDiagram"),
    ("process_deployment_diagram", "DeploymentDiagram"),
    ("process_nn_diagram", "NNDiagram"),
])
def test_converters_refuse_v3_json(process, diagram_type):
    from besser.utilities.web_modeling_editor.backend.services import converters
    with pytest.raises(ConversionError):
        getattr(converters, process)({"title": "T", "model": _v3_model(diagram_type)})


def test_object_converter_refuses_v3_json():
    from besser.BUML.metamodel.structural import Class, DomainModel
    from besser.utilities.web_modeling_editor.backend.services.converters import process_object_diagram
    with pytest.raises(ConversionError):
        process_object_diagram({"title": "T", "model": _v3_model("ObjectDiagram")},
                               DomainModel(name="D", types={Class(name="A")}))


_CLASS_B = {"id": "b", "type": "class", "data": {"name": "B"}}
_MALFORMED_CLASS = {
    "attrs_not_list": ({"nodes": [{"id": "a", "type": "class", "data": {
        "name": "A", "attributes": "x", "methods": {"a": 1}}}]}, "must be a list of objects"),
    "methods_not_list": ({"nodes": [{"id": "a", "type": "class", "data": {
        "name": "A", "methods": {"a": 1}}}]}, "must be a list of objects"),
    "attribute_row_not_object": ({"nodes": [{"id": "a", "type": "class", "data": {
        "name": "A", "attributes": [1]}}]}, "must be a list of objects"),
    "inherit_cycle": ({"nodes": [_CLASS_NODE, _CLASS_B], "edges": [
        {"id": "e1", "type": "ClassInheritance", "source": "a", "target": "b", "data": {}},
        {"id": "e2", "type": "ClassInheritance", "source": "b", "target": "a", "data": {}}]},
        "Inheritance cycle"),
}


# Class-diagram content the v4 shape check lets through; a StateMachineDiagram
# body sent to a class generator takes the same converter path.
@pytest.mark.parametrize("diagram_type", ["ClassDiagram", "StateMachineDiagram"])
@pytest.mark.parametrize("case", sorted(_MALFORMED_CLASS))
def test_malformed_class_content_is_a_400(diagram_type, case):
    parts, message = _MALFORMED_CLASS[case]
    body = {"title": "T", "model": _v4(diagram_type, **parts)}
    urls = [("/besser_api/generate-output", {"generator": "python"})]
    if diagram_type == "ClassDiagram":
        urls.append(("/besser_api/export-buml", {}))
    for url, extra in urls:
        response = _post(url, json={**body, **extra})
        assert response.status_code == 400, (url, response.status_code, response.text[:200])
        assert message in response.json()["detail"]
