"""End-to-end export-to-BUML tests across the BPMN + Component + Deployment
metamodels that now coexist on ``feature/agentic-inter-diagram``.

These tests drive the **real FastAPI export path** (the same endpoints the
WME frontend calls), not the converter functions directly:

* ``POST /besser_api/export-buml``            -- single diagram -> BUML ``.py``
* ``POST /besser_api/export-project-as-buml`` -- multi-diagram project -> BUML ``.py``

"Well-generated" is checked the only way that actually proves it: the
returned BUML source is ``compile()``-d and ``exec()``-d in a fresh namespace,
and the reconstructed metamodel objects are asserted. The combined-project
test is the load-bearing one for the merge -- it exercises the spliced
``project_builder.py`` that emits BPMN + Component + Deployment sections into
a single executable ``project.py``.

The payloads use the v4 wire shape (``{nodes, edges}``) the editor sends. They are kept inline so the test is self-contained and CI-safe while
still exercising the same shape as real WME exports.

Async/ASGI harness mirrors ``test_api_integration.py`` (the installed
starlette/httpx versions do not support the legacy ``TestClient(app=...)``).
"""

import asyncio

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.uml_component import ComponentModel
from besser.BUML.metamodel.uml_component.agentic import (
    AgenticComponentModel,
    AgenticComponent,
)
from besser.BUML.metamodel.uml_deployment import DeploymentModel
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.project_converter import project_to_json



API = "/besser_api"
BASE_URL = "http://testserver"


# ---------------------------------------------------------------------------
# Minimal sync wrapper around httpx.AsyncClient + ASGITransport
# ---------------------------------------------------------------------------

def _post(url: str, **kwargs) -> httpx.Response:
    async def _run():
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
            return await ac.post(url, **kwargs)
    return asyncio.run(_run())


def _exec_buml(source: str) -> dict:
    """compile() + exec() generated BUML in a fresh namespace; return it.

    A SyntaxError or any runtime error fails the test loudly -- that is the
    "is it well-generated?" assertion.
    """
    compile(source, "<generated-buml>", "exec")  # explicit syntax check first
    namespace: dict = {}
    exec(source, namespace)  # noqa: S102 -- executing generated code is the point
    return namespace


def _models_of(namespace: dict, cls) -> list:
    return [v for v in namespace.values() if isinstance(v, cls)]


@pytest.fixture(autouse=True)
def _isolate_artifacts(tmp_path, monkeypatch):
    """Keep any stray generated artifact out of the repo root."""
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# Wire-shape fixtures (mirror the converter unit-test fixtures)
# ---------------------------------------------------------------------------

def _component_diagram() -> dict:
    """Agentic Component diagram: a Subsystem holding two agents, a `solution`
    agent delegating (with a permission suffix) to a `collaboration` agent."""
    return {
        "title": "Component Diagram",
        "model": {
            "version": "4.0.0",
            "type": "ComponentDiagram",
            "size": {"width": 1400, "height": 740},
            "nodes": [
                {
                    "id": "grp", "type": "componentSubsystem",
                    "position": {"x": 100, "y": 100}, "width": 520, "height": 280,
                    "data": {"name": "ConsensusGroup", "stereotype": "subsystem",
                             "isComponentSubsystemHeaderShown": True},
                },
                {
                    "id": "orch", "type": "component", "parentId": "grp",
                    "position": {"x": 40, "y": 80}, "width": 160, "height": 80,
                    "data": {"name": "OrchestratorAgent", "stereotype": "solution",
                             "isComponentHeaderShown": True},
                },
                {
                    "id": "worker", "type": "component", "parentId": "grp",
                    "position": {"x": 320, "y": 80}, "width": 160, "height": 80,
                    "data": {"name": "WorkerAgent", "stereotype": "collaboration",
                             "isComponentHeaderShown": True},
                },
            ],
            "edges": [
                {
                    "id": "rel", "type": "ComponentDependency",
                    "source": "orch", "target": "worker",
                    "sourceHandle": "right", "targetHandle": "left",
                    "data": {
                        "name": "", "isManuallyLayouted": False,
                        "points": [{"x": 300, "y": 220}, {"x": 420, "y": 220}],
                        "stereotype": "delegates {permission: repo:merge:approve}",
                    },
                },
            ],
            "interactive": {"elements": {}, "relationships": {}},
            "assessments": {},
        },
    }


def _deployment_diagram() -> dict:
    """Deployment diagram: a `device` Node holding a `[3]`-multiplicity
    Artifact, plus a free Node and a CommunicationPath between them."""
    return {
        "title": "Deployment Diagram",
        "model": {
            "version": "4.0.0",
            "type": "DeploymentDiagram",
            "size": {"width": 1040, "height": 640},
            "nodes": [
                {
                    "id": "n1", "type": "deploymentNode",
                    "position": {"x": -500, "y": -300}, "width": 440, "height": 340,
                    "data": {"name": "EdgeDevice", "stereotype": "device",
                             "isComponentHeaderShown": True},
                },
                {
                    "id": "a1", "type": "deploymentArtifact", "parentId": "n1",
                    "position": {"x": 50, "y": 100}, "width": 160, "height": 40,
                    "data": {"name": "agent-runtime [3]"},
                },
                {
                    "id": "n2", "type": "deploymentNode",
                    "position": {"x": 230, "y": 0}, "width": 160, "height": 100,
                    "data": {"name": "LLMHost", "stereotype": "executionEnvironment external",
                             "isComponentHeaderShown": True},
                },
            ],
            "edges": [
                # Explicit artifact -> node edge: carries the `[3]` multiplicity
                # (parsed off the artifact name onto the DeploymentRelation).
                {
                    "id": "r-deploys", "type": "DeploymentAssociation",
                    "source": "a1", "target": "n1",
                    "sourceHandle": "bottom", "targetHandle": "bottom",
                    "data": {"name": "", "stereotype": "", "isManuallyLayouted": False,
                             "points": [{"x": 0, "y": 0}, {"x": 0, "y": 120}]},
                },
                # Node <-> node edge: becomes a CommunicationPath, `HTTPS` rides
                # through as a free-form stereotype.
                {
                    "id": "r-comm", "type": "DeploymentAssociation",
                    "source": "n1", "target": "n2",
                    "sourceHandle": "right-bottom", "targetHandle": "left",
                    "data": {"name": "", "stereotype": "HTTPS", "isManuallyLayouted": False,
                             "points": [{"x": 0, "y": 0}, {"x": 290, "y": 0}]},
                },
            ],
            "interactive": {"elements": {}, "relationships": {}},
            "assessments": {},
        },
    }


def _bpmn_diagram() -> dict:
    """Pool-less BPMN process: StartEvent -> Task -> EndEvent."""
    def _node(eid, typ, name, **data):
        return {
            "id": eid, "type": typ,
            "position": {"x": 100, "y": 100}, "width": 110, "height": 60,
            "data": {"name": name, **data},
        }

    def _flow(rid, src, tgt):
        return {
            "id": rid, "type": "BPMNSequenceFlow", "source": src, "target": tgt,
            "sourceHandle": "right", "targetHandle": "left",
            "data": {"name": "", "isManuallyLayouted": False, "isDefault": False,
                     "points": [{"x": 0, "y": 0}, {"x": 100, "y": 0}]},
        }

    return {
        "title": "BPMN Diagram",
        "model": {
            "version": "4.0.0",
            "type": "BPMNDiagram",
            "size": {"width": 1400, "height": 700},
            "nodes": [
                _node("s1", "bpmnStartEvent", "go", eventType="default"),
                _node("t1", "bpmnTask", "Review", taskType="user", marker="none"),
                _node("e1", "bpmnEndEvent", "done", eventType="default"),
            ],
            "edges": [
                _flow("f1", "s1", "t1"),
                _flow("f2", "t1", "e1"),
            ],
            "interactive": {"elements": {}, "relationships": {}},
            "assessments": {},
        },
    }


def _project_payload() -> dict:
    """A WME project bundling all three diagram types (new-format arrays)."""
    return {
        "id": "proj-e2e",
        "type": "Project",
        "name": "InterDiagramProject",
        "description": "BPMN + Component + Deployment in one project",
        "createdAt": "2026-05-31T00:00:00Z",
        "currentDiagramType": "ComponentDiagram",
        "currentDiagramIndices": {
            "BPMN": 0, "ComponentDiagram": 0, "DeploymentDiagram": 0,
        },
        "diagrams": {
            "BPMN": [_bpmn_diagram()],
            "ComponentDiagram": [_component_diagram()],
            "DeploymentDiagram": [_deployment_diagram()],
        },
    }


# ---------------------------------------------------------------------------
# Single-diagram /export-buml
# ---------------------------------------------------------------------------

class TestExportBumlSingleDiagram:
    @pytest.mark.parametrize("diagram_type, element_type", [
        ("ComponentDiagram", "Component"),
        ("DeploymentDiagram", "DeploymentNode"),
    ])
    def test_legacy_v3_diagram_is_rejected(self, diagram_type, element_type):
        legacy = {
            "title": "Legacy",
            "model": {
                "version": "3.0.0", "type": diagram_type,
                "elements": {"x": {"id": "x", "name": "X", "type": element_type, "owner": None,
                                   "bounds": {"x": 0, "y": 0, "width": 160, "height": 100}}},
                "relationships": {},
            },
        }
        resp = _post(f"{API}/export-buml", json=legacy)
        assert resp.status_code == 400
        assert "legacy v3 editor format" in resp.json()["detail"]

    def test_component_exports_and_reconstructs(self):
        resp = _post(f"{API}/export-buml", json=_component_diagram())
        assert resp.status_code == 200, resp.text
        source = resp.text

        # The agentic profile must survive into the generated source.
        assert "AgenticComponent" in source
        assert "AgentCategory" in source
        assert "AgenticEdge" in source
        assert "repo:merge:approve" in source  # the permission scope

        ns = _exec_buml(source)
        models = _models_of(ns, ComponentModel)
        assert models, "no ComponentModel reconstructed from generated BUML"
        model = models[0]
        # Agentic diagram -> AgenticComponentModel; the two agents are agents.
        assert isinstance(model, AgenticComponentModel)
        agents = [c for c in model.components if isinstance(c, AgenticComponent)]
        assert {a.name for a in agents} == {"OrchestratorAgent", "WorkerAgent"}

    def test_deployment_exports_and_reconstructs(self):
        resp = _post(f"{API}/export-buml", json=_deployment_diagram())
        assert resp.status_code == 200, resp.text
        source = resp.text

        assert "DeploymentModel" in source
        # The `[3]` multiplicity is split off the artifact name onto the
        # DeploymentRelation: the Artifact is constructed with the bare name,
        # and the count rides as a Multiplicity on the relation. (The literal
        # "agent-runtime [3]" survives only inside the layout `original_name`
        # field, for byte-stable round-trip — so we assert the split form.)
        assert "Artifact(name='agent-runtime')" in source
        assert "Multiplicity(3, 3)" in source

        ns = _exec_buml(source)
        models = _models_of(ns, DeploymentModel)
        assert models, "no DeploymentModel reconstructed from generated BUML"
        model = models[0]
        artifact_names = {a.name for a in model.all_artifacts()}
        assert "agent-runtime" in artifact_names


# ---------------------------------------------------------------------------
# Multi-diagram /export-project-as-buml  (the load-bearing merge test)
# ---------------------------------------------------------------------------

class TestExportProjectAllThree:
    def test_project_with_three_diagram_families_reconstructs(self):
        resp = _post(f"{API}/export-project-as-buml", json=_project_payload())
        assert resp.status_code == 200, resp.text
        source = resp.text

        # All three section emitters fired (the spliced project_builder.py).
        assert "ComponentModel" in source
        assert "DeploymentModel" in source
        assert "BPMNModel" in source

        ns = _exec_buml(source)

        projects = _models_of(ns, Project)
        assert projects, "no Project reconstructed from generated BUML"

        # All three metamodel roots are reconstructable from the same file.
        assert _models_of(ns, ComponentModel), "ComponentModel missing in project export"
        assert _models_of(ns, DeploymentModel), "DeploymentModel missing in project export"

        # The exported project must also import back into both WME diagram families.
        imported_project = project_to_json(source)
        imported_diagrams = imported_project["diagrams"]

        component_model = imported_diagrams["ComponentDiagram"][0]["model"]
        deployment_model = imported_diagrams["DeploymentDiagram"][0]["model"]

        assert component_model["type"] == "ComponentDiagram"
        assert deployment_model["type"] == "DeploymentDiagram"
        assert component_model["nodes"]
        assert deployment_model["nodes"]
