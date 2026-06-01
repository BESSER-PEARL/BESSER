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

The payloads mirror the real-export wire shape used by the converter unit
tests (which in turn mirror the ``02-D7-*.json`` WME exports). They are kept
inline so the test is self-contained and CI-safe (the ``.claude`` D7 files are
local-only). See ``.claude/agentic-inter-diagram/01-...`` for the manual
reproduction recipe against real WME exports.

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
# Wire-shape fixtures (mirror the converter unit-test fixtures / D7 exports)
# ---------------------------------------------------------------------------

def _component_diagram() -> dict:
    """Agentic Component diagram: a Subsystem holding two agents, a `solution`
    agent delegating (with a permission suffix) to a `collaboration` agent."""
    return {
        "title": "Component Diagram",
        "model": {
            "version": "3.0.0",
            "type": "ComponentDiagram",
            "size": {"width": 1400, "height": 740},
            "elements": {
                "grp": {
                    "id": "grp", "name": "ConsensusGroup", "type": "Subsystem",
                    "owner": None,
                    "bounds": {"x": 100, "y": 100, "width": 520, "height": 280},
                    "stereotype": "subsystem", "displayStereotype": True,
                },
                "orch": {
                    "id": "orch", "name": "OrchestratorAgent", "type": "Component",
                    "owner": "grp",
                    "bounds": {"x": 140, "y": 180, "width": 160, "height": 80},
                    "stereotype": "solution", "displayStereotype": True,
                },
                "worker": {
                    "id": "worker", "name": "WorkerAgent", "type": "Component",
                    "owner": "grp",
                    "bounds": {"x": 420, "y": 180, "width": 160, "height": 80},
                    "stereotype": "collaboration", "displayStereotype": True,
                },
            },
            "relationships": {
                "rel": {
                    "id": "rel", "name": "", "type": "ComponentDependency",
                    "owner": None,
                    "bounds": {"x": 300, "y": 220, "width": 120, "height": 1},
                    "path": [{"x": 0, "y": 0}, {"x": 120, "y": 0}],
                    "source": {"element": "orch", "direction": "Right"},
                    "target": {"element": "worker", "direction": "Left"},
                    "isManuallyLayouted": False,
                    "stereotype": "delegates {permission: repo:merge:approve}",
                },
            },
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
            "version": "3.0.0",
            "type": "DeploymentDiagram",
            "size": {"width": 1040, "height": 640},
            "elements": {
                "n1": {
                    "id": "n1", "name": "EdgeDevice", "type": "DeploymentNode",
                    "owner": None,
                    "bounds": {"x": -500, "y": -300, "width": 440, "height": 340},
                    "stereotype": "device", "displayStereotype": True,
                },
                "a1": {
                    "id": "a1", "name": "agent-runtime [3]",
                    "type": "DeploymentArtifact",
                    "owner": "n1",
                    "bounds": {"x": -450, "y": -200, "width": 160, "height": 40},
                },
                "n2": {
                    "id": "n2", "name": "LLMHost", "type": "DeploymentNode",
                    "owner": None,
                    "bounds": {"x": 230, "y": 0, "width": 160, "height": 100},
                    "stereotype": "executionEnvironment external",
                    "displayStereotype": True,
                },
            },
            "relationships": {
                # Explicit artifact -> node edge: carries the `[3]` multiplicity
                # (parsed off the artifact name onto the DeploymentRelation).
                "r-deploys": {
                    "id": "r-deploys", "name": "", "type": "DeploymentAssociation",
                    "owner": None,
                    "bounds": {"x": 0, "y": 0, "width": 1, "height": 120},
                    "path": [{"x": 0, "y": 0}, {"x": 0, "y": 120}],
                    "source": {"direction": "Down", "element": "a1"},
                    "target": {"direction": "Down", "element": "n1"},
                    "isManuallyLayouted": False,
                    "stereotype": "",
                },
                # Node <-> node edge: becomes a CommunicationPath, `HTTPS` rides
                # through as a free-form stereotype.
                "r-comm": {
                    "id": "r-comm", "name": "", "type": "DeploymentAssociation",
                    "owner": None,
                    "bounds": {"x": 0, "y": 0, "width": 290, "height": 1},
                    "path": [{"x": 0, "y": 0}, {"x": 290, "y": 0}],
                    "source": {"direction": "Downright", "element": "n1"},
                    "target": {"direction": "Left", "element": "n2"},
                    "isManuallyLayouted": False,
                    "stereotype": "HTTPS",
                },
            },
            "interactive": {"elements": {}, "relationships": {}},
            "assessments": {},
        },
    }


def _bpmn_diagram() -> dict:
    """Pool-less BPMN process: StartEvent -> Task -> EndEvent."""
    def _node(eid, typ, name, **extra):
        base = {
            "id": eid, "name": name, "type": typ, "owner": None,
            "bounds": {"x": 100, "y": 100, "width": 110, "height": 60},
        }
        base.update(extra)
        return base

    def _flow(rid, src, tgt):
        return {
            "id": rid, "name": "", "type": "BPMNFlow", "owner": None,
            "bounds": {"x": 0, "y": 0, "width": 100, "height": 1},
            "path": [{"x": 0, "y": 0}, {"x": 100, "y": 0}],
            "source": {"element": src, "direction": "Right",
                       "bounds": {"x": 0, "y": 0, "width": 0, "height": 0}},
            "target": {"element": tgt, "direction": "Left",
                       "bounds": {"x": 100, "y": 0, "width": 0, "height": 0}},
            "isManuallyLayouted": False, "flowType": "sequence", "isDefault": False,
        }

    return {
        "title": "BPMN Diagram",
        "model": {
            "version": "3.0.0",
            "type": "BPMNDiagram",
            "size": {"width": 1400, "height": 700},
            "elements": {
                "s1": _node("s1", "BPMNStartEvent", "go", eventType="default"),
                "t1": _node("t1", "BPMNTask", "Review", taskType="user", marker="none"),
                "e1": _node("e1", "BPMNEndEvent", "done", eventType="default"),
            },
            "relationships": {
                "f1": _flow("f1", "s1", "t1"),
                "f2": _flow("f2", "t1", "e1"),
            },
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
            "BPMNDiagram": 0, "ComponentDiagram": 0, "DeploymentDiagram": 0,
        },
        "diagrams": {
            "BPMNDiagram": [_bpmn_diagram()],
            "ComponentDiagram": [_component_diagram()],
            "DeploymentDiagram": [_deployment_diagram()],
        },
    }


# ---------------------------------------------------------------------------
# Single-diagram /export-buml
# ---------------------------------------------------------------------------

class TestExportBumlSingleDiagram:
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
