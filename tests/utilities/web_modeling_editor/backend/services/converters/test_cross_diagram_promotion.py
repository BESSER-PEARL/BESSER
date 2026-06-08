"""Tests for project-level cross-diagram-reference promotion (02b-...).

Covers the eight unit cases for ``_validate_cross_diagram_references``
plus one integration test through ``json_to_buml_project``.
"""

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
)
from besser.BUML.metamodel.uml_component import (
    AgenticComponent,
    Component,
    ComponentModel,
)
from besser.BUML.metamodel.uml_deployment import (
    Artifact,
    DeploymentModel,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.project_converter import (
    _validate_cross_diagram_references,
    json_to_buml_project,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_component_model_with_id(component_id: str, name: str = "C") -> ComponentModel:
    """Build a ComponentModel holding one Component whose WME id is set."""
    c = Component(name=name)
    c.layout = {"id": component_id}
    return ComponentModel(name="cm", components={c})


def _make_deployment_model_with_artifact(
    manifests: list, artifact_name: str = "A",
) -> DeploymentModel:
    """Build a DeploymentModel holding one root-level Artifact with manifests."""
    a = Artifact(name=artifact_name, manifests=manifests)
    return DeploymentModel(name="dm", artifacts={a})


def _make_class_model_with_id(class_id: str, name: str = "Book") -> DomainModel:
    """Build a DomainModel holding one Class plus the WME-id side-map the class
    processor stashes (``_wme_class_index``), so realizes resolves by id (04-... D1)."""
    cls = Class(name=name)
    dm = DomainModel(name="dm_class", types={cls})
    dm._wme_class_index = {class_id: cls}
    return dm


# ---------------------------------------------------------------------------
# Unit tests — _validate_cross_diagram_references
# ---------------------------------------------------------------------------

class TestValidateCrossDiagramReferences:
    def test_empty_input(self):
        """No models at all → no errors and no warnings."""
        result = _validate_cross_diagram_references()
        assert result == {"errors": [], "warnings": []}

    def test_component_only_no_deployment(self):
        """Component diagram alone — Artifact.manifests path skipped."""
        cm = _make_component_model_with_id("uuid-1")
        result = _validate_cross_diagram_references(component_models=[cm])
        assert result["errors"] == []

    def test_deployment_only_with_bogus_manifest(self):
        """Deployment with manifests but no Component peer → no errors (skipped)."""
        dm = _make_deployment_model_with_artifact(manifests=["uuid-bogus"])
        result = _validate_cross_diagram_references(deployment_models=[dm])
        assert result["errors"] == []

    def test_valid_artifact_manifests(self):
        """Component has UUID, Artifact manifests that UUID → no errors."""
        cm = _make_component_model_with_id("uuid-1")
        dm = _make_deployment_model_with_artifact(manifests=["uuid-1"])
        result = _validate_cross_diagram_references(
            component_models=[cm], deployment_models=[dm],
        )
        assert result["errors"] == []

    def test_dangling_artifact_manifests(self):
        """Component has UUID, Artifact manifests a different UUID → error."""
        cm = _make_component_model_with_id("uuid-1")
        dm = _make_deployment_model_with_artifact(
            manifests=["uuid-bogus"], artifact_name="MyArtifact",
        )
        result = _validate_cross_diagram_references(
            component_models=[cm], deployment_models=[dm],
        )
        assert len(result["errors"]) == 1
        msg = result["errors"][0]
        assert "MyArtifact" in msg
        assert "uuid-bogus" in msg

    def test_valid_component_realizes(self):
        """Class diagram has a Class with WME id 'cls-1';
        Component.realizes=['cls-1'] → no errors (id-keyed, 04-... D4)."""
        c = Component(name="C")
        c.realizes = ["cls-1"]
        cm = ComponentModel(name="cm", components={c})
        dm_class = _make_class_model_with_id("cls-1")
        result = _validate_cross_diagram_references(
            component_models=[cm], class_models=[dm_class],
        )
        assert result["errors"] == []

    def test_dangling_component_realizes(self):
        """Class diagram has a Class with id 'cls-1';
        Component.realizes=['cls-bogus'] → error."""
        c = Component(name="MyComp")
        c.realizes = ["cls-bogus"]
        cm = ComponentModel(name="cm", components={c})
        dm_class = _make_class_model_with_id("cls-1")
        result = _validate_cross_diagram_references(
            component_models=[cm], class_models=[dm_class],
        )
        assert len(result["errors"]) == 1
        msg = result["errors"][0]
        assert "MyComp" in msg
        assert "cls-bogus" in msg

    def test_realizes_resolves_by_id_not_name(self):
        """Collision-safety (04-... D4): a Class named 'Book' lives in two
        diagrams under different WME ids. A Component realizing diagram-A's id
        resolves; the same-named Class in diagram B does NOT satisfy a wrong id."""
        c_ok = Component(name="Realizer")
        c_ok.realizes = ["book-A"]
        c_bad = Component(name="MisRealizer")
        c_bad.realizes = ["book-B-typo"]
        cm = ComponentModel(name="cm", components={c_ok, c_bad})
        dm_a = _make_class_model_with_id("book-A", name="Book")
        dm_b = _make_class_model_with_id("book-B", name="Book")
        result = _validate_cross_diagram_references(
            component_models=[cm], class_models=[dm_a, dm_b],
        )
        # 'book-A' resolves; 'book-B-typo' does not (even though a 'Book' exists).
        assert len(result["errors"]) == 1
        assert "book-B-typo" in result["errors"][0]
        assert "MisRealizer" in result["errors"][0]

    def test_multiple_component_diagrams_unioned(self):
        """IDs split across two Component diagrams — Artifact.manifests
        references IDs from both → no errors."""
        cm1 = _make_component_model_with_id("uuid-1", name="C1")
        cm2 = _make_component_model_with_id("uuid-2", name="C2")
        dm = _make_deployment_model_with_artifact(
            manifests=["uuid-1", "uuid-2"],
        )
        result = _validate_cross_diagram_references(
            component_models=[cm1, cm2], deployment_models=[dm],
        )
        assert result["errors"] == []

    # -- process_model_refs (item 2): agentic Component -> BPMN diagram --------

    def test_valid_process_model_refs(self):
        """AgenticComponent.process_model_refs=[bpmn diagram uuid] resolves to
        the BPMNModel in the diagram index → no errors."""
        from besser.BUML.metamodel.bpmn import BPMNModel, Process

        comp = AgenticComponent(name="Agent1")
        comp.process_model_refs = ["bpmn-diag-1"]
        cm = ComponentModel(name="cm", components={comp})
        bpmn = BPMNModel(name="b", processes={Process(name="P")})
        result = _validate_cross_diagram_references(
            component_models=[cm], diagram_index={"bpmn-diag-1": bpmn},
        )
        assert result["errors"] == []

    def test_dangling_process_model_refs(self):
        """A processModelRefs id absent from the index → error."""
        from besser.BUML.metamodel.bpmn import BPMNModel, Process

        comp = AgenticComponent(name="LonelyAgent")
        comp.process_model_refs = ["missing-diag"]
        cm = ComponentModel(name="cm", components={comp})
        bpmn = BPMNModel(name="b", processes={Process(name="P")})
        result = _validate_cross_diagram_references(
            component_models=[cm], diagram_index={"bpmn-diag-1": bpmn},
        )
        assert len(result["errors"]) == 1
        msg = result["errors"][0]
        assert "LonelyAgent" in msg
        assert "missing-diag" in msg

    def test_process_model_refs_no_index_noop(self):
        """No diagram index (no BPMN/Agent peers) → silently skipped (D5)."""
        comp = AgenticComponent(name="Agent1")
        comp.process_model_refs = ["whatever"]
        cm = ComponentModel(name="cm", components={comp})
        result = _validate_cross_diagram_references(component_models=[cm])
        assert result["errors"] == []

    # -- agent_diagram_ref (item 4): agentic task -> Agent --------------------

    def test_valid_agent_diagram_ref(self):
        """An agentic task's agent_diagram_ref resolves to the Agent in the
        diagram index → no errors."""
        from besser.BUML.metamodel.bpmn import AgenticTask, BPMNModel, Process
        from besser.BUML.metamodel.state_machine.agent import Agent

        task = AgenticTask(name="DoThing", agent_diagram_ref="agent-diag-1")
        bpmn = BPMNModel(name="b", processes={Process(name="P", flow_nodes={task})})
        agent = Agent(name="MyAgent")
        result = _validate_cross_diagram_references(
            bpmn_models=[bpmn], diagram_index={"agent-diag-1": agent},
        )
        assert result["errors"] == []

    def test_dangling_agent_diagram_ref(self):
        """An agent_diagram_ref id absent from the index → error."""
        from besser.BUML.metamodel.bpmn import AgenticTask, BPMNModel, Process
        from besser.BUML.metamodel.state_machine.agent import Agent

        task = AgenticTask(name="DoThing", agent_diagram_ref="missing-agent")
        bpmn = BPMNModel(name="b", processes={Process(name="P", flow_nodes={task})})
        agent = Agent(name="MyAgent")
        result = _validate_cross_diagram_references(
            bpmn_models=[bpmn], diagram_index={"agent-diag-1": agent},
        )
        assert len(result["errors"]) == 1
        msg = result["errors"][0]
        assert "DoThing" in msg
        assert "missing-agent" in msg


# ---------------------------------------------------------------------------
# Integration test — json_to_buml_project wire-up
# ---------------------------------------------------------------------------

class TestJsonToBumlProjectWireUp:
    def test_dangling_manifest_surfaces_on_project(self):
        """End-to-end: a Project with one Component diagram (UUID c1) +
        one Deployment diagram (Artifact manifesting bogus UUID) →
        ``project._cross_diagram_errors`` carries the dangling-ID error."""
        from besser.utilities.web_modeling_editor.backend.models import (
            ProjectInput,
        )

        project_input = ProjectInput.model_validate({
            "id": "p1",
            "type": "Project",
            "schemaVersion": 3,
            "name": "TestProject",
            "description": "",
            "owner": "tester",
            "createdAt": "2026-05-19T00:00:00Z",
            "currentDiagramType": "ComponentDiagram",
            "currentDiagramIndices": {
                "ClassDiagram": 0,
                "ObjectDiagram": 0,
                "StateMachineDiagram": 0,
                "AgentDiagram": 0,
                "ComponentDiagram": 0,
                "DeploymentDiagram": 0,
                "GUINoCodeDiagram": 0,
                "QuantumCircuitDiagram": 0,
            },
            "diagrams": {
                "ComponentDiagram": [{
                    "id": "comp-diag",
                    "title": "Component Diagram",
                    "model": {
                        "version": "3.0.0",
                        "type": "ComponentDiagram",
                        "size": {"width": 800, "height": 600},
                        "elements": {
                            "c1": {
                                "id": "c1", "name": "Frontend",
                                "type": "Component", "owner": None,
                                "bounds": {
                                    "x": 0, "y": 0, "width": 160, "height": 100,
                                },
                            },
                        },
                        "relationships": {},
                        "interactive": {"elements": {}, "relationships": {}},
                        "assessments": {},
                    },
                }],
                "DeploymentDiagram": [{
                    "id": "dep-diag",
                    "title": "Deployment Diagram",
                    "model": {
                        "version": "3.0.0",
                        "type": "DeploymentDiagram",
                        "size": {"width": 800, "height": 600},
                        "elements": {
                            "a1": {
                                "id": "a1",
                                "name": "Orphan",
                                "type": "DeploymentComponent",
                                "owner": None,
                                "bounds": {
                                    "x": 0, "y": 0, "width": 160, "height": 100,
                                },
                            },
                        },
                        "relationships": {},
                        "interactive": {"elements": {}, "relationships": {}},
                        "assessments": {},
                    },
                }],
            },
            "settings": {
                "defaultDiagramType": "ComponentDiagram",
                "autoSave": True,
                "collaborationEnabled": False,
            },
        })

        # The DeploymentComponent "a1" synthesises as Artifact(manifests=["a1"]).
        # "a1" is NOT a Component id (the only Component id is "c1") — that's
        # the dangling cross-ref.
        project = json_to_buml_project(project_input)
        assert hasattr(project, "_cross_diagram_errors")
        errors = project._cross_diagram_errors["errors"]
        assert len(errors) == 1
        assert "a1" in errors[0]
        assert "Orphan" in errors[0]
