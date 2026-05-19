"""Tests for project-level cross-diagram-reference promotion (02b-...).

Covers the eight unit cases for ``_validate_cross_diagram_references``
plus one integration test through ``json_to_buml_project``.
"""

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
)
from besser.BUML.metamodel.uml_component import (
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
        """Class diagram has Book, Component.realizes=['Book'] → no errors."""
        c = Component(name="C")
        c.realizes = ["Book"]
        cm = ComponentModel(name="cm", components={c})
        book = Class(name="Book")
        dm_class = DomainModel(name="dm_class", types={book})
        result = _validate_cross_diagram_references(
            component_models=[cm], class_models=[dm_class],
        )
        assert result["errors"] == []

    def test_dangling_component_realizes(self):
        """Class diagram has Book, Component.realizes=['Magazine'] → error."""
        c = Component(name="MyComp")
        c.realizes = ["Magazine"]
        cm = ComponentModel(name="cm", components={c})
        book = Class(name="Book")
        dm_class = DomainModel(name="dm_class", types={book})
        result = _validate_cross_diagram_references(
            component_models=[cm], class_models=[dm_class],
        )
        assert len(result["errors"]) == 1
        msg = result["errors"][0]
        assert "MyComp" in msg
        assert "Magazine" in msg

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
