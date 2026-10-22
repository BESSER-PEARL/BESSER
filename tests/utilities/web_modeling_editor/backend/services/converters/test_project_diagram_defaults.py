"""Every imported project gets one slot and one active index per diagram type.

``project_to_json`` fills in an empty placeholder for each diagram type the
BUML file does not define and sets ``currentDiagramIndices`` for it. The multi-
diagram path and the single-diagram fallback share one list of types, so a
project export holding a Component or Deployment diagram re-imports with an
active index for it, and a project without one still gets a placeholder.
"""

import pytest

from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural import Class, DomainModel, Metadata, Property, StringType
from besser.BUML.metamodel.uml_component import Component, ComponentModel
from besser.BUML.metamodel.uml_deployment import DeploymentModel, Node
from besser.utilities.buml_code_builder.project_builder import project_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.project_converter import (
    DIAGRAM_DEFAULT_TITLES,
    project_to_json,
)

# Frontend bucket keys: the BPMN bucket is keyed "BPMN", not "BPMNDiagram".
EXPECTED_BUCKETS = {
    "BPMN" if diagram_type == "BPMNDiagram" else diagram_type
    for diagram_type in DIAGRAM_DEFAULT_TITLES
}


def _domain_model() -> DomainModel:
    book = Class(name="Book", attributes={Property(name="title", type=StringType)})
    return DomainModel(name="Library", types={book})


def _export(project: Project, tmp_path) -> str:
    out = tmp_path / "project.py"
    project_to_code(project, str(out))
    return out.read_text(encoding="utf-8")


def _assert_well_formed(imported: dict) -> None:
    diagrams = imported["diagrams"]
    indices = imported["currentDiagramIndices"]
    assert set(diagrams) == EXPECTED_BUCKETS
    # No bucket without an active index, no index without a bucket.
    assert set(indices) == set(diagrams)
    for bucket, entries in diagrams.items():
        assert entries, bucket
        assert 0 <= indices[bucket] < len(entries)


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_project_with_component_and_deployment_round_trips(tmp_path):
    component_model = ComponentModel(
        name="Services", components={Component(name="Api"), Component(name="Worker")},
    )
    deployment_model = DeploymentModel(name="Infra", nodes={Node(name="Server")})
    project = Project(
        name="CompDeploy",
        models=[_domain_model(), component_model, deployment_model],
        metadata=Metadata(description="Component and deployment"),
    )

    imported = project_to_json(_export(project, tmp_path))

    _assert_well_formed(imported)
    diagrams = imported["diagrams"]

    component = diagrams["ComponentDiagram"]
    assert len(component) == 1
    assert component[0]["model"]["type"] == "ComponentDiagram"
    component_names = {
        el["name"] for el in component[0]["model"]["elements"].values()
        if el.get("type") == "Component"
    }
    assert component_names == {"Api", "Worker"}

    deployment = diagrams["DeploymentDiagram"]
    assert len(deployment) == 1
    assert deployment[0]["model"]["type"] == "DeploymentDiagram"
    assert any(
        el.get("name") == "Server" for el in deployment[0]["model"]["elements"].values()
    )

    class_names = {
        el["name"] for el in diagrams["ClassDiagram"][0]["model"]["elements"].values()
        if el.get("type") == "Class"
    }
    assert class_names == {"Book"}


def test_project_without_component_or_deployment_gets_placeholders(tmp_path):
    project = Project(
        name="ClassOnly", models=[_domain_model()], metadata=Metadata(description="Class only"),
    )

    imported = project_to_json(_export(project, tmp_path))

    _assert_well_formed(imported)
    for diagram_type in ("ComponentDiagram", "DeploymentDiagram"):
        placeholder = imported["diagrams"][diagram_type][0]
        assert placeholder["title"] == DIAGRAM_DEFAULT_TITLES[diagram_type]
        assert placeholder["model"]["type"] == diagram_type
        assert placeholder["model"]["elements"] == {}


def test_single_diagram_fallback_uses_same_slots(tmp_path):
    out = tmp_path / "domain.py"
    from besser.utilities.buml_code_builder import domain_model_to_code
    domain_model_to_code(_domain_model(), str(out))

    imported = project_to_json(out.read_text(encoding="utf-8"))

    _assert_well_formed(imported)
    assert imported["currentDiagramType"] == "ClassDiagram"
    # Keys come out in the canonical order, not with the filled type first.
    assert list(imported["diagrams"])[:2] == ["ClassDiagram", "ObjectDiagram"]
