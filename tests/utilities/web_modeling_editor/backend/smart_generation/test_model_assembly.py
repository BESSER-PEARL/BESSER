"""Tests for assemble_models_from_project — ProjectInput → BUML models."""

import pytest

from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
from besser.utilities.web_modeling_editor.backend.services.smart_generation.model_assembly import (
    assemble_models_from_project,
)


CLASS_DIAGRAM_MODEL = {
    "type": "ClassDiagram",
    "elements": {
        "class-1": {
            "type": "Class",
            "name": "Author",
            "attributes": ["attr-1"],
            "methods": [],
        },
        "attr-1": {
            "type": "Attribute",
            "name": "+ name: str",
        },
        "class-2": {
            "type": "Class",
            "name": "Book",
            "attributes": ["attr-2"],
            "methods": [],
        },
        "attr-2": {
            "type": "Attribute",
            "name": "+ title: str",
        },
    },
    "relationships": {
        "rel-1": {
            "type": "ClassBidirectional",
            "source": {
                "element": "class-1",
                "multiplicity": "1..*",
                "role": "authors",
            },
            "target": {
                "element": "class-2",
                "multiplicity": "0..*",
                "role": "books",
            },
        },
    },
}


def _project_with(diagrams: dict) -> ProjectInput:
    """Build a minimal ProjectInput for testing."""
    return ProjectInput(
        id="test-project",
        type="BesserProject",
        name="TestProject",
        createdAt="2026-04-15T00:00:00Z",
        diagrams=diagrams,
        currentDiagramIndices={k: 0 for k in diagrams},
    )


class TestAssembleModels:
    def test_class_only_project_succeeds(self):
        project = _project_with({
            "ClassDiagram": [
                {
                    "id": "cd1",
                    "title": "Library",
                    "model": CLASS_DIAGRAM_MODEL,
                }
            ]
        })

        result = assemble_models_from_project(project)

        assert result.domain_model is not None
        assert result.gui_model is None
        assert result.agent_model is None
        assert result.agent_config is None

    def test_project_without_class_diagram_raises(self):
        """A project with only an AgentDiagram (no class) is rejected."""
        project = _project_with({
            "AgentDiagram": [
                {
                    "id": "ad1",
                    "title": "Agent",
                    "model": {"type": "AgentDiagram", "elements": {}, "relationships": {}},
                }
            ]
        })

        with pytest.raises(ValueError, match="ClassDiagram"):
            assemble_models_from_project(project)

    def test_empty_project_raises(self):
        project = _project_with({})

        with pytest.raises(ValueError, match="ClassDiagram"):
            assemble_models_from_project(project)

    def test_domain_model_has_expected_classes(self):
        project = _project_with({
            "ClassDiagram": [
                {
                    "id": "cd1",
                    "title": "Library",
                    "model": CLASS_DIAGRAM_MODEL,
                }
            ]
        })

        result = assemble_models_from_project(project)
        class_names = {t.name for t in result.domain_model.types if hasattr(t, "name")}
        assert "Author" in class_names
        assert "Book" in class_names
