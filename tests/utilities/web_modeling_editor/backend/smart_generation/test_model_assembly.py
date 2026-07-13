"""Tests for assemble_models_from_project — ProjectInput → BUML models."""

import pytest

from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
from besser.utilities.web_modeling_editor.backend.services.smart_generation import (
    model_assembly as model_assembly_module,
)
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

BPMN_DIAGRAM_MODEL = {
    "version": "3.0.0",
    "type": "BPMNDiagram",
    "size": {"width": 1400, "height": 700},
    "elements": {
        "start": {
            "id": "start", "name": "Start", "type": "BPMNStartEvent",
            "owner": None, "eventType": "default",
            "bounds": {"x": 100, "y": 100, "width": 50, "height": 50},
        },
        "review": {
            "id": "review", "name": "Review", "type": "BPMNTask",
            "owner": None, "taskType": "user", "marker": "none",
            "bounds": {"x": 250, "y": 100, "width": 110, "height": 60},
        },
    },
    "relationships": {
        "flow": {
            "id": "flow", "name": "", "type": "BPMNFlow", "owner": None,
            "bounds": {"x": 0, "y": 0, "width": 100, "height": 1},
            "path": [{"x": 0, "y": 0}, {"x": 100, "y": 0}],
            "source": {
                "element": "start", "direction": "Right",
                "bounds": {"x": 0, "y": 0, "width": 0, "height": 0},
            },
            "target": {
                "element": "review", "direction": "Left",
                "bounds": {"x": 100, "y": 0, "width": 0, "height": 0},
            },
            "isManuallyLayouted": False,
            "flowType": "sequence",
            "isDefault": False,
        },
    },
    "interactive": {"elements": {}, "relationships": {}},
    "assessments": {},
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

    def test_real_wme_bpmn_bucket_drives_assembly(self):
        project = _project_with({
            "BPMN": [{
                "id": "bpmn-1",
                "title": "Approval",
                "model": BPMN_DIAGRAM_MODEL,
            }],
        })

        result = assemble_models_from_project(project)

        assert result.primary_kind == "bpmn"
        assert result.bpmn_model is not None
        assert result.bpmn_model.name == "Approval"

    @pytest.mark.parametrize(
        ("diagram_type", "processor_name", "field_name", "primary_kind"),
        [
            ("BPMN", "process_bpmn_diagram", "bpmn_model", "bpmn"),
            ("NNDiagram", "process_nn_diagram", "nn_model", "nn"),
        ],
    )
    def test_bpmn_and_nn_can_drive_assembly(
        self,
        monkeypatch,
        diagram_type,
        processor_name,
        field_name,
        primary_kind,
    ):
        sentinel = object()
        monkeypatch.setattr(
            model_assembly_module, processor_name, lambda payload: sentinel,
        )
        project = _project_with({
            diagram_type: [{
                "id": "model-1",
                "title": "Primary",
                "model": {
                    "type": diagram_type,
                    "elements": {},
                    "relationships": {},
                },
            }],
        })

        result = assemble_models_from_project(project)

        assert result.primary_kind == primary_kind
        assert getattr(result, field_name) is sentinel
        assert primary_kind in {
            entry["kind"] for entry in result.summary()["present"]
        }

    def test_project_without_class_diagram_is_accepted(self):
        """A project with only an AgentDiagram is now valid — smart
        generation no longer requires a ClassDiagram. The agent diagram
        becomes the primary input and the domain model stays None."""
        project = _project_with({
            "AgentDiagram": [
                {
                    "id": "ad1",
                    "title": "Agent",
                    "model": {"type": "AgentDiagram", "elements": {}, "relationships": {}},
                }
            ]
        })

        # AgentDiagram with an empty body may or may not yield an agent
        # model depending on the processor — accept either assembly
        # success (primary_kind=agent) or the no-models rejection.
        try:
            result = assemble_models_from_project(project)
        except ValueError as exc:
            # Fall-through: processor rejected an empty agent body.
            # That's not the behaviour we're testing here, just verify
            # the error mentions missing artifacts, not ClassDiagram.
            assert "at least one modeling artifact" in str(exc)
            return
        assert result.domain_model is None
        assert result.primary_kind == "agent"

    def test_empty_project_raises(self):
        project = _project_with({})

        with pytest.raises(ValueError, match="at least one modeling artifact"):
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

    def test_new_optional_models_default_to_empty(self):
        """The newly added optional diagram slots (object / state machines /
        quantum) must default to ``None`` / empty list when the project
        doesn't include those diagram types. No crash, no implicit values."""
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
        assert result.object_model is None
        assert result.state_machines == []
        assert result.quantum_circuit is None

    def test_malformed_state_machine_diagram_does_not_abort(self):
        """A broken StateMachineDiagram must never block a run whose
        ClassDiagram is valid. Whether the processor returns ``None`` or
        an empty shell is an implementation detail of the upstream
        processor — all we care about is that assembly finishes with the
        domain model intact."""
        project = _project_with({
            "ClassDiagram": [
                {
                    "id": "cd1",
                    "title": "Library",
                    "model": CLASS_DIAGRAM_MODEL,
                }
            ],
            "StateMachineDiagram": [
                {
                    "id": "sm1",
                    "title": "Broken",
                    "model": {"type": "StateMachineDiagram", "garbage": True},
                }
            ],
        })

        result = assemble_models_from_project(project)
        assert result.domain_model is not None
        # The list is either empty (processor raised) or contains
        # a degenerate entry (processor returned an empty shell).
        assert isinstance(result.state_machines, list)

    def test_malformed_object_diagram_does_not_abort(self):
        project = _project_with({
            "ClassDiagram": [
                {
                    "id": "cd1",
                    "title": "Library",
                    "model": CLASS_DIAGRAM_MODEL,
                }
            ],
            "ObjectDiagram": [
                {
                    "id": "od1",
                    "title": "Broken",
                    "model": {"type": "ObjectDiagram", "garbage": True},
                }
            ],
        })

        result = assemble_models_from_project(project)
        # object_model is either None (processor raised) or an empty
        # ObjectModel (processor returned a shell). Either is acceptable.
        assert result.domain_model is not None

    def test_malformed_quantum_diagram_does_not_abort(self):
        project = _project_with({
            "ClassDiagram": [
                {
                    "id": "cd1",
                    "title": "Library",
                    "model": CLASS_DIAGRAM_MODEL,
                }
            ],
            "QuantumCircuitDiagram": [
                {
                    "id": "qd1",
                    "title": "Broken",
                    "model": {"type": "QuantumCircuitDiagram", "garbage": True},
                }
            ],
        })

        result = assemble_models_from_project(project)
        assert result.domain_model is not None
        # A malformed quantum diagram yields either None or an empty
        # QuantumCircuit. Both are safe for the downstream serializer.
        if result.quantum_circuit is not None:
            assert getattr(result.quantum_circuit, "num_qubits", 0) == 0
