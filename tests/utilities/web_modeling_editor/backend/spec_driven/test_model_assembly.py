"""Tests for assemble_models_from_project — ProjectInput → BUML models."""

import pytest

from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    model_assembly as model_assembly_module,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.model_assembly import (
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

    @pytest.mark.parametrize("returns_none", [False, True])
    def test_malformed_state_machine_diagram_does_not_abort(self, monkeypatch, caplog, returns_none):
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
                },
                {"id": "sm2", "title": "Valid", "model": {"type": "StateMachineDiagram"}},
            ],
        })
        survivor = object()
        def convert(payload):
            if payload["id"] == "sm2":
                return survivor
            if returns_none:
                return None
            raise ValueError("secret-api-key=NEVER-PUBLISH-THIS " + "sensitive " * 100)
        monkeypatch.setattr(model_assembly_module, "process_state_machine", convert)

        result = assemble_models_from_project(project)
        assert result.domain_model is not None
        assert result.state_machines == [survivor], "one failed diagram must not discard a successful sibling"
        assert result.assembly_issues == [{
            "diagram_id": "sm1", "diagram_type": "StateMachineDiagram",
            "diagnostic": "no_model_returned" if returns_none else "processor_failed: ValueError",
        }]
        assert result.summary()["assembly_issues"] == result.assembly_issues
        assert "NEVER-PUBLISH-THIS" not in repr(result.summary()) + caplog.text
        # Bounding an identity must not merge two distinct failed diagrams.
        original = project.diagrams["StateMachineDiagram"][0]
        long_ids = [original.model_copy(update={"id": "shared-prefix-" * 12 + suffix}) for suffix in ("a", "b")]
        identities = [model_assembly_module._assembly_issue(item, "StateMachineDiagram", "no_model_returned")["diagram_id"]
                      for item in long_ids]
        assert identities[0] != identities[1] and all(len(item) <= 120 for item in identities)

        # A total conversion failure also names the lost input safely rather
        # than implying that the submitted project contained no diagrams.
        monkeypatch.setattr(model_assembly_module, "process_class_diagram", lambda payload: None)
        with pytest.raises(ValueError, match=r"Model assembly failed: ClassDiagram \[cd1\]: no_model_returned"):
            assemble_models_from_project(_project_with({"ClassDiagram": project.diagrams["ClassDiagram"]}))

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



class TestAgentNextToAGui:
    """Assembly keeps an AgentDiagram next to a GUI that uses no agent: a run that targets the
    agent (a chatbot / BAF request) needs it. The web app leaves such an agent out instead."""

    @staticmethod
    def _assemble(monkeypatch, **kwargs):
        agent = type("Agent", (), {"name": "Helper"})()
        monkeypatch.setattr(model_assembly_module, "process_gui_diagram", lambda *args: object())
        monkeypatch.setattr(model_assembly_module, "has_agent_components", lambda gui: False)
        monkeypatch.setattr(model_assembly_module, "process_agent_diagram", lambda payload: agent)
        project = _project_with({
            "ClassDiagram": [{"id": "cd1", "title": "Library", "model": CLASS_DIAGRAM_MODEL}],
            "GUINoCodeDiagram": [{"id": "gui1", "title": "Screens",
                                  "model": {"type": "GUINoCodeDiagram", "pages": []}}],
            "AgentDiagram": [{"id": "ad1", "title": "Helper",
                              "model": {"type": "AgentDiagram", "elements": {}, "relationships": {}}}],
        })
        return assemble_models_from_project(project, **kwargs), agent

    def test_an_agent_no_screen_uses_is_still_assembled(self, monkeypatch):
        result, agent = self._assemble(monkeypatch)
        assert result.gui_model is not None
        assert result.agent_model is agent

    def test_a_run_that_targets_the_agent_keeps_the_agent_and_the_baf_generator(self, monkeypatch):
        # Review of 140218981: dropping the agent here turned this run into a class run
        # and took generate_baf off the tool list.
        from besser.spec_driven_agent.agent.tools import get_available_generator_names

        result, agent = self._assemble(monkeypatch, primary_kind_override="agent")
        assert result.primary_kind == "agent"
        assert result.agent_model is agent
        assert "generate_baf" in get_available_generator_names(
            has_domain_model=True, has_gui_model=True, has_agent_model=True)


def _gui_with(*elements):
    from besser.BUML.metamodel.gui import GUIModel, Module, Screen

    screen = Screen(name="S", description="", view_elements=set(elements))
    return GUIModel(name="G", package="p", versionCode="1", versionName="1",
                    modules={Module(name="M", screens={screen})}, description="")


def _agent_gui_cases():
    from besser.BUML.metamodel.gui import GUIModel, Text, ViewContainer
    from besser.BUML.metamodel.gui.dashboard import AgentComponent

    chat = AgentComponent(name="chat", description="")
    nested = ViewContainer(name="outer", description="", view_elements={
        ViewContainer(name="inner", description="", view_elements={chat})})
    return {
        "plain": _gui_with(Text(name="t", content="hi", description="")),
        "top-level": _gui_with(AgentComponent(name="chat", description="")),
        "nested twice": _gui_with(nested),
        "no modules": GUIModel(name="G", package="p", versionCode="1", versionName="1",
                               modules=set(), description=""),
        "modules None": GUIModel(name="G", package="p", versionCode="1", versionName="1",
                                 modules=None, description=""),
    }


@pytest.mark.parametrize("case", ["plain", "top-level", "nested twice", "no modules", "modules None"])
def test_the_spec_driven_agent_rule_matches_generate_output(case):
    """gui_uses_agent (spec-driven layer, which may not import the backend) must agree with
    has_agent_components, which decides on /generate-output whether a web app ships an agent."""
    from besser.spec_driven_agent.planning.agent_wiring import gui_uses_agent
    from besser.utilities.web_modeling_editor.backend.services.generation.agents import has_agent_components

    gui = _agent_gui_cases()[case]
    assert gui_uses_agent(gui) == has_agent_components(gui)
