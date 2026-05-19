"""Tests for the Component diagram converters (02-... §12).

Covers:
  * JSON -> ``ComponentModel`` (``process_component_diagram``).
  * ``ComponentModel`` -> JSON (``component_object_to_json``).
  * Round-trip identity for the load-bearing fields (ids, types, ownership,
    stereotypes-after-token-resolution, agentic-edge dispatch).
  * Stereotype helpers: empty-stereotype guard, consumed-default tokens,
    typed-slot resolution, AgenticEdge permission suffix.
  * Multiplicity is exercised in the Deployment tests.
"""

import pytest

from besser.BUML.metamodel.uml_component import (
    AgentCategory,
    AgenticEdge,
    AgenticEdgeKind,
    Component,
    ComponentDependency,
    ComponentModel,
    Interface,
    InterfaceProvided,
    InterfaceRequired,
    Locality,
    Permission,
    Skill,
    Subsystem,
    Tool,
)
from besser.utilities.buml_code_builder.component_model_builder import (
    component_model_to_code,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.component_diagram_converter import (
    component_buml_to_json,
    component_object_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.component_diagram_processor import (
    process_component_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.stereotype_tokens import (
    apply_component_stereotype_tokens,
    extract_permission_scopes,
    tokenise,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
)


# ---------------------------------------------------------------------------
# Fixtures (mirror the real-export wire shape from 02-D7-Component.json
# and the agentic hand-built snapshot b1-component-snapshot.json)
# ---------------------------------------------------------------------------

@pytest.fixture
def real_component_diagram():
    """A base-Apollon Component diagram (real-export style: compact field
    omission, no `stereotype` key on bare Component)."""
    return {
        "title": "Component Diagram",
        "model": {
            "version": "3.0.0",
            "type": "ComponentDiagram",
            "size": {"width": 920, "height": 560},
            "elements": {
                "c1": {
                    "id": "c1", "name": "Frontend", "type": "Component",
                    "owner": None,
                    "bounds": {"x": -440, "y": -240, "width": 160, "height": 100},
                    "displayStereotype": True,
                },
                "c2": {
                    "id": "c2", "name": "Backend", "type": "Component",
                    "owner": None,
                    "bounds": {"x": 90, "y": -260, "width": 160, "height": 100},
                    "displayStereotype": True,
                },
                "s1": {
                    "id": "s1", "name": "ServiceLayer", "type": "Subsystem",
                    "owner": None,
                    "bounds": {"x": -350, "y": -40, "width": 160, "height": 100},
                    "stereotype": "subsystem", "displayStereotype": True,
                },
                "i1": {
                    "id": "i1", "name": "Api", "type": "ComponentInterface",
                    "owner": None,
                    "bounds": {"x": 0, "y": -50, "width": 20, "height": 20},
                },
            },
            "relationships": {
                "r1": {
                    "id": "r1", "name": "", "type": "ComponentDependency",
                    "owner": None,
                    "bounds": {"x": -280, "y": -200, "width": 370, "height": 1},
                    "path": [{"x": 0, "y": 0}, {"x": 370, "y": 0}],
                    "source": {"direction": "Right", "element": "c1"},
                    "target": {"direction": "Left", "element": "c2"},
                    "isManuallyLayouted": False,
                },
                "r2": {
                    "id": "r2", "name": "", "type": "ComponentInterfaceProvided",
                    "owner": None,
                    "bounds": {"x": 15, "y": -160, "width": 155, "height": 170},
                    "path": [{"x": 155, "y": 0}, {"x": 0, "y": 130}],
                    "source": {"direction": "Down", "element": "c2"},
                    "target": {"direction": "Bottomright", "element": "i1"},
                    "isManuallyLayouted": False,
                },
            },
            "interactive": {"elements": {}, "relationships": {}},
            "assessments": {},
        },
    }


@pytest.fixture
def agentic_component_diagram():
    """Hand-built style: Subsystem holding two Components with `solution`
    stereotype, a `delegates`-stereotyped AgenticEdge."""
    return {
        "title": "Component Diagram",
        "model": {
            "version": "3.0.0",
            "type": "ComponentDiagram",
            "size": {"width": 1400, "height": 740},
            "elements": {
                "elem-consensus-group": {
                    "id": "elem-consensus-group",
                    "name": "ConsensusGroup", "type": "Subsystem",
                    "owner": None,
                    "bounds": {"x": 100, "y": 100, "width": 520, "height": 280},
                    "stereotype": "subsystem", "displayStereotype": True,
                },
                "elem-orchestrator-agent": {
                    "id": "elem-orchestrator-agent",
                    "name": "OrchestratorAgent", "type": "Component",
                    "owner": "elem-consensus-group",
                    "bounds": {"x": 140, "y": 180, "width": 160, "height": 80},
                    "stereotype": "solution", "displayStereotype": True,
                },
                "elem-worker-agent": {
                    "id": "elem-worker-agent",
                    "name": "WorkerAgent", "type": "Component",
                    "owner": "elem-consensus-group",
                    "bounds": {"x": 420, "y": 180, "width": 160, "height": 80},
                    "stereotype": "component", "displayStereotype": True,
                },
            },
            "relationships": {
                "rel-delegates": {
                    "id": "rel-delegates",
                    "name": "", "type": "ComponentDependency",
                    "owner": None,
                    "bounds": {"x": 300, "y": 220, "width": 120, "height": 1},
                    "path": [{"x": 0, "y": 0}, {"x": 120, "y": 0}],
                    "source": {
                        "element": "elem-orchestrator-agent", "direction": "Right",
                    },
                    "target": {
                        "element": "elem-worker-agent", "direction": "Left",
                    },
                    "isManuallyLayouted": False,
                    "stereotype": "delegates {permission: repo:merge:approve, repo:write}",
                },
            },
            "interactive": {"elements": {}, "relationships": {}},
            "assessments": {},
        },
    }


@pytest.fixture
def skill_component_diagram():
    """Component with `stereotype: "skill"` should promote to Skill subclass."""
    return {
        "title": "Skill Diagram",
        "model": {
            "version": "3.0.0",
            "type": "ComponentDiagram",
            "size": {"width": 800, "height": 600},
            "elements": {
                "c-skill": {
                    "id": "c-skill", "name": "WebSearch", "type": "Component",
                    "owner": None,
                    "bounds": {"x": 0, "y": 0, "width": 160, "height": 80},
                    "stereotype": "skill external",
                    "displayStereotype": True,
                },
                "c-tool": {
                    "id": "c-tool", "name": "Calculator", "type": "Component",
                    "owner": None,
                    "bounds": {"x": 200, "y": 0, "width": 160, "height": 80},
                    "stereotype": "tool",
                    "displayStereotype": True,
                },
            },
            "relationships": {},
            "interactive": {"elements": {}, "relationships": {}},
            "assessments": {},
        },
    }


# ---------------------------------------------------------------------------
# JSON -> BUML
# ---------------------------------------------------------------------------

class TestProcessComponentDiagram:
    def test_base_apollon_round_trip(self, real_component_diagram):
        model = process_component_diagram(real_component_diagram)
        assert isinstance(model, ComponentModel)
        assert model.name == "Component Diagram"
        # 4 elements: 3 components (Frontend, Backend, ServiceLayer-subsystem) + 1 interface
        assert len(model.components) == 3
        assert len(model.interfaces) == 1
        # Subsystem is in components since Subsystem extends Component
        subsystems = [c for c in model.components if isinstance(c, Subsystem)]
        assert len(subsystems) == 1
        assert subsystems[0].name == "ServiceLayer"
        # 2 relationships: ComponentDependency, InterfaceProvided
        assert len(model.relationships) == 2

    def test_layout_preserved(self, real_component_diagram):
        model = process_component_diagram(real_component_diagram)
        for component in model.components:
            assert component.layout is not None
            assert "id" in component.layout
            assert "bounds" in component.layout

    def test_agentic_edge_with_permissions(self, agentic_component_diagram):
        model = process_component_diagram(agentic_component_diagram)
        # OrchestratorAgent has agent_category=SOLUTION
        agents = [c for c in model.components
                  if c.agent_category is AgentCategory.SOLUTION]
        assert len(agents) == 1
        assert agents[0].name == "OrchestratorAgent"
        # One AgenticEdge with kind=DELEGATES and two permissions
        edges = [r for r in model.relationships if isinstance(r, AgenticEdge)]
        assert len(edges) == 1
        edge = edges[0]
        assert edge.kind is AgenticEdgeKind.DELEGATES
        scopes = sorted(p.scope for p in edge.permissions)
        assert scopes == ["repo:merge:approve", "repo:write"]
        # Permissions auto-promoted into the model
        assert len(model.permissions) == 2

    def test_subsystem_children(self, agentic_component_diagram):
        model = process_component_diagram(agentic_component_diagram)
        subsystems = [c for c in model.components if isinstance(c, Subsystem)]
        assert len(subsystems) == 1
        assert len(subsystems[0].children) == 2
        for child in subsystems[0].children:
            assert child.parent is subsystems[0]

    def test_subtype_promotion(self, skill_component_diagram):
        model = process_component_diagram(skill_component_diagram)
        skills = [c for c in model.components if isinstance(c, Skill)]
        tools = [c for c in model.components if isinstance(c, Tool)]
        assert len(skills) == 1
        assert skills[0].name == "WebSearch"
        # The Skill should also have locality=EXTERNAL from the "skill external" stereotype.
        assert skills[0].locality is Locality.EXTERNAL
        assert len(tools) == 1
        assert tools[0].name == "Calculator"

    def test_missing_model_raises(self):
        with pytest.raises(ConversionError):
            process_component_diagram({"title": "x"})


# ---------------------------------------------------------------------------
# BUML -> JSON
# ---------------------------------------------------------------------------

class TestComponentObjectToJson:
    def test_envelope_shape(self):
        model = ComponentModel(name="Empty")
        out = component_object_to_json(model)
        assert out["type"] == "ComponentDiagram"
        assert out["version"] == "3.0.0"
        assert "size" in out
        assert out["elements"] == {}
        assert out["relationships"] == {}
        assert out["interactive"] == {"elements": {}, "relationships": {}}
        assert out["assessments"] == {}

    def test_emits_component_with_typed_stereotype(self):
        component = Component(
            name="Agent",
            agent_category=AgentCategory.SOLUTION,
            locality=Locality.EXTERNAL,
        )
        model = ComponentModel(name="m", components={component})
        out = component_object_to_json(model)
        # Find the emitted Component entry.
        entries = [e for e in out["elements"].values()
                   if e["type"] == "Component"]
        assert len(entries) == 1
        stereotype = entries[0].get("stereotype", "")
        # "solution external" — agent_category first, locality second.
        assert "solution" in stereotype.split()
        assert "external" in stereotype.split()

    def test_agentic_edge_emits_dependency_with_kind_and_permissions(self):
        a = Component(name="A", agent_category=AgentCategory.SOLUTION)
        b = Component(name="B", agent_category=AgentCategory.SOLUTION)
        p = Permission(name="r1", scope="r1")
        edge = AgenticEdge(
            source=a, target=b, kind=AgenticEdgeKind.DELEGATES,
            permissions=[p],
        )
        model = ComponentModel(
            name="m", components={a, b},
            permissions={p}, relationships={edge},
        )
        out = component_object_to_json(model)
        edge_entries = [r for r in out["relationships"].values()]
        assert len(edge_entries) == 1
        # AgenticEdge is wire-typed as ComponentDependency.
        assert edge_entries[0]["type"] == "ComponentDependency"
        stereotype = edge_entries[0]["stereotype"]
        assert stereotype.startswith("delegates")
        assert "{permission: r1}" in stereotype


# ---------------------------------------------------------------------------
# Round-trip identity
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def _ids(self, json_dict, key):
        """Return the sorted set of element ids from elements or relationships."""
        return sorted((json_dict.get(key) or {}).keys())

    def test_real_diagram_ids_survive(self, real_component_diagram):
        model = process_component_diagram(real_component_diagram)
        out = component_object_to_json(model)
        in_elements = self._ids(real_component_diagram["model"], "elements")
        out_elements = self._ids(out, "elements")
        assert in_elements == out_elements
        in_rels = self._ids(real_component_diagram["model"], "relationships")
        out_rels = self._ids(out, "relationships")
        assert in_rels == out_rels

    def test_real_diagram_types_survive(self, real_component_diagram):
        model = process_component_diagram(real_component_diagram)
        out = component_object_to_json(model)
        for elem_id, original in (
            real_component_diagram["model"]["elements"].items()
        ):
            assert out["elements"][elem_id]["type"] == original["type"]
        for rel_id, original in (
            real_component_diagram["model"]["relationships"].items()
        ):
            assert out["relationships"][rel_id]["type"] == original["type"]

    def test_real_diagram_owner_chains_survive(self, real_component_diagram):
        model = process_component_diagram(real_component_diagram)
        out = component_object_to_json(model)
        for elem_id, original in (
            real_component_diagram["model"]["elements"].items()
        ):
            assert out["elements"][elem_id].get("owner") == original.get("owner")

    def test_agentic_diagram_stereotype_resolves_and_re_emits(
        self, agentic_component_diagram,
    ):
        model = process_component_diagram(agentic_component_diagram)
        out = component_object_to_json(model)
        # The AgenticEdge re-emits with kind + sorted permission suffix.
        rel = out["relationships"]["rel-delegates"]
        assert rel["type"] == "ComponentDependency"
        assert rel["stereotype"].startswith("delegates")
        assert "{permission: repo:merge:approve, repo:write}" in rel["stereotype"]
        # The OrchestratorAgent emits with `solution`.
        orchestrator = out["elements"]["elem-orchestrator-agent"]
        assert "solution" in orchestrator["stereotype"].split()

    def test_agentic_subsystem_children_survive(self, agentic_component_diagram):
        model = process_component_diagram(agentic_component_diagram)
        out = component_object_to_json(model)
        # The two child components must keep `owner: "elem-consensus-group"`.
        for child_id in ("elem-orchestrator-agent", "elem-worker-agent"):
            assert out["elements"][child_id]["owner"] == "elem-consensus-group"


# ---------------------------------------------------------------------------
# Stereotype-helper unit tests (02-... §3.3 / Appendix B.2)
# ---------------------------------------------------------------------------

class TestStereotypeHelpers:
    def test_empty_stereotype_guard_short_circuits(self):
        c = Component(name="X")
        # Both empty-string and None should leave the component untouched.
        apply_component_stereotype_tokens(c, "")
        assert c.agent_category is AgentCategory.NONE
        assert c.stereotypes == []
        apply_component_stereotype_tokens(c, None)
        assert c.stereotypes == []

    def test_consumed_defaults_dropped(self):
        c = Component(name="X")
        apply_component_stereotype_tokens(c, "component")
        assert c.stereotypes == []
        s = Subsystem(name="Y")
        apply_component_stereotype_tokens(s, "subsystem")
        assert s.stereotypes == []

    def test_unknown_token_lands_in_stereotypes_list(self):
        c = Component(name="X")
        apply_component_stereotype_tokens(c, "weirdo")
        assert c.stereotypes == ["weirdo"]

    def test_case_insensitive_and_guillemets_stripped(self):
        c = Component(name="X")
        apply_component_stereotype_tokens(c, "«SOLUTION»")
        assert c.agent_category is AgentCategory.SOLUTION

    def test_locality_token_sets_typed_slot(self):
        c = Component(name="X")
        apply_component_stereotype_tokens(c, "external")
        assert c.locality is Locality.EXTERNAL

    def test_permission_suffix_extracted(self):
        scopes = extract_permission_scopes(
            "delegates {permission: a:b, c:d}"
        )
        assert scopes == ["a:b", "c:d"]
        assert extract_permission_scopes("delegates") == []
        assert extract_permission_scopes("") == []

    def test_tokenise_skips_empty_and_strips(self):
        assert tokenise("  delegates  ") == ["delegates"]
        assert tokenise("delegates, solution") == ["delegates", "solution"]
        assert tokenise("") == []
        assert tokenise(None) == []


# ---------------------------------------------------------------------------
# component_buml_to_json exec wrapper (03-... §7)
# ---------------------------------------------------------------------------

class TestComponentBumlToJson:
    def test_round_trips_via_builder(self, real_component_diagram):
        """Full pipeline: JSON -> ComponentModel -> .py source -> JSON."""
        model = process_component_diagram(real_component_diagram)
        source = component_model_to_code(model)
        result = component_buml_to_json(source)
        assert result["type"] == "ComponentDiagram"
        # ids survive because component_object_to_json reuses layout["id"]
        # (which the builder emits in .layout = {...} round-trip).
        in_ids = sorted(real_component_diagram["model"]["elements"].keys())
        out_ids = sorted(result["elements"].keys())
        assert in_ids == out_ids

    def test_exec_failure_raises_conversion_error(self):
        with pytest.raises(ConversionError):
            component_buml_to_json("raise NameError('boom')")

    def test_missing_model_raises_conversion_error(self):
        with pytest.raises(ConversionError) as exc_info:
            component_buml_to_json("x = 1")
        assert "no ComponentModel" in str(exc_info.value)

    def test_finds_model_under_any_var_name(self):
        # The wrapper prefers `component_model` but falls back to any
        # ComponentModel instance in the namespace.
        source = (
            "from besser.BUML.metamodel.uml_component import ComponentModel\n"
            "xyz = ComponentModel(name='Found')\n"
        )
        result = component_buml_to_json(source)
        assert result["type"] == "ComponentDiagram"

    def test_runtime_error_propagates_as_500(self):
        # RuntimeError is NOT in the four-tuple — it surfaces as a 500
        # via @handle_endpoint_errors, not a ConversionError 400.
        # The test here just verifies the exception is not caught.
        with pytest.raises(RuntimeError):
            component_buml_to_json("raise RuntimeError('boom')")
