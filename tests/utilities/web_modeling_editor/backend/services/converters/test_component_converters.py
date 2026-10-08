"""Tests for the Component diagram converters (v4 ``{nodes, edges}`` wire shape).

Covers:
  * JSON -> ``ComponentModel`` (``process_component_diagram``).
  * ``ComponentModel`` -> JSON (``component_object_to_json``).
  * Round-trip identity for the load-bearing fields (ids, types, parents,
    stereotypes-after-token-resolution, agentic-edge dispatch).
  * Stereotype helpers: empty-stereotype guard, consumed-default tokens,
    typed-slot resolution, AgenticEdge permission suffix.
  * Multiplicity is exercised in the Deployment tests.
"""

import pytest

from besser.BUML.metamodel.uml_component import (
    AgentCategory,
    AgenticComponent,
    AgenticComponentModel,
    AgenticEdge,
    AgenticEdgeKind,
    Component,
    ComponentModel,
    Database,
    InterfaceRequired,
    LLM,
    Locality,
    Permission,
    RAG,
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
    format_component_stereotype,
    tokenise,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
)


# ---------------------------------------------------------------------------
# v4 fixture builders
# ---------------------------------------------------------------------------

def _node(node_id, node_type, x, y, width, height, data, parent=None):
    node = {
        "id": node_id, "type": node_type,
        "position": {"x": x, "y": y},
        "width": width, "height": height,
        "measured": {"width": width, "height": height},
        "data": data,
    }
    if parent is not None:
        node["parentId"] = parent
    return node


def _edge(edge_id, edge_type, source, target, points, stereotype=None,
          source_handle="right", target_handle="left"):
    data = {"name": "", "label": "", "points": points, "isManuallyLayouted": False}
    if stereotype is not None:
        data["stereotype"] = stereotype
    return {
        "id": edge_id, "source": source, "target": target, "type": edge_type,
        "sourceHandle": source_handle, "targetHandle": target_handle, "data": data,
    }


def _diagram(title, nodes, edges):
    return {
        "title": title,
        "model": {
            "version": "4.0.0", "type": "ComponentDiagram", "title": title,
            "nodes": nodes, "edges": edges,
            "interactive": {"elements": {}, "relationships": {}},
            "assessments": {},
        },
    }


def _nodes_by_id(payload):
    return {n["id"]: n for n in payload["nodes"]}


def _edges_by_id(payload):
    return {e["id"]: e for e in payload["edges"]}


# ---------------------------------------------------------------------------
# Fixtures (mirror the editor's real Component-diagram export and a
# hand-built agentic diagram)
# ---------------------------------------------------------------------------

@pytest.fixture
def real_component_diagram():
    """A plain UML Component diagram (bare Components carry an empty stereotype)."""
    return _diagram(
        "Component Diagram",
        [
            _node("c1", "component", -440, -240, 160, 100,
                  {"name": "Frontend", "stereotype": "", "isComponentHeaderShown": True}),
            _node("c2", "component", 90, -260, 160, 100,
                  {"name": "Backend", "stereotype": "", "isComponentHeaderShown": True}),
            _node("s1", "componentSubsystem", -350, -40, 160, 100,
                  {"name": "ServiceLayer", "stereotype": "subsystem",
                   "isComponentSubsystemHeaderShown": True}),
            _node("i1", "componentInterface", 0, -50, 20, 20, {"name": "Api"}),
        ],
        [
            _edge("r1", "ComponentDependency", "c1", "c2",
                  [{"x": -280, "y": -200}, {"x": 90, "y": -200}]),
            _edge("r2", "ComponentProvidedInterface", "c2", "i1",
                  [{"x": 170, "y": -160}, {"x": 15, "y": -30}],
                  source_handle="bottom", target_handle="right-bottom"),
        ],
    )


@pytest.fixture
def agentic_component_diagram():
    """Hand-built style: Subsystem holding two Components with agent
    stereotypes, a `delegates`-stereotyped AgenticEdge."""
    return _diagram(
        "Component Diagram",
        [
            _node("elem-consensus-group", "componentSubsystem", 100, 100, 520, 280,
                  {"name": "ConsensusGroup", "stereotype": "subsystem",
                   "isComponentSubsystemHeaderShown": True}),
            _node("elem-orchestrator-agent", "component", 40, 80, 160, 80,
                  {"name": "OrchestratorAgent", "stereotype": "solution",
                   "isComponentHeaderShown": True},
                  parent="elem-consensus-group"),
            _node("elem-worker-agent", "component", 320, 80, 160, 80,
                  {"name": "WorkerAgent", "stereotype": "collaboration",
                   "isComponentHeaderShown": True},
                  parent="elem-consensus-group"),
        ],
        [
            _edge("rel-delegates", "ComponentDependency",
                  "elem-orchestrator-agent", "elem-worker-agent",
                  [{"x": 300, "y": 220}, {"x": 420, "y": 220}],
                  stereotype="delegates {permission: repo:merge:approve, repo:write}"),
        ],
    )


@pytest.fixture
def skill_component_diagram():
    """Component with `stereotype: "skill"` should promote to Skill subclass."""
    return _diagram(
        "Skill Diagram",
        [
            _node("c-skill", "component", 0, 0, 160, 80,
                  {"name": "WebSearch", "stereotype": "skill external",
                   "isComponentHeaderShown": True}),
            _node("c-tool", "component", 200, 0, 160, 80,
                  {"name": "Calculator", "stereotype": "tool",
                   "isComponentHeaderShown": True}),
        ],
        [],
    )


# ---------------------------------------------------------------------------
# JSON -> BUML
# ---------------------------------------------------------------------------

class TestProcessComponentDiagram:
    def test_base_apollon_round_trip(self, real_component_diagram):
        model = process_component_diagram(real_component_diagram)
        assert isinstance(model, ComponentModel)
        assert model.name == "Component Diagram"
        # 4 nodes: 3 components (Frontend, Backend, ServiceLayer-subsystem) + 1 interface
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

    def test_layout_bounds_are_absolute(self, agentic_component_diagram):
        """A child's parent-relative v4 position is stashed as absolute bounds."""
        model = process_component_diagram(agentic_component_diagram)
        orchestrator = next(c for c in model.components if c.name == "OrchestratorAgent")
        assert orchestrator.layout["parentId"] == "elem-consensus-group"
        assert orchestrator.layout["bounds"] == {"x": 140, "y": 180, "width": 160, "height": 80}

    def test_agentic_edge_with_permissions(self, agentic_component_diagram):
        model = process_component_diagram(agentic_component_diagram)
        # An agentic diagram yields an AgenticComponentModel.
        assert isinstance(model, AgenticComponentModel)
        # OrchestratorAgent has agent_category=SOLUTION
        agents = [c for c in model.components
                  if isinstance(c, AgenticComponent)
                  and c.agent_category is AgentCategory.SOLUTION]
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
        assert out["version"] == "4.0.0"
        assert out["title"] == "Empty"
        assert "size" in out
        assert out["nodes"] == []
        assert out["edges"] == []
        assert out["interactive"] == {"elements": {}, "relationships": {}}
        assert out["assessments"] == {}

    def test_emits_component_with_typed_stereotype(self):
        component = AgenticComponent(
            name="Agent",
            agent_category=AgentCategory.SOLUTION,
            locality=Locality.EXTERNAL,
        )
        model = AgenticComponentModel(name="m", components={component})
        out = component_object_to_json(model)
        # Find the emitted component node.
        entries = [n for n in out["nodes"] if n["type"] == "component"]
        assert len(entries) == 1
        stereotype = entries[0]["data"].get("stereotype", "")
        # "solution external" — agent_category first, locality second.
        assert "solution" in stereotype.split()
        assert "external" in stereotype.split()

    def test_agentic_edge_emits_dependency_with_kind_and_permissions(self):
        a = AgenticComponent(name="A", agent_category=AgentCategory.SOLUTION)
        b = AgenticComponent(name="B", agent_category=AgentCategory.SOLUTION)
        p = Permission(name="r1", scope="r1")
        edge = AgenticEdge(
            source=a, target=b, kind=AgenticEdgeKind.DELEGATES,
            permissions=[p],
        )
        model = AgenticComponentModel(
            name="m", components={a, b},
            permissions={p}, relationships={edge},
        )
        out = component_object_to_json(model)
        assert len(out["edges"]) == 1
        # AgenticEdge is wire-typed as ComponentDependency.
        assert out["edges"][0]["type"] == "ComponentDependency"
        stereotype = out["edges"][0]["data"]["stereotype"]
        assert stereotype.startswith("delegates")
        assert "{permission: r1}" in stereotype

    def test_model_without_layout_gets_non_overlapping_positions(self):
        """A model built in Python has no editor layout: children are placed
        inside their Subsystem and no two siblings overlap."""
        child_a = Component(name="A")
        child_b = Component(name="B")
        group = Subsystem(name="Group")
        group.add_child(child_a)
        group.add_child(child_b)
        loose = Component(name="Loose")
        model = ComponentModel(name="m", components={group, child_a, child_b, loose})
        out = component_object_to_json(model)
        nodes = {n["data"]["name"]: n for n in out["nodes"]}
        group_node = nodes["Group"]

        # Parents precede their children (React Flow requirement).
        order = [n["data"]["name"] for n in out["nodes"]]
        assert order.index("Group") < order.index("A")
        assert order.index("Group") < order.index("B")

        for name in ("A", "B"):
            child = nodes[name]
            assert child["parentId"] == group_node["id"]
            # Parent-relative position fully inside the parent.
            assert child["position"]["x"] >= 0 and child["position"]["y"] >= 0
            assert child["position"]["x"] + child["width"] <= group_node["width"]
            assert child["position"]["y"] + child["height"] <= group_node["height"]

        def overlaps(n1, n2):
            (x1, y1), (x2, y2) = (n1["position"]["x"], n1["position"]["y"]), (n2["position"]["x"], n2["position"]["y"])
            return x1 < x2 + n2["width"] and x2 < x1 + n1["width"] and y1 < y2 + n2["height"] and y2 < y1 + n1["height"]

        assert not overlaps(nodes["A"], nodes["B"])
        assert not overlaps(group_node, nodes["Loose"])


# ---------------------------------------------------------------------------
# Round-trip identity
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_real_diagram_ids_survive(self, real_component_diagram):
        model = process_component_diagram(real_component_diagram)
        out = component_object_to_json(model)
        assert sorted(_nodes_by_id(out)) == sorted(_nodes_by_id(real_component_diagram["model"]))
        assert sorted(_edges_by_id(out)) == sorted(_edges_by_id(real_component_diagram["model"]))

    def test_real_diagram_types_survive(self, real_component_diagram):
        model = process_component_diagram(real_component_diagram)
        out = component_object_to_json(model)
        out_nodes, out_edges = _nodes_by_id(out), _edges_by_id(out)
        for node_id, original in _nodes_by_id(real_component_diagram["model"]).items():
            assert out_nodes[node_id]["type"] == original["type"]
        for edge_id, original in _edges_by_id(real_component_diagram["model"]).items():
            assert out_edges[edge_id]["type"] == original["type"]

    def test_real_diagram_parent_chains_survive(self, real_component_diagram):
        model = process_component_diagram(real_component_diagram)
        out_nodes = _nodes_by_id(component_object_to_json(model))
        for node_id, original in _nodes_by_id(real_component_diagram["model"]).items():
            assert out_nodes[node_id].get("parentId") == original.get("parentId")

    def test_agentic_diagram_stereotype_resolves_and_re_emits(
        self, agentic_component_diagram,
    ):
        model = process_component_diagram(agentic_component_diagram)
        out = component_object_to_json(model)
        # The AgenticEdge re-emits with kind + sorted permission suffix.
        rel = _edges_by_id(out)["rel-delegates"]
        assert rel["type"] == "ComponentDependency"
        assert rel["data"]["stereotype"].startswith("delegates")
        assert "{permission: repo:merge:approve, repo:write}" in rel["data"]["stereotype"]
        # The OrchestratorAgent emits with `solution`.
        orchestrator = _nodes_by_id(out)["elem-orchestrator-agent"]
        assert "solution" in orchestrator["data"]["stereotype"].split()

    def test_agentic_subsystem_children_survive(self, agentic_component_diagram):
        model = process_component_diagram(agentic_component_diagram)
        out_nodes = _nodes_by_id(component_object_to_json(model))
        # The two child components keep `parentId: "elem-consensus-group"` and
        # their parent-relative positions.
        for child_id in ("elem-orchestrator-agent", "elem-worker-agent"):
            assert out_nodes[child_id]["parentId"] == "elem-consensus-group"
        assert out_nodes["elem-orchestrator-agent"]["position"] == {"x": 40, "y": 80}

    def test_required_interface_socket_variant_survives(self):
        """A ComponentRequiredQuarterInterface edge reads as InterfaceRequired and
        comes back with its drawn socket variant, also through a BUML file."""
        payload = _diagram(
            "Sockets",
            [
                _node("c", "component", 0, 0, 160, 100,
                      {"name": "Client", "stereotype": "", "isComponentHeaderShown": True}),
                _node("i", "componentInterface", 300, 40, 20, 20, {"name": "Port"}),
            ],
            [_edge("req", "ComponentRequiredQuarterInterface", "c", "i",
                   [{"x": 160, "y": 50}, {"x": 300, "y": 50}])],
        )
        model = process_component_diagram(payload)
        (rel,) = model.relationships
        assert isinstance(rel, InterfaceRequired)

        direct = _edges_by_id(component_object_to_json(model))["req"]
        assert direct["type"] == "ComponentRequiredQuarterInterface"
        via_file = _edges_by_id(component_buml_to_json(component_model_to_code(model)))["req"]
        assert via_file["type"] == "ComponentRequiredQuarterInterface"
        assert via_file["data"]["points"] == [{"x": 160, "y": 50}, {"x": 300, "y": 50}]


# ---------------------------------------------------------------------------
# Stereotype-helper unit tests
# ---------------------------------------------------------------------------

class TestStereotypeHelpers:
    def test_empty_stereotype_guard_short_circuits(self):
        c = Component(name="X")
        # Both empty-string and None should leave the component untouched.
        apply_component_stereotype_tokens(c, "")
        assert c.stereotypes == []
        apply_component_stereotype_tokens(c, None)
        assert c.stereotypes == []

    def test_editor_defaults_are_kept_so_they_round_trip(self):
        c = Component(name="X")
        apply_component_stereotype_tokens(c, "component")
        assert c.stereotypes == ["component"]
        assert format_component_stereotype(c) == "component"
        s = Subsystem(name="Y")
        apply_component_stereotype_tokens(s, "subsystem")
        assert s.stereotypes == ["subsystem"]
        assert format_component_stereotype(s) == "subsystem"

    def test_unknown_token_lands_in_stereotypes_list(self):
        c = Component(name="X")
        apply_component_stereotype_tokens(c, "weirdo")
        assert c.stereotypes == ["weirdo"]

    def test_case_insensitive_and_guillemets_stripped(self):
        c = AgenticComponent(name="X")
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
# component_buml_to_json exec wrapper
# ---------------------------------------------------------------------------

class TestComponentBumlToJson:
    def test_round_trips_via_builder(self, real_component_diagram):
        """Full pipeline: JSON -> ComponentModel -> .py source -> JSON."""
        model = process_component_diagram(real_component_diagram)
        source = component_model_to_code(model)
        result = component_buml_to_json(source)
        assert result["type"] == "ComponentDiagram"
        assert result["version"] == "4.0.0"
        # ids survive because component_object_to_json reuses layout["id"]
        # (which the builder emits in .layout = {...} round-trip).
        assert sorted(_nodes_by_id(result)) == sorted(_nodes_by_id(real_component_diagram["model"]))

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

    def test_executable_statement_is_rejected(self):
        with pytest.raises(ConversionError, match="Top-level statement not allowed: Raise"):
            component_buml_to_json("raise RuntimeError('boom')")

    def test_v3_layout_stash_from_older_exports_still_loads(self):
        """`.py` files exported before the v4 migration carry the v3 layout stash
        (``owner``, ``path`` relative to ``bounds``); they still convert, with the
        edge points made absolute."""
        source = (
            "c1 = Component(name='A')\n"
            "c1.layout = {'id': 'a', 'owner': None, 'bounds': {'x': 0, 'y': 0, 'width': 160, 'height': 100},"
            " 'displayStereotype': True, 'wme_type': 'Component'}\n"
            "c2 = Component(name='B')\n"
            "c2.layout = {'id': 'b', 'owner': None, 'bounds': {'x': 300, 'y': 0, 'width': 160, 'height': 100},"
            " 'displayStereotype': True, 'wme_type': 'Component'}\n"
            "component_model = ComponentModel(name='M', components={c1, c2})\n"
            "r = ComponentDependency(source=c1, target=c2)\n"
            "r.layout = {'id': 'r', 'bounds': {'x': 160, 'y': 50, 'width': 140, 'height': 1},"
            " 'path': [{'x': 0, 'y': 0}, {'x': 140, 'y': 0}], 'source_direction': 'Right',"
            " 'target_direction': 'Left', 'isManuallyLayouted': False,"
            " 'wme_type': 'ComponentDependency'}\n"
            "component_model.add_relationship(r)\n"
        )
        result = component_buml_to_json(source)
        edge = _edges_by_id(result)["r"]
        assert edge["sourceHandle"] == "right" and edge["targetHandle"] == "left"
        assert edge["data"]["points"] == [{"x": 160, "y": 50}, {"x": 300, "y": 50}]
        assert sorted(_nodes_by_id(result)) == ["a", "b"]


# ---------------------------------------------------------------------------
# LLM / Database / RAG round-trip
# ---------------------------------------------------------------------------

@pytest.fixture
def llm_db_rag_component_diagram():
    """Diagram with llm/db/rag Components + uses edges from an agent.

    Mirrors the editor's BPMN→Component derivation output for an agent that
    calls an LLM, queries a DB, and reads a RAG store.
    """
    def uses(edge_id, target, y):
        return _edge(edge_id, "ComponentDependency", "agent-1", target,
                     [{"x": 260, "y": y}, {"x": 400, "y": y}], stereotype="uses")

    return _diagram(
        "LLM DB RAG Diagram",
        [
            _node("agent-1", "component", 100, 200, 160, 80,
                  {"name": "ResearchAgent", "stereotype": "solution",
                   "isComponentHeaderShown": True}),
            _node("llm-1", "component", 400, 80, 160, 80,
                  {"name": "GPT4", "stereotype": "llm", "isComponentHeaderShown": True}),
            _node("db-1", "component", 400, 200, 160, 80,
                  {"name": "OrdersDB", "stereotype": "db", "isComponentHeaderShown": True}),
            _node("rag-1", "component", 400, 320, 160, 80,
                  {"name": "KnowledgeBase", "stereotype": "rag",
                   "isComponentHeaderShown": True}),
        ],
        [uses("uses-llm", "llm-1", 120), uses("uses-db", "db-1", 240),
         uses("uses-rag", "rag-1", 360)],
    )


class TestLLMDBRAGRoundTrip:
    def test_import_promotes_to_correct_subtypes(self, llm_db_rag_component_diagram):
        model = process_component_diagram(llm_db_rag_component_diagram)
        components_by_id = {c.layout["id"]: c for c in model.components
                            if c.layout and "id" in c.layout}
        # llm-1 must be LLM, not bare Component and not plain Tool.
        assert isinstance(components_by_id["llm-1"], LLM)
        assert type(components_by_id["llm-1"]) is LLM
        # db-1 must be Database.
        assert isinstance(components_by_id["db-1"], Database)
        assert type(components_by_id["db-1"]) is Database
        # rag-1 must be RAG.
        assert isinstance(components_by_id["rag-1"], RAG)
        assert type(components_by_id["rag-1"]) is RAG

    def test_uses_edges_survive_not_silently_dropped(self, llm_db_rag_component_diagram):
        """Regression guard: the uses edges were once silently dropped."""
        model = process_component_diagram(llm_db_rag_component_diagram)
        agentic_edges = [r for r in model.relationships
                         if isinstance(r, AgenticEdge)]
        assert len(agentic_edges) == 3
        for edge in agentic_edges:
            assert edge.kind is AgenticEdgeKind.USES

    def test_export_emits_correct_subtype_tokens(self, llm_db_rag_component_diagram):
        """Round-trip: import then export must emit llm/db/rag, never 'tool'."""
        model = process_component_diagram(llm_db_rag_component_diagram)
        out_nodes = _nodes_by_id(component_object_to_json(model))
        llm_tokens = out_nodes["llm-1"]["data"]["stereotype"].split()
        db_tokens = out_nodes["db-1"]["data"]["stereotype"].split()
        rag_tokens = out_nodes["rag-1"]["data"]["stereotype"].split()
        assert "llm" in llm_tokens
        assert "db" in db_tokens
        assert "rag" in rag_tokens
        # Most-derived-first: none of these should emit as plain 'tool'.
        assert "tool" not in llm_tokens
        assert "tool" not in db_tokens
        assert "tool" not in rag_tokens
