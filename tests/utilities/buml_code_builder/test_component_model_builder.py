"""Tests for the Component model code builder (03-... §9.A).

Covers exec round-trip, AgenticEdge with permissions, Subsystem children,
Skill/Tool subtype promotion, layout passthrough, cross-diagram-ref
emission, banner stability, determinism.
"""

import pytest

from besser.BUML.metamodel.uml_component import (
    AgenticEdge,
    AgenticEdgeKind,
    AgentCategory,
    Component,
    ComponentDependency,
    ComponentModel,
    Interface,
    InterfaceProvided,
    Locality,
    Permission,
    Skill,
    Subsystem,
    Tool,
)
from besser.utilities.buml_code_builder.component_model_builder import (
    component_model_to_code,
)


def _exec_and_get_model(source: str) -> ComponentModel:
    """Exec the emitted source and return the rebuilt ComponentModel."""
    namespace: dict = {}
    exec(source, namespace)
    return namespace["component_model"]


class TestComponentModelBuilder:
    def test_empty_model_emits_valid_python(self):
        source = component_model_to_code(ComponentModel(name="Empty"))
        compile(source, "<test>", "exec")
        rebuilt = _exec_and_get_model(source)
        assert isinstance(rebuilt, ComponentModel)
        assert rebuilt.name == "Empty"
        assert rebuilt.components == set()

    def test_minimal_model_round_trips(self):
        c1 = Component(name="Frontend")
        c2 = Component(name="Backend")
        rel = ComponentDependency(source=c1, target=c2, name="uses")
        model = ComponentModel(
            name="Minimal", components={c1, c2}, relationships={rel},
        )
        source = component_model_to_code(model)
        rebuilt = _exec_and_get_model(source)
        assert sorted(c.name for c in rebuilt.components) == ["Backend", "Frontend"]
        assert len(rebuilt.relationships) == 1
        assert isinstance(next(iter(rebuilt.relationships)), ComponentDependency)

    def test_agentic_edge_with_permissions_round_trips(self):
        a = Component(name="Solver", agent_category=AgentCategory.SOLUTION)
        b = Component(name="Worker", agent_category=AgentCategory.SOLUTION)
        p1 = Permission(name="p1", scope="repo:read")
        p2 = Permission(name="p2", scope="repo:write")
        edge = AgenticEdge(
            source=a, target=b, kind=AgenticEdgeKind.DELEGATES,
            permissions=[p1, p2],
        )
        model = ComponentModel(
            name="Delegation", components={a, b},
            permissions={p1, p2}, relationships={edge},
        )
        source = component_model_to_code(model)
        rebuilt = _exec_and_get_model(source)
        # AgenticEdge survives with kind + permissions resolved by identity.
        edges = [r for r in rebuilt.relationships if isinstance(r, AgenticEdge)]
        assert len(edges) == 1
        rebuilt_edge = edges[0]
        assert rebuilt_edge.kind is AgenticEdgeKind.DELEGATES
        scopes = sorted(p.scope for p in rebuilt_edge.permissions)
        assert scopes == ["repo:read", "repo:write"]

    def test_subsystem_children_round_trip(self):
        c1 = Component(name="A")
        c2 = Component(name="B")
        sub = Subsystem(name="Group")
        sub.add_child(c1)
        sub.add_child(c2)
        model = ComponentModel(name="m", components={c1, c2, sub})
        source = component_model_to_code(model)
        rebuilt = _exec_and_get_model(source)
        rebuilt_subs = [c for c in rebuilt.components if isinstance(c, Subsystem)]
        assert len(rebuilt_subs) == 1
        assert sorted(c.name for c in rebuilt_subs[0].children) == ["A", "B"]
        for child in rebuilt_subs[0].children:
            assert child.parent is rebuilt_subs[0]

    def test_skill_and_tool_subtype_promotion(self):
        s = Skill(name="WebSearch", locality=Locality.EXTERNAL)
        t = Tool(name="Calculator")
        model = ComponentModel(name="m", components={s, t})
        source = component_model_to_code(model)
        rebuilt = _exec_and_get_model(source)
        skills = [c for c in rebuilt.components if isinstance(c, Skill)]
        tools = [c for c in rebuilt.components if isinstance(c, Tool)]
        assert len(skills) == 1
        assert skills[0].name == "WebSearch"
        assert skills[0].locality is Locality.EXTERNAL
        assert len(tools) == 1

    def test_interface_provided_round_trips(self):
        c = Component(name="Provider")
        iface = Interface(name="Api")
        rel = InterfaceProvided(source=c, target=iface)
        model = ComponentModel(
            name="m", components={c}, interfaces={iface}, relationships={rel},
        )
        source = component_model_to_code(model)
        rebuilt = _exec_and_get_model(source)
        assert len(rebuilt.interfaces) == 1
        rebuilt_rels = [r for r in rebuilt.relationships
                        if isinstance(r, InterfaceProvided)]
        assert len(rebuilt_rels) == 1

    def test_layout_passthrough(self):
        c = Component(name="X", layout={"id": "u1", "bounds": {"x": 1}})
        model = ComponentModel(name="m", components={c})
        source = component_model_to_code(model)
        assert ".layout = " in source
        rebuilt = _exec_and_get_model(source)
        rebuilt_c = next(iter(rebuilt.components))
        assert rebuilt_c.layout == {"id": "u1", "bounds": {"x": 1}}

    def test_cross_diagram_refs_via_dot_assignment(self):
        c = Component(name="X")
        c.realizes = ["class-uuid-1"]
        c.process_model_refs = ["bpmn-uuid-1", "bpmn-uuid-2"]
        model = ComponentModel(name="m", components={c})
        source = component_model_to_code(model)
        assert ".realizes = ['class-uuid-1']" in source
        assert ".process_model_refs = ['bpmn-uuid-1', 'bpmn-uuid-2']" in source
        rebuilt = _exec_and_get_model(source)
        rebuilt_c = next(iter(rebuilt.components))
        assert rebuilt_c.realizes == ["class-uuid-1"]
        assert rebuilt_c.process_model_refs == ["bpmn-uuid-1", "bpmn-uuid-2"]

    def test_cross_diagram_refs_omitted_when_empty(self):
        c = Component(name="X")
        model = ComponentModel(name="m", components={c})
        source = component_model_to_code(model)
        # Q1=(a) — line is omitted entirely when the list is empty.
        assert ".realizes = " not in source
        assert ".process_model_refs = " not in source

    def test_banner_and_imports_stable(self):
        source = component_model_to_code(ComponentModel(name="X"))
        lines = source.splitlines()
        assert lines[0] == "####################"
        assert lines[1] == "# COMPONENT MODEL  #"
        assert lines[2] == "####################"
        # Only ComponentModel imported when nothing else is used.
        assert "ComponentModel" in source
        assert "AgentCategory" not in source

    def test_determinism_same_model_same_source(self):
        c1 = Component(name="A")
        c2 = Component(name="B")
        model = ComponentModel(name="m", components={c1, c2})
        s1 = component_model_to_code(model)
        s2 = component_model_to_code(model)
        assert s1 == s2

    def test_name_with_special_chars_escapes(self):
        # Names with apostrophes / backslashes must be escaped to avoid
        # producing syntactically invalid output that exec()'s differently.
        c = Component(name="O'Brien's Component")
        model = ComponentModel(name="m", components={c})
        source = component_model_to_code(model)
        compile(source, "<test>", "exec")
        rebuilt = _exec_and_get_model(source)
        assert next(iter(rebuilt.components)).name == "O'Brien's Component"

    def test_rejects_non_component_model(self):
        with pytest.raises(TypeError):
            component_model_to_code("not a model")
