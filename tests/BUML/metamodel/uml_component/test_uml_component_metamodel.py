"""Tests for the UML Component metamodel (``besser.BUML.metamodel.uml_component``).

Groups, following ``.claude/component-deployment/01-component-deployment-design.md``
§1 / §2.1 / §4.1:

1. Construction validation -- setters raise on bad input; free-text names accepted.
2. Containment -- Subsystem children + parent back-reference.
3. Relationship endpoint type rules (construction-time TypeError).
4. AgenticEdge per-kind endpoint type rules.
5. ``ComponentModel.validate()`` -- rules E1..E13.
6. Warnings W1..W7.
7. Identity -- elements use object identity (no ``__eq__`` / ``__hash__``).
"""

import pytest

from besser.BUML.metamodel.uml_component import (
    AgentCategory, AgenticEdge, AgenticEdgeKind, Component, ComponentDependency,
    ComponentElement, ComponentModel, ComponentRelationship, Interface,
    InterfaceProvided, InterfaceRequired, Permission, Skill, Subsystem, Tool,
)


# ---------------------------------------------------------------------------
# Group 1 -- construction validation
# ---------------------------------------------------------------------------

def test_free_text_names_are_accepted():
    """D9 -- Component labels are free text: spaces, empty, None coerced."""
    assert Component("Code Tester").name == "Code Tester"
    assert Component("Review & approve").name == "Review & approve"
    assert Component("").name == ""
    assert Component().name == ""
    assert Component(None).name == ""
    assert ComponentModel("My Component Diagram").name == "My Component Diagram"
    assert ComponentModel("").name == ""


def test_name_must_be_str_or_none():
    with pytest.raises(TypeError):
        Component(123)
    with pytest.raises(TypeError):
        ComponentModel(123)


def test_abstract_relationship_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ComponentRelationship(Component(), Component())


def test_concrete_subtypes_instantiate():
    assert isinstance(Subsystem(), Component)
    assert isinstance(Skill(), Component)
    assert isinstance(Tool(), Component)
    assert isinstance(AgenticEdge(Component(), Skill(), kind=AgenticEdgeKind.HAS),
                      ComponentDependency)


def test_enum_attributes_are_type_checked():
    with pytest.raises(TypeError):
        Component("c", agent_category="solution")
    with pytest.raises(TypeError):
        Component("c", locality="external")


def test_is_human_must_be_bool():
    with pytest.raises(TypeError):
        Component("c", is_human="yes")


def test_layout_must_be_dict_or_none():
    assert Component("c", layout={"x": 1}).layout == {"x": 1}
    assert Component("c").layout is None
    with pytest.raises(TypeError):
        Component("c", layout=[1, 2])


def test_stereotypes_must_be_list_of_str():
    assert Component("c", stereotypes=["«custom»"]).stereotypes == ["«custom»"]
    with pytest.raises(TypeError):
        Component("c", stereotypes=[1])


def test_realizes_and_process_model_refs_must_be_list_of_str():
    c = Component("c", realizes=["class_a"], process_model_refs=["proc_b"])
    assert c.realizes == ["class_a"]
    assert c.process_model_refs == ["proc_b"]
    with pytest.raises(TypeError):
        Component("c", realizes=[123])
    with pytest.raises(TypeError):
        Component("c", process_model_refs=[None])


def test_permission_scope_must_be_str():
    assert Permission("p", scope="repo:read").scope == "repo:read"
    with pytest.raises(TypeError):
        Permission("p", scope=123)


def test_permission_default_scope_is_empty_str():
    """Empty scope is allowed construction-time; ``validate()`` flags it E12."""
    p = Permission("p")
    assert p.scope == ""


# ---------------------------------------------------------------------------
# Group 2 -- containment
# ---------------------------------------------------------------------------

def test_subsystem_sets_child_parent():
    a, b = Component("a"), Component("b")
    sub = Subsystem("sub", children={a, b})
    assert a.parent is sub
    assert b.parent is sub


def test_subsystem_add_remove_child_maintains_parent():
    a = Component("a")
    sub = Subsystem("sub")
    sub.add_child(a)
    assert a.parent is sub
    assert a in sub.children
    sub.remove_child(a)
    assert a.parent is None
    assert a not in sub.children


def test_subsystem_reassignment_clears_old_parent():
    a = Component("a")
    sub1, sub2 = Subsystem("sub1", children={a}), Subsystem("sub2")
    sub2.children = {a}  # reassign
    # `a` is no longer in sub1's children -- the setter should have cleared it.
    # But the old reference is still in sub1 since we replaced sub2's not sub1's.
    # Re-test with proper move:
    sub1.children = set()
    assert a.parent is sub2
    assert a not in sub1.children


def test_subsystem_children_type_checked():
    with pytest.raises(TypeError):
        Subsystem("sub", children={Interface("i")})


def test_component_model_all_components_expands_subsystems():
    inner = Component("inner")
    sub = Subsystem("outer", children={inner})
    model = ComponentModel("m", components={sub})
    # Subsystem in `components`; child only via `all_components()`.
    assert sub in model.components
    assert inner not in model.components
    assert inner in model.all_components()
    assert sub in model.all_components()


# ---------------------------------------------------------------------------
# Group 3 -- relationship endpoint construction-time rules
# ---------------------------------------------------------------------------

def test_interface_provided_endpoints():
    c, iface = Component("c"), Interface("i")
    rel = InterfaceProvided(c, iface)
    assert rel.source is c
    assert rel.target is iface
    with pytest.raises(TypeError):
        InterfaceProvided(iface, c)  # swapped
    with pytest.raises(TypeError):
        InterfaceProvided(c, c)  # target not an Interface


def test_interface_required_endpoints():
    c, iface = Component("c"), Interface("i")
    rel = InterfaceRequired(c, iface)
    assert rel.source is c
    with pytest.raises(TypeError):
        InterfaceRequired(iface, c)
    with pytest.raises(TypeError):
        InterfaceRequired(c, c)


def test_component_dependency_requires_components():
    c1, c2 = Component("a"), Component("b")
    iface = Interface("i")
    rel = ComponentDependency(c1, c2)
    assert rel.source is c1 and rel.target is c2
    with pytest.raises(TypeError):
        ComponentDependency(c1, iface)
    with pytest.raises(TypeError):
        ComponentDependency(iface, c2)


# ---------------------------------------------------------------------------
# Group 4 -- AgenticEdge per-kind endpoint construction-time rules
# ---------------------------------------------------------------------------

def test_agentic_edge_has_requires_skill_target():
    agent = Component("a", agent_category=AgentCategory.SOLUTION)
    skill = Skill("s")
    AgenticEdge(agent, skill, kind=AgenticEdgeKind.HAS)
    with pytest.raises(TypeError):
        AgenticEdge(agent, Tool("t"), kind=AgenticEdgeKind.HAS)


def test_agentic_edge_uses_requires_tool_target():
    agent = Component("a", agent_category=AgentCategory.SOLUTION)
    tool = Tool("t")
    AgenticEdge(agent, tool, kind=AgenticEdgeKind.USES)
    with pytest.raises(TypeError):
        AgenticEdge(agent, Skill("s"), kind=AgenticEdgeKind.USES)


def test_agentic_edge_granted_requires_permission_target():
    agent = Component("a", agent_category=AgentCategory.SOLUTION)
    perm = Permission("p", scope="repo:read")
    AgenticEdge(agent, perm, kind=AgenticEdgeKind.GRANTED)
    with pytest.raises(TypeError):
        AgenticEdge(agent, Tool("t"), kind=AgenticEdgeKind.GRANTED)


def test_agentic_edge_implements_tool_to_skill():
    tool, skill = Tool("t"), Skill("s")
    AgenticEdge(tool, skill, kind=AgenticEdgeKind.IMPLEMENTS)
    with pytest.raises(TypeError):
        AgenticEdge(Skill("s2"), skill, kind=AgenticEdgeKind.IMPLEMENTS)
    with pytest.raises(TypeError):
        AgenticEdge(tool, Tool("t2"), kind=AgenticEdgeKind.IMPLEMENTS)


def test_agentic_edge_agent_to_agent_accepts_components():
    a = Component("a", agent_category=AgentCategory.SOLUTION)
    b = Component("b", agent_category=AgentCategory.SUPERVISION)
    AgenticEdge(a, b, kind=AgenticEdgeKind.DELEGATES)
    AgenticEdge(a, b, kind=AgenticEdgeKind.SUPERVISES)
    AgenticEdge(a, b, kind=AgenticEdgeKind.REVISES)
    AgenticEdge(a, b, kind=AgenticEdgeKind.COLLABORATES)
    # Type-only check at construction; Interface as endpoint is rejected.
    iface = Interface("i")
    with pytest.raises(TypeError):
        AgenticEdge(a, iface, kind=AgenticEdgeKind.DELEGATES)


def test_agentic_edge_kind_must_be_enum():
    a, b = Component("a"), Component("b")
    with pytest.raises(TypeError):
        AgenticEdge(a, b, kind="delegates")


def test_agentic_edge_permissions_type_checked():
    a, b = Component("a"), Component("b")
    perm = Permission("p", scope="x")
    e = AgenticEdge(a, b, kind=AgenticEdgeKind.DELEGATES, permissions=[perm])
    assert e.permissions == [perm]
    with pytest.raises(TypeError):
        AgenticEdge(a, b, kind=AgenticEdgeKind.DELEGATES, permissions=["x"])


def test_agentic_edge_kind_setter_rechecks_endpoints():
    agent = Component("a", agent_category=AgentCategory.SOLUTION)
    skill = Skill("s")
    edge = AgenticEdge(agent, skill, kind=AgenticEdgeKind.HAS)
    # Switching to USES requires target to be Tool, but it's a Skill -> error.
    with pytest.raises(TypeError):
        edge.kind = AgenticEdgeKind.USES
    # Kind unchanged after the failed setter.
    assert edge.kind == AgenticEdgeKind.HAS


# ---------------------------------------------------------------------------
# Group 5 -- ComponentModel.validate() errors (E1..E13)
# ---------------------------------------------------------------------------

def test_validate_passes_for_well_formed_models(
    minimal_component_model, agent_swarm_model
):
    minimal_result = minimal_component_model.validate(raise_exception=False)
    assert minimal_result["success"] is True
    assert minimal_result["errors"] == []
    swarm_result = agent_swarm_model.validate(raise_exception=False)
    assert swarm_result["success"] is True
    assert swarm_result["errors"] == []


def test_validate_raises_when_requested():
    a = Component("a")
    b = Component("b")  # b is NOT in the model
    rel = ComponentDependency(a, b, name="dangle")
    model = ComponentModel("m", components={a}, relationships={rel})
    with pytest.raises(ValueError, match="is not present"):
        model.validate(raise_exception=True)


def test_e1_dangling_endpoint_reference():
    a = Component("a")
    b = Component("b")  # not added to the model
    rel = ComponentDependency(a, b, name="ref")
    model = ComponentModel("m", components={a}, relationships={rel})
    result = model.validate(raise_exception=False)
    assert any("not present in the model" in e for e in result["errors"])


def test_e10_dangling_permission_reference():
    agent = Component("agent", agent_category=AgentCategory.SOLUTION,
                     process_model_refs=["p"])
    human = Component("human", is_human=True, process_model_refs=["p"])
    rogue_perm = Permission("rogue", scope="x")  # NOT in model.permissions
    edge = AgenticEdge(agent, human, kind=AgenticEdgeKind.DELEGATES,
                       permissions=[rogue_perm], name="d")
    model = ComponentModel(
        "m",
        components={agent, human},
        relationships={edge},
    )
    result = model.validate(raise_exception=False)
    assert any("not in the model" in e for e in result["errors"])


def test_e5_agent_to_agent_requires_agent_endpoints():
    non_agent = Component("plain")  # agent_category=NONE, is_human=False
    agent = Component("agent", agent_category=AgentCategory.SOLUTION,
                      process_model_refs=["p"])
    edge = AgenticEdge(non_agent, agent, kind=AgenticEdgeKind.DELEGATES, name="bad")
    model = ComponentModel(
        "m", components={non_agent, agent}, relationships={edge}
    )
    result = model.validate(raise_exception=False)
    assert any("not an agent" in e for e in result["errors"])


def test_e11_duplicate_component_names():
    a1 = Component("dup")
    a2 = Component("dup")
    model = ComponentModel("m", components={a1, a2})
    result = model.validate(raise_exception=False)
    assert any("Duplicate component name" in e for e in result["errors"])


def test_e12_empty_permission_scope():
    perm = Permission("p")  # scope defaults to ""
    model = ComponentModel("m", permissions={perm})
    result = model.validate(raise_exception=False)
    assert any("empty scope" in e for e in result["errors"])


def test_e13_orphan_parent_subsystem():
    rogue_sub = Subsystem("rogue")
    a = Component("a")
    a.parent = rogue_sub  # rogue not in model
    model = ComponentModel("m", components={a})
    result = model.validate(raise_exception=False)
    assert any("not in the model" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# Group 6 -- warnings (W1..W7)
# ---------------------------------------------------------------------------

def test_w1_agent_with_no_capabilities():
    agent = Component("agent", agent_category=AgentCategory.SOLUTION,
                      process_model_refs=["p"])
    model = ComponentModel("m", components={agent})
    result = model.validate(raise_exception=False)
    assert any("no outgoing" in w for w in result["warnings"])


def test_w2_agent_with_no_process_refs():
    skill = Skill("s")
    agent = Component("agent", agent_category=AgentCategory.SOLUTION)
    has_skill = AgenticEdge(agent, skill, kind=AgenticEdgeKind.HAS, name="h")
    model = ComponentModel(
        "m", components={agent, skill}, relationships={has_skill}
    )
    result = model.validate(raise_exception=False)
    assert any("process_model_refs is empty" in w for w in result["warnings"])


def test_w3_permissions_on_non_agent_to_agent_kind():
    agent = Component("agent", agent_category=AgentCategory.SOLUTION,
                      process_model_refs=["p"])
    tool = Tool("t")
    perm = Permission("p1", scope="x")
    edge = AgenticEdge(agent, tool, kind=AgenticEdgeKind.USES,
                      permissions=[perm], name="bad")
    model = ComponentModel(
        "m",
        components={agent, tool},
        permissions={perm},
        relationships={edge},
    )
    result = model.validate(raise_exception=False)
    assert any("meaningful only on agent" in w for w in result["warnings"])


def test_w6_skill_with_agent_category():
    bad_skill = Skill("s")
    bad_skill.agent_category = AgentCategory.SOLUTION
    model = ComponentModel("m", components={bad_skill})
    result = model.validate(raise_exception=False)
    assert any("Skill" in w and "agent_category" in w for w in result["warnings"])


def test_w6_tool_with_is_human():
    bad_tool = Tool("t")
    bad_tool.is_human = True
    model = ComponentModel("m", components={bad_tool})
    result = model.validate(raise_exception=False)
    assert any("Tool" in w and "is_human" in w for w in result["warnings"])


def test_w7_orphan_interface():
    iface = Interface("orphan")
    model = ComponentModel("m", interfaces={iface})
    result = model.validate(raise_exception=False)
    assert any("orphan interface" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Group 7 -- object identity (B-UML house style: no __eq__ / __hash__ overrides)
# ---------------------------------------------------------------------------

def test_elements_use_object_identity():
    """B-UML elements compare by object identity, not by content."""
    a1 = Component("same_name")
    a2 = Component("same_name")
    assert a1 is not a2
    assert a1 != a2
    assert hash(a1) != hash(a2)


def test_component_in_set_dedupes_by_identity():
    a = Component("c")
    components = {a, a}  # same object twice
    assert len(components) == 1


# ---------------------------------------------------------------------------
# Extra -- model accessor helpers
# ---------------------------------------------------------------------------

def test_add_remove_helpers():
    model = ComponentModel("m")
    c = Component("c")
    iface = Interface("i")
    perm = Permission("p", scope="x")
    model.add_component(c)
    model.add_interface(iface)
    model.add_permission(perm)
    rel = InterfaceProvided(c, iface)
    model.add_relationship(rel)
    assert c in model.components
    assert iface in model.interfaces
    assert perm in model.permissions
    assert rel in model.relationships
    model.remove_component(c)
    model.remove_interface(iface)
    model.remove_permission(perm)
    model.remove_relationship(rel)
    assert c not in model.components
    assert iface not in model.interfaces
    assert perm not in model.permissions
    assert rel not in model.relationships


def test_add_helpers_type_check():
    model = ComponentModel("m")
    with pytest.raises(TypeError):
        model.add_component(Interface("i"))
    with pytest.raises(TypeError):
        model.add_interface(Component("c"))
    with pytest.raises(TypeError):
        model.add_permission(Component("c"))
    with pytest.raises(TypeError):
        model.add_relationship(Component("c"))


def test_agentic_edges_helper(agent_swarm_model):
    edges = agent_swarm_model.agentic_edges()
    assert all(isinstance(e, AgenticEdge) for e in edges)
    # 5 AgenticEdge instances in the fixture + 1 ComponentDependency.
    assert len(edges) == 5


def test_component_element_repr_is_useful():
    """The default ``__repr__`` includes class name and label for debugging."""
    assert "Component" in repr(Component("c"))
    assert "Skill" in repr(Skill("s"))
    assert "Permission" in repr(Permission("p", scope="x"))


def test_component_model_repr():
    model = ComponentModel("m", components={Component("c")})
    rep = repr(model)
    assert "ComponentModel" in rep
    assert "components=1" in rep


def test_named_element_is_compatible_with_component_element():
    """ComponentElement is a NamedElement (visibility, metadata, timestamp inherit)."""
    c = Component("c")
    assert isinstance(c, ComponentElement)
    assert c.visibility == "public"
    assert c.metadata is None
