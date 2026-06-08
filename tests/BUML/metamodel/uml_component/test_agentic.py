"""Tests for the UML Component agentic extension
(``besser.BUML.metamodel.uml_component.agentic``).

Covers the agentic-swarm profile extracted from the base metamodel by the
``04-`` split: ``AgenticComponent`` / ``Skill`` / ``Tool`` / ``Permission`` /
``AgenticEdge`` / ``AgenticComponentModel``.

Groups:

1. Construction validation -- agentic setters raise on bad input.
2. AgenticEdge per-kind endpoint type rules (construction-time TypeError).
3. AgenticComponentModel.validate() -- agentic rules (E5-E12) + warnings (W1-W3).
4. Re-export -- the package __init__ exposes both base and extension names.
"""

import pytest

from besser.BUML.metamodel.uml_component import (
    AgentCategory, AgenticComponent, AgenticComponentModel, AgenticEdge,
    AgenticEdgeKind, Component, ComponentDependency, ComponentModel, Interface,
    Locality, Permission, Skill, Tool,
)


# ---------------------------------------------------------------------------
# Group 1 -- construction validation
# ---------------------------------------------------------------------------

def test_agentic_subtypes_instantiate():
    assert isinstance(AgenticComponent(), Component)
    assert isinstance(Skill(), Component)
    assert isinstance(Tool(), Component)
    assert isinstance(
        AgenticEdge(AgenticComponent(), Skill(), kind=AgenticEdgeKind.HAS),
        ComponentDependency,
    )
    assert isinstance(AgenticComponentModel("m"), ComponentModel)


def test_agentic_component_carries_agent_profile():
    a = AgenticComponent(
        "agent", agent_category=AgentCategory.SOLUTION,
        process_model_refs=["proc_a"],
    )
    assert a.agent_category is AgentCategory.SOLUTION
    assert a.process_model_refs == ["proc_a"]


def test_agentic_component_default_category_is_none():
    """NONE = agent whose role is not yet specified."""
    assert AgenticComponent("a").agent_category is AgentCategory.NONE


def test_agentic_component_inherits_base_fields():
    a = AgenticComponent("a", locality=Locality.EXTERNAL, realizes=["ClassA"])
    assert a.locality is Locality.EXTERNAL
    assert a.realizes == ["ClassA"]


def test_agent_category_is_type_checked():
    with pytest.raises(TypeError):
        AgenticComponent("c", agent_category="solution")


def test_process_model_refs_must_be_list_of_str():
    a = AgenticComponent("c", process_model_refs=["proc_b"])
    assert a.process_model_refs == ["proc_b"]
    with pytest.raises(TypeError):
        AgenticComponent("c", process_model_refs=[None])


def test_permission_scope_must_be_str():
    assert Permission("p", scope="repo:read").scope == "repo:read"
    with pytest.raises(TypeError):
        Permission("p", scope=123)


def test_permission_default_scope_is_empty_str():
    """Empty scope is allowed construction-time; ``validate()`` flags it E12."""
    assert Permission("p").scope == ""


# ---------------------------------------------------------------------------
# Group 2 -- AgenticEdge per-kind endpoint construction-time rules
# ---------------------------------------------------------------------------

def test_agentic_edge_has_requires_agent_source_skill_target():
    agent = AgenticComponent("a", agent_category=AgentCategory.SOLUTION)
    AgenticEdge(agent, Skill("s"), kind=AgenticEdgeKind.HAS)
    with pytest.raises(TypeError):
        AgenticEdge(agent, Tool("t"), kind=AgenticEdgeKind.HAS)
    with pytest.raises(TypeError):
        # plain Component is not an agent
        AgenticEdge(Component("plain"), Skill("s"), kind=AgenticEdgeKind.HAS)


def test_agentic_edge_uses_requires_agent_source_tool_target():
    agent = AgenticComponent("a", agent_category=AgentCategory.SOLUTION)
    AgenticEdge(agent, Tool("t"), kind=AgenticEdgeKind.USES)
    with pytest.raises(TypeError):
        AgenticEdge(agent, Skill("s"), kind=AgenticEdgeKind.USES)


def test_agentic_edge_granted_requires_permission_target():
    agent = AgenticComponent("a", agent_category=AgentCategory.SOLUTION)
    AgenticEdge(agent, Permission("p", scope="repo:read"),
                kind=AgenticEdgeKind.GRANTED)
    with pytest.raises(TypeError):
        AgenticEdge(agent, Tool("t"), kind=AgenticEdgeKind.GRANTED)


def test_agentic_edge_implements_tool_to_skill():
    tool, skill = Tool("t"), Skill("s")
    AgenticEdge(tool, skill, kind=AgenticEdgeKind.IMPLEMENTS)
    with pytest.raises(TypeError):
        AgenticEdge(Skill("s2"), skill, kind=AgenticEdgeKind.IMPLEMENTS)
    with pytest.raises(TypeError):
        AgenticEdge(tool, Tool("t2"), kind=AgenticEdgeKind.IMPLEMENTS)


def test_agentic_edge_agent_to_agent_requires_agentic_components():
    a = AgenticComponent("a", agent_category=AgentCategory.SOLUTION)
    b = AgenticComponent("b", agent_category=AgentCategory.SUPERVISION)
    AgenticEdge(a, b, kind=AgenticEdgeKind.DELEGATES)
    AgenticEdge(a, b, kind=AgenticEdgeKind.SUPERVISES)
    AgenticEdge(a, b, kind=AgenticEdgeKind.REVISES)
    AgenticEdge(a, b, kind=AgenticEdgeKind.COLLABORATES)
    # A plain Component is not an agent -- rejected at construction (subsumes
    # the old runtime agentic-state check E5).
    with pytest.raises(TypeError):
        AgenticEdge(Component("plain"), b, kind=AgenticEdgeKind.DELEGATES)
    with pytest.raises(TypeError):
        AgenticEdge(a, Interface("i"), kind=AgenticEdgeKind.DELEGATES)


def test_agentic_edge_kind_must_be_enum():
    a = AgenticComponent("a")
    b = AgenticComponent("b")
    with pytest.raises(TypeError):
        AgenticEdge(a, b, kind="delegates")


def test_agentic_edge_permissions_type_checked():
    a, b = AgenticComponent("a"), AgenticComponent("b")
    perm = Permission("p", scope="x")
    e = AgenticEdge(a, b, kind=AgenticEdgeKind.DELEGATES, permissions=[perm])
    assert e.permissions == [perm]
    with pytest.raises(TypeError):
        AgenticEdge(a, b, kind=AgenticEdgeKind.DELEGATES, permissions=["x"])


def test_agentic_edge_kind_setter_rechecks_endpoints():
    agent = AgenticComponent("a", agent_category=AgentCategory.SOLUTION)
    skill = Skill("s")
    edge = AgenticEdge(agent, skill, kind=AgenticEdgeKind.HAS)
    # Switching to USES requires target to be Tool, but it's a Skill -> error.
    with pytest.raises(TypeError):
        edge.kind = AgenticEdgeKind.USES
    # Kind unchanged after the failed setter.
    assert edge.kind == AgenticEdgeKind.HAS


# ---------------------------------------------------------------------------
# Group 3 -- AgenticComponentModel.validate()
# ---------------------------------------------------------------------------

def test_validate_passes_for_well_formed_swarm(agent_swarm_model):
    result = agent_swarm_model.validate(raise_exception=False)
    assert result["success"] is True
    assert result["errors"] == []


def test_e10_dangling_permission_reference():
    agent = AgenticComponent("agent", agent_category=AgentCategory.SOLUTION,
                             process_model_refs=["p"])
    delegate = AgenticComponent("delegate", process_model_refs=["p"])
    rogue_perm = Permission("rogue", scope="x")  # NOT in model.permissions
    edge = AgenticEdge(agent, delegate, kind=AgenticEdgeKind.DELEGATES,
                       permissions=[rogue_perm], name="d")
    model = AgenticComponentModel(
        "m", components={agent, delegate}, relationships={edge},
    )
    result = model.validate(raise_exception=False)
    assert any("not in the model" in e for e in result["errors"])


def test_e12_empty_permission_scope():
    perm = Permission("p")  # scope defaults to ""
    model = AgenticComponentModel("m", permissions={perm})
    result = model.validate(raise_exception=False)
    assert any("empty scope" in e for e in result["errors"])


def test_validate_agentic_endpoint_types_second_line():
    """The model re-checks AgenticEdge endpoint types as a second line of
    defence against a model wired through the private slots."""
    a = AgenticComponent("a", agent_category=AgentCategory.SOLUTION)
    b = AgenticComponent("b", agent_category=AgentCategory.SUPERVISION)
    edge = AgenticEdge(a, b, kind=AgenticEdgeKind.DELEGATES, name="d")
    plain = Component("plain")
    # Bypass the setter (which would reject) by writing the private slot.
    edge._ComponentRelationship__source = plain
    model = AgenticComponentModel(
        "m", components={plain, b}, relationships={edge},
    )
    result = model.validate(raise_exception=False)
    assert any("two AgenticComponents" in e for e in result["errors"])


def test_w1_agent_with_no_capabilities():
    agent = AgenticComponent("agent", agent_category=AgentCategory.SOLUTION,
                             process_model_refs=["p"])
    model = AgenticComponentModel("m", components={agent})
    result = model.validate(raise_exception=False)
    assert any("no outgoing" in w for w in result["warnings"])


def test_w2_agent_with_no_process_refs():
    skill = Skill("s")
    agent = AgenticComponent("agent", agent_category=AgentCategory.SOLUTION)
    has_skill = AgenticEdge(agent, skill, kind=AgenticEdgeKind.HAS, name="h")
    model = AgenticComponentModel(
        "m", components={agent, skill}, relationships={has_skill},
    )
    result = model.validate(raise_exception=False)
    assert any("process_model_refs is empty" in w for w in result["warnings"])


def test_w3_permissions_on_non_agent_to_agent_kind():
    agent = AgenticComponent("agent", agent_category=AgentCategory.SOLUTION,
                             process_model_refs=["p"])
    tool = Tool("t")
    perm = Permission("p1", scope="x")
    edge = AgenticEdge(agent, tool, kind=AgenticEdgeKind.USES,
                       permissions=[perm], name="bad")
    model = AgenticComponentModel(
        "m", components={agent, tool}, permissions={perm},
        relationships={edge},
    )
    result = model.validate(raise_exception=False)
    assert any("meaningful only on agent" in w for w in result["warnings"])


def test_agentic_component_model_runs_base_checks_too(minimal_component_model):
    """AgenticComponentModel.validate() also runs the base structural checks."""
    a = AgenticComponent("a")
    b = AgenticComponent("b")  # not in the model
    rel = ComponentDependency(a, b, name="dangle")
    model = AgenticComponentModel("m", components={a}, relationships={rel})
    result = model.validate(raise_exception=False)
    assert any("not present in the model" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# Extra -- helpers and re-export
# ---------------------------------------------------------------------------

def test_agentic_edges_helper(agent_swarm_model):
    edges = agent_swarm_model.agentic_edges()
    assert all(isinstance(e, AgenticEdge) for e in edges)
    # 5 AgenticEdge instances in the fixture + 1 ComponentDependency.
    assert len(edges) == 5


def test_agentic_model_permission_helpers():
    model = AgenticComponentModel("m")
    perm = Permission("p", scope="x")
    model.add_permission(perm)
    assert perm in model.permissions
    model.remove_permission(perm)
    assert perm not in model.permissions
    with pytest.raises(TypeError):
        model.add_permission(Component("c"))


def test_agentic_repr():
    assert "Skill" in repr(Skill("s"))
    assert "Permission" in repr(Permission("p", scope="x"))
    model = AgenticComponentModel("m", permissions={Permission("p", scope="x")})
    assert "AgenticComponentModel" in repr(model)
    assert "permissions=1" in repr(model)


def test_extension_names_are_re_exported():
    """The package __init__ re-exports both base and extension names."""
    from besser.BUML.metamodel import uml_component as pkg
    for name in ("AgenticComponent", "Skill", "Tool", "Permission",
                 "AgenticEdge", "AgenticComponentModel", "AgentCategory",
                 "AgenticEdgeKind", "Component", "ComponentModel", "Locality"):
        assert hasattr(pkg, name), f"{name} not re-exported"
