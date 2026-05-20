"""Shared fixtures for the UML Component metamodel tests.

Three reusable, well-formed models. ``minimal_component_model`` and
``subsystem_model`` are pure-UML ``ComponentModel``s; ``agent_swarm_model`` is
an ``AgenticComponentModel`` -- intentionally complete enough to round-trip the
UNP-style worked example from the thesis milestone (agents in agent
categories, skills, tools, a permission, a human, agentic edges).
"""

import pytest

from besser.BUML.metamodel.uml_component import (
    AgentCategory, AgenticComponent, AgenticComponentModel, AgenticEdge,
    AgenticEdgeKind, Component, ComponentDependency, ComponentModel, Interface,
    InterfaceProvided, InterfaceRequired, Locality, Permission, Skill,
    Subsystem, Tool,
)


@pytest.fixture
def minimal_component_model() -> ComponentModel:
    """Two plain Components dependent on one another via an Interface."""
    a = Component("ComponentA")
    b = Component("ComponentB")
    iface = Interface("Iface")
    provided = InterfaceProvided(a, iface, name="a_provides")
    required = InterfaceRequired(b, iface, name="b_requires")
    return ComponentModel(
        "minimal_component_model",
        components={a, b},
        interfaces={iface},
        relationships={provided, required},
    )


@pytest.fixture
def agent_swarm_model() -> AgenticComponentModel:
    """A simplified UNP-style swarm: one Solution agent, one Supervision
    agent, one human, one skill, one tool, one permission, full wiring."""
    advisor = AgenticComponent(
        "code_advisor",
        agent_category=AgentCategory.SOLUTION,
        process_model_refs=["proc_unp"],
    )
    supervisor = AgenticComponent(
        "reviewer_supervisor",
        agent_category=AgentCategory.SUPERVISION,
        process_model_refs=["proc_unp"],
    )
    human = AgenticComponent(
        "developer", is_human=True, process_model_refs=["proc_unp"]
    )
    llm = Component(
        "llm_endpoint",
        locality=Locality.EXTERNAL,
    )

    skill_search = Skill("code_search")
    tool_git = Tool("git_client")
    permission_approve = Permission(
        "approve_perm", scope="repo:merge:approve"
    )

    has_skill = AgenticEdge(advisor, skill_search, kind=AgenticEdgeKind.HAS,
                            name="advisor_has_search")
    uses_tool = AgenticEdge(advisor, tool_git, kind=AgenticEdgeKind.USES,
                            name="advisor_uses_git")
    sup_has = AgenticEdge(supervisor, skill_search, kind=AgenticEdgeKind.HAS,
                          name="supervisor_has_search")
    sup_uses = AgenticEdge(supervisor, tool_git, kind=AgenticEdgeKind.USES,
                           name="supervisor_uses_git")
    delegates = AgenticEdge(
        supervisor, human, kind=AgenticEdgeKind.DELEGATES,
        permissions=[permission_approve], name="sup_delegates_human",
    )
    llm_dep = ComponentDependency(advisor, llm, name="advisor_uses_llm")

    return AgenticComponentModel(
        "agent_swarm_model",
        components={advisor, supervisor, human, llm, skill_search, tool_git},
        permissions={permission_approve},
        relationships={
            has_skill, uses_tool, sup_has, sup_uses, delegates, llm_dep,
        },
    )


@pytest.fixture
def subsystem_model() -> ComponentModel:
    """A Subsystem containing two child Components."""
    a = Component("ChildA")
    b = Component("ChildB")
    sub = Subsystem("Cluster", children={a, b})
    return ComponentModel(
        "subsystem_model",
        components={sub, a, b},
    )
