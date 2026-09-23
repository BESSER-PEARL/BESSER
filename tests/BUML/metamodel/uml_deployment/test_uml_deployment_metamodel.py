"""Tests for the UML Deployment metamodel (``besser.BUML.metamodel.uml_deployment``).

Groups, following ``.claude/component-deployment/01-component-deployment-design.md``
§1 / §2.2 / §4.2:

1. Construction validation -- setters raise on bad input; free-text names accepted.
2. Containment -- Node.nested_artifacts / nested_nodes + parent back-reference.
3. Relationship endpoint type rules (construction-time TypeError).
4. ``Multiplicity`` reuse on DeploymentRelation.
5. ``DeploymentModel.validate()`` -- rules E1..E8.
6. Warnings W1..W4.
7. Identity -- elements use object identity.
"""

import pytest

from besser.BUML.metamodel.structural import (
    Multiplicity, UNLIMITED_MAX_MULTIPLICITY,
)
from besser.BUML.metamodel.uml_deployment import (
    Artifact, CommunicationPath, DeploymentDependency, DeploymentElement,
    DeploymentModel, DeploymentRelation, DeploymentRelationship, Interface,
    InterfaceProvided, InterfaceRequired, Locality, Node, NodeKind,
)


# ---------------------------------------------------------------------------
# Group 1 -- construction validation
# ---------------------------------------------------------------------------

def test_free_text_names_are_accepted():
    """D9 -- Deployment labels are free text."""
    assert Node("UNP sandbox VM").name == "UNP sandbox VM"
    assert Artifact("Code Tester [3]").name == "Code Tester [3]"
    assert Node("").name == ""
    assert Node().name == ""
    assert Node(None).name == ""
    assert DeploymentModel("My Deployment").name == "My Deployment"
    assert DeploymentModel("").name == ""


def test_name_must_be_str_or_none():
    with pytest.raises(TypeError):
        Node(123)
    with pytest.raises(TypeError):
        DeploymentModel(123)


def test_abstract_relationship_cannot_be_instantiated():
    with pytest.raises(TypeError):
        DeploymentRelationship(Node(), Node())


def test_node_kind_must_be_enum():
    with pytest.raises(TypeError):
        Node("n", kind="device")


def test_locality_must_be_enum():
    with pytest.raises(TypeError):
        Node("n", locality="external")
    with pytest.raises(TypeError):
        Artifact("a", locality="external")


def test_layout_must_be_dict_or_none():
    assert Node("n", layout={"x": 1}).layout == {"x": 1}
    assert Node("n").layout is None
    with pytest.raises(TypeError):
        Node("n", layout=[1, 2])


def test_stereotypes_must_be_list_of_str():
    assert Node("n", stereotypes=["«device»"]).stereotypes == ["«device»"]
    with pytest.raises(TypeError):
        Node("n", stereotypes=[1])


def test_artifact_manifests_must_be_list_of_str():
    a = Artifact("a", manifests=["component_id_1"])
    assert a.manifests == ["component_id_1"]
    with pytest.raises(TypeError):
        Artifact("a", manifests=[123])


def test_artifact_agent_model_ref_default_none():
    """6b-2 — agent_model_ref defaults to None."""
    assert Artifact("a").agent_model_ref is None


def test_artifact_agent_model_ref_str_accepted():
    """6b-2 — agent_model_ref accepts a str UUID."""
    a = Artifact("a", agent_model_ref="some-uuid")
    assert a.agent_model_ref == "some-uuid"


def test_artifact_agent_model_ref_none_accepted():
    """6b-2 — agent_model_ref accepts None explicitly."""
    a = Artifact("a", agent_model_ref=None)
    assert a.agent_model_ref is None


def test_artifact_agent_model_ref_type_error():
    """6b-2 — agent_model_ref raises TypeError on non-str, non-None values."""
    with pytest.raises(TypeError):
        Artifact("a", agent_model_ref=5)


# ---------------------------------------------------------------------------
# Group 2 -- containment (Node.nested_artifacts / nested_nodes + parent)
# ---------------------------------------------------------------------------

def test_node_sets_nested_artifact_parent():
    a = Artifact("a")
    n = Node("n", nested_artifacts={a})
    assert a.parent is n
    assert a in n.nested_artifacts


def test_node_add_remove_artifact_maintains_parent():
    a = Artifact("a")
    n = Node("n")
    n.add_artifact(a)
    assert a.parent is n
    n.remove_artifact(a)
    assert a.parent is None
    assert a not in n.nested_artifacts


def test_node_nested_nodes_set_parent():
    inner = Node("inner", kind=NodeKind.EXECUTION_ENVIRONMENT)
    outer = Node("outer", kind=NodeKind.DEVICE, nested_nodes={inner})
    assert inner.parent is outer
    assert inner in outer.nested_nodes


def test_node_cannot_be_nested_in_itself():
    n = Node("n")
    with pytest.raises(ValueError):
        n.nested_nodes = {n}
    with pytest.raises(ValueError):
        n.add_nested_node(n)


def test_node_nested_artifacts_type_checked():
    with pytest.raises(TypeError):
        Node("n", nested_artifacts={Interface("i")})


def test_node_nested_nodes_type_checked():
    with pytest.raises(TypeError):
        Node("n", nested_nodes={Artifact("a")})


def test_all_artifacts_and_all_nodes_traversals():
    inner_node = Node("inner", kind=NodeKind.EXECUTION_ENVIRONMENT)
    outer_node = Node("outer", kind=NodeKind.DEVICE,
                      nested_nodes={inner_node})
    nested_artifact = Artifact("nested_a", manifests=["c"])
    inner_node.add_artifact(nested_artifact)

    root_artifact = Artifact("root_a", manifests=["c"])

    model = DeploymentModel(
        "m", nodes={outer_node}, artifacts={root_artifact}
    )
    assert outer_node in model.all_nodes()
    assert inner_node in model.all_nodes()
    assert root_artifact in model.all_artifacts()
    assert nested_artifact in model.all_artifacts()


# ---------------------------------------------------------------------------
# Group 3 -- relationship endpoint construction-time rules
# ---------------------------------------------------------------------------

def test_deployment_relation_artifact_to_node():
    a = Artifact("a")
    n = Node("n")
    rel = DeploymentRelation(a, n)
    assert rel.source is a and rel.target is n
    with pytest.raises(TypeError):
        DeploymentRelation(n, a)  # swapped
    with pytest.raises(TypeError):
        DeploymentRelation(a, a)  # target not a Node
    with pytest.raises(TypeError):
        DeploymentRelation(n, n)  # source not an Artifact


def test_communication_path_node_to_node():
    n1, n2 = Node("a"), Node("b")
    rel = CommunicationPath(n1, n2)
    assert rel.source is n1 and rel.target is n2
    with pytest.raises(TypeError):
        CommunicationPath(Artifact("x"), n2)
    with pytest.raises(TypeError):
        CommunicationPath(n1, Artifact("x"))


def test_interface_provided_endpoints_accept_node_or_artifact():
    iface = Interface("i")
    InterfaceProvided(Node("n"), iface)
    InterfaceProvided(Artifact("a"), iface)
    with pytest.raises(TypeError):
        InterfaceProvided(iface, iface)
    with pytest.raises(TypeError):
        InterfaceProvided(Node("n"), Artifact("a"))


def test_interface_required_endpoints_accept_node_or_artifact():
    iface = Interface("i")
    InterfaceRequired(Node("n"), iface)
    InterfaceRequired(Artifact("a"), iface)
    with pytest.raises(TypeError):
        InterfaceRequired(iface, Node("n"))


def test_deployment_dependency_accepts_any_deployment_element():
    """Plain UML dependency -- base ``DeploymentElement`` endpoint type
    suffices."""
    DeploymentDependency(Artifact("a"), Node("n"))
    DeploymentDependency(Node("n"), Interface("i"))
    DeploymentDependency(Interface("i"), Artifact("a"))


# ---------------------------------------------------------------------------
# Group 4 -- Multiplicity reuse on DeploymentRelation (D6)
# ---------------------------------------------------------------------------

def test_deployment_relation_default_multiplicity_is_one_one():
    rel = DeploymentRelation(Artifact("a"), Node("n"))
    assert rel.multiplicity.min == 1
    assert rel.multiplicity.max == 1


def test_deployment_relation_carries_three_multiplicity():
    """R-20: the load-bearing test -- Code Tester [3] in the UNP scenario."""
    rel = DeploymentRelation(
        Artifact("CodeTester"), Node("CIRunner"),
        multiplicity=Multiplicity(3, 3),
    )
    assert rel.multiplicity.min == 3
    assert rel.multiplicity.max == 3


def test_deployment_relation_unlimited_multiplicity():
    """``[1..*]`` is representable via UNLIMITED_MAX_MULTIPLICITY."""
    rel = DeploymentRelation(
        Artifact("a"), Node("n"),
        multiplicity=Multiplicity(1, "*"),
    )
    assert rel.multiplicity.min == 1
    assert rel.multiplicity.max == UNLIMITED_MAX_MULTIPLICITY


def test_deployment_relation_multiplicity_type_checked():
    a, n = Artifact("a"), Node("n")
    with pytest.raises(TypeError):
        DeploymentRelation(a, n, multiplicity=(1, 3))


# ---------------------------------------------------------------------------
# Group 5 -- DeploymentModel.validate() errors (E1..E8)
# ---------------------------------------------------------------------------

def test_validate_passes_for_well_formed_models(
    minimal_deployment_model, unp_topology_model, interface_model
):
    """The three fixtures validate cleanly (no errors)."""
    for model in (minimal_deployment_model, unp_topology_model, interface_model):
        result = model.validate(raise_exception=False)
        assert result["success"] is True, (
            f"{model.name} unexpected errors: {result['errors']}"
        )


def test_validate_raises_when_requested():
    a, n = Artifact("a"), Node("n")
    rogue_node = Node("rogue")
    rel = DeploymentRelation(a, rogue_node, name="dangle")
    model = DeploymentModel("m", nodes={n}, artifacts={a}, relationships={rel})
    with pytest.raises(ValueError, match="not present"):
        model.validate(raise_exception=True)


def test_e1_dangling_endpoint_reference():
    a = Artifact("a")
    rogue_node = Node("rogue")  # not added to the model
    rel = DeploymentRelation(a, rogue_node, name="bad")
    model = DeploymentModel("m", artifacts={a}, relationships={rel})
    result = model.validate(raise_exception=False)
    assert any("not present in the model" in e for e in result["errors"])


def test_e7_orphan_parent_node():
    """Artifact.parent set to a Node not in the model."""
    rogue_parent = Node("rogue")
    a = Artifact("a")
    a.parent = rogue_parent
    model = DeploymentModel("m", artifacts={a})
    result = model.validate(raise_exception=False)
    assert any("not in the model" in e for e in result["errors"])


def test_e8_duplicate_node_names():
    n1, n2 = Node("dup"), Node("dup")
    model = DeploymentModel("m", nodes={n1, n2})
    result = model.validate(raise_exception=False)
    assert any("Duplicate node name" in e for e in result["errors"])


def test_e6_invalid_multiplicity_after_mutation():
    """The Multiplicity setters block invalid construction; we induce a bad
    state via direct private-slot mutation to exercise validate's safety net."""
    rel = DeploymentRelation(Artifact("a"), Node("n"),
                             multiplicity=Multiplicity(2, 5))
    # Force min > max via the private slot to simulate a buggy mutation.
    rel.multiplicity._Multiplicity__min = 9
    model = DeploymentModel("m", nodes={rel.target}, artifacts={rel.source},
                            relationships={rel})
    result = model.validate(raise_exception=False)
    assert any("invalid multiplicity" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# Group 6 -- warnings (W1..W4)
# ---------------------------------------------------------------------------

def test_w1_empty_node():
    """A Node with no nested artifacts and no incoming DeploymentRelation."""
    empty_node = Node("empty", kind=NodeKind.DEVICE)
    model = DeploymentModel("m", nodes={empty_node})
    result = model.validate(raise_exception=False)
    assert any(
        "empty" in w and "no nested artifacts" in w
        for w in result["warnings"]
    )


def test_w2_artifact_manifests_nothing():
    artifact = Artifact("a")  # no manifests
    node = Node("n", kind=NodeKind.DEVICE, nested_artifacts={artifact})
    model = DeploymentModel("m", nodes={node})
    result = model.validate(raise_exception=False)
    assert any("manifests no Component" in w for w in result["warnings"])


def test_w4_generic_node_with_no_stereotypes():
    """A Node with kind=GENERIC and no stereotype hints is under-typed."""
    n = Node("plain", kind=NodeKind.GENERIC)
    n.add_artifact(Artifact("a", manifests=["c"]))  # avoid W1
    model = DeploymentModel("m", nodes={n})
    result = model.validate(raise_exception=False)
    assert any("kind=GENERIC" in w for w in result["warnings"])


def test_unp_topology_has_no_warnings(unp_topology_model):
    """The UNP fixture is deliberately complete enough to avoid all warnings."""
    result = unp_topology_model.validate(raise_exception=False)
    assert result["warnings"] == [], (
        f"unexpected warnings: {result['warnings']}"
    )


# ---------------------------------------------------------------------------
# Group 7 -- object identity
# ---------------------------------------------------------------------------

def test_elements_use_object_identity():
    n1 = Node("same")
    n2 = Node("same")
    assert n1 is not n2
    assert n1 != n2
    assert hash(n1) != hash(n2)


def test_artifact_in_set_dedupes_by_identity():
    a = Artifact("a")
    artifacts = {a, a}
    assert len(artifacts) == 1


# ---------------------------------------------------------------------------
# Extra -- model accessor helpers + repr
# ---------------------------------------------------------------------------

def test_add_remove_helpers():
    model = DeploymentModel("m")
    n = Node("n", kind=NodeKind.DEVICE)
    a = Artifact("a", manifests=["c"])
    iface = Interface("i")
    rel = DeploymentRelation(a, n)
    model.add_node(n)
    model.add_artifact(a)
    model.add_interface(iface)
    model.add_relationship(rel)
    assert n in model.nodes
    assert a in model.artifacts
    assert iface in model.interfaces
    assert rel in model.relationships
    model.remove_node(n)
    model.remove_artifact(a)
    model.remove_interface(iface)
    model.remove_relationship(rel)
    assert n not in model.nodes


def test_add_helpers_type_check():
    model = DeploymentModel("m")
    with pytest.raises(TypeError):
        model.add_node(Artifact("a"))
    with pytest.raises(TypeError):
        model.add_artifact(Node("n"))
    with pytest.raises(TypeError):
        model.add_interface(Node("n"))
    with pytest.raises(TypeError):
        model.add_relationship(Node("n"))


def test_deployment_element_is_named_element():
    """DeploymentElement inherits NamedElement -- visibility / metadata accessible."""
    n = Node("n")
    assert isinstance(n, DeploymentElement)
    assert n.visibility == "public"
    assert n.metadata is None


def test_deployment_model_repr():
    n = Node("n", kind=NodeKind.DEVICE)
    n.add_artifact(Artifact("a", manifests=["c"]))
    model = DeploymentModel("m", nodes={n})
    rep = repr(model)
    assert "DeploymentModel" in rep
    assert "nodes=1" in rep


def test_repr_for_relationships_includes_endpoints():
    rel = DeploymentRelation(Artifact("a"), Node("n"), name="r")
    assert "Artifact" not in repr(rel)  # type name is on the rel, not endpoints
    assert "DeploymentRelation" in repr(rel)
    assert "'a'" in repr(rel)
    assert "'n'" in repr(rel)
