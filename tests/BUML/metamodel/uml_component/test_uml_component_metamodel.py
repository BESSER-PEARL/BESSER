"""Tests for the base UML Component metamodel
(``besser.BUML.metamodel.uml_component.uml_component`` -- pure UML 2.5).

The agentic-extension tests (``AgenticComponent`` / ``Skill`` / ``Tool`` /
``Permission`` / ``AgenticEdge`` / ``AgenticComponentModel``) live in
``test_agentic.py`` after the ``04-`` base/agentic split.

Groups:

1. Construction validation -- setters raise on bad input; free-text names accepted.
2. Containment -- Subsystem children + parent back-reference.
3. Relationship endpoint type rules (construction-time TypeError).
4. ``ComponentModel.validate()`` -- base rules (E1-E4, E11, E13, W7).
5. Identity -- elements use object identity (no ``__eq__`` / ``__hash__``).
"""

import pytest

from besser.BUML.metamodel.uml_component import (
    Component, ComponentDependency, ComponentElement, ComponentModel,
    ComponentRelationship, Interface, InterfaceProvided, InterfaceRequired,
    Subsystem,
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


def test_subsystem_is_a_component():
    assert isinstance(Subsystem(), Component)


def test_base_component_has_no_agentic_fields():
    """04- split: agentic fields live on AgenticComponent, not base Component."""
    c = Component("c")
    assert not hasattr(c, "agent_category")
    assert not hasattr(c, "is_human")
    assert not hasattr(c, "process_model_refs")


def test_locality_is_type_checked():
    with pytest.raises(TypeError):
        Component("c", locality="external")


def test_layout_must_be_dict_or_none():
    assert Component("c", layout={"x": 1}).layout == {"x": 1}
    assert Component("c").layout is None
    with pytest.raises(TypeError):
        Component("c", layout=[1, 2])


def test_stereotypes_must_be_list_of_str():
    assert Component("c", stereotypes=["«custom»"]).stereotypes == ["«custom»"]
    with pytest.raises(TypeError):
        Component("c", stereotypes=[1])


def test_realizes_must_be_list_of_str():
    c = Component("c", realizes=["class_a"])
    assert c.realizes == ["class_a"]
    with pytest.raises(TypeError):
        Component("c", realizes=[123])


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
# Group 4 -- ComponentModel.validate() (base rules)
# ---------------------------------------------------------------------------

def test_validate_passes_for_well_formed_model(minimal_component_model):
    result = minimal_component_model.validate(raise_exception=False)
    assert result["success"] is True
    assert result["errors"] == []


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


def test_e11_duplicate_component_names():
    a1 = Component("dup")
    a2 = Component("dup")
    model = ComponentModel("m", components={a1, a2})
    result = model.validate(raise_exception=False)
    assert any("Duplicate component name" in e for e in result["errors"])


def test_e13_orphan_parent_subsystem():
    rogue_sub = Subsystem("rogue")
    a = Component("a")
    a.parent = rogue_sub  # rogue not in model
    model = ComponentModel("m", components={a})
    result = model.validate(raise_exception=False)
    assert any("not in the model" in e for e in result["errors"])


def test_w7_orphan_interface():
    iface = Interface("orphan")
    model = ComponentModel("m", interfaces={iface})
    result = model.validate(raise_exception=False)
    assert any("orphan interface" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Group 5 -- object identity (B-UML house style: no __eq__ / __hash__ overrides)
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
    model.add_component(c)
    model.add_interface(iface)
    rel = InterfaceProvided(c, iface)
    model.add_relationship(rel)
    assert c in model.components
    assert iface in model.interfaces
    assert rel in model.relationships
    model.remove_component(c)
    model.remove_interface(iface)
    model.remove_relationship(rel)
    assert c not in model.components
    assert iface not in model.interfaces
    assert rel not in model.relationships


def test_add_helpers_type_check():
    model = ComponentModel("m")
    with pytest.raises(TypeError):
        model.add_component(Interface("i"))
    with pytest.raises(TypeError):
        model.add_interface(Component("c"))
    with pytest.raises(TypeError):
        model.add_relationship(Component("c"))


def test_component_element_repr_is_useful():
    """The default ``__repr__`` includes class name and label for debugging."""
    assert "Component" in repr(Component("c"))
    assert "Subsystem" in repr(Subsystem("s"))


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
