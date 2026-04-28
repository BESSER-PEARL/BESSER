"""Tests for ``besser.BUML.notations.kg_to_buml.kg_to_object_diagram``."""

import pytest

from besser.BUML.metamodel.kg import (
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KGProperty,
    KnowledgeGraph,
)
from besser.BUML.metamodel.object import Link
from besser.BUML.metamodel.structural import Class
from besser.BUML.notations.kg_to_buml import kg_to_class_diagram, kg_to_object_diagram


RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
XSD = "http://www.w3.org/2001/XMLSchema#"
EX = "http://example.org/"


def _make_kg() -> KnowledgeGraph:
    """KG with: two individuals, one literal slot, one object link, one blank,
    one untyped individual, and one individual with two literal values for
    the same property (multi-valued)."""
    kg = KnowledgeGraph(name="ti")
    nodes = {
        "Person": KGClass(id="Person", label="Person", iri=EX + "Person"),
        "name": KGProperty(id="name", label="name", iri=EX + "name"),
        "age": KGProperty(id="age", label="age", iri=EX + "age"),
        "knows": KGProperty(id="knows", label="knows", iri=EX + "knows"),
        "xsd_str": KGClass(id="xsd_str", label="string", iri=XSD + "string"),
        "xsd_int": KGClass(id="xsd_int", label="integer", iri=XSD + "integer"),
        "alice": KGIndividual(id="alice", label="alice", iri=EX + "alice"),
        "bob": KGIndividual(id="bob", label="bob", iri=EX + "bob"),
        "ghost": KGIndividual(id="ghost", label="ghost", iri=EX + "ghost"),
        "lit1": KGLiteral(id="lit1", value="Alice"),
        "lit2": KGLiteral(id="lit2", value="Alicia"),
        "lit3": KGLiteral(id="lit3", value="30", datatype=XSD + "integer"),
        "lit_bad": KGLiteral(id="lit_bad", value="not-a-number", datatype=XSD + "integer"),
        "blank1": KGBlank(id="blank1", label="_:b1"),
    }
    for n in nodes.values():
        kg.add_node(n)
    for eid, src, tgt, pred in [
        ("e1", "name", "Person", RDFS + "domain"),
        ("e2", "name", "xsd_str", RDFS + "range"),
        ("e3", "age", "Person", RDFS + "domain"),
        ("e4", "age", "xsd_int", RDFS + "range"),
        ("e5", "knows", "Person", RDFS + "domain"),
        ("e6", "knows", "Person", RDFS + "range"),
        ("t1", "alice", "Person", RDF + "type"),
        ("t2", "bob", "Person", RDF + "type"),
        # ghost has no rdf:type → INDIVIDUAL_NO_TYPE
        ("a_n1", "alice", "lit1", EX + "name"),
        ("a_n2", "alice", "lit2", EX + "name"),  # multi-valued
        ("b_age_bad", "bob", "lit_bad", EX + "age"),  # type-mismatch coercion
        ("a_age", "alice", "lit3", EX + "age"),
        ("a_knows_b", "alice", "bob", EX + "knows"),
        ("a_knows_blank", "alice", "blank1", EX + "knows"),  # dropped (blank)
    ]:
        kg.add_edge(KGEdge(id=eid, source=nodes[src], target=nodes[tgt], iri=pred))
    return kg


def test_individuals_become_objects():
    cr = kg_to_class_diagram(_make_kg())
    obj = kg_to_object_diagram(_make_kg(), class_result=cr)
    names = {o.name_ for o in obj.object_model.objects}
    assert {"alice", "bob"} <= names


def test_blank_nodes_are_skipped():
    obj = kg_to_object_diagram(_make_kg())
    names = {o.name_ for o in obj.object_model.objects}
    assert all(not n.startswith("_blank") for n in names)
    assert any(w.code == "BLANK_SKIPPED" for w in obj.warnings)


def test_untyped_individual_falls_back_to_thing():
    obj = kg_to_object_diagram(_make_kg())
    ghost = next(o for o in obj.object_model.objects if o.name_ == "ghost")
    assert ghost.classifier.name == "Thing"
    assert any(w.code == "INDIVIDUAL_NO_TYPE" for w in obj.warnings)


def test_literal_slot_is_typed_correctly():
    obj = kg_to_object_diagram(_make_kg())
    alice = next(o for o in obj.object_model.objects if o.name_ == "alice")
    age_slot = next((s for s in alice.slots if s.attribute.name == "age"), None)
    assert age_slot is not None
    assert age_slot.value.value == 30


def test_multi_valued_literal_creates_multiple_slots():
    obj = kg_to_object_diagram(_make_kg())
    alice = next(o for o in obj.object_model.objects if o.name_ == "alice")
    name_slots = [s for s in alice.slots if s.attribute.name == "name"]
    assert len(name_slots) == 2
    assert {s.value.value for s in name_slots} == {"Alice", "Alicia"}


def test_link_is_created_through_correct_association():
    obj = kg_to_object_diagram(_make_kg())
    links = [l for l in obj.object_model.links if isinstance(l, Link)]
    assert len(links) == 1
    link = links[0]
    assert link.association.name == "knows"
    object_names = sorted(le.object.name_ for le in link.connections)
    assert object_names == ["alice", "bob"]


def test_blank_target_does_not_create_link():
    obj = kg_to_object_diagram(_make_kg())
    # alice → blank1 was the only blank-targeting edge; should not appear.
    for link in obj.object_model.links:
        for le in link.connections:
            assert not le.object.name_.startswith("_blank")


def test_link_direction_for_self_referential_association():
    """Alice → Bob should map alice to the source-end and bob to the target-end."""
    cr = kg_to_class_diagram(_make_kg())
    obj = kg_to_object_diagram(_make_kg(), class_result=cr)
    knows = next(a for a in cr.domain_model.associations if a.name == "knows")
    source_end = cr.assoc_source_end[id(knows)]
    link = next(iter(obj.object_model.links))
    alice_le = next(le for le in link.connections if le.object.name_ == "alice")
    assert alice_le.association_end is source_end


def test_type_mismatch_literal_is_coerced_to_string_with_warning():
    obj = kg_to_object_diagram(_make_kg())
    bob = next(o for o in obj.object_model.objects if o.name_ == "bob")
    bad_age_slot = next((s for s in bob.slots if s.attribute.name == "age"), None)
    if bad_age_slot is not None:
        assert bad_age_slot.value.classifier.name == "str"
        assert any(w.code == "LITERAL_TYPE_COERCED" for w in obj.warnings)
    else:
        # Acceptable behaviour: the slot was dropped entirely with a coercion warning.
        assert any(w.code == "LITERAL_TYPE_COERCED" for w in obj.warnings)
