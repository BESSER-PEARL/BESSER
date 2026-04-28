"""Tests for ``besser.BUML.notations.kg_to_buml.kg_to_class_diagram``."""

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
from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    Generalization,
    IntegerType,
    StringType,
    UNLIMITED_MAX_MULTIPLICITY,
)
from besser.BUML.notations.kg_to_buml import kg_to_class_diagram


RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
XSD = "http://www.w3.org/2001/XMLSchema#"
EX = "http://example.org/"


def _make_kg() -> KnowledgeGraph:
    """Hand-build a small but representative KG covering most edge cases."""
    kg = KnowledgeGraph(name="test_kg")
    nodes = {
        # Real classes
        "Person": KGClass(id="Person", label="Person", iri=EX + "Person"),
        "Employee": KGClass(id="Employee", label="Employee", iri=EX + "Employee"),
        "Organization": KGClass(id="Organization", label="Organization", iri=EX + "Organization"),
        # XSD datatype "classes" (the OWL parser produces them as KGClass)
        "xsd_int": KGClass(id="xsd_int", label="integer", iri=XSD + "integer"),
        "xsd_str": KGClass(id="xsd_str", label="string", iri=XSD + "string"),
        # Properties
        "knows": KGProperty(id="knows", label="knows", iri=EX + "knows"),
        "name": KGProperty(id="name", label="name", iri=EX + "name"),
        "age": KGProperty(id="age", label="age", iri=EX + "age"),
        "noDomain": KGProperty(id="noDomain", label="noDomain", iri=EX + "noDomain"),
        "noRange": KGProperty(id="noRange", label="noRange", iri=EX + "noRange"),
        "worksFor": KGProperty(id="worksFor", label="worksFor", iri=EX + "worksFor"),
        # Individuals & literal (for ABox-driven multiplicity bump)
        "alice": KGIndividual(id="alice", label="alice", iri=EX + "alice"),
        "bob": KGIndividual(id="bob", label="bob", iri=EX + "bob"),
        "lit1": KGLiteral(id="lit1", value="Alice"),
        "lit2": KGLiteral(id="lit2", value="Alicia"),
    }
    for n in nodes.values():
        kg.add_node(n)

    edges = [
        # knows: Person -> Person (self-referential object property)
        ("e1", "knows", "Person", RDFS + "domain"),
        ("e2", "knows", "Person", RDFS + "range"),
        # name: Person -> string (datatype property)
        ("e3", "name", "Person", RDFS + "domain"),
        ("e4", "name", "xsd_str", RDFS + "range"),
        # age: Person -> integer
        ("e5", "age", "Person", RDFS + "domain"),
        ("e6", "age", "xsd_int", RDFS + "range"),
        # noRange: Person -> ?  (no rdfs:range)
        ("e7", "noRange", "Person", RDFS + "domain"),
        # noDomain: ? -> string
        ("e8", "noDomain", "xsd_str", RDFS + "range"),
        # worksFor: Person -> Organization
        ("e9", "worksFor", "Person", RDFS + "domain"),
        ("e10", "worksFor", "Organization", RDFS + "range"),
        # subClassOf: Employee -> Person
        ("e11", "Employee", "Person", RDFS + "subClassOf"),
        # ABox: alice, bob with multi-valued name → should bump multiplicity
        ("at1", "alice", "Person", RDF + "type"),
        ("at2", "bob", "Person", RDF + "type"),
        ("an1", "alice", "lit1", EX + "name"),
        ("an2", "alice", "lit2", EX + "name"),
    ]
    for eid, src, tgt, pred in edges:
        kg.add_edge(KGEdge(id=eid, source=nodes[src], target=nodes[tgt], iri=pred))
    return kg


def _classes(domain) -> dict:
    return {c.name: c for c in domain.types if isinstance(c, Class) and c.name != "Thing"}


def test_kg_to_class_diagram_creates_real_classes_and_skips_xsd():
    cr = kg_to_class_diagram(_make_kg())
    classes = _classes(cr.domain_model)
    assert {"Person", "Employee", "Organization"} <= set(classes)
    # XSD datatype IRIs are not turned into classes.
    assert "integer" not in classes
    assert "string" not in classes


def test_subclass_becomes_generalization():
    cr = kg_to_class_diagram(_make_kg())
    classes = _classes(cr.domain_model)
    gens = [(g.specific.name, g.general.name) for g in cr.domain_model.generalizations]
    assert ("Employee", "Person") in gens
    assert classes["Employee"] in {g.specific for g in cr.domain_model.generalizations}


def test_object_property_becomes_binary_association():
    cr = kg_to_class_diagram(_make_kg())
    works_for = next((a for a in cr.domain_model.associations if a.name == "worksFor"), None)
    assert works_for is not None
    types = {e.type.name for e in works_for.ends}
    assert types == {"Person", "Organization"}


def test_self_referential_association_has_distinct_ends():
    cr = kg_to_class_diagram(_make_kg())
    knows = next((a for a in cr.domain_model.associations if a.name == "knows"), None)
    assert knows is not None
    end_names = sorted(e.name for e in knows.ends)
    assert len(set(end_names)) == 2
    assert all(e.type.name == "Person" for e in knows.ends)
    # Source-end is recorded for disambiguation in object diagrams.
    assert id(knows) in cr.assoc_source_end


def test_datatype_property_becomes_attribute():
    cr = kg_to_class_diagram(_make_kg())
    person = _classes(cr.domain_model)["Person"]
    by_name = {a.name: a for a in person.attributes}
    assert "age" in by_name and by_name["age"].type is IntegerType
    assert "name" in by_name and by_name["name"].type is StringType


def test_multivalued_literal_bumps_multiplicity():
    cr = kg_to_class_diagram(_make_kg())
    person = _classes(cr.domain_model)["Person"]
    name_attr = next(a for a in person.attributes if a.name == "name")
    age_attr = next(a for a in person.attributes if a.name == "age")
    # alice has two :name literals → bump.
    assert name_attr.multiplicity.max == UNLIMITED_MAX_MULTIPLICITY
    # age is single-valued in the ABox.
    assert age_attr.multiplicity.max == 1
    codes = {w.code for w in cr.warnings}
    assert "MULTIVALUED_LITERAL" in codes


def test_property_without_domain_emits_warning():
    cr = kg_to_class_diagram(_make_kg())
    codes = {w.code for w in cr.warnings}
    assert "PROPERTY_NO_DOMAIN" in codes
    # The property is attached to the synthetic 'Thing' class.
    thing = next((c for c in cr.domain_model.types if isinstance(c, Class) and c.name == "Thing"), None)
    assert thing is not None
    assert any(a.name == "noDomain" for a in thing.attributes)


def test_property_without_range_warns_and_uses_string():
    cr = kg_to_class_diagram(_make_kg())
    codes = {w.code for w in cr.warnings}
    assert "PROPERTY_NO_RANGE" in codes
    person = _classes(cr.domain_model)["Person"]
    no_range = next((a for a in person.attributes if a.name == "noRange"), None)
    assert no_range is not None
    assert no_range.type is StringType


def test_cyclic_subclass_is_dropped():
    kg = KnowledgeGraph(name="cyclic")
    a = KGClass(id="a", label="A", iri=EX + "A")
    b = KGClass(id="b", label="B", iri=EX + "B")
    for n in (a, b):
        kg.add_node(n)
    kg.add_edge(KGEdge(id="ab", source=a, target=b, iri=RDFS + "subClassOf"))
    kg.add_edge(KGEdge(id="ba", source=b, target=a, iri=RDFS + "subClassOf"))
    cr = kg_to_class_diagram(kg)
    # Only one of the two edges may have been kept.
    assert len(cr.domain_model.generalizations) == 1
    assert any(w.code == "CYCLIC_SUBCLASS" for w in cr.warnings)


def test_blank_nodes_are_ignored_in_class_pass():
    kg = KnowledgeGraph(name="blank")
    person = KGClass(id="p", label="Person", iri=EX + "Person")
    blank = KGBlank(id="b", label="_:b1")
    kg.add_node(person)
    kg.add_node(blank)
    kg.add_edge(KGEdge(id="b_sc_p", source=blank, target=person, iri=RDFS + "subClassOf"))
    cr = kg_to_class_diagram(kg)
    # No generalization involving the blank.
    assert all(
        not (g.specific.name.startswith("blank") or g.general.name.startswith("blank"))
        for g in cr.domain_model.generalizations
    )
