"""Tests for the KnowledgeGraph → rdflib projection.

The projection is the joint between the KG metamodel (which the editor owns)
and the paper's rule engine (which is defined over RDF triples). Its correctness
invariant is *semantic isomorphism*: converting a KG that came from a TTL file
must produce the same UML model as running the rules on that TTL directly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from besser.BUML.metamodel.kg import (
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KGNodeConstraint,
    KGProperty,
    KGPropertyConstraint,
    KnowledgeGraph,
)
from besser.BUML.metamodel.kg.constants import (
    CONSTRAINT_TARGET_CLASS,
    CONSTRAINT_TARGET_PROPERTY,
    SH_PROPERTY,
)
from besser.BUML.notations.kg_to_buml.kg_to_rdf import kg_to_rdf
from besser.BUML.notations.kg_to_buml.owl2uml import build_uml_model
from besser.utilities.owl_to_buml import owl_file_to_knowledge_graph

SH = "http://www.w3.org/ns/shacl#"
EX = "http://ex.org/"
XSD = "http://www.w3.org/2001/XMLSchema#"

FIXTURES = Path(__file__).parent / "fixtures" / "bibo"


def _write_ttl(tmp_path: Path, content: str) -> str:
    path = tmp_path / "ontology.ttl"
    path.write_text(content.strip(), encoding="utf-8")
    return str(path)


def _project(tmp_path: Path, ttl: str):
    warnings: list = []
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    graph, base = kg_to_rdf(kg, warnings=warnings)
    return graph, warnings


# ---------------------------------------------------------------------------
# Pass A — raw edges
# ---------------------------------------------------------------------------


def test_editor_wiring_predicates_are_not_projected():
    """``constraintTargetClass`` / ``constraintTargetProperty`` are editor-only;
    pass C turns them back into real OWL/SHACL triples instead."""
    kg = KnowledgeGraph(name="wiring")
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    nc = KGNodeConstraint(
        id="nc", label="PersonShape",
        metadata={"constraintSpecs": [], "source": "shacl"},
    )
    kg.add_node(person)
    kg.add_node(nc)
    kg.add_edge(KGEdge(id="e1", source=nc, target=person, iri=CONSTRAINT_TARGET_CLASS))

    graph, _ = kg_to_rdf(kg)
    assert (None, URIRef(CONSTRAINT_TARGET_CLASS), None) not in graph
    # ...but the SHACL form it stands for is present.
    assert (None, URIRef(SH + "targetClass"), URIRef(EX + "Person")) in graph


def test_literal_subjects_are_skipped_with_a_warning():
    kg = KnowledgeGraph(name="bad")
    literal = KGLiteral(id="lit", value="oops")
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    kg.add_node(literal)
    kg.add_node(person)
    kg.add_edge(KGEdge(id="e1", source=literal, target=person, iri=str(RDFS.subClassOf)))

    warnings: list = []
    graph, _ = kg_to_rdf(kg, warnings=warnings)
    assert len(graph) == 0 or (None, RDFS.subClassOf, None) not in graph
    assert any(w.code == "KG_LITERAL_SUBJECT" for w in warnings)


def test_sh_property_edges_are_projected():
    """The TTL exporter treats ``sh:property`` as internal because it rebuilds
    shapes from scratch; the projection must keep it or node shapes lose their
    property shapes."""
    kg = KnowledgeGraph(name="shapes")
    nc = KGNodeConstraint(id="nc", metadata={"constraintSpecs": [], "source": "shacl"})
    pc = KGPropertyConstraint(
        id="pc", metadata={"constraintSpecs": [{"kind": "minCardinality", "value": 1}], "source": "shacl"}
    )
    kg.add_node(nc)
    kg.add_node(pc)
    kg.add_edge(KGEdge(id="e1", source=nc, target=pc, iri=SH_PROPERTY))

    graph, _ = kg_to_rdf(kg)
    assert (None, URIRef(SH + "property"), None) in graph


# ---------------------------------------------------------------------------
# Pass A2 — declaration synthesis
# ---------------------------------------------------------------------------


def test_editor_authored_nodes_get_owl_declarations():
    """A KG drawn in the editor has no ``rdf:type owl:Class`` triples; without
    synthesising them the rule engine would see an empty ontology."""
    kg = KnowledgeGraph(name="drawn")
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    pet = KGClass(id="Pet", label="Pet", iri=EX + "Pet")
    owns = KGProperty(id="owns", label="owns", iri=EX + "owns")
    age = KGProperty(id="age", label="age", iri=EX + "age")
    xsd_int = KGClass(id="xsd_int", label="integer", iri=XSD + "integer")
    alice = KGIndividual(id="alice", label="alice", iri=EX + "alice")
    for node in (person, pet, owns, age, xsd_int, alice):
        kg.add_node(node)
    kg.add_edge(KGEdge(id="e1", source=owns, target=person, iri=str(RDFS.domain)))
    kg.add_edge(KGEdge(id="e2", source=owns, target=pet, iri=str(RDFS.range)))
    kg.add_edge(KGEdge(id="e3", source=age, target=person, iri=str(RDFS.domain)))
    kg.add_edge(KGEdge(id="e4", source=age, target=xsd_int, iri=str(RDFS.range)))

    graph, _ = kg_to_rdf(kg)
    assert (URIRef(EX + "Person"), RDF.type, OWL.Class) in graph
    assert (URIRef(EX + "owns"), RDF.type, OWL.ObjectProperty) in graph
    assert (URIRef(EX + "age"), RDF.type, OWL.DatatypeProperty) in graph
    assert (URIRef(EX + "alice"), RDF.type, OWL.NamedIndividual) in graph
    # xsd:integer is a datatype the importer models as a KGClass for display;
    # declaring it a class would put it in the diagram as one.
    assert (URIRef(XSD + "integer"), RDF.type, OWL.Class) not in graph


def test_declared_nodes_are_not_redeclared(tmp_path: Path):
    """An imported ontology keeps its own declarations, so classes it merely
    references stay recognisable as external stubs."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Person a owl:Class .
    :knows a owl:ObjectProperty ; rdfs:domain :Person ; rdfs:range :Undeclared .
    """
    graph, _ = _project(tmp_path, ttl)
    assert (URIRef(EX + "Undeclared"), RDF.type, OWL.Class) not in graph
    model = build_uml_model(graph, None, shapes=graph)
    assert model.classes["Undeclared"].is_stub is True


# ---------------------------------------------------------------------------
# Pass B — list re-collection
# ---------------------------------------------------------------------------


def test_flattened_union_is_rebuilt_in_source_order(tmp_path: Path):
    """The importer replaces the ``rdf:List`` spine with direct member edges.
    Rebuilding it in the recorded order matters: the auxiliary class is *named*
    after its operands."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Pet a owl:Class ; owl:equivalentClass [ a owl:Class ; owl:unionOf ( :Cat :Dog ) ] .
    :Cat a owl:Class .
    :Dog a owl:Class .
    """
    graph, warnings = _project(tmp_path, ttl)
    from rdflib.collection import Collection

    head = next(graph.objects(None, OWL.unionOf))
    assert [str(m) for m in Collection(graph, head)] == [EX + "Cat", EX + "Dog"]
    assert not any(w.code == "LIST_ORDER_INFERRED" for w in warnings)


def test_one_of_literals_stays_literal(tmp_path: Path):
    """The importer records a literal member by its lexical form, so rebuilding
    the list from the recorded strings alone would turn it into an IRI."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Color a owl:Class ; owl:equivalentClass [ a owl:Class ; owl:oneOf ( "red" "green" ) ] .
    """
    graph, _ = _project(tmp_path, ttl)
    from rdflib.collection import Collection

    head = next(graph.objects(None, OWL.oneOf))
    members = list(Collection(graph, head))
    assert all(isinstance(m, Literal) for m in members)
    assert [str(m) for m in members] == ["red", "green"]


def test_editor_authored_list_warns_about_inferred_order():
    kg = KnowledgeGraph(name="drawn_union")
    pet = KGClass(id="Pet", label="Pet", iri=EX + "Pet")
    cat = KGClass(id="Cat", label="Cat", iri=EX + "Cat")
    dog = KGClass(id="Dog", label="Dog", iri=EX + "Dog")
    for node in (pet, cat, dog):
        kg.add_node(node)
    kg.add_edge(KGEdge(id="e1", source=pet, target=cat, iri=str(OWL.unionOf)))
    kg.add_edge(KGEdge(id="e2", source=pet, target=dog, iri=str(OWL.unionOf)))

    warnings: list = []
    graph, _ = kg_to_rdf(kg, warnings=warnings)
    assert any(w.code == "LIST_ORDER_INFERRED" for w in warnings)
    from rdflib.collection import Collection

    head = next(graph.objects(None, OWL.unionOf))
    assert [str(m) for m in Collection(graph, head)] == [EX + "Cat", EX + "Dog"]


# ---------------------------------------------------------------------------
# Pass C / D — constraint structure and spec back-fill
# ---------------------------------------------------------------------------


def test_owl_restriction_is_reattached_to_its_owner(tmp_path: Path):
    """The importer deletes ``Class rdfs:subClassOf _:restriction`` when it lifts
    the restriction into a KGPropertyConstraint. The mapper needs it back."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Person a owl:Class ; rdfs:subClassOf [
        a owl:Restriction ; owl:onProperty :owns ;
        owl:minCardinality "1"^^xsd:nonNegativeInteger ] .
    :Pet a owl:Class .
    :owns a owl:ObjectProperty ; rdfs:domain :Person ; rdfs:range :Pet .
    """
    graph, _ = _project(tmp_path, ttl)
    restriction = next(graph.subjects(RDF.type, OWL.Restriction))
    assert (URIRef(EX + "Person"), RDFS.subClassOf, restriction) in graph
    assert (restriction, OWL.onProperty, URIRef(EX + "owns")) in graph


def test_spec_backfill_does_not_duplicate_a_raw_triple(tmp_path: Path):
    """A raw ``sh:maxCount`` edge and the equivalent spec must collapse to one
    triple, not two with different XSD types."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Person a owl:Class .
    :name a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
    :PersonShape a sh:NodeShape ; sh:targetClass :Person ;
        sh:property [ sh:path :name ; sh:maxCount 1 ] .
    """
    graph, _ = _project(tmp_path, ttl)
    assert len(list(graph.subject_objects(URIRef(SH + "maxCount")))) == 1


def test_consumed_shacl_list_is_recovered_from_specs(tmp_path: Path):
    """``sh:or`` has its list spine deleted by the importer with no replacement
    edge, so it can only come back from the constraint specs."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .

    :Person a owl:Class . :Pet a owl:Class . :Toy a owl:Class .
    :owns a owl:ObjectProperty ; rdfs:domain :Person ; rdfs:range :Pet .
    :PersonShape a sh:NodeShape ; sh:targetClass :Person ;
        sh:property [ sh:path :owns ; sh:or ( [ sh:class :Pet ] [ sh:class :Toy ] ) ] .
    """
    graph, _ = _project(tmp_path, ttl)
    from rdflib.collection import Collection

    head = next(graph.objects(None, URIRef(SH + "or")))
    members = list(Collection(graph, head))
    assert len(members) == 2
    classes = {str(graph.value(m, URIRef(SH + "class"))) for m in members}
    assert classes == {EX + "Pet", EX + "Toy"}


def test_sh_in_list_is_not_double_wrapped(tmp_path: Path):
    """Regression: ``sh:in``'s spine is consumed, not flattened, so pass B must
    not try to rebuild it from the (now dangling) list-head edge."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Person a owl:Class .
    :name a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
    :PersonShape a sh:NodeShape ; sh:targetClass :Person ;
        sh:property [ sh:path :name ; sh:in ( "Alice" "Bob" ) ] .
    """
    graph, _ = _project(tmp_path, ttl)
    from rdflib.collection import Collection

    head = next(graph.objects(None, URIRef(SH + "in")))
    members = list(Collection(graph, head))
    assert [str(m) for m in members] == ["Alice", "Bob"]


def test_editor_authored_node_shape_gets_type_and_target():
    """A node shape whose only job is to hold ``sh:property`` children still
    needs its ``rdf:type`` and ``sh:targetClass`` — that is what the SHACL phase
    keys on."""
    kg = KnowledgeGraph(name="drawn_shape")
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    name = KGProperty(id="name", label="name", iri=EX + "name")
    nc = KGNodeConstraint(id="nc", metadata={"constraintSpecs": [], "source": "shacl"})
    pc = KGPropertyConstraint(
        id="pc",
        metadata={"constraintSpecs": [{"kind": "minCardinality", "value": 1}], "source": "shacl"},
    )
    for node in (person, name, nc, pc):
        kg.add_node(node)
    kg.add_edge(KGEdge(id="e1", source=nc, target=person, iri=CONSTRAINT_TARGET_CLASS))
    kg.add_edge(KGEdge(id="e2", source=nc, target=pc, iri=SH_PROPERTY))
    kg.add_edge(KGEdge(id="e3", source=pc, target=name, iri=CONSTRAINT_TARGET_PROPERTY))

    graph, _ = kg_to_rdf(kg)
    shape = next(graph.subjects(RDF.type, URIRef(SH + "NodeShape")))
    assert (shape, URIRef(SH + "targetClass"), URIRef(EX + "Person")) in graph
    property_shape = next(graph.objects(shape, URIRef(SH + "property")))
    assert (property_shape, URIRef(SH + "path"), URIRef(EX + "name")) in graph
    assert (property_shape, URIRef(SH + "minCount"), None) in graph


def test_unresolvable_constraint_target_is_reported():
    kg = KnowledgeGraph(name="orphan")
    literal = KGLiteral(id="lit", value="not-a-class")
    nc = KGNodeConstraint(
        id="nc", label="Bogus", metadata={"constraintSpecs": [], "source": "shacl"}
    )
    kg.add_node(literal)
    kg.add_node(nc)
    kg.add_edge(KGEdge(id="e1", source=nc, target=literal, iri=CONSTRAINT_TARGET_CLASS))

    warnings: list = []
    kg_to_rdf(kg, warnings=warnings)
    assert any(w.code == "ORPHANED_CONSTRAINT" and w.node_id == "nc" for w in warnings)


def test_nested_shape_reference_to_unknown_node_is_reported():
    kg = KnowledgeGraph(name="dangling_ref")
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    name = KGProperty(id="name", label="name", iri=EX + "name")
    nc = KGNodeConstraint(id="nc", metadata={"constraintSpecs": [], "source": "shacl"})
    pc = KGPropertyConstraint(
        id="pc",
        metadata={
            "constraintSpecs": [{"kind": "shaclOr", "value": [{"ref": "does-not-exist"}]}],
            "source": "shacl",
        },
    )
    for node in (person, name, nc, pc):
        kg.add_node(node)
    kg.add_edge(KGEdge(id="e1", source=nc, target=person, iri=CONSTRAINT_TARGET_CLASS))
    kg.add_edge(KGEdge(id="e2", source=nc, target=pc, iri=SH_PROPERTY))
    kg.add_edge(KGEdge(id="e3", source=pc, target=name, iri=CONSTRAINT_TARGET_PROPERTY))

    warnings: list = []
    kg_to_rdf(kg, warnings=warnings)
    assert any(w.code == "SHAPE_REF_UNRESOLVED" for w in warnings)


# ---------------------------------------------------------------------------
# The isomorphism invariant
# ---------------------------------------------------------------------------


SYNTHETIC = """
@prefix : <http://ex.org/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:Agent a owl:Class .
:Person a owl:Class ; rdfs:subClassOf :Agent .
:Org a owl:Class ; rdfs:subClassOf :Agent ; owl:disjointWith :Person .
:Pet a owl:Class .
:Cat a owl:Class ; rdfs:subClassOf :Pet .
:Dog a owl:Class ; rdfs:subClassOf :Pet .

:owns a owl:ObjectProperty ; rdfs:domain :Person ; rdfs:range :Pet .
:ownedBy a owl:ObjectProperty ; owl:inverseOf :owns .
:knows a owl:ObjectProperty, owl:SymmetricProperty ; rdfs:domain :Person ; rdfs:range :Person .
:ancestorOf a owl:ObjectProperty, owl:TransitiveProperty ; rdfs:domain :Person ; rdfs:range :Person .
:name a owl:DatatypeProperty, owl:FunctionalProperty ; rdfs:domain :Agent ; rdfs:range xsd:string .
:age a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:integer .
:nickname a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .

:PetOwner a owl:Class ; owl:equivalentClass [
    a owl:Class ; owl:intersectionOf ( :Person [
        a owl:Restriction ; owl:onProperty :owns ; owl:someValuesFrom :Pet ] ) ] .
:Petless a owl:Class ; owl:equivalentClass [ a owl:Class ; owl:complementOf :PetOwner ] .
:AnyPet a owl:Class ; owl:equivalentClass [ a owl:Class ; owl:unionOf ( :Cat :Dog ) ] .
:Color a owl:Class ; owl:equivalentClass [ a owl:Class ; owl:oneOf ( "red" "green" ) ] .

:Bounded a owl:Class ; rdfs:subClassOf
    [ a owl:Restriction ; owl:onProperty :nickname ;
      owl:maxCardinality "3"^^xsd:nonNegativeInteger ] ,
    [ a owl:Restriction ; owl:onProperty :age ;
      owl:cardinality "1"^^xsd:nonNegativeInteger ] ,
    [ a owl:Restriction ; owl:onProperty :name ; owl:hasValue "fixed" ] ,
    [ a owl:Restriction ; owl:onProperty :knows ; owl:allValuesFrom :Person ] .

:Person owl:hasKey ( :name ) .

:PersonShape a sh:NodeShape ; sh:targetClass :Person ;
    sh:property [ sh:path :name ; sh:minCount 1 ; sh:maxCount 1 ;
                  sh:datatype xsd:string ; sh:pattern "^[A-Z]" ] ,
                [ sh:path :age ; sh:minInclusive 0 ; sh:maxInclusive 150 ] ,
                [ sh:path :nickname ; sh:in ( "Al" "Bob" ) ; sh:languageIn ( "en" ) ] ,
                [ sh:path :owns ; sh:or ( [ sh:class :Cat ] [ sh:class :Dog ] ) ] ,
                [ sh:path :knows ; sh:not [ sh:class :Org ] ] ,
                [ sh:path :ancestorOf ; sh:xone ( [ sh:class :Person ] [ sh:class :Org ] ) ] .
"""


def _digest(model) -> str:
    payload = {
        "classes": sorted(model.classes),
        "abstract": sorted(c.name for c in model.classes.values() if c.is_abstract),
        "datatypes": sorted(model.datatypes),
        "enumerations": sorted(model.enumerations),
        "generalizations": sorted(
            [g.subclass, g.superclass] for g in model.generalizations
        ),
        "associations": sorted(
            [a.source.type, a.target.type, a.target.role or "", a.source.role or ""]
            for a in model.associations
        ),
        "attributes": sorted(
            [c.name, a.name, a.type, a.multiplicity()]
            for c in model.classes.values() for a in c.attributes
        ),
        "invariants": sorted([i.context, i.body] for i in model.ocl_constraints()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _model_from_ttl(*paths: Path):
    graph = Graph()
    for path in paths:
        graph.parse(str(path), format="turtle")
    return build_uml_model(graph, None, shapes=graph)


def _model_via_kg(*paths: Path, tmp_path: Path):
    """Merge the sources, import them as a KG, then project and convert."""
    merged = Graph()
    for path in paths:
        merged.parse(str(path), format="turtle")
    combined = tmp_path / "combined.ttl"
    merged.serialize(destination=str(combined), format="turtle")
    kg = owl_file_to_knowledge_graph(str(combined))
    graph, base = kg_to_rdf(kg)
    return build_uml_model(graph, base, shapes=graph)


LOAD_BEARING_PREDICATES = [
    OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty,
]

#: Predicates whose count must match the source exactly.
EXACT_PARITY_LINKS = [
    RDFS.domain, RDFS.range,
    OWL.unionOf, OWL.intersectionOf, OWL.oneOf, OWL.complementOf,
    OWL.disjointWith, OWL.inverseOf, OWL.onProperty, OWL.hasKey,
    URIRef(SH + "path"), URIRef(SH + "targetClass"), URIRef(SH + "property"),
    URIRef(SH + "datatype"), URIRef(SH + "pattern"), URIRef(SH + "minCount"),
    URIRef(SH + "maxCount"), URIRef(SH + "or"), URIRef(SH + "not"),
    URIRef(SH + "xone"), URIRef(SH + "in"),
]

#: ``rdfs:subClassOf`` may legitimately *grow*. The importer records both
#: ``rdfs:subClassOf`` and ``owl:equivalentClass`` links to an anonymous class
#: expression the same way, so pass C re-emits every one of them as
#: ``rdfs:subClassOf``. That is lossless for the rule engine, which routes both
#: through the same ``resolve_class(..., subsuming=...)`` call.
NO_LOSS_LINKS = [RDFS.subClassOf]


@pytest.mark.parametrize("dataset", ["synthetic", "bibo"])
def test_projection_preserves_load_bearing_predicates(tmp_path: Path, dataset: str):
    """(a) no loss, and (b) exact parity on the predicates the rules read."""
    if dataset == "synthetic":
        sources = [Path(_write_ttl(tmp_path, SYNTHETIC))]
    else:
        sources = [FIXTURES / "bibo.ttl", FIXTURES / "bibo-shapes.ttl"]

    source = Graph()
    for path in sources:
        source.parse(str(path), format="turtle")
    combined = tmp_path / "combined.ttl"
    source.serialize(destination=str(combined), format="turtle")
    projected, _ = kg_to_rdf(owl_file_to_knowledge_graph(str(combined)))

    for rdf_type in LOAD_BEARING_PREDICATES:
        expected = {s for s in source.subjects(RDF.type, rdf_type) if isinstance(s, URIRef)}
        actual = {s for s in projected.subjects(RDF.type, rdf_type) if isinstance(s, URIRef)}
        assert expected <= actual, f"lost {rdf_type}: {sorted(expected - actual)[:5]}"

    for predicate in EXACT_PARITY_LINKS:
        expected = len(list(source.subject_objects(predicate)))
        actual = len(list(projected.subject_objects(predicate)))
        assert actual == expected, f"{predicate}: expected {expected}, projected {actual}"

    for predicate in NO_LOSS_LINKS:
        expected = len(list(source.subject_objects(predicate)))
        actual = len(list(projected.subject_objects(predicate)))
        assert actual >= expected, f"{predicate}: expected >= {expected}, projected {actual}"


@pytest.mark.parametrize("dataset", ["synthetic", "bibo"])
def test_projection_is_semantically_isomorphic(tmp_path: Path, dataset: str):
    """The headline invariant: converting via the KG metamodel produces exactly
    the same UML model as running the rules on the source TTL directly."""
    if dataset == "synthetic":
        sources = [Path(_write_ttl(tmp_path, SYNTHETIC))]
    else:
        sources = [FIXTURES / "bibo.ttl", FIXTURES / "bibo-shapes.ttl"]

    direct = _model_from_ttl(*sources)
    via_kg = _model_via_kg(*sources, tmp_path=tmp_path)
    assert _digest(via_kg) == _digest(direct)
