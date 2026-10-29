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
from rdflib import Graph, Literal, URIRef
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
from besser.BUML.notations.kg_rdf import kg_to_rdf
from besser.BUML.notations.kg_to_buml.owl2uml import plan_domain_model
from besser.BUML.notations.kg_to_buml.to_buml import build_domain_model
from besser.BUML.notations.kg_rdf import owl_file_to_knowledge_graph

SH = "http://www.w3.org/ns/shacl#"
EX = "http://ex.org/"
XSD = "http://www.w3.org/2001/XMLSchema#"

FIXTURES = Path(__file__).resolve().parents[1] / "kg_to_buml" / "fixtures" / "bibo"


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
    model = plan_domain_model(graph, None, shapes=graph)
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


def test_restriction_on_an_anonymous_inverse_keeps_its_filler(tmp_path: Path):
    """The importer keeps ``owl:onProperty [ owl:inverseOf :p ]`` only as a raw
    edge, with no ``onPropertyIri``; the restriction's filler, which lives in its
    spec, used to be dropped for want of a property."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Org a owl:Class .
    :employs a owl:ObjectProperty ; rdfs:domain :Org .
    :Employee a owl:Class ; rdfs:subClassOf [
        a owl:Restriction ; owl:onProperty [ owl:inverseOf :employs ] ;
        owl:someValuesFrom :Org ] .
    """
    graph, _ = _project(tmp_path, ttl)
    (restriction,) = graph.objects(URIRef(EX + "Employee"), RDFS.subClassOf)
    assert graph.value(restriction, OWL.someValuesFrom) == URIRef(EX + "Org")
    assert graph.value(graph.value(restriction, OWL.onProperty), OWL.inverseOf) == URIRef(EX + "employs")


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


def _digest(domain_model) -> str:
    """Hash the *lowered* model, i.e. what a user gets, not the internal plan.

    Digesting the pre-lowering plan tied this test to how that plan is
    represented; the parity it asserts (direct vs. via the KG metamodel) is a
    property of the result.
    """
    classes = sorted(domain_model.get_classes(), key=lambda c: c.name)
    payload = {
        "classes": [c.name for c in classes],
        "abstract": [c.name for c in classes if c.is_abstract],
        "enumerations": sorted(e.name for e in domain_model.get_enumerations()),
        "generalizations": sorted(
            [g.specific.name, g.general.name] for g in domain_model.generalizations
        ),
        "associations": sorted(
            [a.name, sorted(
                [e.name, e.type.name, e.multiplicity.min, e.multiplicity.max, e.is_navigable]
                for e in a.ends
            )]
            for a in domain_model.associations
        ),
        "attributes": sorted(
            [c.name, a.name, a.type.name, a.multiplicity.min, a.multiplicity.max]
            for c in classes for a in c.attributes
        ),
        "constraints": sorted([c.context.name, c.expression] for c in domain_model.constraints),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _lowered(plan):
    return build_domain_model(plan, model_name="Parity", warnings=[]).domain_model


def _model_from_ttl(*paths: Path):
    graph = Graph()
    for path in paths:
        graph.parse(str(path), format="turtle")
    return _lowered(plan_domain_model(graph, None, shapes=graph))


def _model_via_kg(*paths: Path, tmp_path: Path):
    """Merge the sources, import them as a KG, then project and convert."""
    merged = Graph()
    for path in paths:
        merged.parse(str(path), format="turtle")
    combined = tmp_path / "combined.ttl"
    merged.serialize(destination=str(combined), format="turtle")
    kg = owl_file_to_knowledge_graph(str(combined))
    graph, base = kg_to_rdf(kg)
    return _lowered(plan_domain_model(graph, base, shapes=graph))


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


# ---------------------------------------------------------------------------
# List-valued constructs that used to come out of the projection incomplete
# ---------------------------------------------------------------------------


_LIST_TTL = """
@prefix : <http://ex.org/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

:P a owl:Class .
:c a owl:DatatypeProperty ; rdfs:domain :P ; rdfs:range [ a rdfs:Datatype ; owl:oneOf ( "S" "M" 3 ) ] .
:hasParent a owl:ObjectProperty ; rdfs:domain :P ; rdfs:range :P .
:hasGrandparent a owl:ObjectProperty ; rdfs:domain :P ; rdfs:range :P ;
    owl:propertyChainAxiom ( :hasParent :hasParent ) .
:a1 a owl:ObjectProperty . :a2 a owl:ObjectProperty .
[ a owl:AllDisjointProperties ; owl:members ( :a1 :a2 ) ] .
"""


def _projected(tmp_path: Path) -> Graph:
    graph, _ = kg_to_rdf(owl_file_to_knowledge_graph(_write_ttl(tmp_path, _LIST_TTL)))
    return graph


def test_data_range_one_of_keeps_its_literals(tmp_path: Path):
    from rdflib.collection import Collection

    graph = _projected(tmp_path)
    ranges = [r for r in graph.objects(URIRef(EX + "c"), RDFS.range)]
    members = [list(Collection(graph, graph.value(r, OWL.oneOf))) for r in ranges if graph.value(r, OWL.oneOf)]
    assert len(members) == 1
    assert sorted(str(m) for m in members[0]) == ["3", "M", "S"]


def test_property_chain_keeps_a_repeated_property(tmp_path: Path):
    from rdflib.collection import Collection

    graph = _projected(tmp_path)
    head = graph.value(URIRef(EX + "hasGrandparent"), OWL.propertyChainAxiom)
    assert list(Collection(graph, head)) == [URIRef(EX + "hasParent"), URIRef(EX + "hasParent")]
    assert len(list(graph.objects(URIRef(EX + "hasGrandparent"), OWL.propertyChainAxiom))) == 1


def test_all_disjoint_properties_members_are_the_properties_themselves(tmp_path: Path):
    from rdflib.collection import Collection

    graph = _projected(tmp_path)
    (axiom,) = list(graph.subjects(RDF.type, OWL.AllDisjointProperties))
    assert sorted(Collection(graph, graph.value(axiom, OWL.members))) == [URIRef(EX + "a1"), URIRef(EX + "a2")]


# ---------------------------------------------------------------------------
# Several chains / keys on one subject, one list per axiom
# ---------------------------------------------------------------------------


_MULTI_LIST_TTL = """
@prefix : <http://ex.org/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:P a owl:Class ; owl:hasKey ( :ssn ) ; owl:hasKey ( :first :last ) .
:ssn a owl:DatatypeProperty ; rdfs:domain :P ; rdfs:range xsd:string .
:first a owl:DatatypeProperty ; rdfs:domain :P ; rdfs:range xsd:string .
:last a owl:DatatypeProperty ; rdfs:domain :P ; rdfs:range xsd:string .

:q a owl:ObjectProperty . :r a owl:ObjectProperty . :s a owl:ObjectProperty .
:p a owl:ObjectProperty ;
   owl:propertyChainAxiom ( :q :r ) ;
   owl:propertyChainAxiom ( :s :q ) .
:a1 a owl:ObjectProperty . :a2 a owl:ObjectProperty . :a3 a owl:ObjectProperty .
[] a owl:AllDisjointProperties ; owl:members ( :a1 :a2 :a3 ) .
"""


def _lists(graph: Graph, subject: str, predicate) -> list:
    from rdflib.collection import Collection

    return sorted(
        [list(Collection(graph, head)) for head in graph.objects(URIRef(EX + subject), predicate)],
        key=str,
    )


def test_two_property_chains_on_one_property_are_both_kept(tmp_path: Path):
    """Each chain record used to clear the one before it, so only the last
    survived — and O19 then stated the property *equals* that one chain."""
    graph, _ = _project(tmp_path, _MULTI_LIST_TTL)

    assert _lists(graph, "p", OWL.propertyChainAxiom) == sorted(
        [[URIRef(EX + "q"), URIRef(EX + "r")], [URIRef(EX + "s"), URIRef(EX + "q")]], key=str,
    )


def test_two_keys_on_one_class_stay_two_keys(tmp_path: Path):
    """The flattened member edges of both keys were merged into one composite key."""
    graph, _ = _project(tmp_path, _MULTI_LIST_TTL)

    assert _lists(graph, "P", OWL.hasKey) == sorted(
        [[URIRef(EX + "ssn")], [URIRef(EX + "first"), URIRef(EX + "last")]], key=str,
    )


def test_two_keys_do_not_become_one_composite_key(tmp_path: Path):
    from besser.BUML.notations.kg_to_buml import kg_to_class_diagram

    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, _MULTI_LIST_TTL))
    bodies = [c.expression for c in kg_to_class_diagram(kg).domain_model.constraints]

    # The merged key used to come out as one invariant over ssn, first and last.
    # (Whether the mapper then emits the second key is owl2uml's business.)
    assert not any("ssn" in b and "first" in b for b in bodies), bodies


def test_rebuilt_lists_leave_no_stray_list_cells(tmp_path: Path):
    """Every ``rdf:first`` cell hangs off a list head something points at."""
    graph, _ = _project(tmp_path, _MULTI_LIST_TTL)

    referenced = set(graph.objects())
    orphans = [cell for cell in graph.subjects(RDF.first, None) if cell not in referenced]
    assert orphans == []


def test_an_import_does_not_warn_about_list_order(tmp_path: Path):
    """``LIST_ORDER_INFERRED`` fired on every TTL import with a key, a chain or
    an ``owl:members`` list: keys and member sets have no order, and chains are
    rebuilt from their records."""
    _, warnings = _project(tmp_path, _MULTI_LIST_TTL)

    assert [w for w in warnings if w.code == "LIST_ORDER_INFERRED"] == []


# ---------------------------------------------------------------------------
# Constraint specs are the source of truth for constraint parameters
# ---------------------------------------------------------------------------


_MIN_COUNT_TTL = """
@prefix ex: <http://ex.org/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:name a owl:DatatypeProperty ; rdfs:domain ex:Person ; rdfs:range xsd:string .
ex:Person a owl:Class, sh:NodeShape ;
   sh:property [ sh:path ex:name ; sh:minCount 1 ] .
ex:bob a ex:Person ; ex:name "Bob" .
"""


def _with_min_count(tmp_path: Path, value):
    """Import, then change (or, with ``None``, delete) the minCount spec the way
    the inspector does: through ``metadata.constraintSpecs`` only."""
    from besser.utilities.web_modeling_editor.backend.services.converters import (
        kg_to_json,
        process_kg_diagram,
    )

    payload = kg_to_json(owl_file_to_knowledge_graph(_write_ttl(tmp_path, _MIN_COUNT_TTL)))
    for node in payload["model"]["nodes"]:
        meta = node.get("metadata") or {}
        specs = meta.get("constraintSpecs")
        if not specs:
            continue
        if value is None:
            meta["constraintSpecs"] = [s for s in specs if s["kind"] != "minCardinality"]
        else:
            for spec in specs:
                if spec["kind"] == "minCardinality":
                    spec["value"] = value
    return process_kg_diagram(payload)


@pytest.mark.parametrize("mirror", [False, True], ids=["conversion", "consistency"])
def test_an_edited_spec_wins_over_the_imported_edge(tmp_path: Path, mirror: bool):
    graph, _ = kg_to_rdf(_with_min_count(tmp_path, 3), mirror_shapes=mirror)

    assert [o.toPython() for o in graph.objects(None, URIRef(SH + "minCount"))] == [3]


@pytest.mark.parametrize("mirror", [False, True], ids=["conversion", "consistency"])
def test_a_deleted_spec_takes_its_parameter_with_it(tmp_path: Path, mirror: bool):
    graph, _ = kg_to_rdf(_with_min_count(tmp_path, None), mirror_shapes=mirror)

    assert list(graph.objects(None, URIRef(SH + "minCount"))) == []


def test_an_unedited_parameter_is_written_from_its_spec(tmp_path: Path):
    """The importer keeps no raw edge for a spec-held parameter, so the value
    is the spec's rendering (``xsd:nonNegativeInteger``, as the exporter writes
    it), not the source's ``xsd:integer``."""
    graph, _ = _project(tmp_path, _MIN_COUNT_TTL)

    assert list(graph.objects(None, URIRef(SH + "minCount"))) == [Literal("1", datatype=XSD_NNI)]


XSD_NNI = URIRef(XSD + "nonNegativeInteger")
