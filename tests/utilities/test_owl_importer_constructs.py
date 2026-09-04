"""Tests for the OWL/TTL/OWL2 importer's construct-by-construct handling.

Each test feeds a small inline TTL fragment to ``owl_file_to_knowledge_graph``
and asserts the resulting KG nodes/edges have the expected shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from besser.BUML.metamodel.kg import (
    DisjointClassesAxiom,
    DisjointUnionAxiom,
    EquivalentClassesAxiom,
    HasKeyAxiom,
    ImportAxiom,
    InversePropertiesAxiom,
    KGBlank,
    KGClass,
    KGIndividual,
    KGNodeConstraint,
    KGPropertyConstraint,
    KGProperty,
    PropertyChainAxiom,
    SubPropertyOfAxiom,
)
from besser.BUML.metamodel.kg.constants import CONSTRAINT_TARGET_CLASS
from besser.utilities.owl_to_buml import owl_file_to_knowledge_graph


def _write_ttl(tmp_path: Path, content: str, name: str = "ontology.ttl") -> str:
    p = tmp_path / name
    p.write_text(content.strip(), encoding="utf-8")
    return str(p)


def test_xsd_integer_used_as_range_becomes_class_not_individual(tmp_path: Path):
    """``xsd:integer`` referenced via ``rdfs:range`` should classify as KGClass.

    Regression: before the fix in ``_classify`` / first-pass class seeding,
    datatype IRIs referenced as ``rdfs:range`` were silently imported as
    KGIndividual nodes, which then leaked into downstream BUML conversions.
    """
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Person a owl:Class .
    :age    a owl:DatatypeProperty ;
            rdfs:domain :Person ;
            rdfs:range  xsd:integer .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))

    xsd_integer_iri = "http://www.w3.org/2001/XMLSchema#integer"
    matches = [n for n in kg.nodes if n.id == xsd_integer_iri]
    assert len(matches) == 1, "xsd:integer should appear exactly once in the KG"
    node = matches[0]
    assert isinstance(node, KGClass), (
        f"xsd:integer must be a KGClass (got {type(node).__name__})"
    )
    assert not isinstance(node, KGIndividual), (
        "xsd:integer must not be imported as a KGIndividual"
    )


@pytest.mark.parametrize(
    "datatype_iri",
    [
        "http://www.w3.org/2001/XMLSchema#string",
        "http://www.w3.org/2001/XMLSchema#boolean",
        "http://www.w3.org/2001/XMLSchema#dateTime",
        "http://www.w3.org/2001/XMLSchema#decimal",
        "http://www.w3.org/2000/01/rdf-schema#Literal",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#PlainLiteral",
    ],
)
def test_other_datatype_iris_classify_as_class(tmp_path: Path, datatype_iri: str):
    """Every datatype IRI we recognise must become a KGClass when used as range."""
    ttl = f"""
    @prefix : <http://ex.org/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Thing a owl:Class .
    :prop  a owl:DatatypeProperty ;
           rdfs:domain :Thing ;
           rdfs:range  <{datatype_iri}> .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))

    matches = [n for n in kg.nodes if n.id == datatype_iri]
    assert len(matches) == 1
    assert isinstance(matches[0], KGClass)


def test_non_datatype_iri_used_as_range_still_classifies_as_class(tmp_path: Path):
    """Non-datatype URIRefs used as range should classify as a class.

    A range pointing to a user-defined class (no rdf:type owl:Class
    declaration on it) should be treated as a class so it can become a
    BUML association target. This already worked because of the
    ``rdf:type`` second-pass logic — the test guards that the datatype
    fix didn't regress it.
    """
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Person a owl:Class .
    :Pet    a owl:Class .
    :owns   a owl:ObjectProperty ;
            rdfs:domain :Person ;
            rdfs:range  :Pet .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    pet_iri = "http://ex.org/Pet"
    pet_nodes = [n for n in kg.nodes if n.id == pet_iri]
    assert len(pet_nodes) == 1
    assert isinstance(pet_nodes[0], KGClass)
    assert not isinstance(pet_nodes[0], KGIndividual)


# --- OWL property characteristics -----------------------------------------


@pytest.mark.parametrize(
    "owl_type, expected_label",
    [
        ("owl:FunctionalProperty", "Functional"),
        ("owl:InverseFunctionalProperty", "InverseFunctional"),
        ("owl:TransitiveProperty", "Transitive"),
        ("owl:SymmetricProperty", "Symmetric"),
        ("owl:AsymmetricProperty", "Asymmetric"),
        ("owl:ReflexiveProperty", "Reflexive"),
        ("owl:IrreflexiveProperty", "Irreflexive"),
    ],
)
def test_property_characteristic_marked_in_metadata(tmp_path: Path, owl_type: str, expected_label: str):
    ttl = f"""
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :prop a owl:ObjectProperty , {owl_type} .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    prop = next(n for n in kg.nodes if isinstance(n, KGProperty) and n.id == "http://ex.org/prop")
    assert expected_label in prop.metadata.get("characteristics", [])
    assert prop.metadata.get("kind") == "Object"


# --- Restrictions ----------------------------------------------------------


def _restriction_blank(kg) -> KGBlank:
    blanks = [n for n in kg.nodes if isinstance(n, KGBlank) and n.metadata.get("kind") == "restriction"]
    assert len(blanks) == 1, f"expected exactly one restriction blank, got {len(blanks)}"
    return blanks[0]


def _restriction_pc(kg) -> KGPropertyConstraint:
    """Return the single PropertyConstraint emitted for an OWL restriction.

    OWL restrictions are imported as KGPropertyConstraint nodes (with the
    constraint payload in ``metadata['constraintSpecs']``); the legacy
    ``_restriction_blank`` helper is kept only for tests still asserting the
    pre-constraint-nodes representation.
    """
    pcs = [n for n in kg.nodes if isinstance(n, KGPropertyConstraint)]
    assert len(pcs) == 1, f"expected exactly one PropertyConstraint, got {len(pcs)}"
    return pcs[0]


def test_restriction_min_cardinality_annotated_on_blank(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Person a owl:Class ;
        rdfs:subClassOf [
            a owl:Restriction ;
            owl:onProperty :hasName ;
            owl:minCardinality "1"^^xsd:nonNegativeInteger
        ] .
    :hasName a owl:DatatypeProperty .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    pc = _restriction_pc(kg)
    specs = pc.get_specs()
    assert len(specs) == 1
    assert specs[0]["kind"] == "minCardinality"
    assert specs[0]["value"] == 1
    assert pc.metadata.get("onPropertyIri") == "http://ex.org/hasName"


def test_restriction_some_values_from_annotated_on_blank(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Pet a owl:Class .
    :Person a owl:Class ;
        rdfs:subClassOf [
            a owl:Restriction ;
            owl:onProperty :owns ;
            owl:someValuesFrom :Pet
        ] .
    :owns a owl:ObjectProperty .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    pc = _restriction_pc(kg)
    specs = pc.get_specs()
    assert specs[0]["kind"] == "someValuesFrom"
    assert specs[0]["value"] == "http://ex.org/Pet"
    assert pc.metadata.get("onPropertyIri") == "http://ex.org/owns"


def test_restriction_has_value_annotated_on_blank(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Adult a owl:Class ;
        rdfs:subClassOf [
            a owl:Restriction ;
            owl:onProperty :country ;
            owl:hasValue "US"
        ] .
    :country a owl:DatatypeProperty .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    pc = _restriction_pc(kg)
    specs = pc.get_specs()
    assert specs[0]["kind"] == "hasValue"
    assert specs[0]["value"] == "US"


def test_restriction_qualified_cardinality(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Person a owl:Class ;
        rdfs:subClassOf [
            a owl:Restriction ;
            owl:onProperty :owns ;
            owl:onClass :Pet ;
            owl:minQualifiedCardinality "2"^^xsd:nonNegativeInteger
        ] .
    :owns a owl:ObjectProperty .
    :Pet a owl:Class .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    pc = _restriction_pc(kg)
    specs = pc.get_specs()
    assert specs[0]["kind"] == "minQualifiedCardinality"
    assert specs[0]["value"] == 2
    assert specs[0]["on_class"] == "http://ex.org/Pet"


# --- Class combinators ----------------------------------------------------


def _class_expr_nc(kg, owner_iri: str, kind: str) -> KGNodeConstraint:
    """Find the KGNodeConstraint carrying a ``kind`` class-expression spec and
    targeting ``owner_iri`` via a ``constraintTargetClass`` edge."""
    owner = next(n for n in kg.nodes if isinstance(n, KGClass) and n.iri == owner_iri)
    ncs = [
        e.source
        for e in kg.edges
        if e.iri == CONSTRAINT_TARGET_CLASS
        and e.target is owner
        and isinstance(e.source, KGNodeConstraint)
    ]
    matches = [nc for nc in ncs if any(s.get("kind") == kind for s in nc.get_specs())]
    assert len(matches) == 1
    return matches[0]


def test_unionOf_decoded(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Cat a owl:Class .
    :Dog a owl:Class .
    :Pet a owl:Class ;
        owl:equivalentClass [
            a owl:Class ;
            owl:unionOf ( :Cat :Dog )
        ] .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    nc = _class_expr_nc(kg, "http://ex.org/Pet", "unionOf")
    spec = next(s for s in nc.get_specs() if s["kind"] == "unionOf")
    assert spec["value"] == ["http://ex.org/Cat", "http://ex.org/Dog"]
    assert nc.metadata["source"] == "owl"


def test_intersectionOf_decoded(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Mammal a owl:Class .
    :Pet    a owl:Class .
    :PetMammal a owl:Class ;
        owl:equivalentClass [
            a owl:Class ;
            owl:intersectionOf ( :Mammal :Pet )
        ] .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    nc = _class_expr_nc(kg, "http://ex.org/PetMammal", "intersectionOf")
    spec = next(s for s in nc.get_specs() if s["kind"] == "intersectionOf")
    assert set(spec["value"]) == {"http://ex.org/Mammal", "http://ex.org/Pet"}


def test_complementOf_decoded(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Adult a owl:Class .
    :Minor a owl:Class ;
        owl:equivalentClass [
            a owl:Class ;
            owl:complementOf :Adult
        ] .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    nc = _class_expr_nc(kg, "http://ex.org/Minor", "complementOf")
    spec = next(s for s in nc.get_specs() if s["kind"] == "complementOf")
    assert spec["value"] == "http://ex.org/Adult"


def test_oneOf_decoded(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :red a :Color .
    :green a :Color .
    :Color a owl:Class ;
        owl:equivalentClass [
            a owl:Class ;
            owl:oneOf ( :red :green )
        ] .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    nc = _class_expr_nc(kg, "http://ex.org/Color", "oneOf")
    spec = next(s for s in nc.get_specs() if s["kind"] == "oneOf")
    assert set(spec["value"]) == {"http://ex.org/red", "http://ex.org/green"}


def test_list_spine_blanks_swept_for_every_list_valued_construct(tmp_path: Path):
    """RDF-list "cons cell" blanks behind a decoded list-valued construct
    (unionOf/oneOf member lists, hasKey/disjointUnionOf id lists) must not
    survive into the KG once their content has been captured as structured
    metadata/axioms — leaving them behind made the KG-refinement preflight
    wrongly recommend promoting them to individuals (they carry no domain
    content, just RDF-list encoding)."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :name a owl:DatatypeProperty .
    :email a owl:DatatypeProperty .
    :Cat a owl:Class .
    :Dog a owl:Class .
    :red a :Color .
    :green a :Color .

    :Pet a owl:Class ;
        owl:equivalentClass [ a owl:Class ; owl:unionOf ( :Cat :Dog ) ] .
    :Color a owl:Class ;
        owl:equivalentClass [ a owl:Class ; owl:oneOf ( :red :green ) ] .
    :Person a owl:Class ;
        owl:hasKey ( :name :email ) .
    :PetKind a owl:Class ; owl:disjointUnionOf ( :Cat :Dog ) .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    blanks = [n for n in kg.nodes if isinstance(n, KGBlank)]
    assert blanks == [], f"list-spine blanks leaked into the KG: {[(b.id, b.metadata) for b in blanks]}"
    # The decoded content itself must still be intact.
    nc = _class_expr_nc(kg, "http://ex.org/Pet", "unionOf")
    spec = next(s for s in nc.get_specs() if s["kind"] == "unionOf")
    assert spec["value"] == ["http://ex.org/Cat", "http://ex.org/Dog"]
    key_axioms = _find_axioms(kg, HasKeyAxiom)
    assert key_axioms[0].property_ids == ["http://ex.org/name", "http://ex.org/email"]
    union_axioms = _find_axioms(kg, DisjointUnionAxiom)
    assert union_axioms[0].part_class_ids == ["http://ex.org/Cat", "http://ex.org/Dog"]


# --- Axioms ---------------------------------------------------------------


def _find_axioms(kg, axiom_type):
    return [a for a in kg.axioms if isinstance(a, axiom_type)]


def test_equivalentClass_emitted_as_axiom(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Human  a owl:Class .
    :Person a owl:Class ; owl:equivalentClass :Human .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    axioms = _find_axioms(kg, EquivalentClassesAxiom)
    assert len(axioms) == 1
    assert set(axioms[0].class_ids) == {"http://ex.org/Person", "http://ex.org/Human"}


def test_disjointWith_emitted_as_axiom(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Cat a owl:Class .
    :Dog a owl:Class ; owl:disjointWith :Cat .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    axioms = _find_axioms(kg, DisjointClassesAxiom)
    assert len(axioms) == 1
    assert set(axioms[0].class_ids) == {"http://ex.org/Dog", "http://ex.org/Cat"}


def test_disjointUnionOf_decoded(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Cat a owl:Class .
    :Dog a owl:Class .
    :Pet a owl:Class ; owl:disjointUnionOf ( :Cat :Dog ) .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    axioms = _find_axioms(kg, DisjointUnionAxiom)
    assert len(axioms) == 1
    assert axioms[0].union_class_id == "http://ex.org/Pet"
    assert axioms[0].part_class_ids == ["http://ex.org/Cat", "http://ex.org/Dog"]


def test_subPropertyOf_emitted_as_axiom(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :hasParent a owl:ObjectProperty .
    :hasFather a owl:ObjectProperty ; rdfs:subPropertyOf :hasParent .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    axioms = _find_axioms(kg, SubPropertyOfAxiom)
    assert len(axioms) == 1
    assert axioms[0].sub_property_id == "http://ex.org/hasFather"
    assert axioms[0].super_property_id == "http://ex.org/hasParent"


def test_inverseOf_emitted_as_axiom(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :hasParent a owl:ObjectProperty ; owl:inverseOf :hasChild .
    :hasChild  a owl:ObjectProperty .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    axioms = _find_axioms(kg, InversePropertiesAxiom)
    assert len(axioms) == 1
    assert {axioms[0].property_a_id, axioms[0].property_b_id} == {
        "http://ex.org/hasParent",
        "http://ex.org/hasChild",
    }


def test_propertyChainAxiom_decoded_as_list(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :hasParent  a owl:ObjectProperty .
    :hasBrother a owl:ObjectProperty .
    :hasUncle   a owl:ObjectProperty ;
        owl:propertyChainAxiom ( :hasParent :hasBrother ) .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    axioms = _find_axioms(kg, PropertyChainAxiom)
    assert len(axioms) == 1
    assert axioms[0].property_id == "http://ex.org/hasUncle"
    assert axioms[0].chain_property_ids == ["http://ex.org/hasParent", "http://ex.org/hasBrother"]


def test_hasKey_emitted_as_axiom(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :ssn a owl:DatatypeProperty .
    :Person a owl:Class ; owl:hasKey ( :ssn ) .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    axioms = _find_axioms(kg, HasKeyAxiom)
    assert len(axioms) == 1
    assert axioms[0].class_id == "http://ex.org/Person"
    assert axioms[0].property_ids == ["http://ex.org/ssn"]


def test_imports_logged_not_followed(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/onto> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    <http://ex.org/onto> a owl:Ontology ;
        owl:imports <http://xmlns.com/foaf/0.1/> .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    imports = _find_axioms(kg, ImportAxiom)
    assert len(imports) == 1
    assert imports[0].target_iri == "http://xmlns.com/foaf/0.1/"


# --- rdfs:comment / isDefinedBy --------------------------------------------


def test_rdfs_comment_populates_metadata_description(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Person a owl:Class ;
        rdfs:comment "A human being."@en ;
        rdfs:isDefinedBy <http://ex.org/onto> .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    person = next(n for n in kg.nodes if n.id == "http://ex.org/Person")
    assert person.metadata.get("description") == "A human being."
    assert person.metadata.get("defined_by") == "http://ex.org/onto"


# --- OWL2 punning ----------------------------------------------------------


def test_punning_class_and_individual_both_present(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Eagle a owl:Class , owl:NamedIndividual .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    eagle_class = [n for n in kg.nodes if n.id == "http://ex.org/Eagle"]
    eagle_indiv = [n for n in kg.nodes if n.id == "http://ex.org/Eagle#individual"]
    assert len(eagle_class) == 1 and isinstance(eagle_class[0], KGClass)
    assert len(eagle_indiv) == 1 and isinstance(eagle_indiv[0], KGIndividual)
    # Cross-references via metadata.
    assert eagle_class[0].metadata.get("punned_with") == "http://ex.org/Eagle#individual"
    assert eagle_indiv[0].metadata.get("punned_with") == "http://ex.org/Eagle"


def test_punning_class_and_node_shape_both_present(tmp_path: Path):
    """The standard SHACL idiom ``<C> a sh:NodeShape ; sh:targetClass <C>``.

    Regression: the shape used to *replace* the class node, so an ontology
    written this way silently lost every class that had a shape — along with
    their ``rdfs:subClassOf``, ``rdfs:domain`` and ``rdfs:range`` links on TTL
    re-export. ``test_bibo_regression.py`` covers the same idiom at scale.
    """
    from besser.BUML.metamodel.kg import KGNodeConstraint
    from besser.BUML.metamodel.kg.constants import CONSTRAINT_TARGET_CLASS

    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Agent a owl:Class .
    :Person a owl:Class ; rdfs:subClassOf :Agent .
    :name a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .

    :Person a sh:NodeShape ;
        sh:targetClass :Person ;
        sh:property [ sh:path :name ; sh:maxCount 1 ] .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    nodes = [n for n in kg.nodes if n.iri == "http://ex.org/Person"]
    classes = [n for n in nodes if isinstance(n, KGClass)]
    shapes = [n for n in nodes if isinstance(n, KGNodeConstraint)]
    assert len(classes) == 1, "the class declaration must survive the shape"
    assert len(shapes) == 1
    assert classes[0].id != shapes[0].id, "distinct ids keep the JSON round trip unambiguous"

    # The shape targets the surviving class.
    assert any(
        e.source is shapes[0] and e.target is classes[0] and e.iri == CONSTRAINT_TARGET_CLASS
        for e in kg.edges
    )
    # ...and the class keeps its own structural links.
    assert any(
        e.source is classes[0] and e.target.iri == "http://ex.org/Agent"
        and e.iri == "http://www.w3.org/2000/01/rdf-schema#subClassOf"
        for e in kg.edges
    )


def test_punned_class_and_shape_round_trip_through_rdf(tmp_path: Path):
    """Re-exporting must reproduce both assertions the source made."""
    from rdflib import Graph, URIRef
    from rdflib.namespace import OWL, RDF, RDFS

    from besser.utilities.kg_to_owl import knowledge_graph_to_rdf

    SH = "http://www.w3.org/ns/shacl#"
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .

    :Agent a owl:Class .
    :Person a owl:Class ; rdfs:subClassOf :Agent .
    :Person a sh:NodeShape ; sh:targetClass :Person .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    exported = knowledge_graph_to_rdf(kg, vocab="both")
    person = URIRef("http://ex.org/Person")

    assert (person, RDF.type, OWL.Class) in exported
    assert (person, RDFS.subClassOf, URIRef("http://ex.org/Agent")) in exported
    assert (person, URIRef(SH + "targetClass"), person) in exported
