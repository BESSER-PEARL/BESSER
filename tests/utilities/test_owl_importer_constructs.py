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
    KGProperty,
    PropertyChainAxiom,
    SubPropertyOfAxiom,
)
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
    blank = _restriction_blank(kg)
    assert blank.metadata["restriction_type"] == "minCardinality"
    assert blank.metadata["value"] == 1
    assert blank.metadata["on_property"] == "http://ex.org/hasName"


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
    blank = _restriction_blank(kg)
    assert blank.metadata["restriction_type"] == "someValuesFrom"
    assert blank.metadata["value"] == "http://ex.org/Pet"
    assert blank.metadata["on_property"] == "http://ex.org/owns"


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
    blank = _restriction_blank(kg)
    assert blank.metadata["restriction_type"] == "hasValue"
    assert blank.metadata["value"] == "US"


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
    blank = _restriction_blank(kg)
    assert blank.metadata["restriction_type"] == "minQualifiedCardinality"
    assert blank.metadata["value"] == 2
    assert blank.metadata["on_class"] == "http://ex.org/Pet"


# --- Class combinators ----------------------------------------------------


def _class_expr_blank(kg) -> KGBlank:
    blanks = [n for n in kg.nodes if isinstance(n, KGBlank) and n.metadata.get("kind") == "class_expression"]
    assert len(blanks) == 1
    return blanks[0]


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
    blank = _class_expr_blank(kg)
    assert blank.metadata["combinator"] == "unionOf"
    assert blank.metadata["members"] == ["http://ex.org/Cat", "http://ex.org/Dog"]


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
    blank = _class_expr_blank(kg)
    assert blank.metadata["combinator"] == "intersectionOf"
    assert set(blank.metadata["members"]) == {"http://ex.org/Mammal", "http://ex.org/Pet"}


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
    blank = _class_expr_blank(kg)
    assert blank.metadata["combinator"] == "complementOf"
    assert blank.metadata["members"] == ["http://ex.org/Adult"]


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
    blank = _class_expr_blank(kg)
    assert blank.metadata["combinator"] == "oneOf"
    assert set(blank.metadata["members"]) == {"http://ex.org/red", "http://ex.org/green"}


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
