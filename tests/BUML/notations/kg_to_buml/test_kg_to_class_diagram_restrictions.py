"""Tests for OWL restriction handling in the KG → class-diagram converter.

Following rules D22-D25 and O07-O15, every ``owl:Restriction`` materialises a
dedicated auxiliary class (e.g. ``_min_1_hasName_str``) that the restricted
class generalizes to, rather than lifting the cardinality onto the property's
own ``Multiplicity``. An object-property restriction gives the aux class its
own association carrying the restricted multiplicity; a data-property
restriction gives it an OCL invariant.

OWL property characteristics (``owl:FunctionalProperty``) *are* lifted directly
onto the property's own attribute/association — those aren't restrictions but
global property-level facts, so they don't get an aux class. A data property
that is not functional stays many-valued, per D11/D36.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from besser.BUML.notations.kg_to_buml import kg_to_class_diagram
from besser.utilities.owl_to_buml import owl_file_to_knowledge_graph


def _write_ttl(tmp_path: Path, content: str) -> str:
    p = tmp_path / "ontology.ttl"
    p.write_text(content.strip(), encoding="utf-8")
    return str(p)


def _attribute(domain_model, class_name: str, attr_name: str):
    cls = next(c for c in domain_model.types if getattr(c, "name", None) == class_name)
    return next(a for a in cls.attributes if a.name == attr_name)


def _association(domain_model, name: str):
    return next(a for a in domain_model.associations if a.name == name)


def _class(domain_model, name: str):
    return next(c for c in domain_model.types if getattr(c, "name", None) == name)


def _generalizes_to(domain_model, specific_name: str, general_name: str) -> bool:
    return any(
        g.specific.name == specific_name and g.general.name == general_name
        for g in domain_model.generalizations
    )


def test_min_cardinality_materializes_aux_class(tmp_path: Path):
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
    :hasName a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    result = kg_to_class_diagram(kg)
    # Person's own "hasName" attribute is unaffected by the restriction; it
    # stays many-valued because the property is not owl:FunctionalProperty.
    attr = _attribute(result.domain_model, "Person", "hasName")
    assert (attr.multiplicity.min, attr.multiplicity.max) == (0, 9999)
    # The restriction lives on a dedicated aux class Person generalizes to,
    # with its own "hasName" attribute and an OCL invariant enforcing >= 1.
    aux = _class(result.domain_model, "_min_1_hasName_str")
    assert _generalizes_to(result.domain_model, "Person", "_min_1_hasName_str")
    bodies = [c.expression for c in result.domain_model.constraints if c.context is aux]
    assert any("self.hasName->asSet()->select(v | v.oclIsKindOf(str))->size() >= 1" in b for b in bodies)


def test_max_cardinality_materializes_aux_class(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Person a owl:Class ;
        rdfs:subClassOf [
            a owl:Restriction ;
            owl:onProperty :hasNickname ;
            owl:maxCardinality "3"^^xsd:nonNegativeInteger
        ] .
    :hasNickname a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    result = kg_to_class_diagram(kg)
    aux = _class(result.domain_model, "_max_3_hasNickname_str")
    assert _generalizes_to(result.domain_model, "Person", "_max_3_hasNickname_str")
    bodies = [c.expression for c in result.domain_model.constraints if c.context is aux]
    assert any("self.hasNickname->asSet()->select(v | v.oclIsKindOf(str))->size() <= 3" in b for b in bodies)


def test_exact_cardinality_materializes_aux_class(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Person a owl:Class ;
        rdfs:subClassOf [
            a owl:Restriction ;
            owl:onProperty :ssn ;
            owl:cardinality "1"^^xsd:nonNegativeInteger
        ] .
    :ssn a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    result = kg_to_class_diagram(kg)
    aux = _class(result.domain_model, "_exact_1_ssn_str")
    assert _generalizes_to(result.domain_model, "Person", "_exact_1_ssn_str")
    bodies = [c.expression for c in result.domain_model.constraints if c.context is aux]
    assert any("self.ssn->asSet()->select(v | v.oclIsKindOf(str))->size() = 1" in b for b in bodies)


def test_some_values_from_materializes_aux_class(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Pet    a owl:Class .
    :Person a owl:Class ;
        rdfs:subClassOf [
            a owl:Restriction ;
            owl:onProperty :owns ;
            owl:someValuesFrom :Pet
        ] .
    :owns a owl:ObjectProperty ; rdfs:domain :Person ; rdfs:range :Pet .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    result = kg_to_class_diagram(kg)
    # The someValuesFrom restriction materialises an aux class that Person
    # specialises, carrying the association to Pet with target end 1..*.
    aux = _class(result.domain_model, "_some_owns_Pet")
    assert _generalizes_to(result.domain_model, "Person", "_some_owns_Pet")
    aux_assoc = _association(result.domain_model, "owns")
    aux_src_end = next(e for e in aux_assoc.ends if e.type is aux)
    aux_target_end = next(e for e in aux_assoc.ends if e is not aux_src_end)
    assert aux_target_end.type.name == "Pet"
    assert (aux_target_end.multiplicity.min, aux_target_end.multiplicity.max) == (1, 9999)
    # Person's own rdfs:domain/range "owns" is subsumed by the aux's: BUML
    # forbids a class and its ancestor both owning the role name, and the
    # ancestor's is the one that carries the restriction's tighter bound, so
    # Person inherits `owns [1..*]` rather than redeclaring `owns [0..*]`.
    assert sum(1 for a in result.domain_model.associations if a.name == "owns") == 1
    assert any(
        w.code == "ASSOC_INHERITED_SHADOWED" and "Person.owns" in w.message
        for w in result.warnings
    )


def test_functional_property_caps_max_to_one(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Person a owl:Class .
    :birthDate a owl:DatatypeProperty , owl:FunctionalProperty ;
        rdfs:domain :Person ;
        rdfs:range xsd:date .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    result = kg_to_class_diagram(kg)
    attr = _attribute(result.domain_model, "Person", "birthDate")
    assert attr.multiplicity.max == 1


def test_abox_bump_and_restriction_aux_are_independent(tmp_path: Path):
    """A cardinality restriction no longer touches the property's own
    ``Multiplicity`` at all (it's enforced by an independent OCL invariant on
    its aux class instead — see test_exact_cardinality_materializes_aux_class),
    so the ABox multi-valued-literal heuristic is free to bump Person's own
    plain "name" attribute same as it would for any other property."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Person a owl:Class ;
        rdfs:subClassOf [
            a owl:Restriction ;
            owl:onProperty :name ;
            owl:cardinality "1"^^xsd:nonNegativeInteger
        ] .
    :name a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .

    :alice a :Person ; :name "Alice" , "Alicia" .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    result = kg_to_class_diagram(kg)
    attr = _attribute(result.domain_model, "Person", "name")
    assert attr.multiplicity.max == 9999
    # The exact-cardinality restriction still independently enforces "= 1" on
    # its own aux class, regardless of what Person's own attribute allows.
    aux = _class(result.domain_model, "_exact_1_name_str")
    assert _generalizes_to(result.domain_model, "Person", "_exact_1_name_str")
    bodies = [c.expression for c in result.domain_model.constraints if c.context is aux]
    assert any("self.name->asSet()->select(v | v.oclIsKindOf(str))->size() = 1" in b for b in bodies)


def test_has_value_data_restriction_materializes_aux_class(tmp_path: Path):
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
    :country a owl:DatatypeProperty ; rdfs:domain :Adult .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    result = kg_to_class_diagram(kg)
    aux = _class(result.domain_model, "_hasValue_country_US")
    assert _generalizes_to(result.domain_model, "Adult", "_hasValue_country_US")
    bodies = [c.expression for c in result.domain_model.constraints if c.context is aux]
    assert any('self.country->asSet()->includes("US")' in b for b in bodies)
    codes = {w.code for w in result.warnings}
    assert "ADV_RESTRICTION_UNSUPPORTED" not in codes


def test_restriction_on_unknown_property_does_not_crash(tmp_path: Path):
    """Restriction whose ``on_property`` we never lifted into BUML must be
    silently skipped (the preflight surfaces this as an ``UNATTACHED``
    issue; the converter must not crash)."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Thing a owl:Class ;
        rdfs:subClassOf [
            a owl:Restriction ;
            owl:onProperty :ghost ;
            owl:cardinality "1"^^xsd:nonNegativeInteger
        ] .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    result = kg_to_class_diagram(kg)
    # No crash; "ghost" is not a property of the domain model.
    cls = next(c for c in result.domain_model.types if getattr(c, "name", None) == "Thing")
    assert cls is not None
