"""Tests for the OWL restriction → BUML Multiplicity lifting.

The KG → class-diagram converter reads structured ``owl:Restriction``
metadata that the importer attaches to ``KGBlank`` nodes (kind='restriction',
on_property, restriction_type, value, on_class) and lifts the corresponding
cardinality into the matching ``Property.multiplicity`` (datatype attributes)
or association target-end multiplicity (object properties).

OWL property characteristics on ``KGProperty.metadata.characteristics``
(currently only ``Functional``) are also lifted into ``max=1``.

ABox-driven multiplicity bumps (Step 5 in the converter) must defer to
explicit OWL-derived multiplicities rather than overwrite them.
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


def test_min_cardinality_lifted_into_multiplicity(tmp_path: Path):
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
    attr = _attribute(result.domain_model, "Person", "hasName")
    assert attr.multiplicity.min == 1
    assert attr.is_optional is False


def test_max_cardinality_lifted_into_multiplicity(tmp_path: Path):
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
    attr = _attribute(result.domain_model, "Person", "hasNickname")
    assert attr.multiplicity.max == 3


def test_exact_cardinality_lifted_into_multiplicity(tmp_path: Path):
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
    attr = _attribute(result.domain_model, "Person", "ssn")
    assert attr.multiplicity.min == 1
    assert attr.multiplicity.max == 1


def test_some_values_from_lifts_min_to_one(tmp_path: Path):
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
    assoc = _association(result.domain_model, "owns")
    src_end = result.assoc_source_end[id(assoc)]
    target_end = next(e for e in assoc.ends if e is not src_end)
    assert target_end.multiplicity.min == 1


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


def test_restriction_overrides_abox_multiplicity_bump(tmp_path: Path):
    """If a restriction explicitly caps cardinality at 1, ABox multi-valued
    data must NOT bump the multiplicity to 0..*. The OWL-declared semantics
    win over inferred-from-data heuristics."""
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
    # cardinality=1 wins; multi-valued ABox does not bump to 0..*.
    assert attr.multiplicity.min == 1
    assert attr.multiplicity.max == 1
    # Ensure no MULTIVALUED_LITERAL warning fired.
    codes = {w.code for w in result.warnings}
    assert "MULTIVALUED_LITERAL" not in codes


def test_unsupported_restriction_emits_advisory(tmp_path: Path):
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
    codes = {w.code for w in result.warnings}
    assert "ADV_RESTRICTION_UNSUPPORTED" in codes


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
