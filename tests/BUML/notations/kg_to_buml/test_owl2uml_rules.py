"""Tests for the kg2uml/owl2uml-aligned rules adopted in the KG → BUML
class-diagram converter: equivalentClass/Property canonicalization (D27/D28),
class-expression aux classes (D17-D21/O06), domain/range union direct-linking
(D29-D32), composite datatypes (O01-O03), and the SHACL fixes/additions
(`sh:class`, `sh:equals`/`lessThan`/`lessThanOrEquals`).

Each test feeds a small inline TTL fragment through the real importer
(``owl_file_to_knowledge_graph``) and the real converter
(``kg_to_class_diagram``), mirroring the style of
``test_kg_to_class_diagram_restrictions.py``.
"""

from __future__ import annotations

from pathlib import Path

from besser.BUML.notations.kg_to_buml import kg_to_class_diagram
from besser.utilities.owl_to_buml import owl_file_to_knowledge_graph


def _write_ttl(tmp_path: Path, content: str) -> str:
    p = tmp_path / "ontology.ttl"
    p.write_text(content.strip(), encoding="utf-8")
    return str(p)


def _class(domain_model, name: str):
    return next(c for c in domain_model.types if getattr(c, "name", None) == name)


def _classes_by_name(domain_model):
    return {c.name: c for c in domain_model.types if hasattr(c, "attributes")}


def _generalizes_to(domain_model, specific_name: str, general_name: str) -> bool:
    return any(
        g.specific.name == specific_name and g.general.name == general_name
        for g in domain_model.generalizations
    )


def _bodies_for(domain_model, ctx_name: str):
    return [c.expression for c in domain_model.constraints if c.context.name == ctx_name]


# ---------------------------------------------------------------------------
# D27/D28: equivalentClass / equivalentProperty canonicalization
# ---------------------------------------------------------------------------


def test_equivalent_class_canonicalizes_to_one_class(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Person a owl:Class .
    :Human  a owl:Class ; owl:equivalentClass :Person .
    :name a owl:DatatypeProperty ; rdfs:domain :Human .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    result = kg_to_class_diagram(kg)
    names = [c.name for c in result.domain_model.types if hasattr(c, "attributes")]
    # Only one merged class survives, not two.
    assert names.count("Person") + names.count("Human") == 1
    merged_name = "Person" if "Person" in names else "Human"
    merged = _class(result.domain_model, merged_name)
    # The property declared on the alias (:Human) ended up on the same
    # merged class.
    assert any(a.name == "name" for a in merged.attributes)
    # No tautological "equivalentClasses" OCL note.
    bodies = _bodies_for(result.domain_model, merged_name)
    assert not any("oclIsKindOf(Person)" in b or "oclIsKindOf(Human)" in b for b in bodies)


def test_equivalent_property_canonicalizes_to_one_attribute(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Person a owl:Class .
    :name  a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
    :fullName a owl:DatatypeProperty ; owl:equivalentProperty :name .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    result = kg_to_class_diagram(kg)
    person = _class(result.domain_model, "Person")
    # Only one attribute for the merged name/fullName property, not two.
    attr_names = {a.name for a in person.attributes}
    assert len(attr_names & {"name", "fullName"}) == 1


# ---------------------------------------------------------------------------
# D17-D21/O06: class expressions
# ---------------------------------------------------------------------------


def test_union_of_class_axiom_materializes_aux_and_generalizations(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Cat a owl:Class .
    :Dog a owl:Class .
    :Pet a owl:Class ;
        owl:equivalentClass [
            a owl:Class ;
            owl:unionOf ( :Cat :Dog )
        ] .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    result = kg_to_class_diagram(kg)
    aux = _class(result.domain_model, "_Cat_Dog_Union")
    assert aux.is_abstract is True
    # Each member generalizes TO the union aux, and Pet generalizes to it too.
    assert _generalizes_to(result.domain_model, "Cat", "_Cat_Dog_Union")
    assert _generalizes_to(result.domain_model, "Dog", "_Cat_Dog_Union")
    assert _generalizes_to(result.domain_model, "Pet", "_Cat_Dog_Union")


def test_intersection_of_class_axiom_materializes_aux_and_generalizations(tmp_path: Path):
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
    result = kg_to_class_diagram(kg)
    aux = _class(result.domain_model, "_Mammal_Pet_Intersection")
    assert aux.is_abstract is False
    # The aux is a subclass of EACH member (multiple inheritance).
    assert _generalizes_to(result.domain_model, "_Mammal_Pet_Intersection", "Mammal")
    assert _generalizes_to(result.domain_model, "_Mammal_Pet_Intersection", "Pet")
    assert _generalizes_to(result.domain_model, "PetMammal", "_Mammal_Pet_Intersection")


def test_one_of_literals_becomes_enumeration(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Color a owl:Class ;
        owl:equivalentClass [
            a owl:Class ;
            owl:oneOf ( "red" "green" )
        ] .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    result = kg_to_class_diagram(kg)
    enums = [t for t in result.domain_model.types if hasattr(t, "literals")]
    assert len(enums) == 1
    literal_names = {lit.name for lit in enums[0].literals}
    assert literal_names == {"red", "green"}


def test_complement_of_materializes_abstract_aux_with_ocl(tmp_path: Path):
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
    result = kg_to_class_diagram(kg)
    aux = _class(result.domain_model, "_not_Adult")
    assert aux.is_abstract is True
    assert _generalizes_to(result.domain_model, "Minor", "_not_Adult")
    bodies = _bodies_for(result.domain_model, "_not_Adult")
    assert any("not self.oclIsKindOf(Adult)" in b for b in bodies)


# ---------------------------------------------------------------------------
# D29-D32 special case: domain/range unionOf -> direct links, no aux class
# ---------------------------------------------------------------------------


def test_union_range_links_through_the_union_class(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Person a owl:Class .
    :Person1 a owl:Class .
    :Org a owl:Class .
    :worksFor a owl:ObjectProperty ;
        rdfs:domain :Person ;
        rdfs:range [ a owl:Class ; owl:unionOf ( :Person1 :Org ) ] .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    result = kg_to_class_diagram(kg)
    # D30/D31 resolve the union range bottom-up and link the property to the
    # class that expression resolves to — the D19 union class: a single
    # association to an abstract auxiliary that both members specialise, which
    # preserves both the navigation name and `oclIsKindOf(<member>)`.
    #
    # Linking to each member instead put the association *below* every invariant
    # emitted for the same property (whose context is that same resolved class),
    # where no OCL evaluator could reach it — they walk a context's ancestors,
    # never its subclasses.
    linked_target_types = set()
    for assoc in result.domain_model.associations:
        src_end = result.assoc_source_end.get(id(assoc))
        if src_end is None or src_end.type.name != "Person":
            continue
        target_end = next(e for e in assoc.ends if e is not src_end)
        linked_target_types.add(target_end.type.name)
    assert linked_target_types == {"_Org_Person1_Union"}
    assert _generalizes_to(result.domain_model, "Org", "_Org_Person1_Union")
    assert _generalizes_to(result.domain_model, "Person1", "_Org_Person1_Union")
    assert _class(result.domain_model, "_Org_Person1_Union").is_abstract is True
    # Built once, up front — not fanned out and merged back together during the
    # lowering, which is what `_merge_fanout` used to have to repair here.
    assert not any(w.code == "ASSOC_FANOUT_MERGED" for w in result.warnings)


def test_union_domain_drops_subclass_redundant_member(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Super a owl:Class .
    :Sub   a owl:Class ; rdfs:subClassOf :Super .
    :D a owl:Class .
    :p a owl:ObjectProperty ;
        rdfs:domain [ a owl:Class ; owl:unionOf ( :Super :Sub ) ] ;
        rdfs:range :D .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    result = kg_to_class_diagram(kg)
    owners = set()
    for assoc in result.domain_model.associations:
        if assoc.name != "p":
            continue
        src_end = result.assoc_source_end[id(assoc)]
        owners.add(src_end.type.name)
    # Only Super is linked (Sub is redundant, already covered by its superclass).
    assert owners == {"Super"}


# ---------------------------------------------------------------------------
# O01-O03: composite data ranges
# ---------------------------------------------------------------------------


def test_union_of_datatypes_materializes_composite_aux(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Event a owl:Class .
    :startDate a owl:DatatypeProperty ;
        rdfs:domain :Event ;
        rdfs:range [ a rdfs:Datatype ; owl:unionOf ( xsd:date xsd:dateTime ) ] .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    result = kg_to_class_diagram(kg)
    event = _class(result.domain_model, "Event")
    # startDate is an association to the composite aux (not a plain attribute)
    # since composite data ranges are represented as an aux Class.
    aux_names = {getattr(c, "name", "") for c in result.domain_model.types} - {"Event"}
    composite = next(n for n in aux_names if "Union" in n)
    aux = _class(result.domain_model, composite)
    value_attr = next(a for a in aux.attributes if a.name == "value")
    assert value_attr.type.name == "any"
    bodies = _bodies_for(result.domain_model, composite)
    assert any(
        "oclIsTypeOf(date)" in b and "oclIsTypeOf(datetime)" in b and " or " in b
        for b in bodies
    )
    # startDate stays an attribute on Event, typed by the composite class —
    # exactly the shape of Figure 2b in the paper. (The composite must be a
    # Class rather than a DataType because BUML's Property.owner rejects a
    # DataType owner, so it could not carry the `value` attribute otherwise.)
    start_date = next(a for a in event.attributes if a.name == "startDate")
    assert start_date.type is aux


# ---------------------------------------------------------------------------
# SHACL fixes/additions
# ---------------------------------------------------------------------------


def test_sh_class_translates_to_forall_not_exists(tmp_path: Path):
    """Regression test for the sh:class bug: it must mean "every value is of
    this type" (forAll), not "some value is" (exists)."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .

    :Person a owl:Class .
    :Pet a owl:Class .
    :owns a owl:ObjectProperty ; rdfs:domain :Person ; rdfs:range :Pet .
    :PersonShape a sh:NodeShape ; sh:targetClass :Person ;
        sh:property [ sh:path :owns ; sh:class :Pet ] .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    result = kg_to_class_diagram(kg)
    bodies = _bodies_for(result.domain_model, "Person")
    assert any("self.owns->forAll(v | v.oclIsKindOf(Pet))" in b for b in bodies)
    assert not any("->exists(" in b and "Pet" in b for b in bodies)


def test_sh_equals_and_less_than_translate_to_ocl(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Event a owl:Class .
    :startDate a owl:DatatypeProperty ; rdfs:domain :Event ; rdfs:range xsd:date .
    :endDate a owl:DatatypeProperty ; rdfs:domain :Event ; rdfs:range xsd:date .
    :EventShape a sh:NodeShape ; sh:targetClass :Event ;
        sh:property [ sh:path :startDate ; sh:lessThan :endDate ] .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    result = kg_to_class_diagram(kg)
    bodies = _bodies_for(result.domain_model, "Event")
    assert any(
        "self.startDate->forAll(v | self.endDate->forAll(w | v < w))" in b for b in bodies
    )
