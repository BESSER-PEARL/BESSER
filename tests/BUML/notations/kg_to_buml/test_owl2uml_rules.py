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

from besser.BUML.metamodel.structural import Class, Enumeration
from besser.BUML.notations.kg_to_buml import kg_to_class_diagram
from besser.BUML.notations.owl_to_kg import owl_file_to_knowledge_graph


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


def test_object_max_cardinality_zero_emits_an_ocl_constraint(tmp_path: Path):
    """``owl:maxCardinality 0`` on an object property forbids every link.

    BUML's Multiplicity cannot express an upper bound of 0, so the lowering
    clamps (0, 0) to (0, 1) -- which would let the model hold exactly the one
    link the ontology rules out. The data-property branch already guarded its
    own 0 case with ``size() <= 0``; this is the object-property equivalent.
    """
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Pet a owl:Class .
    :owns a owl:ObjectProperty ; rdfs:domain :Person ; rdfs:range :Pet .
    :Person a owl:Class .
    :PetFreePerson a owl:Class ; rdfs:subClassOf [
        a owl:Restriction ;
        owl:onProperty :owns ;
        owl:maxCardinality "0"^^xsd:nonNegativeInteger
    ] .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    result = kg_to_class_diagram(kg)

    all_bodies = [c.expression for c in result.domain_model.constraints]
    assert any("self.owns->isEmpty()" in b for b in all_bodies), all_bodies


# ---------------------------------------------------------------------------
# Rules added after the paper audit: D17, D20, D21, D32, D37, D38, O04, O05,
# O19-O21, O27, O28
# ---------------------------------------------------------------------------

_PREFIXES = """
@prefix : <http://ex.org/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""


def _convert(tmp_path: Path, body: str):
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, _PREFIXES + body))
    return kg_to_class_diagram(kg)


def _constraints(result, ctx_name: str):
    return [c.expression for c in result.domain_model.constraints if c.context.name == ctx_name]


def _attr(result, class_name: str, attr_name: str):
    return next(a for a in _class(result.domain_model, class_name).attributes if a.name == attr_name)


def _end(result, assoc_name: str, end_name: str):
    association = next(a for a in result.domain_model.associations if a.name == assoc_name)
    return next(e for e in association.ends if e.name == end_name)


def test_d37_datatype_definition_keeps_its_name_and_facets(tmp_path: Path):
    """A datatype defined by restriction is one type, named as declared -- not a ghost class beside a facet-less one."""
    result = _convert(tmp_path, """
    :Person a owl:Class .
    :Age18 a rdfs:Datatype ; owl:equivalentClass [
        a rdfs:Datatype ; owl:onDatatype xsd:integer ;
        owl:withRestrictions ( [ xsd:minInclusive 18 ] [ xsd:maxInclusive 120 ] ) ] .
    :adultAge a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range :Age18 .
    """)
    names = [t.name for t in result.domain_model.types]
    assert names.count("Age18") == 1
    assert "external" not in result.class_stereotypes.get("Age18", [])
    assert _attr(result, "Person", "adultAge").type.name == "Age18"
    assert _attr(result, "Age18", "value").type.name == "int"
    assert any("v >= 18 and v <= 120" in b for b in _constraints(result, "Age18"))


def test_d37_datatype_defined_as_another_datatype_carries_its_type(tmp_path: Path):
    result = _convert(tmp_path, """
    :Person a owl:Class .
    :Alias a rdfs:Datatype ; owl:equivalentClass xsd:string .
    :alias a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range :Alias .
    """)
    assert _attr(result, "Alias", "value").type.name == "str"
    assert _attr(result, "Person", "alias").type.name == "Alias"


def test_o05_anonymous_datatype_restriction_emits_facet_invariant(tmp_path: Path):
    result = _convert(tmp_path, """
    :Person a owl:Class .
    :code a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range [
        a rdfs:Datatype ; owl:onDatatype xsd:string ;
        owl:withRestrictions ( [ xsd:pattern "[A-Z]+" ] [ xsd:minLength 2 ] ) ] .
    """)
    restricted = _attr(result, "Person", "code").type
    bodies = _constraints(result, restricted.name)
    assert any("v.matches('[A-Z]+') and v.size() >= 2" in b for b in bodies), bodies


def test_o05_unsupported_facet_is_reported(tmp_path: Path):
    result = _convert(tmp_path, """
    :Person a owl:Class .
    :price a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range [
        a rdfs:Datatype ; owl:onDatatype xsd:decimal ;
        owl:withRestrictions ( [ xsd:totalDigits 5 ] ) ] .
    """)
    assert any(w.code == "DATATYPE_FACET_UNSUPPORTED" for w in result.warnings)


def test_d17_data_one_of_strings_becomes_an_enumeration(tmp_path: Path):
    result = _convert(tmp_path, """
    :Shirt a owl:Class .
    :size a owl:DatatypeProperty ; rdfs:domain :Shirt ;
        rdfs:range [ a rdfs:Datatype ; owl:oneOf ( "S" "M" "L" ) ] .
    """)
    size = _attr(result, "Shirt", "size").type
    assert {lit.name for lit in size.literals} == {"S", "M", "L"}


def test_o04_data_one_of_typed_literals_restricts_the_value(tmp_path: Path):
    result = _convert(tmp_path, """
    :Shirt a owl:Class .
    :level a owl:DatatypeProperty ; rdfs:domain :Shirt ;
        rdfs:range [ a rdfs:Datatype ; owl:oneOf ( 1 2 3 ) ] .
    """)
    level = _attr(result, "Shirt", "level").type
    assert _attr(result, level.name, "value").type.name == "int"
    assert any("v = 1 or v = 2 or v = 3" in b for b in _constraints(result, level.name))


def test_d20_named_one_of_class_is_the_enumeration(tmp_path: Path):
    result = _convert(tmp_path, """
    :Color a owl:Class ; owl:equivalentClass [ a owl:Class ; owl:oneOf ( :red :green ) ] .
    :red a owl:NamedIndividual, :Color . :green a owl:NamedIndividual, :Color .
    :Car a owl:Class .
    :hasColor a owl:ObjectProperty, owl:FunctionalProperty ; rdfs:domain :Car ; rdfs:range :Color .
    """)
    color = next(t for t in result.domain_model.types if t.name == "Color")
    assert isinstance(color, Enumeration)
    assert {lit.name for lit in color.literals} == {"red", "green"}
    assert [t.name for t in result.domain_model.types].count("Color") == 1
    has_color = _attr(result, "Car", "hasColor")
    assert has_color.type is color and has_color.multiplicity.max == 1
    assert not any(a.name == "hasColor" for a in result.domain_model.associations)


def test_d20_one_of_class_with_other_axioms_stays_a_class(tmp_path: Path):
    result = _convert(tmp_path, """
    :Pet a owl:Class .
    :Sized a owl:Class ; owl:equivalentClass [ owl:oneOf ( :s1 ) ] ; rdfs:subClassOf :Pet .
    :s1 a owl:NamedIndividual .
    """)
    assert isinstance(_class(result.domain_model, "Sized"), Class)
    assert _generalizes_to(result.domain_model, "Sized", "Pet")
    assert any(w.code == "ONEOF_CLASS_KEPT" for w in result.warnings)


def test_d21_relational_one_of_links_every_individual(tmp_path: Path):
    result = _convert(tmp_path, """
    :Star a owl:Class . :Planet a owl:Class ; owl:equivalentClass [ owl:oneOf ( :earth :mars ) ] .
    :orbits a owl:ObjectProperty ; rdfs:domain :Planet ; rdfs:range :Star .
    :sun a owl:NamedIndividual, :Star .
    :earth a owl:NamedIndividual, :Planet ; :orbits :sun .
    :mars a owl:NamedIndividual, :Planet ; :orbits :sun .
    """)
    aux = "_earth_mars_OneOf"
    assert _generalizes_to(result.domain_model, "Planet", aux)
    for role in ("earth", "mars"):
        end = next(e for a in result.domain_model.associations for e in a.ends if e.name == role)
        assert end.type.name == "Planet" and (end.multiplicity.min, end.multiplicity.max) == (1, 1)


def test_d32_functional_and_inverse_functional_on_an_inverse_pair(tmp_path: Path):
    result = _convert(tmp_path, """
    :A a owl:Class . :B a owl:Class .
    :p a owl:ObjectProperty, owl:FunctionalProperty ; rdfs:domain :A ; rdfs:range :B ; owl:inverseOf :q .
    :q a owl:ObjectProperty, owl:InverseFunctionalProperty ; rdfs:domain :B ; rdfs:range :A .
    """)
    assert _end(result, "p", "p").multiplicity.max == 1                     # p is functional
    assert _end(result, "p", "q").multiplicity.max == 9999                  # nothing bounds q's side
    assert any("c.q->includes(d)" in b for b in _constraints(result, "A"))  # O22 for q


def test_d38_single_functional_key_is_unique_and_an_identifier(tmp_path: Path):
    result = _convert(tmp_path, """
    :P a owl:Class ; owl:hasKey ( :ssn ) .
    :ssn a owl:DatatypeProperty, owl:FunctionalProperty ; rdfs:domain :P ; rdfs:range xsd:string .
    """)
    assert any("P.allInstances()->isUnique(ssn)" in b for b in _constraints(result, "P"))
    assert _attr(result, "P", "ssn").is_id is True


def test_d38_compound_key_is_an_invariant_only(tmp_path: Path):
    result = _convert(tmp_path, """
    :P a owl:Class ; owl:hasKey ( :ssn :passport ) .
    :ssn a owl:DatatypeProperty, owl:FunctionalProperty ; rdfs:domain :P ; rdfs:range xsd:string .
    :passport a owl:DatatypeProperty, owl:FunctionalProperty ; rdfs:domain :P ; rdfs:range xsd:string .
    """)
    assert any("a.ssn <> b.ssn or a.passport <> b.passport" in b for b in _constraints(result, "P"))
    assert not _attr(result, "P", "ssn").is_id and not _attr(result, "P", "passport").is_id


def test_o19_chain_of_an_unasserted_property_derives_it(tmp_path: Path):
    result = _convert(tmp_path, """
    :P a owl:Class .
    :hasParent a owl:ObjectProperty ; rdfs:domain :P ; rdfs:range :P .
    :hasGrandparent a owl:ObjectProperty ; rdfs:domain :P ; rdfs:range :P ;
        owl:propertyChainAxiom ( :hasParent :hasParent ) .
    """)
    bodies = _constraints(result, "P")
    assert any("self.hasParent.hasParent->forAll(o | self.hasGrandparent->includes(o)) and "
               "self.hasGrandparent->forAll(o | self.hasParent.hasParent->includes(o))" in b for b in bodies), bodies
    assert _end(result, "hasGrandparent", "hasGrandparent").is_derived is True
    assert _end(result, "hasParent", "hasParent").is_derived is False


def test_o20_chain_of_an_asserted_property_only_implies_it(tmp_path: Path):
    result = _convert(tmp_path, """
    :P a owl:Class .
    :hasParent a owl:ObjectProperty ; rdfs:domain :P ; rdfs:range :P .
    :hasUncle a owl:ObjectProperty ; rdfs:domain :P ; rdfs:range :P ;
        owl:propertyChainAxiom ( :hasParent :hasParent ) .
    :x a owl:NamedIndividual, :P ; :hasUncle :x .
    """)
    bodies = _constraints(result, "P")
    assert any(b.endswith("self.hasParent.hasParent->forAll(o | self.hasUncle->includes(o))") for b in bodies), bodies
    assert _end(result, "hasUncle", "hasUncle").is_derived is False


def test_o21_all_disjoint_properties_is_every_pair(tmp_path: Path):
    result = _convert(tmp_path, """
    :P a owl:Class .
    :a1 a owl:ObjectProperty ; rdfs:domain :P ; rdfs:range :P .
    :a2 a owl:ObjectProperty ; rdfs:domain :P ; rdfs:range :P .
    :a3 a owl:ObjectProperty ; rdfs:domain :P ; rdfs:range :P .
    [ a owl:AllDisjointProperties ; owl:members ( :a1 :a2 :a3 ) ] .
    """)
    bodies = " ".join(_constraints(result, "P"))
    for pair in ("self.a1->forAll(o | self.a2->excludes(o))",
                 "self.a1->forAll(o | self.a3->excludes(o))",
                 "self.a2->forAll(o | self.a3->excludes(o))"):
        assert pair in bodies


def test_o27_union_of_sub_properties_derives_the_super_property(tmp_path: Path):
    result = _convert(tmp_path, """
    :P a owl:Class .
    :name a owl:DatatypeProperty ; rdfs:domain :P ; rdfs:range xsd:string .
    :given a owl:DatatypeProperty ; rdfs:domain :P ; rdfs:range xsd:string ; rdfs:subPropertyOf :name .
    :family a owl:DatatypeProperty ; rdfs:domain :P ; rdfs:range xsd:string ; rdfs:subPropertyOf :name .
    """)
    assert any("self.name->asSet() = self.family->asSet()->union(self.given->asSet())" in b
               for b in _constraints(result, "P"))
    assert _attr(result, "P", "name").is_derived is True


def test_o27_single_sub_property_is_not_a_derivation(tmp_path: Path):
    result = _convert(tmp_path, """
    :P a owl:Class .
    :name a owl:DatatypeProperty ; rdfs:domain :P ; rdfs:range xsd:string .
    :given a owl:DatatypeProperty ; rdfs:domain :P ; rdfs:range xsd:string ; rdfs:subPropertyOf :name .
    """)
    assert _attr(result, "P", "name").is_derived is False


def test_o28_disjoint_data_properties(tmp_path: Path):
    result = _convert(tmp_path, """
    :P a owl:Class .
    :a a owl:DatatypeProperty ; rdfs:domain :P ; rdfs:range xsd:string .
    :b a owl:DatatypeProperty ; rdfs:domain :P ; rdfs:range xsd:string ; owl:propertyDisjointWith :a .
    """)
    assert any("self.b->asSet()->intersection(self.a->asSet())->isEmpty()" in x
               or "self.a->asSet()->intersection(self.b->asSet())->isEmpty()" in x
               for x in _constraints(result, "P"))


def test_exact_cardinality_zero_on_an_object_property_emits_is_empty(tmp_path: Path):
    result = _convert(tmp_path, """
    :Pet a owl:Class . :Person a owl:Class .
    :owns a owl:ObjectProperty ; rdfs:domain :Person ; rdfs:range :Pet .
    :Free a owl:Class ; rdfs:subClassOf [ a owl:Restriction ; owl:onProperty :owns ;
        owl:cardinality "0"^^xsd:nonNegativeInteger ] .
    """)
    assert any("self.owns->isEmpty()" in c.expression for c in result.domain_model.constraints)


# ---------------------------------------------------------------------------
# D37 datatype definitions that refer back to themselves
# ---------------------------------------------------------------------------


def _plan(body: str, warnings: list):
    from rdflib import Graph

    from besser.BUML.notations.kg_to_buml.owl2uml import plan_domain_model

    graph = Graph().parse(data=_PREFIXES + body, format="turtle")
    return plan_domain_model(graph, "http://ex.org/", warnings=warnings)


def test_d37_datatype_equivalent_to_itself_is_a_bare_declaration(tmp_path: Path):
    """``A ≡ A`` used to recurse until RecursionError."""
    result = _convert(tmp_path, """
    :P a owl:Class .
    :A a rdfs:Datatype ; owl:equivalentClass :A .
    :x a owl:DatatypeProperty ; rdfs:domain :P ; rdfs:range :A .
    """)
    assert _attr(result, "P", "x").type.name == "A"
    assert _class(result.domain_model, "A").attributes == set()


def test_d37_mutually_equivalent_datatypes_merge_without_recursing(tmp_path: Path):
    """``A ≡ B ≡ A`` is one canonical datatype (D27); the equivalence is the
    merge, not a definition to follow."""
    result = _convert(tmp_path, """
    :P a owl:Class .
    :A a rdfs:Datatype ; owl:equivalentClass :B .
    :B a rdfs:Datatype ; owl:equivalentClass :A .
    :x a owl:DatatypeProperty ; rdfs:domain :P ; rdfs:range :A .
    """)
    assert _attr(result, "P", "x").type.name == "A"
    # Not a ``value`` slot typed by itself.
    assert _class(result.domain_model, "A").attributes == set()


def test_d37_definition_stated_on_an_alias_is_kept(tmp_path: Path):
    result = _convert(tmp_path, """
    :P a owl:Class .
    :A a rdfs:Datatype ; owl:equivalentClass :B .
    :B a rdfs:Datatype ; owl:equivalentClass xsd:string .
    :x a owl:DatatypeProperty ; rdfs:domain :P ; rdfs:range :A .
    """)
    assert _attr(result, "A", "value").type.name == "str"


def test_d37_cyclic_datatype_definitions_are_cut_with_a_warning():
    """``A ≡ unionOf(B, int)``, ``B ≡ unionOf(A, string)``: a genuine definition
    cycle (illegal in OWL 2) is followed once, then reported."""
    warnings: list = []
    plan = _plan("""
    :P a owl:Class .
    :A a rdfs:Datatype ; owl:equivalentClass [ a rdfs:Datatype ; owl:unionOf ( :B xsd:integer ) ] .
    :B a rdfs:Datatype ; owl:equivalentClass [ a rdfs:Datatype ; owl:unionOf ( :A xsd:string ) ] .
    :x a owl:DatatypeProperty ; rdfs:domain :P ; rdfs:range :A .
    """, warnings)
    assert {"A", "B"} <= set(plan.datatypes)
    assert [w.code for w in warnings].count("DATATYPE_CYCLIC") == 1


def test_malformed_cardinality_skips_the_restriction_with_a_warning(tmp_path: Path):
    """``int("abc")`` used to abort the whole conversion with a ValueError."""
    result = _convert(tmp_path, """
    :Pet a owl:Class . :Person a owl:Class .
    :owns a owl:ObjectProperty ; rdfs:domain :Person ; rdfs:range :Pet .
    :Q a owl:Class ; rdfs:subClassOf
        [ a owl:Restriction ; owl:onProperty :owns ; owl:maxCardinality "abc"^^xsd:nonNegativeInteger ] ,
        [ a owl:Restriction ; owl:onProperty :owns ; owl:minCardinality "-1"^^xsd:integer ] ,
        [ a owl:Restriction ; owl:onProperty :owns ; owl:cardinality "2"^^xsd:nonNegativeInteger ] .
    """)
    messages = [w.message for w in result.warnings if w.code == "OWL_CARDINALITY_INVALID"]
    assert len(messages) == 2
    assert any("'abc'" in m for m in messages) and any("'-1'" in m for m in messages)
    # The well-formed restriction beside them is still mapped.
    assert "_exact_2_owns_Pet" in _classes_by_name(result.domain_model)


def test_d38_key_on_an_unmodelled_property_names_it():
    warnings: list = []
    _plan("""
    :P a owl:Class ; owl:hasKey ( :ghost ) .
    """, warnings)
    assert [(w.code, w.message) for w in warnings if w.code == "KEY_SKIPPED"] == [
        ("KEY_SKIPPED", "Key of 'P' skipped: ghost is not modelled as a feature of it."),
    ]


def test_disjoint_object_and_data_property_is_reported_not_dropped_silently():
    warnings: list = []
    plan = _plan("""
    :P a owl:Class .
    :o a owl:ObjectProperty ; rdfs:domain :P ; rdfs:range :P .
    :d a owl:DatatypeProperty ; rdfs:domain :P ; owl:propertyDisjointWith :o .
    """, warnings)
    assert "DISJOINT_PROPERTIES_SKIPPED" in {w.code for w in warnings}
    assert not [i for i in plan.ocl_constraints() if i.origin_rule in ("O21", "O28")]
