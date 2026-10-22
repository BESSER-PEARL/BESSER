"""Tests for the UMLModel → BUML lowering.

The paper's rules target plain UML, which is more permissive than BUML's
metamodel. This suite pins the concessions the lowering makes — each one forced
by a specific validation rule in ``structural.py`` — and checks that no
generated OCL survives referencing something the model does not have.
"""

from __future__ import annotations

from pathlib import Path


from besser.BUML.metamodel.structural import (
    UNLIMITED_MAX_MULTIPLICITY,
    Class,
    Enumeration,
    PrimitiveDataType,
)
from besser.BUML.notations.kg_to_buml import kg_to_class_diagram
from besser.BUML.notations.kg_to_buml.owl2uml._plan import (
    ModelPlan,
    PlannedAssociation,
    PlannedAttribute,
    PlannedClass,
    PlannedDataType,
    PlannedEnumeration,
    PlannedInvariant,
)
from besser.BUML.notations.kg_to_buml.to_buml import build_domain_model
from besser.BUML.notations.owl_to_kg import owl_file_to_knowledge_graph


def _write_ttl(tmp_path: Path, content: str) -> str:
    path = tmp_path / "ontology.ttl"
    path.write_text(content.strip(), encoding="utf-8")
    return str(path)


def _lower(model: ModelPlan, **kwargs):
    warnings: list = []
    result = build_domain_model(model, model_name="Test", warnings=warnings, **kwargs)
    return result, warnings


def _classes(result):
    return {t.name: t for t in result.domain_model.types if isinstance(t, Class)}


def _codes(warnings):
    return {w.code for w in warnings}


# --- Builders for the internal plan ----------------------------------------
# Tests describe the pre-lowering model through these, never through the record
# classes directly, so a change to how that plan is represented touches only
# this block and not the thirty-odd tests that use it.


def _attr(name, type="str", lower=0, upper="*", **kwargs):
    return PlannedAttribute(name=name, type=type, lower=lower, upper=upper, **kwargs)


def _cls(name, *attributes, invariants=(), **flags):
    cls = PlannedClass(name=name, **flags)
    cls.attributes.extend(attributes)
    cls.invariants.extend(invariants)
    return cls


def _inv(context, body, name="check"):
    return PlannedInvariant(context=context, body=body, name=name)


def _datatype(name, *attributes, **flags):
    datatype = PlannedDataType(name=name, **flags)
    datatype.attributes.extend(attributes)
    return datatype


def _enum(name, literals):
    return PlannedEnumeration(name=name, literals=list(literals))


def _assoc(name, source, target, role, *, source_navigable=False, tgt_mult=(0, "*")):
    return PlannedAssociation(
        name=name, source=source, target=target, role=role, navigable_source=source_navigable,
        tgt_mult=tgt_mult,
    )


def _gen(subclass, superclass):
    return (subclass, superclass)


def _model(*items) -> ModelPlan:
    """Assemble a plan from classes, datatypes, enumerations, associations and generalizations."""
    model = ModelPlan()
    for item in items:
        if isinstance(item, PlannedClass):
            model.classes[item.name] = item
        elif isinstance(item, PlannedDataType):
            model.datatypes[item.name] = item
        elif isinstance(item, PlannedEnumeration):
            model.enumerations[item.name] = item
        elif isinstance(item, PlannedAssociation):
            model.associations.append(item)
        elif isinstance(item, tuple):          # a (subclass, superclass) pair, see _gen
            model.generalizations.append(item)
        else:
            raise TypeError(f"not a plan element: {item!r}")
    return model


# ---------------------------------------------------------------------------
# Names and types
# ---------------------------------------------------------------------------


def test_class_named_after_a_primitive_is_renamed():
    """``DomainModel.types`` reads a primitive-named type as "primitives already
    supplied", skips its own injection, then raises on the duplicate name."""
    model = _model(_cls("str"))
    result, warnings = _lower(model)

    assert "str_" in _classes(result)
    assert "NAME_SHADOWS_PRIMITIVE" in _codes(warnings)
    # The nine real primitives are still present and untouched.
    primitives = {t.name for t in result.domain_model.types if isinstance(t, PrimitiveDataType)}
    assert "str" in primitives


def test_materialised_data_range_becomes_a_class_not_a_datatype():
    """``Property.owner`` raises for a DataType owner, so a materialised data
    range could not carry its ``value`` attribute as a DataType."""
    model = _model(_datatype("_date_str_Union", _attr("value", "any")))
    result, _ = _lower(model)

    lowered = _classes(result)["_date_str_Union"]
    assert isinstance(lowered, Class)
    value = next(a for a in lowered.attributes if a.name == "value")
    assert value.type.name == "any"


def test_primitive_datatypes_resolve_to_the_shared_singletons():
    model = _model(
        _datatype("str", is_primitive=True),
        _cls("Person", _attr("name", "str", 0, 1)),
    )
    result, _ = _lower(model)

    name = next(a for a in _classes(result)["Person"].attributes if a.name == "name")
    assert isinstance(name.type, PrimitiveDataType)
    # Exactly one `str` type in the model, i.e. the singleton was reused.
    assert sum(1 for t in result.domain_model.types if t.name == "str") == 1


def test_enumeration_literals_are_sanitised_and_deduplicated():
    """``Mapper.literal_value`` quotes strings, and a value like ``"foo bar"``
    would be rejected outright by ``NamedElement.name``."""
    model = _model(_enum("Color", ['"deep red"', '"deep-red"', '"green"']))
    result, warnings = _lower(model)

    enum = next(t for t in result.domain_model.types if isinstance(t, Enumeration))
    names = {literal.name for literal in enum.literals}
    assert names == {"deep_red", "deep_red_2", "green"}
    # Renames are reported, once per enumeration; a literal kept as-is is not.
    renamed = [w.message for w in warnings if w.code == "ENUM_LITERAL_RENAMED"]
    assert renamed == [
        "Enumeration Color: 2 literal(s) renamed to valid, unique BUML names: "
        "'deep red' -> deep_red, 'deep-red' -> deep_red_2."
    ]


def test_enumeration_with_valid_literals_reports_nothing():
    result, warnings = _lower(_model(_enum("Color", ['"red"', '"green"'])))
    assert "ENUM_LITERAL_RENAMED" not in _codes(warnings)


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------


def test_zero_upper_bound_is_clamped():
    """``Multiplicity.max`` raises on ``<= 0``, but ``owl:maxCardinality 0`` is
    legal OWL."""
    model = _model(_cls("Person", _attr("nickname", "str", 0, 0)))
    result, warnings = _lower(model)

    nickname = next(a for a in _classes(result)["Person"].attributes if a.name == "nickname")
    assert nickname.multiplicity.max == 1
    assert "MULTIPLICITY_CLAMPED" in _codes(warnings)


def _expressions(result):
    return [c.expression for c in result.domain_model.constraints]


def test_clamped_attribute_keeps_the_constraint_that_forbids_the_value():
    """The clamp alone would let the model hold the one value the ontology rules out.

    The data-property rule pairs the ``(0, 0)`` with ``size() <= 0``; the lowering
    must carry that constraint through next to the clamp.
    """
    model = _model(_cls(
        "Person", _attr("nickname", "str", 0, 0),
        invariants=[_inv("Person", "self.nickname->size() <= 0")],
    ))
    result, warnings = _lower(model)

    nickname = next(a for a in _classes(result)["Person"].attributes if a.name == "nickname")
    assert nickname.multiplicity.max == 1 and "MULTIPLICITY_CLAMPED" in _codes(warnings)
    assert any("size() <= 0" in e for e in _expressions(result)), _expressions(result)


def test_clamped_association_end_keeps_the_constraint_that_forbids_every_link():
    """Same for an object property: ``owl:maxCardinality 0`` becomes ``0..1`` plus ``isEmpty()``."""
    model = _model(
        _cls("Person", invariants=[_inv("Person", "self.owns->isEmpty()")]),
        _cls("Pet"),
        _assoc("owns", "Person", "Pet", "owns", tgt_mult=(0, 0)),
    )
    result, warnings = _lower(model)

    ends = {e.name: e for a in result.domain_model.associations for e in a.ends}
    assert ends["owns"].multiplicity.max == 1
    assert "MULTIPLICITY_CLAMPED" in _codes(warnings)
    assert any("isEmpty()" in e for e in _expressions(result)), _expressions(result)


def test_unbounded_upper_becomes_the_buml_sentinel():
    model = _model(_cls("Person", _attr("alias", "str", 0, "*")))
    result, _ = _lower(model)

    alias = next(a for a in _classes(result)["Person"].attributes if a.name == "alias")
    assert alias.multiplicity.max == UNLIMITED_MAX_MULTIPLICITY


def test_attributes_do_not_share_the_default_multiplicity_singleton():
    """``Property``'s default ``Multiplicity(1, 1)`` is a module-level object;
    handing it out would make every attribute alias the same instance."""
    model = _model(_cls("Person", _attr("a", "str", 0, 1), _attr("b", "str", 0, 1)))
    result, _ = _lower(model)

    attributes = sorted(_classes(result)["Person"].attributes, key=lambda a: a.name)
    assert attributes[0].multiplicity is not attributes[1].multiplicity


# ---------------------------------------------------------------------------
# Generalizations
# ---------------------------------------------------------------------------


def test_generalization_cycles_are_broken():
    """``Class.all_parents()`` recurses without a visited set, so one cycle
    hangs the first consumer that walks the hierarchy."""
    model = _model(
        _cls("A"), _cls("B"), _cls("C"),
        _gen("A", "B"), _gen("B", "C"), _gen("C", "A"),
    )
    result, warnings = _lower(model)

    assert len(result.domain_model.generalizations) == 2
    assert "CYCLIC_SUBCLASS" in _codes(warnings)
    # The hierarchy is now safe to walk.
    for cls in _classes(result).values():
        cls.all_parents()


# ---------------------------------------------------------------------------
# Associations
# ---------------------------------------------------------------------------


def _assoc_model(targets):
    """One source class with an association named ``worksFor`` per target."""
    return _model(
        _cls("Person"),
        *[_cls(target) for target in targets],
        *[_assoc("worksFor", "Person", target, "worksFor") for target in targets],
    )


def test_fan_out_is_merged_onto_an_abstract_union_class():
    """BUML forbids two ends named ``worksFor`` on Person, and renaming one
    would break every ``self.worksFor`` in the generated OCL."""
    result, warnings = _lower(_assoc_model(["Org", "School"]))

    assert len(result.domain_model.associations) == 1
    classes = _classes(result)
    aux = classes["_Org_School_Union"]
    assert aux.is_abstract is True
    pairs = {(g.specific.name, g.general.name) for g in result.domain_model.generalizations}
    assert ("Org", "_Org_School_Union") in pairs
    assert ("School", "_Org_School_Union") in pairs
    assert "ASSOC_FANOUT_MERGED" in _codes(warnings)


def test_reverse_end_is_named_and_uniquified():
    """The paper's model leaves the reverse end anonymous; BUML requires a name
    and rejects duplicates across the target's hierarchy."""
    model = _model(
        _cls("Person"), _cls("Org"),
        *[_assoc(role, "Person", "Org", role) for role in ("worksFor", "foundedBy")],
    )
    result, _ = _lower(model)

    reverse_names = {
        result.assoc_source_end[id(a)].name for a in result.domain_model.associations
    }
    assert reverse_names == {"person", "person_2"}


def test_self_association_keeps_two_distinct_ends():
    model = _model(_cls("Person"), _assoc("knows", "Person", "Person", "knows"))
    result, _ = _lower(model)

    association = next(iter(result.domain_model.associations))
    names = {end.name for end in association.ends}
    assert len(names) == 2
    assert "knows" in names


def test_association_names_are_uniquified():
    """``DomainModel.associations`` rejects duplicate association names."""
    model = _model(
        _cls("Person"), _cls("Org"), _cls("School"),
        *[_assoc("memberOf", source, "School", f"memberOf{source}") for source in ("Person", "Org")],
    )
    result, _ = _lower(model)

    names = {a.name for a in result.domain_model.associations}
    assert len(names) == 2


def test_a_role_an_ancestor_already_provides_is_dropped(tmp_path: Path):
    """The restriction's auxiliary class sits *above* the class it constrains
    and carries the tighter bound, so it is the one that survives."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Pet a owl:Class .
    :Person a owl:Class ; rdfs:subClassOf
        [ a owl:Restriction ; owl:onProperty :owns ; owl:someValuesFrom :Pet ] .
    :owns a owl:ObjectProperty ; rdfs:domain :Person ; rdfs:range :Pet .
    """
    result = kg_to_class_diagram(owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl)))

    owns = [a for a in result.domain_model.associations if a.name == "owns"]
    assert len(owns) == 1
    target = next(e for e in owns[0].ends if e.type.name == "Pet")
    assert target.multiplicity.min == 1
    assert "ASSOC_INHERITED_SHADOWED" in _codes(result.warnings)


def _sibling_restrictions_model():
    """Two unrelated parents of ``Code`` each declaring a ``hasPart`` end."""
    return _model(
        _cls("RestrictA"), _cls("RestrictB"), _cls("Code"), _cls("Part"),
        _gen("Code", "RestrictA"), _gen("Code", "RestrictB"),
        _assoc("hasPart", "RestrictA", "Part", "hasPart"),
        _assoc("hasPart", "RestrictB", "Part", "hasPart"),
    )


def test_an_end_a_descendant_inherits_from_another_parent_is_dropped():
    """A descendant may reach one end per name across *all* its parents.

    The metamodel's setter only compares against the descendants' own ends, so
    the second ``hasPart`` used to build here (generalizations first) and then
    fail in every consumer that re-creates generalizations after associations.
    """
    result, warnings = _lower(_sibling_restrictions_model())

    has_part = [a for a in result.domain_model.associations if a.name.startswith("hasPart")]
    assert len(has_part) == 1
    assert "ASSOC_END_CONFLICT_DROPPED" in _codes(warnings)
    message = next(w.message for w in warnings if w.code == "ASSOC_END_CONFLICT_DROPPED")
    assert message.startswith("RestrictB.hasPart would give Code")
    code = _classes(result)["Code"]
    assert [e.name for e in code.all_association_ends()].count("hasPart") == 1


def test_sibling_end_conflict_reloads_in_the_editor_and_as_code(tmp_path: Path):
    """Both reload paths build associations before generalizations."""
    import runpy

    from besser.BUML.metamodel.structural import DomainModel
    from besser.utilities.buml_code_builder import domain_model_to_code
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
        class_buml_to_json,
    )
    from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
        process_class_diagram,
    )

    domain_model = _lower(_sibling_restrictions_model())[0].domain_model
    reloaded = process_class_diagram({"title": "Test", "model": class_buml_to_json(domain_model)})
    assert len(reloaded.associations) == len(domain_model.associations)

    path = tmp_path / "model.py"
    domain_model_to_code(domain_model, str(path))
    namespace = runpy.run_path(str(path))
    assert any(isinstance(value, DomainModel) for value in namespace.values())


# ---------------------------------------------------------------------------
# OCL constraints
# ---------------------------------------------------------------------------


def _model_with_invariant(body: str, *, with_feature: bool = True) -> ModelPlan:
    return _model(_cls(
        "Person",
        *([_attr("name")] if with_feature else []),
        invariants=[_inv("Person", body)],
    ))


def test_expression_uses_the_shape_the_editor_requires():
    """``ocl_parser`` drops any block that does not start with ``context
    <TypeName>`` and carry at least four whitespace-separated tokens."""
    result, _ = _lower(_model_with_invariant("self.name->size() > 0"))

    constraint = next(iter(result.domain_model.constraints))
    assert constraint.language == "OCL"
    assert constraint.context.name == "Person"
    assert constraint.expression == "context Person inv Person_check: self.name->size() > 0"


def test_invariant_naming_an_absent_feature_is_dropped():
    result, warnings = _lower(_model_with_invariant("self.name->size() > 0", with_feature=False))

    assert not result.domain_model.constraints
    assert "OCL_DROPPED_UNRESOLVED_FEATURE" in _codes(warnings)


def test_invariant_the_ocl_parser_rejects_is_dropped_with_its_reason():
    """Validation is the real parser's, not a regex's, so a body that is merely
    malformed (rather than naming an absent feature) is caught too, and the
    warning says why instead of shipping something no evaluator can read."""
    result, warnings = _lower(_model_with_invariant("self.name >"))

    assert not result.domain_model.constraints
    invalid = [w for w in warnings if w.code == "OCL_DROPPED_INVALID"]
    assert len(invalid) == 1
    assert "Person" in invalid[0].message and "not valid OCL" in invalid[0].message
    assert "OCL_DROPPED_UNRESOLVED_FEATURE" not in _codes(warnings)


def test_a_second_hop_is_checked_not_just_the_first():
    """The regex only ever inspected ``self.<first>``; the parser resolves the
    whole path, so an invariant navigating through a real feature into one that
    does not exist is dropped as well."""
    result, warnings = _lower(_model_with_invariant("self.name.nonexistent > 0"))

    assert not result.domain_model.constraints
    assert _codes(warnings) & {"OCL_DROPPED_UNRESOLVED_FEATURE", "OCL_DROPPED_INVALID"}


def test_invariant_naming_an_absent_type_is_dropped():
    result, warnings = _lower(_model_with_invariant("self.name->forAll(v | v.oclIsKindOf(Ghost))"))

    assert not result.domain_model.constraints
    assert "OCL_DROPPED_UNKNOWN_TYPE" in _codes(warnings)


def test_invariant_containing_a_context_token_is_dropped():
    """``ocl_parser`` splits blocks on the word ``context``, so one inside a body
    would silently truncate the constraint."""
    result, warnings = _lower(_model_with_invariant("self.name->size() > 0 and context > 1"))

    assert not result.domain_model.constraints
    assert "OCL_DROPPED_MALFORMED_BODY" in _codes(warnings)


def test_an_aux_class_invariant_resolves_through_its_own_declaration():
    """The † rule in Table 3: the auxiliary class materialised for a restriction
    carries the invariant, and declares the feature it constrains.

    That declaration is what makes the invariant evaluable — OCL resolves
    ``self.name`` by walking the context's ancestors, so an aux class talking
    about an attribute only its subclasses declare could never be checked. The
    class it constrains does not repeat the declaration: the aux class's carries
    the restriction's own multiplicity and is the one to keep."""
    model = _model(
        _cls(
            "_min_1_name_str", _attr("name", "str", 1, "*"),
            invariants=[_inv("_min_1_name_str", "self.name->size() >= 1", "min")],
            is_auxiliary=True,
        ),
        _cls("Person", _attr("name")),
        _gen("Person", "_min_1_name_str"),
    )
    result, warnings = _lower(model)

    assert len(result.domain_model.constraints) == 1
    constraint = result.domain_model.constraints.pop()
    assert constraint.context.name == "_min_1_name_str"
    assert "OCL_DROPPED_UNRESOLVED_FEATURE" not in _codes(warnings)

    lowered_person = next(t for t in result.domain_model.types if t.name == "Person")
    assert [a.name for a in lowered_person.attributes] == []
    assert [a.name for a in lowered_person.all_attributes()] == ["name"]
    assert "ATTR_INHERITED_SHADOWED" in _codes(warnings)


def test_inherited_attribute_is_found_through_a_grandparent_whatever_the_names_sort_like():
    """The ancestor's copy wins however far up it is and whichever class sorts first.

    The lowering fills classes ancestors-first and asks the metamodel what a class
    already inherits, so neither the depth nor the alphabetical order may matter.
    """
    model = _model(
        _cls("A_Child", _attr("name"), _attr("age", "int")),   # sorts before its ancestors
        _cls("M_Middle"),                                       # declares nothing itself
        _cls("Z_Top", _attr("name", "str", 1, 1)),
        _gen("A_Child", "M_Middle"),
        _gen("M_Middle", "Z_Top"),
    )
    result, warnings = _lower(model)

    classes = _classes(result)
    assert [a.name for a in classes["A_Child"].attributes] == ["age"]       # `name` is inherited
    assert sorted(a.name for a in classes["A_Child"].all_attributes()) == ["age", "name"]
    assert [a.name for a in classes["Z_Top"].attributes] == ["name"]
    shadowed = [w for w in warnings if w.code == "ATTR_INHERITED_SHADOWED"]
    assert len(shadowed) == 1 and "A_Child.name" in shadowed[0].message


def test_an_attribute_no_ancestor_declares_is_kept():
    model = _model(
        _cls("Child", _attr("nickname")),
        _cls("Parent", _attr("name")),
        _gen("Child", "Parent"),
    )
    result, warnings = _lower(model)

    assert [a.name for a in _classes(result)["Child"].attributes] == ["nickname"]
    assert "ATTR_INHERITED_SHADOWED" not in _codes(warnings)


def test_attribute_warnings_come_out_in_class_name_order_not_fill_order():
    """Fill order is ancestors-first; the warnings must not reveal it."""
    model = _model(
        _cls("B_Child", _attr("x", "str", 0, "*"), _attr("y", "nosuchtype")),
        _cls("A_Other", _attr("z", "nosuchtype")),
        _cls("Top", _attr("x")),
        _gen("B_Child", "Top"),
    )
    _, warnings = _lower(model)

    messages = [w.message for w in warnings if w.code in {"TYPE_UNRESOLVED", "ATTR_INHERITED_SHADOWED"}]
    assert len(messages) == 3, messages          # A_Other.z, B_Child.x shadowed, B_Child.y
    assert "A_Other" in messages[0] and all("B_Child" in m for m in messages[1:]), messages


def test_an_invariant_naming_a_feature_only_a_subclass_has_is_dropped():
    """The other half of that contract. A rule that emits an invariant is
    responsible for putting it on a class that owns what it navigates; the
    lowering does not go looking downwards for a class it would fit, because
    picking one silently narrows what the rule said."""
    model = _model(
        _cls(
            "_min_1_name_str",
            invariants=[_inv("_min_1_name_str", "self.name->size() >= 1", "min")],
            is_auxiliary=True,
        ),
        _cls("Person", _attr("name")),
        _gen("Person", "_min_1_name_str"),
    )
    result, warnings = _lower(model)

    assert not result.domain_model.constraints
    assert "OCL_DROPPED_UNRESOLVED_FEATURE" in _codes(warnings)


def test_constraint_names_are_uniquified():
    model = _model(_cls(
        "Person", _attr("name"),
        invariants=[_inv("Person", "self.name->size() > 0") for _ in range(3)],
    ))
    result, _ = _lower(model)

    names = {c.name for c in result.domain_model.constraints}
    assert len(names) == 3


def test_emit_ocl_false_produces_structure_only():
    result, _ = _lower(_model_with_invariant("self.name->size() > 0"), emit_ocl=False)
    assert not result.domain_model.constraints
    assert "Person" in _classes(result)


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------


def test_result_exposes_the_maps_the_object_diagram_needs(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Person a owl:Class .
    :Org a owl:Class .
    :name a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
    :worksFor a owl:ObjectProperty ; rdfs:domain :Person ; rdfs:range :Org .
    """
    result = kg_to_class_diagram(owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl)))

    assert result.iri_to_class["http://ex.org/Person"].name == "Person"
    assert result.property_iri_to_attribute["http://ex.org/name"].name == "name"
    association = result.property_iri_to_association["http://ex.org/worksFor"]
    assert result.assoc_source_end[id(association)].type.name == "Person"


def test_stereotypes_are_reported_rather_than_written_into_metadata(tmp_path: Path):
    """BUML has no stereotype concept, and ``Metadata.description`` round-trips
    into a visible comment box — misrepresenting a classification as prose."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Pet a owl:Class ; owl:equivalentClass [ a owl:Class ; owl:unionOf ( :Cat :Dog ) ] .
    :Cat a owl:Class .
    :Dog a owl:Class .
    """
    result = kg_to_class_diagram(owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl)))

    assert result.class_stereotypes["_Cat_Dog_Union"] == ["aux"]
    aux = next(t for t in result.domain_model.types if t.name == "_Cat_Dog_Union")
    assert aux.metadata is None or aux.metadata.description is None
    assert "AUX_CLASSES_MATERIALISED" in _codes(result.warnings)
