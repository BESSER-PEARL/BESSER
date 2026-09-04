"""SHACL shapes → OCL invariants (rules S01-S31 of the KG2UML paper).

Each test feeds a small TTL fragment through the real importer
(``owl_file_to_knowledge_graph``) and the real converter
(``kg_to_class_diagram``), so it exercises the whole chain: the KG projection
(``kg_to_rdf``), the rule engine (``owl2uml.shacl``), and the BUML lowering
(``to_buml``).

Replaces the old ``test_constraint_to_ocl.py``, which tested a hand-rolled
emitter that no longer exists. Constraints with no OCL equivalent
(``sh:nodeKind``, ``sh:closed``, ``sh:flags``, ``sh:languageIn``,
``sh:severity``, ``sh:message``) are intentionally not translated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from besser.BUML.notations.kg_to_buml import kg_to_class_diagram
from besser.utilities.owl_to_buml import owl_file_to_knowledge_graph


PREAMBLE = """
@prefix : <http://ex.org/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:Person a owl:Class .
:Pet a owl:Class .
:name a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
:age a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:integer .
:nickname a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
:owns a owl:ObjectProperty ; rdfs:domain :Person ; rdfs:range :Pet .
"""


def _convert(tmp_path: Path, shape_body: str):
    """Convert the shared Person/Pet ontology plus ``shape_body``."""
    ttl = f"{PREAMBLE}\n:PersonShape a sh:NodeShape ; sh:targetClass :Person ;\n{shape_body}\n"
    path = tmp_path / "ontology.ttl"
    path.write_text(ttl.strip(), encoding="utf-8")
    return kg_to_class_diagram(owl_file_to_knowledge_graph(str(path)))


def _bodies(result, context_name: str = "Person"):
    return [
        c.expression for c in result.domain_model.constraints
        if c.context.name == context_name
    ]


def _assert_body(result, fragment: str, context_name: str = "Person"):
    bodies = _bodies(result, context_name)
    assert any(fragment in b for b in bodies), (
        f"{fragment!r} not found in {context_name} invariants:\n"
        + "\n".join(f"  {b}" for b in bodies)
    )


# ---------------------------------------------------------------------------
# Cardinality and value range (S19-S26)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "predicate, operator",
    [("sh:minCount", ">="), ("sh:maxCount", "<=")],
)
def test_count_constraints(tmp_path: Path, predicate: str, operator: str):
    result = _convert(tmp_path, f'    sh:property [ sh:path :name ; {predicate} 2 ] .')
    _assert_body(result, f"self.name->size() {operator} 2")


@pytest.mark.parametrize(
    "predicate, operator",
    [
        ("sh:minInclusive", ">="),
        ("sh:maxInclusive", "<="),
        ("sh:minExclusive", ">"),
        ("sh:maxExclusive", "<"),
    ],
)
def test_range_constraints(tmp_path: Path, predicate: str, operator: str):
    result = _convert(tmp_path, f'    sh:property [ sh:path :age ; {predicate} 18 ] .')
    _assert_body(result, f"self.age->forAll(v | v {operator} 18)")


@pytest.mark.parametrize(
    "predicate, operator",
    [("sh:minLength", ">="), ("sh:maxLength", "<=")],
)
def test_length_constraints(tmp_path: Path, predicate: str, operator: str):
    result = _convert(tmp_path, f'    sh:property [ sh:path :name ; {predicate} 3 ] .')
    _assert_body(result, f"self.name->forAll(v | v.size() {operator} 3)")


# ---------------------------------------------------------------------------
# Type and value constraints (S01, S03, S04, S14, S18)
# ---------------------------------------------------------------------------


def test_datatype_maps_to_ocl_is_type_of(tmp_path: Path):
    result = _convert(tmp_path, '    sh:property [ sh:path :name ; sh:datatype xsd:string ] .')
    _assert_body(result, "self.name->forAll(v | v.oclIsTypeOf(str))")


def test_class_means_for_all_not_exists(tmp_path: Path):
    """``sh:class`` constrains *every* value, so it is forAll, never exists."""
    result = _convert(tmp_path, '    sh:property [ sh:path :owns ; sh:class :Pet ] .')
    _assert_body(result, "self.owns->forAll(v | v.oclIsKindOf(Pet))")
    assert not any("->exists(" in b for b in _bodies(result))


def test_has_value_maps_to_includes(tmp_path: Path):
    result = _convert(tmp_path, '    sh:property [ sh:path :name ; sh:hasValue "Alice" ] .')
    _assert_body(result, "self.name->includes('Alice')")


def test_pattern_maps_to_matches(tmp_path: Path):
    result = _convert(tmp_path, '    sh:property [ sh:path :name ; sh:pattern "^[A-Z]" ] .')
    _assert_body(result, "self.name->forAll(v | v.matches('^[A-Z]'))")


def test_in_maps_to_disjunction_of_equalities(tmp_path: Path):
    """``sh:in`` expands to ``v = a or v = b``.

    B-OCL has no collection literal, so ``Set{...}->includes(v)`` — which the
    editor could not read back — is emitted as the equivalent disjunction.
    """
    result = _convert(
        tmp_path, '    sh:property [ sh:path :name ; sh:in ( "Alice" "Bob" ) ] .'
    )
    _assert_body(result, "self.name->forAll(v | v = 'Alice' or v = 'Bob')")


def test_empty_in_list_requires_an_empty_property(tmp_path: Path):
    """``sh:in ()`` admits no value at all, which is what ``isEmpty()`` says."""
    result = _convert(tmp_path, '    sh:property [ sh:path :name ; sh:in ( ) ] .')
    _assert_body(result, "self.name->isEmpty()")


def test_unique_lang_maps_to_is_unique(tmp_path: Path):
    """``isUnique(body)`` evaluates ``body`` per element under the implicit
    iterator — the one-argument form is the only one B-OCL accepts."""
    result = _convert(
        tmp_path, '    sh:property [ sh:path :name ; sh:uniqueLang true ] .'
    )
    _assert_body(result, "self.name->isUnique(language)")


# ---------------------------------------------------------------------------
# Property-pair constraints (S06, S27-S29)
# ---------------------------------------------------------------------------


def test_equals_maps_to_mutual_inclusion(tmp_path: Path):
    result = _convert(
        tmp_path, '    sh:property [ sh:path :name ; sh:equals :nickname ] .'
    )
    _assert_body(
        result,
        "self.name->forAll(v | self.nickname->includes(v)) and "
        "self.nickname->forAll(v | self.name->includes(v))",
    )


@pytest.mark.parametrize(
    "predicate, operator",
    [("sh:lessThan", "<"), ("sh:lessThanOrEquals", "<=")],
)
def test_comparison_constraints(tmp_path: Path, predicate: str, operator: str):
    result = _convert(
        tmp_path, f'    sh:property [ sh:path :name ; {predicate} :nickname ] .'
    )
    _assert_body(
        result, f"self.name->forAll(v | self.nickname->forAll(w | v {operator} w))"
    )


def test_disjoint_maps_to_empty_intersection(tmp_path: Path):
    result = _convert(
        tmp_path, '    sh:property [ sh:path :name ; sh:disjoint :nickname ] .'
    )
    _assert_body(result, "self.name->intersection(self.nickname)->isEmpty()")


# ---------------------------------------------------------------------------
# Logical operators (S02, S15-S17)
# ---------------------------------------------------------------------------


def test_or_over_classes_joins_with_or(tmp_path: Path):
    ttl = f"""{PREAMBLE}
:Toy a owl:Class .
:PersonShape a sh:NodeShape ; sh:targetClass :Person ;
    sh:property [ sh:path :owns ; sh:or ( [ sh:class :Pet ] [ sh:class :Toy ] ) ] .
"""
    path = tmp_path / "ontology.ttl"
    path.write_text(ttl.strip(), encoding="utf-8")
    result = kg_to_class_diagram(owl_file_to_knowledge_graph(str(path)))
    _assert_body(result, "self.owns->forAll(v | v.oclIsKindOf(Pet) or v.oclIsKindOf(Toy))")


def test_not_over_a_class_negates(tmp_path: Path):
    result = _convert(
        tmp_path, '    sh:property [ sh:path :owns ; sh:not [ sh:class :Pet ] ] .'
    )
    _assert_body(result, "self.owns->forAll(v | not v.oclIsKindOf(Pet))")


def test_xone_expands_to_exactly_one(tmp_path: Path):
    ttl = f"""{PREAMBLE}
:Toy a owl:Class .
:PersonShape a sh:NodeShape ; sh:targetClass :Person ;
    sh:property [ sh:path :owns ; sh:xone ( [ sh:class :Pet ] [ sh:class :Toy ] ) ] .
"""
    path = tmp_path / "ontology.ttl"
    path.write_text(ttl.strip(), encoding="utf-8")
    result = kg_to_class_diagram(owl_file_to_knowledge_graph(str(path)))
    _assert_body(result, "(if v.oclIsKindOf(Pet) then 1 else 0 endif)")
    _assert_body(result, ") + (")
    _assert_body(result, "= 1)")


def test_qualified_value_shape_counts_matching_values(tmp_path: Path):
    result = _convert(
        tmp_path,
        '    sh:property [ sh:path :owns ; sh:qualifiedValueShape [ sh:class :Pet ] ;\n'
        '                  sh:qualifiedMinCount 1 ] .',
    )
    _assert_body(result, "self.owns->select(v | v.oclIsKindOf(Pet))->size() >= 1")


# ---------------------------------------------------------------------------
# Non-translatable constructs and guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape_body",
    [
        '    sh:property [ sh:path :name ; sh:nodeKind sh:Literal ] .',
        '    sh:property [ sh:path :name ; sh:severity sh:Warning ] .',
        '    sh:property [ sh:path :name ; sh:message "must be set" ] .',
    ],
)
def test_constructs_without_ocl_equivalent_emit_nothing(tmp_path: Path, shape_body: str):
    """S08-S13: validation metadata that expresses no model constraint."""
    result = _convert(tmp_path, shape_body)
    assert _bodies(result) == []


def test_shape_on_unmodelled_property_is_skipped_with_a_warning(tmp_path: Path):
    """A ``sh:path`` the OWL phase never turned into a feature cannot be
    navigated, so the whole property shape is skipped rather than emitting a
    ``self.<x>`` the class does not have."""
    ttl = f"""{PREAMBLE}
:PersonShape a sh:NodeShape ; sh:targetClass :Person ;
    sh:property [ sh:path rdfs:label ; sh:minCount 1 ] .
"""
    path = tmp_path / "ontology.ttl"
    path.write_text(ttl.strip(), encoding="utf-8")
    result = kg_to_class_diagram(owl_file_to_knowledge_graph(str(path)))
    assert _bodies(result) == []
    assert any(w.code == "SHACL_PATH_NOT_MODELLED" for w in result.warnings)


def test_every_constraint_is_parseable_by_the_editor(tmp_path: Path):
    """Every emitted expression must survive the editor's OCL round-trip.

    ``ocl_parser`` drops any block that does not start with ``context
    <TypeName>`` and carry at least four whitespace-separated tokens, so a
    malformed expression would silently vanish between backend and frontend.
    """
    from besser.utilities.web_modeling_editor.backend.services.converters.parsers.ocl_parser import (
        process_ocl_constraints,
    )

    result = _convert(
        tmp_path,
        '    sh:property [ sh:path :name ; sh:minCount 1 ; sh:maxCount 5 ;\n'
        '                  sh:pattern "^[A-Z]" ; sh:datatype xsd:string ] .',
    )
    constraints = list(result.domain_model.constraints)
    assert constraints, "expected at least one invariant"
    for constraint in constraints:
        assert constraint.language == "OCL"
        assert constraint.expression.startswith(f"context {constraint.context.name} inv ")

    text = "\n".join(c.expression for c in constraints)
    recovered, errors = process_ocl_constraints(text, result.domain_model, 0)
    assert not errors
    assert len(recovered) == len(constraints)


#: One shape per rule whose expression shape is not exercised above. Each of
#: these emitted something the editor could not read back at some point:
#: ``sh:in``/``sh:xone`` used ``Set{}``/``Sequence{}`` literals B-OCL has no
#: production for, ``sh:uniqueLang`` used a zero-argument ``->isUnique()``,
#: ``sh:hasValue`` double-quoted its string, and a ``date``-named path could
#: not be navigated at all.
_ROUND_TRIP_SHAPES = {
    "in": '    sh:property [ sh:path :name ; sh:in ( "Alice" "Bob" ) ] .',
    "in_empty": '    sh:property [ sh:path :name ; sh:in ( ) ] .',
    "xone": '    sh:property [ sh:path :owns ; sh:xone ( [ sh:class :Pet ] [ sh:class :Person ] ) ] .',
    "uniqueLang": '    sh:property [ sh:path :name ; sh:uniqueLang true ] .',
    "hasValue": '    sh:property [ sh:path :name ; sh:hasValue "Alice" ] .',
    "hasValue_apostrophe": '    sh:property [ sh:path :name ; sh:hasValue "O\'Brien" ] .',
    "pattern": '    sh:property [ sh:path :name ; sh:pattern "^[A-Z]" ] .',
    "disjoint": '    sh:property [ sh:path :name ; sh:disjoint :nickname ] .',
    "minInclusive": '    sh:property [ sh:path :age ; sh:minInclusive 18 ] .',
    "qualified": ('    sh:property [ sh:path :owns ; sh:qualifiedValueShape [ sh:class :Pet ] ;\n'
                  '                  sh:qualifiedMinCount 1 ] .'),
}


@pytest.mark.parametrize("shape_body", _ROUND_TRIP_SHAPES.values(), ids=_ROUND_TRIP_SHAPES)
def test_each_rule_survives_the_editor_round_trip(tmp_path: Path, shape_body: str):
    """Emitting an invariant is only half the job — it has to parse.

    Each expression is fed to ``process_ocl_constraints`` on its own, because
    ANTLR recovers across ``context`` boundaries and a joined parse comes back
    clean even when individual blocks are unreadable.
    """
    from besser.utilities.web_modeling_editor.backend.services.converters.parsers.ocl_parser import (
        process_ocl_constraints,
    )

    result = _convert(tmp_path, shape_body)
    for constraint in result.domain_model.constraints:
        recovered, errors = process_ocl_constraints(
            constraint.expression, result.domain_model, 0
        )
        assert not errors, f"{constraint.name}: {constraint.expression} -> {errors}"
        assert len(recovered) == 1, constraint.expression
