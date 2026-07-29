"""End-to-end regression over a real, published ontology.

The fixtures are the Bibliographic Ontology (BIBO 1.3) — 95 ``owl:Class``
declarations, 53 object properties, 55 data properties, 23 ``owl:Restriction``
class expressions and 14 named individuals — plus a SHACL shapes graph written
for this suite (BIBO ships none of its own; see the header of
``bibo-shapes.ttl``). Between them they exercise every phase of the pipeline on
input nobody wrote to be convenient: OWL-2 → UML, SHACL → OCL, and the lowering
onto BUML's metamodel.

The counts are asserted at two layers, because they legitimately differ:

* on the intermediate ``UMLModel`` — unconstrained by BUML;
* on the ``DomainModel`` — after the lowering reconciles it with BUML's
  metamodel rules (chiefly association end-name uniqueness).

This is also the regression that pins the failure that motivated the rewrite:
the previous implementation silently dropped classes that carried a shape, and
emitted a handful of OCL invariants where the rules call for scores of them.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from rdflib import Graph

from besser.BUML.metamodel.structural import Class, Enumeration
from besser.BUML.notations.kg_to_buml import kg_to_class_diagram
from besser.BUML.notations.kg_to_buml.kg_to_rdf import kg_to_rdf
from besser.BUML.notations.kg_to_buml.owl2uml import build_uml_model
from besser.utilities.owl_to_buml import owl_file_to_knowledge_graph

FIXTURES = Path(__file__).parent / "fixtures" / "bibo"
SCHEMA = FIXTURES / "bibo.ttl"
SHAPES = FIXTURES / "bibo-shapes.ttl"

#: Cheap, high-signal: a class per corner of the ontology, including the two
#: that only exist because a ``foaf:`` term was declared inside BIBO.
KEY_CLASSES = [
    "Document", "Book", "Journal", "AcademicArticle", "Thesis",
    "LegalDecision", "Collection", "DocumentStatus", "Agent", "Person",
]


@pytest.fixture(scope="module")
def combined_ttl(tmp_path_factory) -> str:
    """Schema + shapes as one file, the way a user would import them."""
    graph = Graph()
    graph.parse(str(SCHEMA), format="turtle")
    graph.parse(str(SHAPES), format="turtle")
    path = tmp_path_factory.mktemp("bibo") / "bibo-combined.ttl"
    graph.serialize(destination=str(path), format="turtle")
    return str(path)


@pytest.fixture(scope="module")
def uml_model(combined_ttl: str):
    """The intermediate model, before BUML's rules apply."""
    kg = owl_file_to_knowledge_graph(combined_ttl)
    graph, base = kg_to_rdf(kg)
    return build_uml_model(graph, base, shapes=graph)


@pytest.fixture(scope="module")
def result(combined_ttl: str):
    return kg_to_class_diagram(owl_file_to_knowledge_graph(combined_ttl), model_name="Bibo")


# ---------------------------------------------------------------------------
# The intermediate model
# ---------------------------------------------------------------------------


def test_uml_model_counts(uml_model):
    assert len(uml_model.classes) == 79
    assert sum(1 for c in uml_model.classes.values() if c.is_abstract) == 1
    assert sum(1 for c in uml_model.classes.values() if c.is_auxiliary) == 12
    assert len(uml_model.generalizations) == 79
    assert len(uml_model.associations) == 71
    assert len(uml_model.ocl_constraints()) == 127
    assert len(uml_model.enumerations) == 0


def test_invariant_origins_match_the_rule_census(uml_model):
    """Which transformation rule produced each invariant.

    The ``O*`` rules come from BIBO's own OWL axioms; the ``S-*`` rules from
    the shapes graph. Between them, 28 of the table's rows fire at least once.
    """
    census = Counter(i.origin_rule for i in uml_model.ocl_constraints())
    assert dict(sorted(census.items())) == {
        "O02": 1,
        "O07": 9,
        "O18": 20,
        "O27": 25,
        "S-and": 1,
        "S-class": 2,
        "S-datatype": 14,
        "S-disjoint": 1,
        "S-equals": 1,
        "S-hasValue": 1,
        "S-in": 3,
        "S-lessThan": 1,
        "S-lessThanOrEquals": 1,
        "S-maxCount": 18,
        "S-maxInclusive": 1,
        "S-maxLength": 2,
        "S-minCount": 5,
        "S-minExclusive": 1,
        "S-minInclusive": 2,
        "S-minLength": 2,
        "S-node": 2,
        "S-not": 1,
        "S-or": 1,
        "S-pattern": 7,
        "S-qualifiedMaxCount": 1,
        "S-qualifiedMinCount": 2,
        "S-uniqueLang": 1,
        "S-xone": 1,
    }


# ---------------------------------------------------------------------------
# The BUML model
# ---------------------------------------------------------------------------


def test_domain_model_counts(result):
    domain_model = result.domain_model
    classes = [t for t in domain_model.types if isinstance(t, Class)]
    # 79 from the intermediate model, plus the materialised data range the
    # ``sh:or`` of two XSD types collapses onto, plus the auxiliary union class
    # the fan-out merge introduces for the List/Seq author-ordering properties.
    assert len(classes) == 81
    assert sum(1 for c in classes if c.is_abstract) == 2
    assert len(domain_model.generalizations) == 81
    # 71 pre-merge; the three ``*List`` groups fan out to List and Seq and
    # collapse onto one union class.
    assert len(domain_model.associations) == 68
    assert len(domain_model.constraints) == 127
    assert len([t for t in domain_model.types if isinstance(t, Enumeration)]) == 0


def test_no_construct_is_dropped(result):
    """Nothing may be silently lost: every warning is informational."""
    codes = Counter(w.code for w in result.warnings)
    assert codes["ASSOC_DROPPED"] == 0
    assert codes["OCL_DROPPED_UNRESOLVED_FEATURE"] == 0
    assert codes["OCL_DROPPED_UNKNOWN_TYPE"] == 0
    assert codes["OCL_DROPPED_MALFORMED_BODY"] == 0
    assert codes["ORPHANED_CONSTRAINT"] == 0
    assert codes["LIST_ORDER_INFERRED"] == 0
    assert codes["SHACL_PATH_NOT_MODELLED"] == 0
    assert codes["SHACL_COMPLEX_PATH"] == 0


@pytest.mark.parametrize("name", KEY_CLASSES)
def test_key_classes_exist(result, name: str):
    assert any(
        isinstance(t, Class) and t.name == name for t in result.domain_model.types
    )


def test_bibo_disambiguates_the_two_event_classes(result):
    """BIBO declares ``bibo:Event`` and imports ``c4dm:Event``. Both must
    survive under distinct UML names rather than one silently winning."""
    names = {t.name for t in result.domain_model.types if isinstance(t, Class)}
    assert {"Event_bibo", "Event_c4dm"} <= names


def test_identifier_subproperties_become_derivation_invariants(result):
    """O27: BIBO declares a dozen identifier schemes as
    ``rdfs:subPropertyOf dc:identifier``. Each one has to constrain the
    inherited attribute on the class that owns it — here the auxiliary union
    that ``bibo:Collection`` and ``bibo:Document`` share."""
    expressions = sorted(
        c.expression for c in result.domain_model.constraints
        if c.context.name == "_Collection_Document_Union" and "_O27" in c.name
    )
    assert len(expressions) == 14
    assert (
        "context _Collection_Document_Union inv _Collection_Document_Union_O27_3: "
        "self.doi->asSet()->forAll(v | self.identifier->asSet()->includes(v))"
    ) in expressions
    # isbn10/isbn13 refine bibo:isbn, not dc:identifier.
    assert (
        "context _Collection_Document_Union inv _Collection_Document_Union_O27_7: "
        "self.isbn10->asSet()->forAll(v | self.isbn->asSet()->includes(v))"
    ) in expressions


def test_document_carries_the_full_shacl_workout(result):
    """``bibo:Document`` is where the shapes graph is densest: it is the one
    context that exercises datatype, pattern, count, length, range, node,
    class, logical and property-pair rules together."""
    expressions = {
        c.expression for c in result.domain_model.constraints
        if c.context.name == "Document" and "_O" not in c.name
    }
    assert expressions == {
        "context Document inv Document_citedBy_not: "
        "self.citedBy->forAll(v | not v.oclIsKindOf(Note))",
        "context Document inv Document_cites_node: "
        "self.cites->forAll(v | v.oclIsKindOf(Document))",
        "context Document inv Document_doi_datatype: "
        "self.doi->forAll(v | v.oclIsTypeOf(str))",
        "context Document inv Document_doi_maxCount: self.doi->size() <= 1",
        "context Document inv Document_doi_pattern: "
        r"self.doi->forAll(v | v.matches('^10\.[0-9]{4,9}/[^ ]+$'))",
        "context Document inv Document_editor_or: "
        "self.editor->forAll(v | v.oclIsKindOf(Agent) or v.oclIsKindOf(Organization))",
        "context Document inv Document_isbn10_datatype: "
        "self.isbn10->forAll(v | v.oclIsTypeOf(str))",
        "context Document inv Document_isbn10_maxCount: self.isbn10->size() <= 1",
        "context Document inv Document_isbn10_pattern: "
        "self.isbn10->forAll(v | v.matches('^[0-9]{9}[0-9X]$'))",
        "context Document inv Document_isbn13_datatype: "
        "self.isbn13->forAll(v | v.oclIsTypeOf(str))",
        "context Document inv Document_isbn13_maxCount: self.isbn13->size() <= 1",
        "context Document inv Document_isbn13_pattern: "
        "self.isbn13->forAll(v | v.matches('^97[89][0-9]{10}$'))",
        "context Document inv Document_numPages_maxCount: self.numPages->size() <= 1",
        "context Document inv Document_numPages_maxInclusive: "
        "self.numPages->forAll(v | v <= 100000)",
        "context Document inv Document_numPages_minInclusive: "
        "self.numPages->forAll(v | v >= 1)",
        "context Document inv Document_pageEnd_datatype: "
        "self.pageEnd->forAll(v | v.oclIsTypeOf(str))",
        "context Document inv Document_pageEnd_maxCount: self.pageEnd->size() <= 1",
        "context Document inv Document_pageStart_datatype: "
        "self.pageStart->forAll(v | v.oclIsTypeOf(str))",
        "context Document inv Document_pageStart_lessThanOrEquals: "
        "self.pageStart->forAll(v | self.pageEnd->forAll(w | v <= w))",
        "context Document inv Document_pageStart_maxCount: self.pageStart->size() <= 1",
        "context Document inv Document_reviewOf_disjoint: "
        "self.reviewOf->intersection(self.transcriptOf)->isEmpty()",
        "context Document inv Document_shortTitle_maxLength: "
        "self.shortTitle->forAll(v | v.size() <= 40)",
        "context Document inv Document_shortTitle_uniqueLang: "
        "self.shortTitle->collect(p | p.language)->isUnique()",
        "context Document inv Document_status_class: "
        "self.status->forAll(v | v.oclIsKindOf(DocumentStatus))",
        "context Document inv Document_volume_datatype: "
        "self.volume->forAll(v | v.oclIsTypeOf(str))",
        "context Document inv Document_volume_minLength: "
        "self.volume->forAll(v | v.size() >= 1)",
    }


def test_shapes_are_inherited_not_copied(result):
    """A shape on a subclass constrains the attribute it inherits, and does not
    restate the superclass's own constraints on it. ``bibo:doi`` is declared
    once, shaped ``maxCount 1`` on ``Document`` and ``minCount 1`` on
    ``AcademicArticle`` five generalizations below it."""
    academic = sorted(
        c.name for c in result.domain_model.constraints
        if c.context.name == "AcademicArticle"
    )
    assert academic == [
        "AcademicArticle_doi_minCount",
        "AcademicArticle_pmid_datatype",
        "AcademicArticle_pmid_maxCount",
        "AcademicArticle_pmid_pattern",
        "AcademicArticle_status_in",
    ]


def test_sh_in_over_iris_resolves_to_individual_names(result):
    """``sh:in`` members that are IRIs must come out as the UML names of the
    individuals BIBO declares, not as raw IRIs or blank placeholders."""
    thesis = next(
        c.expression for c in result.domain_model.constraints
        if c.name == "Thesis_degree_in"
    )
    assert thesis == (
        "context Thesis inv Thesis_degree_in: "
        "self.degree->forAll(v | Set{ma, ms, phd}->includes(v))"
    )
    manuscript = next(
        c.expression for c in result.domain_model.constraints
        if c.name == "Manuscript_status_hasValue"
    )
    assert manuscript == (
        "context Manuscript inv Manuscript_status_hasValue: self.status->includes(draft)"
    )


def test_untranslatable_shacl_constraints_are_ignored(result):
    """``sh:nodeKind``, ``sh:closed``, ``sh:flags``, ``sh:languageIn``,
    ``sh:severity``, ``sh:message``, ``sh:name`` and ``sh:description`` have no
    OCL equivalent. The ``bibo:Report`` shape carries all eight; only its two
    ``sh:maxCount`` constraints may reach the model."""
    report = sorted(
        c.name for c in result.domain_model.constraints if c.context.name == "Report"
    )
    assert report == ["Report_shortDescription_maxCount", "Report_uri_maxCount"]


def test_composite_data_range_is_a_class_with_a_value_attribute(result):
    """The ``sh:or ( xsd:date xsd:dateTime )`` on ``bibo:argued``: every operand
    is a datatype, so the constraint is not inlined on each context class —
    it materialises one shared data range and retargets the attribute at it."""
    union = next(
        t for t in result.domain_model.types
        if isinstance(t, Class) and t.name == "_date_datetime_Union"
    )
    value = next(a for a in union.attributes if a.name == "value")
    assert value.type.name == "any"

    invariants = [c for c in result.domain_model.constraints if c.context is union]
    assert len(invariants) == 1
    assert invariants[0].expression == (
        "context _date_datetime_Union inv _date_datetime_Union_union_invariant: "
        "self.value->forAll(v | v.oclIsTypeOf(date) or v.oclIsTypeOf(datetime))"
    )

    argued = next(
        a for t in result.domain_model.types if isinstance(t, Class) and t.name == "LegalDocument"
        for a in t.attributes if a.name == "argued"
    )
    assert argued.type is union


def test_all_values_from_restrictions_become_auxiliary_classes(result):
    """O07: BIBO constrains ``dc:hasPart`` with ``owl:allValuesFrom`` on nine
    different collection classes. Each becomes an auxiliary superclass carrying
    the type invariant, so the subclasses inherit it."""
    aux = sorted(
        t.name for t in result.domain_model.types
        if isinstance(t, Class) and t.name.startswith("_all_hasPart_")
    )
    assert aux == [
        "_all_hasPart_Article",
        "_all_hasPart_Book",
        "_all_hasPart_Document",
        "_all_hasPart_Issue",
        "_all_hasPart_LegalDocument",
        "_all_hasPart_Legislation",
        "_all_hasPart_Slide",
        "_all_hasPart_Webpage",
        "_all_hasPart__Collection_Document_Union",
    ]
    # bibo:Journal is a Periodical whose parts are Issues.
    assert any(
        g.specific.name == "Journal" and g.general.name == "_all_hasPart_Issue"
        for g in result.domain_model.generalizations
    )


# ---------------------------------------------------------------------------
# Downstream contracts
# ---------------------------------------------------------------------------


def test_every_constraint_is_well_formed(result):
    import re

    type_names = {t.name for t in result.domain_model.types}
    for constraint in result.domain_model.constraints:
        assert constraint.language == "OCL"
        assert re.match(r"^context \w+ inv \w+: \S", constraint.expression)
        assert constraint.context.name in type_names


def test_model_round_trips_through_the_editor(result):
    """The whole point of the conversion is that the editor can render it, and
    ``ocl_parser`` silently drops any expression it cannot re-read."""
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
        class_buml_to_json,
    )
    from besser.utilities.web_modeling_editor.backend.services.converters.parsers.ocl_parser import (
        process_ocl_constraints,
    )

    diagram = class_buml_to_json(result.domain_model)
    elements = diagram["elements"]
    elements = list(elements.values()) if isinstance(elements, dict) else elements
    blocks = [e["constraint"] for e in elements if e["type"] == "ClassOCLConstraint"]
    assert len(blocks) == len(result.domain_model.constraints)

    recovered, errors = process_ocl_constraints("\n".join(blocks), result.domain_model, 0)
    assert not errors
    assert len(recovered) == len(result.domain_model.constraints)


def test_object_diagram_builds_on_the_same_result(result, combined_ttl: str):
    """BIBO declares 14 named individuals — the degrees, the document statuses
    and its own authors — so the object diagram is not empty.

    Only the *number* of objects is asserted. ``kg_to_object_diagram`` is not
    yet hash-seed independent: both an object's name (``bdarcus`` vs the
    ``foaf:name`` literal ``Bruce D'Arcus``) and the classifier it is bound to
    (``DocumentStatus`` vs the ``owl:Thing`` fallback) vary with
    ``PYTHONHASHSEED``, so asserting either would be flaky. The class diagram,
    which is what ``test_determinism.py`` covers, does not have this problem.
    """
    from besser.BUML.notations.kg_to_buml import kg_to_object_diagram

    kg = owl_file_to_knowledge_graph(combined_ttl)
    object_result = kg_to_object_diagram(kg, class_result=result, model_name="Bibo")
    assert object_result.object_model is not None
    assert len(object_result.object_model.objects) == 23
