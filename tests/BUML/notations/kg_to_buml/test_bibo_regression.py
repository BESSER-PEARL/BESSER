"""End-to-end regression over a real, published ontology.

The fixtures are the Bibliographic Ontology (BIBO 1.3) — 95 ``owl:Class``
declarations, 53 object properties, 55 data properties, 23 ``owl:Restriction``
class expressions and 14 named individuals — plus a SHACL shapes graph written
for this suite (BIBO ships none of its own; see the header of
``bibo-shapes.ttl``). Between them they exercise every phase of the pipeline on
input nobody wrote to be convenient: OWL-2 → UML, SHACL → OCL, and the lowering
onto BUML's metamodel.

The counts are asserted at three layers, because they legitimately differ:

* on the intermediate ``UMLModel`` — unconstrained by BUML;
* on the ``DomainModel`` converted straight from the file — after the lowering
  reconciles it with BUML's metamodel rules (chiefly association end-name
  uniqueness) and drops the invariants no context can navigate;
* on the ``DomainModel`` converted after the preflight's recommendations are
  accepted, which is the flow the editor puts users through. BIBO declares 42
  properties with no ``rdfs:domain``, and an invariant that navigates one of
  them can only resolve if the user accepts ``attach_to_thing`` for it. That is
  the only layer where nothing is dropped.

This is also the regression that pins the failure that motivated the rewrite:
the previous implementation silently dropped classes that carried a shape, and
emitted a handful of OCL invariants where the rules call for scores of them.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest
from rdflib import Graph

from besser.BUML.metamodel.structural import Class, Enumeration
from besser.BUML.notations.kg_to_buml import kg_to_class_diagram
from besser.BUML.notations.kg_to_buml.kg_to_rdf import kg_to_rdf
from besser.BUML.notations.kg_to_buml.owl2uml import build_uml_model
from besser.BUML.notations.kg_to_buml.preflight import analyze_kg_for_class_diagram
from besser.BUML.notations.kg_to_buml.resolutions import KGResolution
from besser.BUML.notations.kg_to_buml.to_buml import _OCL_OPERATIONS
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
    """Converted straight from the file, with no preflight decisions applied."""
    return kg_to_class_diagram(owl_file_to_knowledge_graph(combined_ttl), model_name="Bibo")


@pytest.fixture(scope="module")
def refined_result(combined_ttl: str):
    """Converted after accepting every ``PROPERTY_NO_DOMAIN`` recommendation.

    What the editor's Refine-KG panel produces when the user takes the offered
    fix: each domain-less property is attached to ``Thing`` and every top-level
    class is made to inherit from it, so an invariant navigating one of those
    properties resolves from its own context.
    """
    kg = owl_file_to_knowledge_graph(combined_ttl)
    report = analyze_kg_for_class_diagram(kg)
    resolutions = [
        KGResolution(
            issue_id=issue.id,
            choice=issue.recommended_action.key,
            parameters=dict(issue.recommended_action.parameters),
        )
        for issue in report.issues
        if issue.code == "PROPERTY_NO_DOMAIN"
    ]
    assert resolutions, "the fixture is pointless if the detector finds nothing"
    return kg_to_class_diagram(kg, model_name="Bibo", resolutions=resolutions)


# ---------------------------------------------------------------------------
# The intermediate model
# ---------------------------------------------------------------------------


def test_uml_model_counts(uml_model):
    # 79 named classes plus the abstract union D19 materialises for the
    # Collection/Document domain that six BIBO properties share: D30/D31 resolve
    # a union-typed domain or range to that one class instead of linking to each
    # member, which is what collapses 71 associations into 62.
    assert len(uml_model.classes) == 80
    assert sum(1 for c in uml_model.classes.values() if c.is_abstract) == 2
    assert sum(1 for c in uml_model.classes.values() if c.is_auxiliary) == 13
    assert len(uml_model.generalizations) == 81
    assert len(uml_model.associations) == 62
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
    # 80 from the intermediate model, plus the materialised data range the
    # ``sh:or`` of two XSD types collapses onto.
    assert len(classes) == 81
    assert sum(1 for c in classes if c.is_abstract) == 2
    assert len(domain_model.generalizations) == 81
    assert len(domain_model.associations) == 62
    # 127 emitted, 20 dropped — see test_only_domainless_properties_are_dropped.
    assert len(domain_model.constraints) == 107
    assert len([t for t in domain_model.types if isinstance(t, Enumeration)]) == 0


def test_refined_domain_model_keeps_every_constraint(refined_result):
    """Accepting the recommendations costs nothing structurally and loses no
    invariant: the same 81 classes and 62 associations, 13 more generalizations
    (the top-level classes now inherit from ``Thing``), and all 127 invariants."""
    domain_model = refined_result.domain_model
    classes = [t for t in domain_model.types if isinstance(t, Class)]
    assert len(classes) == 81
    assert len(domain_model.associations) == 62
    assert len(domain_model.generalizations) == 94
    assert len(domain_model.constraints) == 127


def test_no_construct_is_dropped(refined_result):
    """Nothing may be silently lost: every warning is informational."""
    codes = Counter(w.code for w in refined_result.warnings)
    assert codes["ASSOC_DROPPED"] == 0
    assert codes["ASSOC_INHERITED_SHADOWED"] == 0
    assert codes["OCL_DROPPED_UNRESOLVED_FEATURE"] == 0
    assert codes["OCL_DROPPED_UNKNOWN_TYPE"] == 0
    assert codes["OCL_DROPPED_MALFORMED_BODY"] == 0
    assert codes["ORPHANED_CONSTRAINT"] == 0
    assert codes["LIST_ORDER_INFERRED"] == 0
    assert codes["SHACL_PATH_NOT_MODELLED"] == 0
    assert codes["SHACL_COMPLEX_PATH"] == 0


def test_only_domainless_properties_are_dropped(result, combined_ttl: str):
    """Converting without refining drops 20 invariants, and every one of them
    is a property the preflight offered to fix.

    A domain-less property is attached to ``Thing``, which nothing inherits from
    unless the user says so — so ``self.<property>`` resolves from no context at
    all and the invariant would fail in any evaluator. Dropping it with a warning
    is the honest outcome; the way to keep it is to accept the recommendation."""
    dropped = [w for w in result.warnings if w.code == "OCL_DROPPED_UNRESOLVED_FEATURE"]
    assert len(dropped) == 20

    report = analyze_kg_for_class_diagram(owl_file_to_knowledge_graph(combined_ttl))
    offered = {
        (issue.recommended_action.parameters["property_iri"] or "").rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        for issue in report.issues
        if issue.code == "PROPERTY_NO_DOMAIN"
    }
    for warning in dropped:
        navigated = warning.message.split("navigates self.")[1].split(",")[0]
        assert navigated in offered, f"{navigated} is dropped but never offered a fix"


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


def test_identifier_subproperties_become_derivation_invariants(refined_result):
    """O27: BIBO declares a dozen identifier schemes as
    ``rdfs:subPropertyOf dc:identifier``. Each one has to constrain the
    inherited attribute on the class that owns it — here the auxiliary union
    that ``bibo:Collection`` and ``bibo:Document`` share."""
    expressions = sorted(
        c.expression for c in refined_result.domain_model.constraints
        if c.context.name == "_Collection_Document_Union" and "_O27" in c.name
    )
    assert len(expressions) == 14
    assert (
        "context _Collection_Document_Union inv _Collection_Document_Union_O27_3: "
        "self.doi->asSet()->forAll(v | self.identifier->asSet()->includes(v))"
    ) in expressions
    # isbn10/isbn13 refine bibo:isbn, not dc:identifier — and bibo:isbn has no
    # rdfs:domain, so this pair only survives on the refined model.
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


@pytest.mark.parametrize("conversion", ["result", "refined_result"])
def test_every_constraint_is_navigable_from_its_context(request, conversion: str):
    """Every ``self.<feature>`` must resolve from the invariant's own context.

    This is the guard the suite was missing. B-OCL resolves a feature through
    ``Class.all_attributes()`` / ``all_association_ends()``
    (``BOCLVisitorImpl._resolve_property``), both of which walk ``all_parents()``
    — never subclasses, and with no implicit ``owl:Thing``. An invariant that
    breaks this rule reaches the editor and the generators and then fails with
    "Property 'x' not found in context 'Y'"; 18 of BIBO's 127 did.

    It deliberately does not parse anything, so it keeps its meaning while
    ``BOCL.g4`` still cannot read every expression — a syntax error used to mask
    the resolution failure behind it.
    """
    domain_model = request.getfixturevalue(conversion).domain_model
    for constraint in domain_model.constraints:
        context = constraint.context
        available = (
            {attribute.name for attribute in context.all_attributes()}
            | {end.name for end in context.all_association_ends()}
        )
        for feature in re.findall(r"self\.([A-Za-z_]\w*)", constraint.expression):
            assert feature in _OCL_OPERATIONS or feature in available, (
                f"{constraint.name}: self.{feature} is not navigable from {context.name}"
            )


def test_every_constraint_is_well_formed(result):
    type_names = {t.name for t in result.domain_model.types}
    for constraint in result.domain_model.constraints:
        assert constraint.language == "OCL"
        assert re.match(r"^context \w+ inv \w+: \S", constraint.expression)
        assert constraint.context.name in type_names


#: Constructs the current ``BOCL.g4`` does not accept. Closing that gap is a
#: separate piece of work; until it lands, an invariant using one of these
#: reaches the editor but cannot be re-read.
_UNSUPPORTED_OCL = re.compile(
    r"->asSet\(\)|Set\{|Sequence\{|->intersection\(|->isUnique\(|oclIsTypeOf\(date\)"
)


def test_model_round_trips_through_the_editor(result):
    """The whole point of the conversion is that the editor can render it, and
    ``ocl_parser`` silently drops any expression it cannot re-read.

    Each block is parsed on its own. Handing the parser all of them joined by
    newlines — which is what this test used to do — reports no error even when
    dozens of the individual expressions are unreadable, because ANTLR recovers
    across the block boundaries: at the time this was written, 50 of 127
    expressions failed on their own while the joined parse came back clean.

    The count of unreadable blocks is asserted exactly, so that finishing the
    grammar work turns this test red and it gets tightened back to zero rather
    than being forgotten."""
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

    unreadable = []
    for block in blocks:
        recovered, errors = process_ocl_constraints(block, result.domain_model, 0)
        if errors:
            unreadable.append(block)
        else:
            assert len(recovered) == 1, block

    assert len(unreadable) == 30
    # Nothing is unreadable for any reason other than the known grammar gap.
    for block in unreadable:
        assert _UNSUPPORTED_OCL.search(block), block
