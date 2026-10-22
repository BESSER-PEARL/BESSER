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
  them can only resolve if the user accepts ``attach_to_thing`` for it.

At both BUML layers several auxiliary restriction classes declare
``dc:hasPart`` (and, refined, ``dc:isPartOf``) for one shared subclass, which
BUML allows only one end of that name. The lowering declares the end once --
on ``Thing`` when it is their common ancestor, on an auxiliary
``_hasPart_Domain`` parent otherwise -- and restates each class's narrower
bound or target as an invariant, so no association's meaning and no invariant
is lost to it (``test_every_planned_end_keeps_its_meaning``).

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

from besser.BUML.metamodel.structural import UNLIMITED_MAX_MULTIPLICITY, Class, Enumeration
from besser.BUML.notations.kg_to_buml import kg_to_class_diagram
from besser.BUML.notations.kg_to_buml.kg_to_rdf import kg_to_rdf
from besser.BUML.notations.kg_to_buml.owl2uml import plan_domain_model
from besser.BUML.notations.kg_to_buml.preflight import analyze_kg_for_class_diagram
from besser.BUML.notations.kg_to_buml.resolutions import KGResolution, apply_resolutions
from besser.BUML.notations.owl_to_kg import owl_file_to_knowledge_graph

#: Names that may follow ``self.`` without being a feature. This audit keeps its
#: own list rather than reading the lowering's: its job is to be an independent
#: second opinion, and one that shares the code under test is not.
_OCL_OPERATIONS = frozenset({
    "oclIsKindOf", "oclIsTypeOf", "oclAsType", "oclIsNew", "oclIsUndefined",
    "allInstances", "size", "language", "asSet", "asBag", "asSequence",
    "isEmpty", "notEmpty", "sum", "count", "first", "last", "flatten",
})

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
    return plan_domain_model(graph, base, shapes=graph)


@pytest.fixture(scope="module")
def result(combined_ttl: str):
    """Converted straight from the file, with no preflight decisions applied."""
    return kg_to_class_diagram(owl_file_to_knowledge_graph(combined_ttl), model_name="Bibo")


def _no_domain_resolutions(kg):
    """Accept every ``PROPERTY_NO_DOMAIN`` recommendation, as the Refine-KG panel does."""
    report = analyze_kg_for_class_diagram(kg)
    return [
        KGResolution(
            issue_id=issue.id,
            choice=issue.recommended_action.key,
            parameters=dict(issue.recommended_action.parameters),
        )
        for issue in report.issues
        if issue.code == "PROPERTY_NO_DOMAIN"
    ]


@pytest.fixture(scope="module")
def refined_uml_model(combined_ttl: str):
    """The intermediate model of the refined conversion."""
    kg = owl_file_to_knowledge_graph(combined_ttl)
    graph, base = kg_to_rdf(apply_resolutions(kg, _no_domain_resolutions(kg)))
    return plan_domain_model(graph, base, shapes=graph)


@pytest.fixture(scope="module")
def refined_result(combined_ttl: str):
    """Converted after accepting every ``PROPERTY_NO_DOMAIN`` recommendation.

    What the editor's Refine-KG panel produces when the user takes the offered
    fix: each domain-less property is attached to ``Thing`` and every top-level
    class is made to inherit from it, so an invariant navigating one of those
    properties resolves from its own context.
    """
    kg = owl_file_to_knowledge_graph(combined_ttl)
    resolutions = _no_domain_resolutions(kg)
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
    assert len(uml_model.ocl_constraints()) == 128
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
        "O27": 26,
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


def _classes(result):
    return {t.name: t for t in result.domain_model.types if isinstance(t, Class)}


def _stated(result):
    """``(context, body)`` of every constraint in the converted model."""
    return {
        (c.context.name, c.expression.split(": ", 1)[1]) for c in result.domain_model.constraints
    }


def _planned(uml_model):
    return {(i.context, " ".join(i.body.split())) for i in uml_model.ocl_constraints()}


def test_domain_model_counts(result):
    domain_model = result.domain_model
    classes = _classes(result)
    # 80 from the intermediate model, the materialised data range the ``sh:or``
    # of two XSD types collapses onto, and the two the shared ``hasPart`` end
    # needs (see test_hasPart_restrictions_share_one_end).
    assert len(classes) == 83
    assert sorted(name for name, c in classes.items() if c.is_abstract) == [
        "_Collection_Document_Union", "_List_Seq_Union",
        "_Thing__Collection_Document_Union_Union", "_hasPart_Domain",
    ]
    # 81 planned, the 9 restriction classes under _hasPart_Domain, 2 targets under its union.
    assert len(domain_model.generalizations) == 92
    # 62 planned; the 9 restriction ends of hasPart become one.
    assert len(domain_model.associations) == 54
    # 128 planned, 20 dropped (test_only_domainless_properties_are_dropped),
    # 2 restated (test_hasPart_restrictions_share_one_end).
    assert len(domain_model.constraints) == 110
    assert len([t for t in domain_model.types if isinstance(t, Enumeration)]) == 0


#: The restriction classes BIBO declares ``dc:hasPart`` on that share a subclass
#: with another of them (``bibo:Issue`` is both an ``_all_hasPart_Article`` and
#: a ``_min_1_hasPart_Thing``). ``_all_hasPart_Slide`` shares none.
_HASPART_RESTRICTIONS = [
    "_all_hasPart_Article", "_all_hasPart_Book", "_all_hasPart_Document",
    "_all_hasPart_Issue", "_all_hasPart_LegalDocument", "_all_hasPart_Legislation",
    "_all_hasPart_Webpage", "_all_hasPart__Collection_Document_Union", "_min_1_hasPart_Thing",
]


def test_hasPart_restrictions_share_one_end(result):
    """Unrefined, the restriction classes have no common ancestor, so the
    ``hasPart`` end is declared once, on an abstract parent of all nine. Its
    target is the union of the restrictions' targets (``Thing`` and the
    Collection/Document union, which every other target specialises), and what
    each restriction said beyond that end is restated on it:
    ``_min_1_hasPart_Thing`` keeps "at least one part, each a Thing"; each
    ``_all_*`` keeps its O07 type invariant, which already says it."""
    classes = _classes(result)
    domain = classes["_hasPart_Domain"]
    assert sorted(c.name for c in domain.specializations()) == _HASPART_RESTRICTIONS
    (association,) = [a for a in result.domain_model.associations if any(e.type is domain for e in a.ends)]
    end = next(e for e in association.ends if e.name == "hasPart")
    assert end.type.name == "_Thing__Collection_Document_Union_Union"
    assert (end.multiplicity.min, end.multiplicity.max) == (0, UNLIMITED_MAX_MULTIPLICITY)

    stated = _stated(result)
    assert ("_min_1_hasPart_Thing", "self.hasPart->size() >= 1") in stated
    assert ("_min_1_hasPart_Thing", "self.hasPart->forAll(o | o.oclIsKindOf(Thing))") in stated
    assert ("_all_hasPart_Issue", "self.hasPart->forAll(o | o.oclIsKindOf(Issue))") in stated
    # The property's own declaration, on Thing, is a hierarchy of its own and stays as it is.
    own = result.property_iri_to_association["http://purl.org/dc/terms/hasPart"]
    assert result.assoc_source_end[id(own)].type.name == "Thing"


def test_refined_domain_model_keeps_every_constraint(refined_result, refined_uml_model):
    """Accepting the recommendations resolves every domain-less invariant, and
    all 128 planned invariants reach the model.

    ``Thing`` now declares ``hasPart`` and ``isPartOf`` and is an ancestor of
    every class, the restriction classes included, so their ends merge into
    Thing's (``ASSOC_INHERITED_SHADOWED``): 62 planned associations, 51 built.
    Two invariants restate the cardinality only those ends carried."""
    domain_model = refined_result.domain_model
    classes = _classes(refined_result)
    assert len(classes) == 81
    assert len(domain_model.associations) == 51
    # 81 planned, plus the 13 top-level classes and 11 restriction classes made Thing's.
    assert len(domain_model.generalizations) == 105
    assert len(domain_model.constraints) == 130

    stated = _stated(refined_result)
    assert len(_planned(refined_uml_model)) == 128
    assert _planned(refined_uml_model) <= stated
    assert stated - _planned(refined_uml_model) == {
        ("_min_1_hasPart_Thing", "self.hasPart->size() >= 1"),
        ("_max_1_isPartOf_Thing", "self.isPartOf->size() <= 1"),
    }

    thing = classes["Thing"]
    for name in [*_HASPART_RESTRICTIONS, "_all_hasPart_Slide", "_max_1_isPartOf_Thing"]:
        assert thing in classes[name].all_parents(), name
    for iri in ("http://purl.org/dc/terms/hasPart", "http://purl.org/dc/terms/isPartOf"):
        association = refined_result.property_iri_to_association[iri]
        assert refined_result.assoc_source_end[id(association)].type is thing


@pytest.mark.parametrize("conversion, plan", [
    ("result", "uml_model"), ("refined_result", "refined_uml_model"),
])
def test_every_planned_end_keeps_its_meaning(request, conversion, plan):
    """Every association the rules planned is still navigable from its source
    under its own name, and whatever the end now serving it does not already
    guarantee -- the target, the bounds -- is stated as an invariant on the
    source. Merging ends must not lose a restriction."""
    result, uml_model = request.getfixturevalue(conversion), request.getfixturevalue(plan)
    classes, stated = _classes(result), _stated(result)
    for planned in uml_model.associations:
        source, target = classes.get(planned.source), classes.get(planned.target)
        if source is None or target is None:
            continue
        role = planned.role or planned.name
        (end,) = [e for e in source.all_association_ends() if e.name == role]
        if end.type is not target and target not in end.type.all_parents():
            assert (planned.source, f"self.{role}->forAll(o | o.oclIsKindOf({planned.target}))") in stated
        lower, upper = planned.tgt_mult
        if lower > end.multiplicity.min:
            assert (planned.source, f"self.{role}->size() >= {lower}") in stated
        if upper != "*" and int(upper) < end.multiplicity.max:
            assert (planned.source, f"self.{role}->size() <= {upper}") in stated


@pytest.mark.parametrize("conversion", ["result", "refined_result"])
def test_the_model_passes_metamodel_validation(request, conversion):
    """Including the end-name rule: no class reaches two ends of one name."""
    domain_model = request.getfixturevalue(conversion).domain_model
    assert domain_model.validate(raise_exception=False)["errors"] == []


def test_no_construct_is_dropped(refined_result):
    """Nothing may be silently lost: every warning is informational."""
    codes = Counter(w.code for w in refined_result.warnings)
    assert codes["ASSOC_DROPPED"] == 0
    assert codes["OCL_DROPPED_UNRESOLVED_FEATURE"] == 0
    assert codes["OCL_DROPPED_INVALID"] == 0
    assert codes["OCL_DROPPED_UNKNOWN_TYPE"] == 0
    assert codes["OCL_DROPPED_MALFORMED_BODY"] == 0
    assert codes["ORPHANED_CONSTRAINT"] == 0
    assert codes["LIST_ORDER_INFERRED"] == 0
    assert codes["SHACL_PATH_NOT_MODELLED"] == 0
    assert codes["SHACL_COMPLEX_PATH"] == 0


def test_only_domainless_properties_are_dropped(result, uml_model, combined_ttl: str):
    """Converting without refining drops 20 invariants, and every one of them
    navigates a property the preflight offered to fix.

    A domain-less property is attached to ``Thing``, which nothing inherits from
    unless the user says so — so ``self.<property>`` resolves from no context at
    all and the invariant would fail in any evaluator. Dropping it with a warning
    is the honest outcome; the way to keep it is to accept the recommendation."""
    dropped = _planned(uml_model) - _stated(result)
    assert len(dropped) == 20
    assert Counter(w.code for w in result.warnings)["OCL_DROPPED_UNRESOLVED_FEATURE"] == 20

    report = analyze_kg_for_class_diagram(owl_file_to_knowledge_graph(combined_ttl))
    offered = {
        (issue.recommended_action.parameters["property_iri"] or "").rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        for issue in report.issues
        if issue.code == "PROPERTY_NO_DOMAIN"
    }
    classes = _classes(result)
    for context, body in dropped:
        reachable = (
            {a.name for a in classes[context].all_attributes()}
            | {e.name for e in classes[context].all_association_ends()}
        )
        unreachable = set(re.findall(r"self\.([A-Za-z_]\w*)", body)) - reachable - _OCL_OPERATIONS
        assert unreachable and unreachable <= offered, f"{context}: {body}"


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
        "self.shortTitle->isUnique(language)",
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
        "self.degree->forAll(v | v = ma or v = ms or v = phd)"
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


@pytest.mark.parametrize("with_shapes", [False, True], ids=["schema", "schema+shapes"])
def test_model_reloads_through_json_and_generated_code(with_shapes, combined_ttl, tmp_path):
    """Both reload paths re-create associations *before* generalizations, the
    opposite of the lowering. BUML checks end names the same way in either
    order, so this pins that the merged ``hasPart`` ends leave nothing for it to
    reject."""
    import runpy

    from besser.BUML.metamodel.structural import DomainModel
    from besser.utilities.buml_code_builder import domain_model_to_code
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
        class_buml_to_json,
    )
    from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
        process_class_diagram,
    )

    source = combined_ttl if with_shapes else str(SCHEMA)
    domain_model = kg_to_class_diagram(owl_file_to_knowledge_graph(source), model_name="Bibo").domain_model

    reloaded = process_class_diagram({"title": "Bibo", "model": class_buml_to_json(domain_model)})
    assert len(reloaded.associations) == len(domain_model.associations)
    assert len(reloaded.generalizations) == len(domain_model.generalizations)
    assert len(reloaded.constraints) == len(domain_model.constraints)

    path = tmp_path / "bibo_model.py"
    domain_model_to_code(domain_model, str(path))
    namespace = runpy.run_path(str(path))
    regenerated = next(v for v in namespace.values() if isinstance(v, DomainModel))
    assert len(regenerated.associations) == len(domain_model.associations)
    assert len(regenerated.generalizations) == len(domain_model.generalizations)


def test_every_constraint_is_well_formed(result):
    type_names = {t.name for t in result.domain_model.types}
    for constraint in result.domain_model.constraints:
        assert constraint.language == "OCL"
        assert re.match(r"^context \w+ inv \w+: \S", constraint.expression)
        assert constraint.context.name in type_names


def test_model_round_trips_through_the_editor(result):
    """The whole point of the conversion is that the editor can render it, and
    ``ocl_parser`` silently drops any expression it cannot re-read.

    Each block is parsed on its own. Handing the parser all of them joined by
    newlines — which is what this test used to do — reports no error even when
    dozens of the individual expressions are unreadable, because ANTLR recovers
    across the block boundaries: at the time this was written, 50 of 127
    expressions failed on their own while the joined parse came back clean.

    This used to assert an exact count of unreadable blocks (30, then 8) while
    the grammar could not read `->asSet()`, `->intersection()`, `.allInstances()`
    or `self.date`, and while the KG emitted `Set{}`/`Sequence{}` literals,
    zero-argument `->isUnique()` and double-quoted strings. All of that is
    fixed, so the count is zero and stays zero."""
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

    assert unreadable == []
