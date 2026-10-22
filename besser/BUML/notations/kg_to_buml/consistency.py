"""OWL2 + SHACL consistency checking for a :class:`KnowledgeGraph`.

Delegates the actual validation to :mod:`pyshacl` (which itself wraps
:mod:`owlrl` for OWL2-RL inference). The KG is first serialised to an
``rdflib.Graph`` via the existing :func:`knowledge_graph_to_rdf`
exporter (``vocab="both"`` so both OWL and SHACL triples are present),
then augmented with a small set of synthetic SHACL shapes for OWL DL
constructs that ``pyshacl + owlrl`` doesn't translate into validation
violations on its own (notably ``owl:disjointWith``,
``owl:complementOf``, ``owl:FunctionalProperty``).

The whole export is the *data* (and ontology) graph. The *shapes* graph is a
separate one, built by :func:`_extract_shapes` from the shapes BESSER itself
writes and nothing else, so a construct an imported ontology carries beyond
BESSER's constraint vocabulary is never read as a shape.

The validation report is parsed back into our
:class:`ConsistencyIssue` shape, with focus-node IRIs mapped to KG
node ids. The shim shapes never escape this module — they live only
in the in-memory graphs passed to pyshacl.

pyshacl runs in a child process (``_shacl_worker.py``) that is killed when its
time budget runs out: ``sh:pattern`` is user-authored and evaluated with
:mod:`re`, which a catastrophically backtracking pattern can pin, GIL and all,
for as long as it likes. Patterns that are recognisably dangerous are dropped
up front and reported as a warning instead.

Requires :mod:`pyshacl`, which is a dependency of the web editor backend only
(``besser/utilities/web_modeling_editor/backend/requirements.txt``), not of the
``besser`` library. Import this module lazily from code that must work without it.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import rdflib
from rdflib import BNode, Literal, RDF, URIRef, XSD
from rdflib.namespace import OWL, RDFS

from besser.BUML.metamodel.kg.iri import local_name as _local_name
from besser.BUML.metamodel.kg import (
    KnowledgeGraph,
    KGBlank,
    KGLiteral,
    KGNodeConstraint,
    KGPropertyConstraint,
)
from besser.BUML.notations.kg_to_buml._common import short_digest
from besser.BUML.notations.kg_to_buml.preflight import kg_signature
from besser.utilities.kg_to_owl import (
    DEFAULT_NAMESPACE,
    SHACL_LOGICAL_KINDS,
    SHACL_NODE_KINDS,
    SHACL_PROPERTY_KINDS,
    _slugify,
    knowledge_graph_to_rdf,
)
from besser.BUML.metamodel.kg.constants import (
    SH as SH_NS,
)


__all__ = [
    "ConsistencyCheckError",
    "ConsistencyCheckTimeout",
    "ConsistencyIssue",
    "ConsistencyReport",
    "DEFAULT_CONSISTENCY_TIMEOUT_SECONDS",
    "MAX_SHACL_PATTERN_LENGTH",
    "check_kg_consistency",
]

logger = logging.getLogger(__name__)

SH = rdflib.Namespace(SH_NS)

#: Seconds the validator child process may run before it is killed.
DEFAULT_CONSISTENCY_TIMEOUT_SECONDS = 60.0

#: Longest ``sh:pattern`` that is evaluated at all; a longer one is skipped with
#: a warning. BESSER's own patterns are a few dozen characters.
MAX_SHACL_PATTERN_LENGTH = 1000

# The child process that runs pyshacl (see ``_validate``).
_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shacl_worker.py")
# How much of the child's stderr is logged when it dies.
_LOGGED_STDERR_CHARS = 2000


class ConsistencyCheckError(RuntimeError):
    """The KG could not be checked, for a reason that lies in the KG itself.

    The message is written for the end user (the web backend returns it as an
    HTTP 400). Raised when pyshacl rejects the KG's shapes, and, through
    :class:`ConsistencyCheckTimeout`, when the check exceeds its time budget.
    """


class ConsistencyCheckTimeout(ConsistencyCheckError):
    """The check ran out of time and its validator process was killed."""


# SHACL-AF constructs that make a validator EXECUTE something the KG author
# wrote, rather than merely check a shape. A SPARQL body may carry
# `SERVICE <http://...>`, which rdflib resolves with a plain urlopen() against
# no allowlist -- so on a shared host an uploaded ontology could make the
# server fetch a URL of the author's choosing (link-local metadata endpoints
# included). BESSER's own constraint vocabulary (``_SHACL_PROPERTY_KINDS`` /
# ``_SHACL_NODE_KINDS`` / ``_SHACL_LOGICAL_KINDS`` in ``kg_to_owl``) cannot
# author any of these, so dropping them costs nothing a BESSER model expresses;
# they can only arrive as raw triples through an imported OWL/TTL file.
#
# ``sh:target`` is the *custom* (SPARQL/JS) target predicate and belongs here.
# The Core targeting predicates -- sh:targetClass, sh:targetNode,
# sh:targetSubjectsOf, sh:targetObjectsOf -- are deliberately NOT listed: they
# are what BESSER's shapes use, and implicit class targeting is a separate
# always-on code path that `advanced=False` does not disable.
_EXECUTABLE_SHACL_PREDICATES = frozenset({
    SH.sparql, SH.rule, SH.target, SH.expression, SH.js,
    SH.select, SH.construct, SH.ask, SH.update,
    SH.jsLibrary, SH.jsFunctionName,
})

_EXECUTABLE_SHACL_TYPES = frozenset({
    SH.SPARQLRule, SH.TripleRule, SH.Rule,
    SH.SPARQLConstraint, SH.SPARQLTarget, SH.SPARQLTargetType,
    SH.SPARQLFunction, SH.SPARQLAskValidator, SH.SPARQLSelectValidator,
    SH.JSRule, SH.JSConstraint, SH.JSTarget, SH.JSTargetType,
    SH.JSFunction, SH.JSValidator, SH.JSLibrary,
})


# The shapes the validator is allowed to read. Stripping the constructs above
# from the KG leaves a *blacklist*; this is the allowlist behind it. Only what
# BESSER itself writes as a shape is copied into the shapes graph: the predicates
# of its constraint vocabulary (the exporter's three kind tables) and the
# structural ones that attach a shape to its targets and to its property shapes
# (the OWL -> SHACL shim below uses exactly these). Any other triple in an
# imported ontology stays in the *data* graph, where it is just data.
_SHAPE_STRUCTURE_PREDICATES = frozenset({
    SH.path, SH.property, SH.targetClass, SH.targetSubjectsOf, SH.qualifiedValueShape,
})
_SHAPE_PREDICATES = frozenset(
    {*SHACL_PROPERTY_KINDS.values(), *SHACL_NODE_KINDS.values(), *SHACL_LOGICAL_KINDS.values()}
    | _SHAPE_STRUCTURE_PREDICATES
)
_SHAPE_TYPES = frozenset({SH.NodeShape, SH.PropertyShape})
# Predicates whose object is itself a shape, so the walk follows it.
_SHAPE_VALUED = frozenset({SH.property, SH.qualifiedValueShape, SH["not"]})
# Predicates whose object is an rdf:List: copied as a list, and followed into
# the members when they are shapes.
_SHAPE_LISTS = frozenset({SH["and"], SH["or"], SH.xone})
_VALUE_LISTS = frozenset({SH["in"], SH.languageIn, SH.ignoredProperties})


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ConsistencyIssue:
    """A single inconsistency surfaced by the OWL/SHACL check."""

    id: str
    code: str
    severity: str           # "info" | "warning" | "violation"
    message: str
    affected_node_ids: List[str]
    affected_edge_ids: List[str] = field(default_factory=list)
    constraint_node_id: Optional[str] = None
    spec_kind: Optional[str] = None
    path_iri: Optional[str] = None
    #: One-line human-readable summary of the violated constraint
    #: (e.g. "hasPrincipalInvestigator: exactly 1 of SeniorResearcher"). Built
    #: from the source-shape's SHACL triples — gives the UI a clear "this is
    #: the constraint" label without the user having to read the verbose
    #: pyshacl message.
    constraint_label: Optional[str] = None


@dataclass
class ConsistencyReport:
    """The outcome of :func:`check_kg_consistency`: every issue, most severe first."""

    issues: List[ConsistencyIssue]
    issue_count: int
    severity_counts: Dict[str, int]
    kg_signature: str
    inference_used: str


# ---------------------------------------------------------------------------
# Constraint-component → ConstraintKind label mapping
# ---------------------------------------------------------------------------


_COMPONENT_TO_KIND: Dict[str, str] = {
    f"{SH}MinCountConstraintComponent": "minCardinality",
    f"{SH}MaxCountConstraintComponent": "maxCardinality",
    f"{SH}QualifiedMinCountConstraintComponent": "minQualifiedCardinality",
    f"{SH}QualifiedMaxCountConstraintComponent": "maxQualifiedCardinality",
    f"{SH}DatatypeConstraintComponent": "datatype",
    f"{SH}NodeKindConstraintComponent": "nodeKind",
    f"{SH}PatternConstraintComponent": "pattern",
    f"{SH}MinLengthConstraintComponent": "minLength",
    f"{SH}MaxLengthConstraintComponent": "maxLength",
    f"{SH}MinInclusiveConstraintComponent": "minInclusive",
    f"{SH}MaxInclusiveConstraintComponent": "maxInclusive",
    f"{SH}MinExclusiveConstraintComponent": "minExclusive",
    f"{SH}MaxExclusiveConstraintComponent": "maxExclusive",
    f"{SH}LanguageInConstraintComponent": "languageIn",
    f"{SH}UniqueLangConstraintComponent": "uniqueLang",
    f"{SH}InConstraintComponent": "in",
    f"{SH}HasValueConstraintComponent": "hasValue",
    f"{SH}ClassConstraintComponent": "someValuesFrom",
    f"{SH}NotConstraintComponent": "shaclNot",
    f"{SH}AndConstraintComponent": "shaclAnd",
    f"{SH}OrConstraintComponent": "shaclOr",
    f"{SH}XoneConstraintComponent": "shaclXone",
    f"{SH}ClosedConstraintComponent": "shaclClosed",
    f"{SH}DisjointConstraintComponent": "shaclDisjoint",
    f"{SH}EqualsConstraintComponent": "shaclEquals",
}


# Most severe first. ``sorted`` on the severity *strings* yields
# info < violation < warning, which is not a severity order at all.
_SEVERITY_RANK: Dict[str, int] = {"violation": 0, "warning": 1, "info": 2}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check_kg_consistency(
    kg: KnowledgeGraph,
    *,
    inference: str = "owlrl",
    timeout_seconds: Optional[float] = DEFAULT_CONSISTENCY_TIMEOUT_SECONDS,
) -> ConsistencyReport:
    """Validate ``kg``'s ABox against its OWL/SHACL constraints.

    Args:
        kg: The KG to check.
        inference: ``"rdfs"`` (faster, SHACL-only checks) or ``"owlrl"``
            (default; runs OWL2-RL pre-inference so OWL DL constructs
            are also checked).
        timeout_seconds: Budget for the validation, which runs in a child
            process that is killed when the budget is spent. ``None`` runs
            pyshacl in this process with no limit: only for a KG whose
            ``sh:pattern`` values are trusted.

    Returns:
        :class:`ConsistencyReport` whose ``issues`` list is empty when
        the KG is fully conformant.

    Raises:
        ConsistencyCheckTimeout: the validation did not finish in time.
        ConsistencyCheckError: pyshacl rejected the KG's shapes.
    """
    data_graph = knowledge_graph_to_rdf(kg, vocab="both")
    _normalize_count_literals(data_graph)
    _augment_with_owl_shim(data_graph)
    _strip_executable_shacl(data_graph)

    # The validator reads its shapes from a graph of its own, holding only what
    # BESSER writes as a shape (see ``_SHAPE_PREDICATES``), so nothing an imported
    # ontology carries beyond that is ever a shape, let alone one that runs.
    shapes_graph = _extract_shapes(data_graph)
    skipped_patterns = _drop_unsafe_patterns(shapes_graph)

    # pyshacl runs with ``advanced=False`` (its own default) as a second line of
    # defence: it gates SHACL-AF features (sh:Function/sh:expression, sh:Rule,
    # custom sh:target) that BESSER's constraint model cannot express. Implicit
    # class targeting, which BESSER's shapes DO rely on, is a separate always-on
    # code path and keeps working. ``_strip_executable_shacl`` above removes the
    # same constructs from the data graph as a third. See ``_shacl_worker``.
    results_graph = _validate(data_graph, shapes_graph, inference, timeout_seconds)

    issues = _parse_validation_report(results_graph, kg, data_graph)
    issues.extend(_skipped_pattern_issues(skipped_patterns, kg, data_graph))
    issues.sort(key=lambda i: (_SEVERITY_RANK.get(i.severity, 99), i.code, i.id))
    issues = _dedupe(issues)

    return ConsistencyReport(
        issues=issues,
        issue_count=len(issues),
        severity_counts=_count_severities(issues),
        kg_signature=kg_signature(kg),
        inference_used=inference,
    )


# ---------------------------------------------------------------------------
# Running pyshacl under a time budget
# ---------------------------------------------------------------------------


def _validate(
    data_graph: rdflib.Graph,
    shapes_graph: rdflib.Graph,
    inference: str,
    timeout_seconds: Optional[float],
) -> rdflib.Graph:
    """Run pyshacl and return its results graph, in a killable child process.

    Graphs cross the process boundary pickled, which keeps blank-node labels
    intact: the report's focus nodes and source shapes still name the nodes
    of ``data_graph``.
    """
    if timeout_seconds is None:
        from besser.BUML.notations.kg_to_buml._shacl_worker import run_validation

        return _unwrap_outcome(run_validation(data_graph, shapes_graph, inference))

    payload = pickle.dumps((data_graph, shapes_graph, inference), protocol=pickle.HIGHEST_PROTOCOL)
    try:
        completed = subprocess.run(
            [sys.executable, "-P", _WORKER_SCRIPT],
            input=payload,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        # ``run`` has already killed the child.
        raise ConsistencyCheckTimeout(
            f"The consistency check did not finish within {timeout_seconds:g} seconds and was "
            "stopped. A SHACL pattern (sh:pattern) that backtracks heavily, or a very large "
            "ontology, can cause this."
        ) from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", "replace")[-_LOGGED_STDERR_CHARS:]
        logger.error("SHACL validator process exited with %s: %s", completed.returncode, stderr)
        raise RuntimeError(f"SHACL validator process exited with status {completed.returncode}.")
    return _unwrap_outcome(pickle.loads(completed.stdout))


def _unwrap_outcome(outcome: Tuple[str, Any]) -> rdflib.Graph:
    """The results graph of a worker outcome, or the error it carries."""
    kind, value = outcome
    if kind == "shape_error":
        # pyshacl's own wording names the offending construct, e.g. "A shape defined
        # as a PropertyShape must include one `sh:path` property."
        raise ConsistencyCheckError(f"The KG's SHACL shapes could not be loaded: {value}")
    return value


# A quantifier that can repeat its atom more than once: ``*``, ``+``, ``{n,}``,
# ``{n,m}`` (m > 1) or ``{n}`` (n > 1). ``?`` cannot, so ``(https?://)?`` is fine.
_REPEAT_RE = re.compile(r"[*+]|\{(\d*)(,?)(\d*)\}")


def _repeat_at(pattern: str, index: int) -> Optional[Tuple[int, bool]]:
    """``(end, variable)`` for a repeating quantifier at ``index``, else ``None``.

    ``variable`` is False for a fixed count (``{3}``), which repeats its atom
    but cannot backtrack over how many times.
    """
    match = _REPEAT_RE.match(pattern, index)
    if match is None:
        return None
    if match.group(0) in ("*", "+"):
        return match.end(), True
    low, comma, high = match.groups()
    if not low and not high:
        return None                         # "{}" / "{,}": literal braces
    if not comma:
        return (match.end(), False) if int(low) > 1 else None
    if not high:
        return match.end(), True
    if int(high) <= 1:
        return None
    return match.end(), int(high) > int(low or 0)


def _has_nested_quantifier(pattern: str) -> bool:
    """True when a repeated group contains a variable repetition: ``(a+)+``, ``(.*a){9}``.

    The classic shape of catastrophic backtracking ("star height" above one, the
    rule tools such as safe-regex apply). A heuristic in both directions: it
    also flags unambiguous patterns such as ``([a-z]+\\.)+``, and it misses
    overlapping alternations such as ``(a|aa)+``, which the validator's time
    budget still bounds.
    """
    group_varies: List[bool] = []   # per open group: does it contain a variable repeat?
    index = 0
    in_class = False
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            index += 2
            continue
        if in_class:
            in_class = char != "]"
            index += 1
            continue
        if char == "[":
            # A "]" first in the class (after an optional "^") is a member, not the end.
            index += 1
            if pattern[index:index + 1] == "^":
                index += 1
            if pattern[index:index + 1] == "]":
                index += 1
            in_class = True
            continue
        if char == "(":
            group_varies.append(False)
            # "(?:", "(?P<name>", "(?=": that "?" is syntax, not a quantifier.
            index += 2 if pattern[index + 1:index + 2] == "?" else 1
            continue
        if char == ")":
            inner = group_varies.pop() if group_varies else False
            repeat = _repeat_at(pattern, index + 1)
            if repeat is not None and inner:
                return True
            if group_varies and (inner or (repeat is not None and repeat[1])):
                group_varies[-1] = True
            index = repeat[0] if repeat is not None else index + 1
            continue
        repeat = _repeat_at(pattern, index)
        if repeat is not None:
            if group_varies and repeat[1]:
                group_varies[-1] = True
            index = repeat[0]
            continue
        index += 1
    return False


def _drop_unsafe_patterns(shapes_graph: rdflib.Graph) -> List[Tuple[Any, str, str]]:
    """Remove every ``sh:pattern`` that could make the validator backtrack for ages.

    Mutates ``shapes_graph``. A shape whose pattern goes also loses its
    ``sh:flags``, which only qualify a pattern. Returns ``(shape, pattern,
    reason)`` for each pattern removed, so it can be reported.
    """
    skipped: List[Tuple[Any, str, str]] = []
    for shape, pattern in list(shapes_graph.subject_objects(SH.pattern)):
        text = str(pattern)
        if len(text) > MAX_SHACL_PATTERN_LENGTH:
            reason = f"it is longer than {MAX_SHACL_PATTERN_LENGTH} characters"
        elif _has_nested_quantifier(text):
            reason = (
                "it repeats a group that itself contains a repetition, which can make "
                "matching take exponential time"
            )
        else:
            continue
        shapes_graph.remove((shape, SH.pattern, pattern))
        skipped.append((shape, text, reason))
    for shape, _text, _reason in skipped:
        if (shape, SH.pattern, None) not in shapes_graph:
            shapes_graph.remove((shape, SH.flags, None))
    return skipped


def _skipped_pattern_issues(
    skipped: List[Tuple[Any, str, str]],
    kg: KnowledgeGraph,
    data_graph: rdflib.Graph,
) -> List[ConsistencyIssue]:
    """One warning per pattern :func:`_drop_unsafe_patterns` removed."""
    if not skipped:
        return []
    iri_to_node_id = _build_iri_to_node_id(kg)
    issues: List[ConsistencyIssue] = []
    for shape, text, reason in skipped:
        constraint_node_id = _term_to_node_id(shape, iri_to_node_id)
        path = _shape_value(data_graph, shape, SH.path)
        path_iri = str(path) if isinstance(path, URIRef) else None
        shown = text if len(text) <= 80 else text[:77] + "..."
        issues.append(ConsistencyIssue(
            id=_hash_id("UnsafePatternSkipped", constraint_node_id or "", path_iri or "", text),
            code="UnsafePatternSkipped",
            severity="warning",
            message=f"The pattern {shown!r} was not checked: {reason}.",
            affected_node_ids=[constraint_node_id] if constraint_node_id else [],
            constraint_node_id=constraint_node_id,
            spec_kind="pattern",
            path_iri=path_iri,
        ))
    return issues


# ---------------------------------------------------------------------------
# Graph pre-processing
# ---------------------------------------------------------------------------


def _extract_shapes(data_graph: rdflib.Graph) -> rdflib.Graph:
    """The shapes graph: every BESSER-written shape in ``data_graph``, and nothing else.

    Walks from each ``sh:NodeShape`` / ``sh:PropertyShape`` and copies the triples
    whose predicate is in ``_SHAPE_PREDICATES``, following the shapes they point
    at (``sh:property``, ``sh:not``, ``sh:qualifiedValueShape``, the members of
    ``sh:and`` / ``sh:or`` / ``sh:xone``) and copying the list spines of the list
    valued ones. A shape node's class type (``rdfs:Class`` / ``owl:Class``) is
    kept too: that is how SHACL spots a shape that is also a class, whose
    instances it targets.
    """
    shapes = rdflib.Graph()
    todo: List[Any] = [
        node for shape_type in _SHAPE_TYPES for node in data_graph.subjects(RDF.type, shape_type)
    ]
    seen: Set[Any] = set()

    def copy_list(head: Any) -> List[Any]:
        members: List[Any] = []
        node = head
        while node is not None and node != RDF.nil and node not in seen:
            seen.add(node)
            first = data_graph.value(node, RDF.first)
            rest = data_graph.value(node, RDF.rest)
            if first is None or rest is None:
                break
            shapes.add((node, RDF.first, first))
            shapes.add((node, RDF.rest, rest))
            members.append(first)
            node = rest
        return members

    while todo:
        node = todo.pop()
        if node in seen:
            continue
        seen.add(node)
        for predicate, obj in data_graph.predicate_objects(node):
            if predicate == RDF.type:
                if obj in _SHAPE_TYPES or obj in (RDFS.Class, OWL.Class):
                    shapes.add((node, predicate, obj))
                continue
            if predicate not in _SHAPE_PREDICATES:
                continue
            shapes.add((node, predicate, obj))
            if predicate in _SHAPE_VALUED:
                todo.append(obj)
            elif predicate in _SHAPE_LISTS:
                todo.extend(copy_list(obj))
            elif predicate in _VALUE_LISTS:
                copy_list(obj)
    return shapes


def _strip_executable_shacl(g: rdflib.Graph) -> int:
    """Remove SHACL-AF constructs that a validator would *execute*.

    The KG under test is its own shapes graph, so any SPARQL a rule or custom
    target carries would run with the server's network access. See
    :data:`_EXECUTABLE_SHACL_PREDICATES` for why this is a security boundary
    and not a capability trade-off.

    Mutates ``g`` in place. Returns the number of triples removed, which is 0
    for every KG built through BESSER's own constraint model.
    """
    doomed: Set[Tuple[Any, Any, Any]] = set()

    # The link that reaches a SPARQL body, and the body itself.
    for predicate in _EXECUTABLE_SHACL_PREDICATES:
        for triple in g.triples((None, predicate, None)):
            doomed.add(triple)

    # Anything declared to BE one of those constructs: drop the type triple and
    # every other statement describing that subject, so no partially-stripped
    # rule is left for a later pyshacl version to pick up.
    for rdf_type in _EXECUTABLE_SHACL_TYPES:
        for subject in set(g.subjects(RDF.type, rdf_type)):
            for triple in g.triples((subject, None, None)):
                doomed.add(triple)

    for triple in doomed:
        g.remove(triple)
    return len(doomed)


def _normalize_count_literals(g: rdflib.Graph) -> None:
    """pyshacl strictly requires ``xsd:integer`` for ``sh:minCount`` /
    ``sh:maxCount`` and other count predicates, but the OWL exporter
    emits ``xsd:nonNegativeInteger`` for the matching OWL cardinality
    predicates. Rewrite in place — semantics are preserved (the value
    range is the same, only the datatype IRI differs)."""
    to_replace = []
    for s, p, o in g:
        if isinstance(o, Literal) and o.datatype == XSD.nonNegativeInteger:
            try:
                lit = Literal(int(o), datatype=XSD.integer)
            except (TypeError, ValueError):
                continue
            to_replace.append((s, p, o, lit))
    for s, p, old, new in to_replace:
        g.remove((s, p, old))
        g.add((s, p, new))


def _augment_with_owl_shim(g: rdflib.Graph) -> None:
    """Add synthetic SHACL shapes for OWL DL constructs that pyshacl +
    owlrl don't surface as validation violations on their own.

    The shapes are anonymous blank nodes whose subjects never leak
    outside the consistency check — the caller-visible RDF graph is
    untouched. Translations:

    - ``:A owl:disjointWith :B`` → ``[a sh:NodeShape ; sh:targetClass :A ;
      sh:not [sh:class :B]]`` (and symmetrically).
    - ``:A owl:complementOf :B`` → ``[a sh:NodeShape ; sh:targetClass :A ;
      sh:not [sh:class :B]]``.
    - ``:p a owl:FunctionalProperty`` → ``[a sh:NodeShape ; sh:targetSubjectsOf
      :p ; sh:property [sh:path :p ; sh:maxCount 1]]``.
    - ``owl:AllDisjointClasses`` axioms expand to all pairwise disjointWith.
    """
    seen_pairs: Set[Tuple[URIRef, URIRef]] = set()

    def _add_disjoint_shape(a: URIRef, b: URIRef) -> None:
        if (a, b) in seen_pairs:
            return
        seen_pairs.add((a, b))
        shape = BNode()
        inner = BNode()
        g.add((shape, RDF.type, SH.NodeShape))
        g.add((shape, SH.targetClass, a))
        g.add((shape, SH["not"], inner))
        g.add((inner, SH["class"], b))

    # owl:disjointWith — symmetric.
    for a, b in list(g.subject_objects(OWL.disjointWith)):
        if isinstance(a, URIRef) and isinstance(b, URIRef):
            _add_disjoint_shape(a, b)
            _add_disjoint_shape(b, a)

    # owl:complementOf.
    for a, b in list(g.subject_objects(OWL.complementOf)):
        if isinstance(a, URIRef) and isinstance(b, URIRef):
            _add_disjoint_shape(a, b)

    # owl:AllDisjointClasses — expand each membership list into pairwise.
    for axiom in list(g.subjects(RDF.type, OWL.AllDisjointClasses)):
        members_head = next(g.objects(axiom, OWL.members), None)
        if members_head is None:
            continue
        members = list(rdflib.collection.Collection(g, members_head))
        members = [m for m in members if isinstance(m, URIRef)]
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                _add_disjoint_shape(a, b)
                _add_disjoint_shape(b, a)

    # owl:FunctionalProperty → per-subject max-1 constraint.
    for prop in list(g.subjects(RDF.type, OWL.FunctionalProperty)):
        if not isinstance(prop, URIRef):
            continue
        shape = BNode()
        ps = BNode()
        g.add((shape, RDF.type, SH.NodeShape))
        g.add((shape, SH.targetSubjectsOf, prop))
        g.add((shape, SH.property, ps))
        g.add((ps, SH.path, prop))
        g.add((ps, SH.maxCount, Literal(1, datatype=XSD.integer)))

    # Our SHACL exporter marks inline anonymous shapes under sh:not / sh:and /
    # sh:or / sh:xone as `rdf:type sh:PropertyShape`, but pyshacl strictly
    # requires every PropertyShape to carry an `sh:path`. Inline shapes
    # nested under a logical operator inherit their path from the outer
    # shape and don't repeat it — strip the explicit type so pyshacl's
    # structural inference treats them as the implicit NodeShape they are.
    bnodes_without_path = [
        bn for bn in g.subjects(RDF.type, SH.PropertyShape)
        if isinstance(bn, BNode) and not any(g.objects(bn, SH.path))
    ]
    for bn in bnodes_without_path:
        g.remove((bn, RDF.type, SH.PropertyShape))


# ---------------------------------------------------------------------------
# Validation-report parsing
# ---------------------------------------------------------------------------


_SEVERITY_MAP = {
    f"{SH}Violation": "violation",
    f"{SH}Warning": "warning",
    f"{SH}Info": "info",
}


def _parse_validation_report(
    results_graph: rdflib.Graph,
    kg: KnowledgeGraph,
    data_graph: rdflib.Graph,
) -> List[ConsistencyIssue]:
    iri_to_node_id = _build_iri_to_node_id(kg)

    issues: List[ConsistencyIssue] = []
    for result in results_graph.subjects(RDF.type, SH.ValidationResult):
        focus = next(results_graph.objects(result, SH.focusNode), None)
        path = next(results_graph.objects(result, SH.resultPath), None)
        message = next(results_graph.objects(result, SH.resultMessage), None)
        severity_iri = next(results_graph.objects(result, SH.resultSeverity), None)
        source_shape = next(results_graph.objects(result, SH.sourceShape), None)
        component = next(results_graph.objects(result, SH.sourceConstraintComponent), None)
        value = next(results_graph.objects(result, SH.value), None)

        focus_id = _term_to_node_id(focus, iri_to_node_id)
        affected = [focus_id] if focus_id else []
        value_id = _term_to_node_id(value, iri_to_node_id)
        if value_id and value_id not in affected:
            affected.append(value_id)

        component_iri = str(component) if component else ""
        component_local = component_iri.split("#")[-1] if component_iri else "Unknown"

        spec_kind = _COMPONENT_TO_KIND.get(component_iri)
        severity = _SEVERITY_MAP.get(str(severity_iri), "violation") if severity_iri else "violation"

        constraint_node_id = _term_to_node_id(source_shape, iri_to_node_id)

        path_iri: Optional[str] = None
        if isinstance(path, URIRef):
            path_iri = str(path)

        msg_text = str(message) if message else _synthesize_message(focus, component_local, path_iri)
        constraint_label = _constraint_label_for(source_shape, path_iri, spec_kind, component_local, data_graph)

        # `constraint_label` stands in for `source_shape`, which must not be
        # hashed: the OWL shim mints fresh BNodes for its shapes, so their
        # labels are random per run and the ids changed on every call.
        #
        # Replacing it rather than merely dropping it is the point. The label
        # is built from the shape's own SHACL triples, so it is stable across
        # runs *and* still tells two shapes apart -- and something has to.
        # Both halves of an owl:disjointWith breach share one focus node, one
        # component and no path, so with nothing shape-derived in the identity
        # they collapse onto one id and `_dedupe` drops a real finding.
        issue_id = _hash_id(
            focus, constraint_node_id or "", path_iri or "",
            component, value, constraint_label or "",
        )

        issues.append(
            ConsistencyIssue(
                id=issue_id,
                code=component_local,
                severity=severity,
                message=msg_text,
                affected_node_ids=affected,
                affected_edge_ids=[],
                constraint_node_id=constraint_node_id,
                spec_kind=spec_kind,
                path_iri=path_iri,
                constraint_label=constraint_label,
            )
        )
    # By rank, not alphabetically: sorting the severity strings put them in
    # info -> violation -> warning order, burying the most severe findings in
    # the middle of the list the UI renders top-down.
    issues.sort(key=lambda i: (_SEVERITY_RANK.get(i.severity, 99), i.code, i.id))
    return issues



def _shape_value(g: rdflib.Graph, shape: Any, predicate: URIRef) -> Optional[Any]:
    """First object of (shape, predicate, *) in the graph, or None."""
    if shape is None:
        return None
    return next(g.objects(shape, predicate), None)


def _format_literal_or_iri(term: Any) -> str:
    """Render a SHACL constraint value for display.

    Blank nodes become a fixed placeholder rather than ``str(term)``. Their
    label is minted per run, so letting it through made the label a different
    string on every call -- and :func:`_parse_validation_report` hashes the
    label into the issue id, so an unstable label means an unstable id. A raw
    ``Nb3f1a...`` told a reader nothing anyway. Reachable through the rare
    ``sh:in``-of-blank-nodes and blank ``sh:hasValue`` shapes an imported
    ontology may carry.
    """
    if isinstance(term, URIRef):
        return _local_name(str(term))
    if isinstance(term, Literal):
        return f'"{term}"'
    if isinstance(term, BNode):
        return "[anonymous]"
    return str(term)


def _constraint_label_for(
    source_shape: Any,
    path_iri: Optional[str],
    spec_kind: Optional[str],
    component_local: str,
    data_graph: rdflib.Graph,
) -> Optional[str]:
    """Compose a short human-readable label for the violated constraint.

    Reads the source shape's SHACL triples (counts, datatypes, value classes,
    …) and combines them with the property path (when present) so users can
    grok the rule at a glance without parsing pyshacl's verbose message.

    Returns ``None`` only when there is nothing useful to say beyond the
    constraint-component name (which the row already shows as a chip).
    """
    if source_shape is None:
        return None
    kind = spec_kind or component_local

    # Pull the most relevant value(s) for each kind.
    parts: List[str] = []
    if kind in ("minCardinality", "MinCountConstraintComponent"):
        n = _shape_value(data_graph, source_shape, SH.minCount)
        parts.append(f"at least {n}" if n is not None else "at least N")
    elif kind in ("maxCardinality", "MaxCountConstraintComponent"):
        n = _shape_value(data_graph, source_shape, SH.maxCount)
        parts.append(f"at most {n}" if n is not None else "at most N")
    elif kind in ("minQualifiedCardinality", "QualifiedMinCountConstraintComponent"):
        n = _shape_value(data_graph, source_shape, SH.qualifiedMinCount)
        qv = _shape_value(data_graph, source_shape, SH.qualifiedValueShape)
        cls = _shape_value(data_graph, qv, SH["class"]) if qv is not None else None
        max_qv = _shape_value(data_graph, source_shape, SH.qualifiedMaxCount)
        if n is not None and max_qv is not None and str(n) == str(max_qv):
            parts.append(f"exactly {n} of {_format_literal_or_iri(cls) if cls is not None else 'class'}")
        elif n is not None:
            parts.append(f"at least {n} of {_format_literal_or_iri(cls) if cls is not None else 'class'}")
    elif kind in ("maxQualifiedCardinality", "QualifiedMaxCountConstraintComponent"):
        n = _shape_value(data_graph, source_shape, SH.qualifiedMaxCount)
        qv = _shape_value(data_graph, source_shape, SH.qualifiedValueShape)
        cls = _shape_value(data_graph, qv, SH["class"]) if qv is not None else None
        parts.append(f"at most {n} of {_format_literal_or_iri(cls) if cls is not None else 'class'}")
    elif kind in ("datatype", "DatatypeConstraintComponent"):
        d = _shape_value(data_graph, source_shape, SH.datatype)
        parts.append(f"datatype {_format_literal_or_iri(d)}" if d is not None else "datatype")
    elif kind in ("pattern", "PatternConstraintComponent"):
        p = _shape_value(data_graph, source_shape, SH.pattern)
        parts.append(f"matches /{p}/" if p is not None else "matches regex")
    elif kind in ("minLength", "MinLengthConstraintComponent"):
        n = _shape_value(data_graph, source_shape, SH.minLength)
        parts.append(f"length ≥ {n}" if n is not None else "minimum length")
    elif kind in ("maxLength", "MaxLengthConstraintComponent"):
        n = _shape_value(data_graph, source_shape, SH.maxLength)
        parts.append(f"length ≤ {n}" if n is not None else "maximum length")
    elif kind in ("minInclusive", "MinInclusiveConstraintComponent"):
        n = _shape_value(data_graph, source_shape, SH.minInclusive)
        parts.append(f"≥ {n}" if n is not None else "minimum inclusive")
    elif kind in ("maxInclusive", "MaxInclusiveConstraintComponent"):
        n = _shape_value(data_graph, source_shape, SH.maxInclusive)
        parts.append(f"≤ {n}" if n is not None else "maximum inclusive")
    elif kind in ("minExclusive", "MinExclusiveConstraintComponent"):
        n = _shape_value(data_graph, source_shape, SH.minExclusive)
        parts.append(f"> {n}" if n is not None else "exclusive minimum")
    elif kind in ("maxExclusive", "MaxExclusiveConstraintComponent"):
        n = _shape_value(data_graph, source_shape, SH.maxExclusive)
        parts.append(f"< {n}" if n is not None else "exclusive maximum")
    elif kind in ("in", "InConstraintComponent"):
        head = _shape_value(data_graph, source_shape, SH["in"])
        if head is not None:
            items = list(rdflib.collection.Collection(data_graph, head))
            parts.append(f"in {{{', '.join(_format_literal_or_iri(i) for i in items)}}}")
        else:
            parts.append("in set")
    elif kind in ("hasValue", "HasValueConstraintComponent"):
        v = _shape_value(data_graph, source_shape, SH.hasValue)
        parts.append(f"must equal {_format_literal_or_iri(v)}" if v is not None else "must equal a value")
    elif kind in ("nodeKind", "NodeKindConstraintComponent"):
        v = _shape_value(data_graph, source_shape, SH.nodeKind)
        parts.append(f"node kind {_format_literal_or_iri(v)}" if v is not None else "node kind")
    elif kind in ("someValuesFrom", "ClassConstraintComponent"):
        v = _shape_value(data_graph, source_shape, SH["class"])
        parts.append(f"of class {_format_literal_or_iri(v)}" if v is not None else "of class")
    elif kind in ("languageIn", "LanguageInConstraintComponent"):
        head = _shape_value(data_graph, source_shape, SH.languageIn)
        if head is not None:
            tags = list(rdflib.collection.Collection(data_graph, head))
            parts.append(f"language in {{{', '.join(str(t) for t in tags)}}}")
    elif kind in ("uniqueLang", "UniqueLangConstraintComponent"):
        parts.append("each language at most once")
    elif kind in ("shaclNot", "NotConstraintComponent"):
        # The negated inner shape commonly carries a sh:class — surface it.
        inner = _shape_value(data_graph, source_shape, SH["not"])
        inner_cls = _shape_value(data_graph, inner, SH["class"]) if inner is not None else None
        if inner_cls is not None:
            parts.append(f"must NOT be of class {_format_literal_or_iri(inner_cls)}")
        else:
            parts.append("must NOT satisfy nested shape")
    elif kind in ("shaclAnd", "AndConstraintComponent"):
        parts.append("must satisfy ALL nested shapes")
    elif kind in ("shaclOr", "OrConstraintComponent"):
        parts.append("must satisfy ANY nested shape")
    elif kind in ("shaclXone", "XoneConstraintComponent"):
        parts.append("must satisfy EXACTLY ONE nested shape")
    elif kind in ("shaclDisjoint", "DisjointConstraintComponent"):
        other = _shape_value(data_graph, source_shape, SH.disjoint)
        parts.append(f"disjoint from {_format_literal_or_iri(other)}" if other is not None else "disjoint")
    elif kind in ("shaclClosed", "ClosedConstraintComponent"):
        parts.append("closed shape (no extra properties)")
    else:
        # Fallback — use the component local name as-is.
        parts.append(component_local.replace("ConstraintComponent", ""))

    body = ", ".join(parts)
    if path_iri:
        return f"{_local_name(path_iri)}: {body}"
    return body


def _build_iri_to_node_id(kg: KnowledgeGraph) -> Dict[str, str]:
    """Map every rdflib-term-as-string to the KG node id that produced it.

    The exporter at :func:`knowledge_graph_to_rdf` uses three strategies in
    ``_term_for_node`` for non-literal nodes:

    1. ``node.iri`` directly when set;
    2. an explicit :class:`BNode` for KGBlank, KGNodeConstraint, and
       KGPropertyConstraint without an iri (KGBlank uses ``node.id``
       without the ``_:`` prefix as the BNode handle);
    3. otherwise (KGIndividual, KGClass, KGProperty without an iri) a
       synthesised IRI ``<DEFAULT_NAMESPACE><slug(label or id)>``.

    We replicate that logic here so the consistency report can map every
    focus-node term pyshacl emits back to a KG node id — without this,
    UI-created individuals (which have no explicit iri) come back with
    empty ``affected_node_ids`` and the "Fix in KG" button never appears.
    """
    mapping: Dict[str, str] = {}
    for node in kg.nodes:
        nid = node.id      # always non-empty: KGNode.id's setter rejects "" / None
        # Identity mapping always.
        mapping[nid] = nid
        iri = node.iri
        if iri:
            mapping[iri] = nid
            continue
        # Mirror the exporter's BNode-vs-synthetic-URI choice.
        if isinstance(node, KGBlank):
            bn_id = nid[2:] if nid.startswith("_:") else nid
            if bn_id:
                mapping[f"_:{bn_id}"] = nid
        elif isinstance(node, (KGNodeConstraint, KGPropertyConstraint)):
            # The exporter mints a deterministic BNode from
            # `_slugify(node.id)` for constraint nodes without an iri so
            # pyshacl's `sh:sourceShape` can be traced back here.
            slug = _slugify(nid)
            if slug:
                mapping[f"_:{slug}"] = nid
        elif isinstance(node, KGLiteral):
            # Literals can't be a focus node anyway.
            continue
        else:
            # KGIndividual / KGClass / KGProperty without an iri.
            slug = _slugify(node.label or nid)
            mapping[f"{DEFAULT_NAMESPACE}{slug}"] = nid
    return mapping


def _term_to_node_id(
    term: Optional[Any], iri_to_node_id: Dict[str, str]
) -> Optional[str]:
    if term is None:
        return None
    if isinstance(term, URIRef):
        return iri_to_node_id.get(str(term))
    if isinstance(term, BNode):
        # The exporter materialises stable BNode handles for KGBlank /
        # KGNodeConstraint / KGPropertyConstraint nodes (see
        # _build_iri_to_node_id). Consult the mapping first.
        #
        # An unresolved handle used to be returned as "_:<label>". Those labels
        # belong to blank nodes rdflib minted during validation (the OWL shim
        # builds its shapes from fresh BNodes), so the value was a different
        # random string on every call and named nothing the frontend could look
        # up. None says "no constraint node" honestly.
        return iri_to_node_id.get(f"_:{str(term)}")
    if isinstance(term, Literal):
        return None
    return iri_to_node_id.get(str(term))


def _synthesize_message(focus: Any, component_local: str, path_iri: Optional[str]) -> str:
    focus_label = str(focus) if focus is not None else "<unknown>"
    if path_iri:
        return f"{focus_label} violates {component_local} on <{path_iri}>"
    return f"{focus_label} violates {component_local}"


def _hash_id(*terms: Any) -> str:
    # md5, not sha1: these ids already ship to the editor, so the digest must not change.
    return short_digest("".join(f"{t}\x00" for t in terms), 16, algo="md5")


def _dedupe(issues: List[ConsistencyIssue]) -> List[ConsistencyIssue]:
    """Drop issues that merely repeat one already reported.

    ``constraint_label`` is part of the key because it is often the only thing
    separating two genuine findings: the two directions of an
    ``owl:disjointWith`` breach agree on code, constraint node and affected
    nodes, and differ only in which class the individual must not be.
    """
    seen: Set[Tuple[str, str, str, Tuple[str, ...]]] = set()
    out: List[ConsistencyIssue] = []
    for i in issues:
        key = (
            i.code,
            i.constraint_node_id or "",
            i.constraint_label or "",
            tuple(i.affected_node_ids),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(i)
    return out


def _count_severities(issues: List[ConsistencyIssue]) -> Dict[str, int]:
    counts = {"info": 0, "warning": 0, "violation": 0}
    for i in issues:
        if i.severity in counts:
            counts[i.severity] += 1
    return counts
