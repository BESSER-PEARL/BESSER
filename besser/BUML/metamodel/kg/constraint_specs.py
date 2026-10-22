"""Normalised constraint specifications for KG NodeConstraint / PropertyConstraint.

A constraint spec captures a single OWL2/SHACL constraint in a vocabulary-
agnostic form. Specs live in ``KGNodeConstraint.metadata['constraintSpecs']``
or ``KGPropertyConstraint.metadata['constraintSpecs']`` as plain dicts (so the
existing JSON round-trip plumbing keeps working); :func:`is_valid_spec` is the
structural check. There is deliberately no spec class: nothing needs one, and a
dataclass next to the dicts would be a second schema to keep in step.

A spec dict has these keys:

``kind``
    The discriminator; a key of :data:`CONSTRAINT_VOCAB_MAP`.
``value``
    Its shape depends on ``kind``:

    - integer kinds (``minCardinality``, ``maxLength`` ...): ``int``
    - IRI kinds (``datatype``, ``hasValue`` IRIs): ``str``
    - list kinds (``oneOf``, ``in``, ``unionOf`` ...): ``List[Any]``
    - nested kinds (``shaclAnd``, ``shaclOr`` ...): ``List[Dict]`` of nested specs
    - ``hasValue`` may also carry ``{"value": "x", "datatype": "...", "language": "en"}``
      for literal values
``on_class``
    Populated for qualified cardinality (``owl:onClass`` /
    ``sh:qualifiedValueShape``); otherwise absent.
``vocab_pref``
    An explicit per-spec vocabulary override. When absent, the exporter picks
    any vocabulary listed in :data:`CONSTRAINT_VOCAB_MAP` that is compatible
    with the user's export-time choice.

The :data:`CONSTRAINT_VOCAB_MAP` table is the single source of truth that
the importer, exporter, and validation layers consult to decide which
vocabularies a given kind can be serialised in. The frontend mirrors this
table in ``constraint-catalog.ts``.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Literal

from besser.BUML.metamodel.kg.constants import OWL, SH


__all__ = [
    "Vocab",
    "CONSTRAINT_VOCAB_MAP",
    "LOGICAL_KINDS",
    "OWL_RESTRICTION_PREDICATES",
    "SHACL_PROPERTY_COMPONENTS",
    "SHACL_NODE_COMPONENTS",
    "is_valid_spec",
    "spec_folded_predicates",
]


Vocab = Literal["owl", "shacl"]


# Which vocabularies (OWL, SHACL, or both) can express each constraint kind.
# Used by the exporter to skip specs whose vocab the user didn't select, and
# by the inspector to show the OWL/SHACL/OWL+SHACL chip on each spec row.
CONSTRAINT_VOCAB_MAP: Dict[str, List[str]] = {
    # NodeConstraint kinds
    "equivalentClasses": ["owl"],
    "disjointWith": ["owl", "shacl"],
    "subClassOf": ["owl"],
    "complementOf": ["owl", "shacl"],
    "unionOf": ["owl", "shacl"],
    "intersectionOf": ["owl", "shacl"],
    "oneOf": ["owl", "shacl"],
    "hasKey": ["owl"],
    "disjointUnionOf": ["owl"],
    "shaclClosed": ["shacl"],
    "shaclIgnoredProperties": ["shacl"],
    "shaclDisjoint": ["shacl"],
    # Cardinality
    "minCardinality": ["owl", "shacl"],
    "maxCardinality": ["owl", "shacl"],
    "exactCardinality": ["owl"],
    "minQualifiedCardinality": ["owl", "shacl"],
    "maxQualifiedCardinality": ["owl", "shacl"],
    "exactQualifiedCardinality": ["owl"],
    # Value
    # SHACL has no faithful existential-quantification predicate — sh:class
    # decodes to allValuesFrom (forall), not someValuesFrom (exists) — so
    # someValuesFrom is owl-only.
    "someValuesFrom": ["owl"],
    # sh:class also decodes to this kind (see SHACL_PROPERTY_COMPONENTS).
    "allValuesFrom": ["owl", "shacl"],
    "hasValue": ["owl", "shacl"],
    "hasSelf": ["owl"],
    # Datatype / literal (SHACL Core)
    "datatype": ["shacl"],
    "nodeKind": ["shacl"],
    "pattern": ["shacl"],
    "flags": ["shacl"],
    "minLength": ["shacl"],
    "maxLength": ["shacl"],
    "minInclusive": ["shacl"],
    "maxInclusive": ["shacl"],
    "minExclusive": ["shacl"],
    "maxExclusive": ["shacl"],
    "languageIn": ["shacl"],
    "uniqueLang": ["shacl"],
    "in": ["owl", "shacl"],
    "equals": ["shacl"],
    "lessThan": ["shacl"],
    "lessThanOrEquals": ["shacl"],
    # Logical — SHACL-only on export. OWL has class-level counterparts
    # (owl:complementOf / unionOf / intersectionOf) but they operate over
    # classes rather than arbitrary SHACL shapes; the class-level variants
    # live under the "classAxiom" entries instead.
    "shaclNot": ["shacl"],
    "shaclAnd": ["shacl"],
    "shaclOr": ["shacl"],
    "shaclXone": ["shacl"],
    # Meta (SHACL)
    "shaclSeverity": ["shacl"],
    "shaclMessage": ["shacl"],
    "shaclName": ["shacl"],
    "shaclDescription": ["shacl"],
    "shaclDeactivated": ["shacl"],
    "shaclOrder": ["shacl"],
    "shaclGroup": ["shacl"],
}


#: The four SHACL Core logical-operator kinds. Their ``value`` is a list of
#: :data:`KG_NESTED_SHAPE` dicts (each either ``{"ref": <node-id>}`` or
#: ``{"specs": [<inline constraint specs>]}``); ``shaclNot`` carries exactly
#: one entry, the others carry zero or more.
LOGICAL_KINDS: set = {"shaclNot", "shaclAnd", "shaclOr", "shaclXone"}


def _is_valid_nested_shape(obj: Any) -> bool:
    """A nested-shape slot is either a reference to an existing constraint
    node (``{"ref": str}``) or an inline anonymous shape that carries its
    own list of constraint specs (``{"specs": [<spec>, ...]}``). The two
    forms are mutually exclusive — a slot with both ``ref`` and ``specs``
    fails validation.
    """
    if not isinstance(obj, dict):
        return False
    has_ref = "ref" in obj
    has_specs = "specs" in obj
    if has_ref and has_specs:
        return False
    if has_ref:
        return isinstance(obj["ref"], str) and bool(obj["ref"])
    if has_specs:
        specs = obj["specs"]
        if not isinstance(specs, list):
            return False
        return all(is_valid_spec(s) for s in specs)
    return False


def is_valid_spec(spec: Any) -> bool:
    """Cheap structural validation used by KGNodeConstraint / KGPropertyConstraint setters.

    Logical-operator specs (``shaclNot``/``shaclAnd``/``shaclOr``/``shaclXone``)
    additionally require their ``value`` to be a list of valid nested shapes;
    ``shaclNot`` is further pinned to a single entry. Other kinds are accepted
    on the basis of the ``kind`` discriminator alone — value-shape checks are
    the UI's responsibility (catalog-driven) and the exporter's responsibility
    (skip unknown shapes safely).
    """
    if not isinstance(spec, dict):
        return False
    kind = spec.get("kind")
    if not isinstance(kind, str):
        return False
    if kind not in CONSTRAINT_VOCAB_MAP:
        return False
    if kind in LOGICAL_KINDS:
        value = spec.get("value")
        if not isinstance(value, list):
            return False
        if not all(_is_valid_nested_shape(v) for v in value):
            return False
        if kind == "shaclNot" and len(value) != 1:
            return False
    return True


# ---------------------------------------------------------------------------
# Which RDF predicates a spec holds
# ---------------------------------------------------------------------------

#: OWL restriction spec kind -> the predicate carrying its value.
OWL_RESTRICTION_PREDICATES: Dict[str, str] = {
    "minCardinality": f"{OWL}minCardinality",
    "maxCardinality": f"{OWL}maxCardinality",
    "exactCardinality": f"{OWL}cardinality",
    "minQualifiedCardinality": f"{OWL}minQualifiedCardinality",
    "maxQualifiedCardinality": f"{OWL}maxQualifiedCardinality",
    "exactQualifiedCardinality": f"{OWL}qualifiedCardinality",
    "someValuesFrom": f"{OWL}someValuesFrom",
    "allValuesFrom": f"{OWL}allValuesFrom",
    "hasValue": f"{OWL}hasValue",
    "hasSelf": f"{OWL}hasSelf",
}

#: SHACL constraint component a shape of either kind can carry -> its spec kind.
#: The SHACL qualified counts are not here: they stay raw edges.
SHACL_PROPERTY_COMPONENTS: Dict[str, str] = {
    f"{SH}minCount": "minCardinality",
    f"{SH}maxCount": "maxCardinality",
    f"{SH}minInclusive": "minInclusive",
    f"{SH}maxInclusive": "maxInclusive",
    f"{SH}minExclusive": "minExclusive",
    f"{SH}maxExclusive": "maxExclusive",
    f"{SH}minLength": "minLength",
    f"{SH}maxLength": "maxLength",
    f"{SH}pattern": "pattern",
    f"{SH}flags": "flags",
    f"{SH}datatype": "datatype",
    f"{SH}nodeKind": "nodeKind",
    f"{SH}hasValue": "hasValue",
    f"{SH}uniqueLang": "uniqueLang",
    f"{SH}languageIn": "languageIn",
    # sh:class means "every value must be of this type" (forall), matching the
    # allValuesFrom kind's OCL translation — NOT someValuesFrom (exists).
    f"{SH}class": "allValuesFrom",
    f"{SH}in": "in",
    f"{SH}equals": "equals",
    f"{SH}lessThan": "lessThan",
    f"{SH}lessThanOrEquals": "lessThanOrEquals",
    f"{SH}severity": "shaclSeverity",
    f"{SH}message": "shaclMessage",
    f"{SH}name": "shaclName",
    f"{SH}description": "shaclDescription",
    f"{SH}deactivated": "shaclDeactivated",
    f"{SH}order": "shaclOrder",
    f"{SH}group": "shaclGroup",
}

#: SHACL component only a node shape carries (closure, plus the shape-level
#: meta it shares with property shapes) -> its spec kind.
SHACL_NODE_COMPONENTS: Dict[str, str] = {
    f"{SH}closed": "shaclClosed",
    f"{SH}ignoredProperties": "shaclIgnoredProperties",
    f"{SH}disjoint": "shaclDisjoint",
    f"{SH}severity": "shaclSeverity",
    f"{SH}message": "shaclMessage",
    f"{SH}name": "shaclName",
    f"{SH}description": "shaclDescription",
    f"{SH}deactivated": "shaclDeactivated",
}

_OWL_RESTRICTION_FOLDED: FrozenSet[str] = frozenset(OWL_RESTRICTION_PREDICATES.values()) | {f"{OWL}onClass"}
_OWL_EXPRESSION_FOLDED: FrozenSet[str] = frozenset(
    f"{OWL}{name}" for name in ("complementOf", "unionOf", "intersectionOf", "oneOf")
)
# The property components are read on every shape, the node components on node
# shapes only; sh:not becomes a shaclNot spec.
_SHACL_PROPERTY_SHAPE_FOLDED: FrozenSet[str] = frozenset(SHACL_PROPERTY_COMPONENTS) | {f"{SH}not"}
_SHACL_NODE_SHAPE_FOLDED: FrozenSet[str] = _SHACL_PROPERTY_SHAPE_FOLDED | frozenset(SHACL_NODE_COMPONENTS)


def spec_folded_predicates(node: Any) -> FrozenSet[str]:
    """The predicates whose value ``node``'s ``constraintSpecs`` hold.

    For an OWL restriction: its cardinality or filler and the qualified
    ``owl:onClass``; for a class expression: its operator (``owl:unionOf``,
    ``owl:intersectionOf``, ``owl:oneOf``, ``owl:complementOf``); for a SHACL
    shape: every component with a spec kind (``sh:qualifiedValueShape`` and the
    qualified counts are not folded and stay raw edges). The spec is what the
    editor edits, so it is the only stored copy: the importer keeps no raw edge
    (and flattens no list) with these predicates, and the exporter and the
    projection write them from the spec, ignoring any such edge drawn by hand.
    Empty for any node that is not a constraint node.
    """
    # Imported here: knowledge_graph imports this module (is_valid_spec).
    from besser.BUML.metamodel.kg.knowledge_graph import KGNodeConstraint, KGPropertyConstraint

    shacl = node.metadata.get("source") == "shacl"
    if isinstance(node, KGPropertyConstraint):
        return _SHACL_PROPERTY_SHAPE_FOLDED if shacl else _OWL_RESTRICTION_FOLDED
    if isinstance(node, KGNodeConstraint):
        return _SHACL_NODE_SHAPE_FOLDED if shacl else _OWL_EXPRESSION_FOLDED
    return frozenset()
