"""Normalised constraint specifications for KG NodeConstraint / PropertyConstraint.

A ``ConstraintSpec`` captures a single OWL2/SHACL constraint in a vocabulary-
agnostic form. Specs live in ``KGNodeConstraint.metadata['constraintSpecs']``
or ``KGPropertyConstraint.metadata['constraintSpecs']`` as plain dicts (so the
existing JSON round-trip plumbing keeps working).

The :data:`CONSTRAINT_VOCAB_MAP` table is the single source of truth that
the importer, exporter, and validation layers consult to decide which
vocabularies a given kind can be serialised in. The frontend mirrors this
table in ``constraint-catalog.ts``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional


__all__ = [
    "ConstraintKind",
    "Vocab",
    "ConstraintSpec",
    "CONSTRAINT_VOCAB_MAP",
    "CONSTRAINT_CATEGORY",
    "is_valid_spec",
]


Vocab = Literal["owl", "shacl"]


ConstraintKind = Literal[
    # NodeConstraint kinds
    "equivalentClasses",
    "disjointWith",
    "subClassOf",
    "complementOf",
    "unionOf",
    "intersectionOf",
    "oneOf",
    "hasKey",
    "disjointUnionOf",
    "shaclClosed",
    "shaclIgnoredProperties",
    "shaclDisjoint",
    # PropertyConstraint cardinality
    "minCardinality",
    "maxCardinality",
    "exactCardinality",
    "minQualifiedCardinality",
    "maxQualifiedCardinality",
    "exactQualifiedCardinality",
    # PropertyConstraint value
    "someValuesFrom",
    "allValuesFrom",
    "hasValue",
    "hasSelf",
    # PropertyConstraint datatype / literal (SHACL Core)
    "datatype",
    "nodeKind",
    "pattern",
    "flags",
    "minLength",
    "maxLength",
    "minInclusive",
    "maxInclusive",
    "minExclusive",
    "maxExclusive",
    "languageIn",
    "uniqueLang",
    "in",
    # Logical
    "shaclNot",
    "shaclAnd",
    "shaclOr",
    "shaclXone",
    # SHACL meta
    "shaclSeverity",
    "shaclMessage",
    "shaclName",
    "shaclDescription",
    "shaclDeactivated",
    "shaclOrder",
    "shaclGroup",
]


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
    "someValuesFrom": ["owl", "shacl"],
    "allValuesFrom": ["owl"],
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


# UI category used to group specs in the inspector picker. The frontend
# mirrors this layout. ``classAxiom`` and ``shapeMeta`` are NodeConstraint-only;
# ``cardinality``, ``value``, ``datatype`` are PropertyConstraint-only;
# ``enumeration`` and ``logical`` and ``meta`` apply to both.
CONSTRAINT_CATEGORY: Dict[str, str] = {
    # Class axioms
    "equivalentClasses": "classAxiom",
    "disjointWith": "classAxiom",
    "subClassOf": "classAxiom",
    "complementOf": "classAxiom",
    "unionOf": "classAxiom",
    "intersectionOf": "classAxiom",
    "hasKey": "classAxiom",
    "disjointUnionOf": "classAxiom",
    "shaclClosed": "classAxiom",
    "shaclIgnoredProperties": "classAxiom",
    "shaclDisjoint": "classAxiom",
    # Cardinality
    "minCardinality": "cardinality",
    "maxCardinality": "cardinality",
    "exactCardinality": "cardinality",
    "minQualifiedCardinality": "cardinality",
    "maxQualifiedCardinality": "cardinality",
    "exactQualifiedCardinality": "cardinality",
    # Value
    "someValuesFrom": "value",
    "allValuesFrom": "value",
    "hasValue": "value",
    "hasSelf": "value",
    "nodeKind": "value",
    # Datatype
    "datatype": "datatype",
    "pattern": "datatype",
    "flags": "datatype",
    "minLength": "datatype",
    "maxLength": "datatype",
    "minInclusive": "datatype",
    "maxInclusive": "datatype",
    "minExclusive": "datatype",
    "maxExclusive": "datatype",
    "languageIn": "datatype",
    "uniqueLang": "datatype",
    # Enumeration
    "oneOf": "enumeration",
    "in": "enumeration",
    # Logical
    "shaclNot": "logical",
    "shaclAnd": "logical",
    "shaclOr": "logical",
    "shaclXone": "logical",
    # Meta
    "shaclSeverity": "meta",
    "shaclMessage": "meta",
    "shaclName": "meta",
    "shaclDescription": "meta",
    "shaclDeactivated": "meta",
    "shaclOrder": "meta",
    "shaclGroup": "meta",
}


@dataclass
class ConstraintSpec:
    """A single normalised constraint instance.

    ``value`` shape depends on ``kind``:

    - integer kinds (``minCardinality``, ``maxLength`` …): ``int``
    - IRI kinds (``datatype``, ``hasValue`` IRIs): ``str``
    - list kinds (``oneOf``, ``in``, ``unionOf`` …): ``List[Any]``
    - nested kinds (``shaclAnd``, ``shaclOr`` …): ``List[Dict]`` of nested specs
    - ``hasValue`` may also carry ``{"value": "x", "datatype": "...", "language": "en"}``
      for literal values

    ``on_class`` is populated for qualified cardinality (``owl:onClass`` /
    ``sh:qualifiedValueShape``) and otherwise left ``None``.

    ``vocab_pref`` is an explicit per-spec override. When ``None``, the
    exporter picks any vocabulary listed in :data:`CONSTRAINT_VOCAB_MAP`
    that's compatible with the user's export-time choice.
    """

    kind: str
    value: Any = None
    on_class: Optional[str] = None
    vocab_pref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None or k == "value"}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstraintSpec":
        return cls(
            kind=data["kind"],
            value=data.get("value"),
            on_class=data.get("on_class"),
            vocab_pref=data.get("vocab_pref"),
        )


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
