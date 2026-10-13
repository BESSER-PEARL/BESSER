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

from typing import Any, Dict, List, Literal


__all__ = [
    "Vocab",
    "CONSTRAINT_VOCAB_MAP",
    "LOGICAL_KINDS",
    "is_valid_spec",
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
    # sh:class also decodes to this kind (see owl_to_buml._SHACL_PROPERTY_COMPONENTS).
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
