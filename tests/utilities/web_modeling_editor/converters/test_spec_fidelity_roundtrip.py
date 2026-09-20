"""Everything a class diagram carries must reach the Spec-Driven Agent.

A weak spec is allowed: "if the model have issue with the spec bad from the
modeling agent but the spec agent should fix it like I use you claude code".
A LOSSY one is not. The difference is that a vague spec is visibly vague,
while a dropped key is invisible - an absent key reads to the downstream
agent as "not shown", never as "there is none". It then invents a value and
is confident about it.

That has already cost two runs. Method return types were dropped because the
editor's newer JSON carries the type in its own ``attributeType`` property
(fixed in ab03f9d7, pinned by ``test_method_return_type.py``), and the same
omission on the PARAMETER list made the agent demand an argument for a
zero-argument action in 16 of 21 runs of one evaluation case.

So this file walks ONE hand-written diagram - read by hand, every expectation
below written from the JSON above and not from the code's output - through
every hop the real pipeline uses:

    editor / modeling-agent JSON
      -> process_class_diagram        (json_to_buml)
      -> DomainModel
      -> serialize_domain_model       (what the agent's system prompt embeds)

and asserts, feature by feature, that what went in comes out. The table is
the enumeration of what a class diagram can express; a feature with no row
here is a feature nobody is watching.

The JSON -> BUML -> JSON leg is covered by ``test_converter_roundtrip.py``;
the rule there ("if JSON->BUML supports a feature, BUML->JSON must too") is
extended here to the spec text, which is the third consumer of the same
model and the only one with no test before this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pytest

from besser.generators.llm.model_serializer import serialize_domain_model
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
    class_buml_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)


# --------------------------------------------------------------------------- #
# The diagram. Every element exists to exercise one row of the table below.
# --------------------------------------------------------------------------- #

# Declaration order is deliberately NOT alphabetical: SCHEDULED is declared
# first and CANCELLED last, so any layer that sorts by name is visible here.
# "The first literal" is a decision the user made when they typed the list.
LITERAL_ORDER = ["SCHEDULED", "CONFIRMED", "CANCELLED"]

DIAGRAM: dict[str, Any] = {
    "title": "Hotel",
    "model": {
        "elements": {
            "e1": {"id": "e1", "type": "Enumeration", "name": "BookingStatus",
                   "attributes": ["l1", "l2", "l3"], "methods": []},
            "l1": {"id": "l1", "type": "ClassAttribute", "name": "SCHEDULED", "owner": "e1"},
            "l2": {"id": "l2", "type": "ClassAttribute", "name": "CONFIRMED", "owner": "e1"},
            "l3": {"id": "l3", "type": "ClassAttribute", "name": "CANCELLED", "owner": "e1"},

            # Abstract class, with description + URI metadata.
            "c0": {"id": "c0", "type": "AbstractClass", "name": "Person",
                   "description": "Anyone the hotel holds a record for",
                   "uri": "https://schema.org/Person",
                   "attributes": ["a0"], "methods": []},
            "a0": {"id": "a0", "type": "ClassAttribute", "name": "fullName",
                   "visibility": "public", "attributeType": "str", "owner": "c0"},

            # Concrete child: visibility, optional, default, natural identifier.
            "c1": {"id": "c1", "type": "Class", "name": "Guest",
                   "attributes": ["a1", "a2"], "methods": []},
            "a1": {"id": "a1", "type": "ClassAttribute", "name": "email",
                   "visibility": "private", "attributeType": "str",
                   "isExternalId": True, "owner": "c1"},
            "a2": {"id": "a2", "type": "ClassAttribute", "name": "loyaltyPoints",
                   "visibility": "public", "attributeType": "int",
                   "defaultValue": "0", "isOptional": True, "owner": "c1"},

            # Declared id, enum-typed attribute, derived attribute, and three
            # method shapes: zero-argument, parameterised, defaulted parameter.
            "c2": {"id": "c2", "type": "Class", "name": "Booking",
                   "attributes": ["a3", "a4", "a6"], "methods": ["m1", "m2"]},
            "a3": {"id": "a3", "type": "ClassAttribute", "name": "reference",
                   "visibility": "public", "attributeType": "str",
                   "isId": True, "owner": "c2"},
            "a4": {"id": "a4", "type": "ClassAttribute", "name": "status",
                   "visibility": "public", "attributeType": "BookingStatus",
                   "owner": "c2"},
            "a6": {"id": "a6", "type": "ClassAttribute", "name": "totalPrice",
                   "visibility": "public", "attributeType": "float",
                   "isDerived": True, "owner": "c2"},
            # Bare name + separate return-type property (the editor's newer shape).
            "m1": {"id": "m1", "type": "ClassMethod", "name": "cancel()",
                   "attributeType": "bool", "owner": "c2"},
            # Legacy signature shape, private, one required and one defaulted arg.
            "m2": {"id": "m2", "type": "ClassMethod",
                   "name": "- addGuest(guest: Guest, nights: int = 1): bool",
                   "owner": "c2"},

            "c3": {"id": "c3", "type": "Class", "name": "Room",
                   "attributes": ["a5"], "methods": []},
            "a5": {"id": "a5", "type": "ClassAttribute", "name": "roomNumber",
                   "visibility": "public", "attributeType": "str",
                   "isExternalId": True, "owner": "c3"},

            "o1": {"id": "o1", "type": "ClassOCLConstraint",
                   "name": "referenceIsSet",
                   "constraint": "context Booking inv referenceIsSet: self.reference <> ''"},

            # Two comment boxes. A linked one becomes the class's description;
            # an unlinked one becomes the model's. Both are prose the user
            # typed and the editor keeps across a save.
            "k1": {"id": "k1", "type": "Comments",
                   "name": "All prices are in EUR, VAT included."},
            "k2": {"id": "k2", "type": "Comments",
                   "name": "A stay is never shorter than one night."},
        },
        "relationships": {
            # Both ends named. Read by hand: the end typed Guest is reached
            # FROM a Booking, so `guest` is a property of Booking; the end
            # typed Booking is reached from a Guest, so `bookings` is a
            # property of Guest.
            "r1": {"id": "r1", "type": "ClassBidirectional", "name": "bookingsFor",
                   "source": {"element": "c1", "multiplicity": "1", "role": "guest"},
                   "target": {"element": "c2", "multiplicity": "0..*", "role": "bookings"}},
            # Composition, no explicit roles: both ends fall back to the
            # lowercased name of the class at that end.
            "r2": {"id": "r2", "type": "ClassComposition", "name": "occupies",
                   "source": {"element": "c2", "multiplicity": "0..*"},
                   "target": {"element": "c3", "multiplicity": "1..*"}},
            "r3": {"id": "r3", "type": "ClassInheritance",
                   "source": {"element": "c1"}, "target": {"element": "c0"}},
            "r5": {"id": "r5", "type": "Link",
                   "source": {"element": "k2"}, "target": {"element": "c2"}},
            # Unidirectional: the source end is not navigable.
            "r4": {"id": "r4", "type": "ClassUnidirectional", "name": "handledBy",
                   "source": {"element": "c2", "multiplicity": "0..*",
                              "role": "bookingsHandled"},
                   "target": {"element": "c0", "multiplicity": "1", "role": "handler"}},
        },
    },
}


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    """The serialized model exactly as the agent's system prompt embeds it."""
    return serialize_domain_model(process_class_diagram(DIAGRAM))


# --------------------------------------------------------------------------- #
# Readers — small accessors so a failing row names the feature, not an index.
# --------------------------------------------------------------------------- #

def _cls(spec: dict, name: str) -> dict:
    return next(c for c in spec["classes"] if c["name"] == name)


def _attr(spec: dict, cls: str, name: str) -> dict:
    return next(a for a in _cls(spec, cls).get("attributes", []) if a["name"] == name)


def _method(spec: dict, cls: str, name: str) -> dict:
    return next(m for m in _cls(spec, cls).get("methods", []) if m["name"] == name)


def _assoc(spec: dict, name: str) -> dict:
    return next(a for a in spec["associations"] if a["name"] == name)


def _end(spec: dict, assoc: str, role: str) -> dict:
    return next(e for e in _assoc(spec, assoc)["ends"] if e["role"] == role)


# --------------------------------------------------------------------------- #
# The table: one row per thing a class diagram can express.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Feature:
    """What the diagram declared, and what the spec must therefore say."""

    name: str
    read: Callable[[dict], Any]
    expected: Any


FEATURES: tuple[Feature, ...] = (
    # -- classes ------------------------------------------------------------
    Feature("class names",
            lambda s: sorted(c["name"] for c in s["classes"]),
            ["Booking", "Guest", "Person", "Room"]),
    Feature("abstract class",
            lambda s: _cls(s, "Person").get("is_abstract"), True),
    Feature("concrete class is not marked abstract",
            lambda s: _cls(s, "Guest").get("is_abstract"), None),
    Feature("class description",
            lambda s: _cls(s, "Person")["metadata"]["description"],
            "Anyone the hotel holds a record for"),
    Feature("class uri",
            lambda s: _cls(s, "Person")["metadata"].get("uri"),
            "https://schema.org/Person"),
    # Comment boxes: the user typed these and nothing else carries them.
    Feature("comment linked to a class",
            lambda s: _cls(s, "Booking")["metadata"]["description"],
            "A stay is never shorter than one night."),
    Feature("comment linked to nothing (model-level note)",
            lambda s: s["metadata"]["description"],
            "All prices are in EUR, VAT included."),

    # -- attributes ---------------------------------------------------------
    Feature("attribute type",
            lambda s: _attr(s, "Guest", "loyaltyPoints")["type"], "int"),
    Feature("attribute enum type",
            lambda s: _attr(s, "Booking", "status")["type"], "BookingStatus"),
    Feature("attribute visibility",
            lambda s: _attr(s, "Guest", "email").get("visibility"), "private"),
    Feature("attribute default",
            lambda s: _attr(s, "Guest", "loyaltyPoints").get("default"), "0"),
    Feature("attribute optional",
            lambda s: _attr(s, "Guest", "loyaltyPoints").get("is_optional"), True),
    Feature("attribute declared id",
            lambda s: _attr(s, "Booking", "reference").get("is_id"), True),
    Feature("attribute derived",
            lambda s: _attr(s, "Booking", "totalPrice").get("is_derived"), True),
    # "every room is identified by its room number" - the flag that makes the
    # generated database reject a second room 101. Dropping it silently turns
    # a stated uniqueness rule into an ordinary string column.
    Feature("attribute natural identifier",
            lambda s: _attr(s, "Room", "roomNumber").get("is_external_id"), True),
    Feature("a plain attribute carries no identifier flag",
            lambda s: _attr(s, "Guest", "loyaltyPoints").get("is_external_id"), None),

    # -- methods ------------------------------------------------------------
    Feature("method names",
            lambda s: sorted(m["name"] for m in _cls(s, "Booking")["methods"]),
            ["addGuest", "cancel"]),
    Feature("method return type (separate property)",
            lambda s: _method(s, "Booking", "cancel")["return_type"], "bool"),
    Feature("method return type (in the signature)",
            lambda s: _method(s, "Booking", "addGuest")["return_type"], "bool"),
    Feature("method visibility",
            lambda s: _method(s, "Booking", "addGuest").get("visibility"), "private"),
    # The 16-of-21 failure: an omitted key reads as "not shown", so the agent
    # invented a required argument. An empty list is a statement.
    Feature("zero-argument method states an empty parameter list",
            lambda s: _method(s, "Booking", "cancel")["parameters"], []),
    Feature("parameter names and types",
            lambda s: [(p["name"], p["type"])
                       for p in _method(s, "Booking", "addGuest")["parameters"]],
            [("guest", "Guest"), ("nights", "int")]),
    # A defaulted parameter is OPTIONAL. Without the default the agent reads
    # it as required and the generated handler rejects a call that omits it -
    # the same failure as the zero-argument case, one argument along.
    Feature("parameter default",
            lambda s: _method(s, "Booking", "addGuest")["parameters"][1].get("default"),
            "1"),
    Feature("a parameter with no default says nothing",
            lambda s: _method(s, "Booking", "addGuest")["parameters"][0].get("default"),
            None),

    # -- enumerations -------------------------------------------------------
    Feature("enumeration name",
            lambda s: [e["name"] for e in s["enumerations"]], ["BookingStatus"]),
    # CANCELLED sorts before SCHEDULED. Sorting by name replaces the user's
    # first literal with the alphabetically first one, and "the first literal"
    # is what an initial state gets read from.
    Feature("enumeration literal order",
            lambda s: s["enumerations"][0]["literals"], LITERAL_ORDER),

    # -- associations -------------------------------------------------------
    Feature("association names",
            lambda s: [a["name"] for a in s["associations"]],
            ["bookingsFor", "handledBy", "occupies"]),
    Feature("association end roles",
            lambda s: sorted(e["role"] for e in _assoc(s, "bookingsFor")["ends"]),
            ["bookings", "guest"]),
    Feature("association end class",
            lambda s: _end(s, "bookingsFor", "guest")["class"], "Guest"),
    Feature("association end multiplicity",
            lambda s: _end(s, "bookingsFor", "bookings")["multiplicity"], "0..*"),
    # A role on one end is a property of the class at the OPPOSITE end. The
    # serialized shape used to leave that to inference, and it has been
    # inferred backwards before (foreign key on the wrong table).
    Feature("association end owner (role belongs to the opposite class)",
            lambda s: _end(s, "bookingsFor", "guest")["owner"], "Booking"),
    Feature("association end owner, other side",
            lambda s: _end(s, "bookingsFor", "bookings")["owner"], "Guest"),
    Feature("unnamed end falls back to the lowercased class name",
            lambda s: sorted(e["role"] for e in _assoc(s, "occupies")["ends"]),
            ["booking", "room"]),
    Feature("composition",
            lambda s: _end(s, "occupies", "room").get("composite"), True),
    Feature("non-navigable end",
            lambda s: _end(s, "handledBy", "bookingsHandled").get("navigable"), False),
    Feature("navigable end says nothing",
            lambda s: _end(s, "handledBy", "handler").get("navigable"), None),

    # -- generalization and constraints --------------------------------------
    Feature("generalization",
            lambda s: s["generalizations"], [{"parent": "Person", "child": "Guest"}]),
    Feature("direct parents on the child",
            lambda s: _cls(s, "Guest")["parents"], ["Person"]),
    Feature("inherited attributes are flattened with their owner",
            lambda s: [(a["name"], a["from"])
                       for a in _cls(s, "Guest")["inherited_attributes"]],
            [("fullName", "Person")]),
    Feature("ocl constraint context",
            lambda s: s["constraints"][0]["context"], "Booking"),
    Feature("ocl constraint expression",
            lambda s: s["constraints"][0]["expression"],
            "context Booking inv referenceIsSet: self.reference <> ''"),
)


@pytest.mark.parametrize("feature", FEATURES, ids=lambda f: f.name)
def test_the_spec_states_what_the_diagram_declared(feature: Feature, spec: dict):
    assert feature.read(spec) == feature.expected


# --------------------------------------------------------------------------- #
# Guards on the table itself. A checker is code that can be wrong.
# --------------------------------------------------------------------------- #

# Every group the enumeration names. A group with no row is a blind spot,
# and this list is the thing a future feature has to be added to.
COVERED_GROUPS = {
    "class", "attribute", "method", "parameter", "enumeration",
    "association", "generalization", "ocl",
}


def test_every_feature_group_has_at_least_one_row():
    named = " ".join(f.name for f in FEATURES)
    missing = sorted(g for g in COVERED_GROUPS if g not in named)
    assert not missing, f"no fidelity row covers: {missing}"


def test_feature_names_are_unique():
    names = [f.name for f in FEATURES]
    assert len(names) == len(set(names))


def test_the_spec_is_byte_identical_across_repeated_serialization():
    """Sets in the metamodel make iteration order a coin flip.

    ``Association.ends`` and ``DomainModel.associations`` are ``set``s, so
    without an explicit sort the two ends of an association swap places
    between processes. That is not only unstable prompt-cache input; it also
    makes "the first end" meaningless to anyone reading the JSON.
    """
    import json

    model = process_class_diagram(DIAGRAM)
    first = json.dumps(serialize_domain_model(model), sort_keys=False)
    for _ in range(25):
        assert json.dumps(serialize_domain_model(model), sort_keys=False) == first

    # And stable across a fresh conversion, where the sets are rebuilt.
    again = json.dumps(serialize_domain_model(process_class_diagram(DIAGRAM)))
    assert again == json.dumps(serialize_domain_model(process_class_diagram(DIAGRAM)))


def test_literal_order_also_survives_back_to_editor_json():
    """The existing round-trip rule, restated for the feature under test.

    If the spec text is going to claim declaration order, the editor has to
    agree with it, or a save/reload would renumber the user's literals.
    """
    model = process_class_diagram(DIAGRAM)
    elements = class_buml_to_json(model)["elements"]

    enum_id = next(k for k, v in elements.items()
                   if v.get("type") == "Enumeration" and v["name"] == "BookingStatus")
    literals = [elements[lid]["name"] for lid in elements[enum_id]["attributes"]]
    assert literals == LITERAL_ORDER


def test_end_ownership_agrees_with_the_metamodel_not_with_this_file():
    """Calibrate the ``owner`` claim against BUML itself.

    The rows above assert ownership from a hand reading of the diagram,
    which is exactly the kind of reading that has been wrong before. This
    asks the metamodel the same question: ``Class.association_ends()``
    returns the ends a class navigates by (it drops the end typed by the
    class itself), so the two answers must be the same set.
    """
    model = process_class_diagram(DIAGRAM)
    spec = serialize_domain_model(model)

    from_spec: dict[str, set[tuple[str, str]]] = {}
    for assoc in spec["associations"]:
        for end in assoc["ends"]:
            from_spec.setdefault(end["owner"], set()).add((end["role"], end["class"]))

    from_metamodel = {
        cls.name: {(e.name, e.type.name) for e in cls.association_ends()}
        for cls in model.get_classes()
    }
    from_metamodel = {k: v for k, v in from_metamodel.items() if v}

    assert from_spec == from_metamodel


def test_a_model_with_no_declared_order_is_still_deterministic():
    """Models built in Python carry no declaration order.

    They must fall back to a stable sort, never to set iteration order -
    otherwise the spec text differs between processes for the same model.
    """
    from besser.BUML.metamodel.structural import (
        DomainModel, Enumeration, EnumerationLiteral,
    )

    enum = Enumeration(name="Status", literals={
        EnumerationLiteral(name="OPEN"),
        EnumerationLiteral(name="CLOSED"),
        EnumerationLiteral(name="ARCHIVED"),
    })
    result = serialize_domain_model(DomainModel(name="M", types={enum}))
    assert result["enumerations"][0]["literals"] == ["ARCHIVED", "CLOSED", "OPEN"]
