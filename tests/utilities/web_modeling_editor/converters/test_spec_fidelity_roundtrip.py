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

from besser.spec_driven_agent.model_serializer import serialize_domain_model
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

def _node(node_id: str, node_type: str, data: dict[str, Any], x: int = 0) -> dict[str, Any]:
    return {"id": node_id, "type": node_type,
            "position": {"x": x, "y": 0}, "width": 160, "height": 100, "data": data}


def _edge(edge_id: str, edge_type: str, source: str, target: str, **data: Any) -> dict[str, Any]:
    return {"id": edge_id, "type": edge_type, "source": source, "target": target,
            "data": {"points": [], **data}}


# v4 wire shape (``{nodes, edges}``), see docs/source/migrations/uml-v4-shape.md.
DIAGRAM: dict[str, Any] = {
    "title": "Hotel",
    "model": {
        "version": "4.0.0",
        "type": "ClassDiagram",
        "nodes": [
            _node("e1", "class", {
                "name": "BookingStatus", "stereotype": "Enumeration", "methods": [],
                "attributes": [
                    {"id": "l1", "name": "SCHEDULED"},
                    {"id": "l2", "name": "CONFIRMED"},
                    {"id": "l3", "name": "CANCELLED"},
                ],
            }),

            # Abstract class, with description + URI metadata.
            _node("c0", "class", {
                "name": "Person", "stereotype": "Abstract",
                "description": "Anyone the hotel holds a record for",
                "uri": "https://schema.org/Person",
                "attributes": [
                    {"id": "a0", "name": "fullName", "visibility": "public", "attributeType": "str"},
                ],
                "methods": [],
            }),

            # Concrete child: visibility, optional, default, natural identifier.
            _node("c1", "class", {
                "name": "Guest", "stereotype": None, "methods": [],
                "attributes": [
                    {"id": "a1", "name": "email", "visibility": "private",
                     "attributeType": "str", "isExternalId": True},
                    {"id": "a2", "name": "loyaltyPoints", "visibility": "public",
                     "attributeType": "int", "defaultValue": "0", "isOptional": True},
                ],
            }),

            # Declared id, enum-typed attribute, derived attribute, and three
            # method shapes: zero-argument, parameterised, defaulted parameter.
            _node("c2", "class", {
                "name": "Booking", "stereotype": None,
                "attributes": [
                    {"id": "a3", "name": "reference", "visibility": "public",
                     "attributeType": "str", "isId": True},
                    {"id": "a4", "name": "status", "visibility": "public",
                     "attributeType": "BookingStatus"},
                    {"id": "a6", "name": "totalPrice", "visibility": "public",
                     "attributeType": "float", "isDerived": True},
                ],
                "methods": [
                    # Bare name + separate return-type property (the editor's newer shape).
                    {"id": "m1", "name": "cancel()", "attributeType": "bool"},
                    # Legacy signature shape, private, one required and one defaulted arg.
                    {"id": "m2", "name": "- addGuest(guest: Guest, nights: int = 1): bool"},
                ],
            }),

            _node("c3", "class", {
                "name": "Room", "stereotype": None, "methods": [],
                "attributes": [
                    {"id": "a5", "name": "roomNumber", "visibility": "public",
                     "attributeType": "str", "isExternalId": True},
                ],
            }),

            _node("o1", "ClassOCLConstraint", {
                "name": "referenceIsSet",
                "expression": "context Booking inv referenceIsSet: self.reference <> ''",
            }),

            # Two comment boxes. A linked one becomes the class's description;
            # an unlinked one becomes the model's. Both are prose the user
            # typed and the editor keeps across a save.
            _node("k1", "comment", {"name": "All prices are in EUR, VAT included."}),
            _node("k2", "comment", {"name": "A stay is never shorter than one night."}),
        ],
        "edges": [
            # Both ends named. Read by hand: the end typed Guest is reached
            # FROM a Booking, so `guest` is a property of Booking; the end
            # typed Booking is reached from a Guest, so `bookings` is a
            # property of Guest.
            _edge("r1", "ClassBidirectional", "c1", "c2", name="bookingsFor",
                  sourceMultiplicity="1", sourceRole="guest",
                  targetMultiplicity="0..*", targetRole="bookings"),
            # Composition, no explicit roles: both ends fall back to the
            # lowercased name of the class at that end.
            _edge("r2", "ClassComposition", "c2", "c3", name="occupies",
                  sourceMultiplicity="0..*", targetMultiplicity="1..*"),
            _edge("r3", "ClassInheritance", "c1", "c0"),
            _edge("r5", "CommentLink", "k2", "c2"),
            # One-way association: the source end is not navigable.
            _edge("r4", "ClassBidirectional", "c2", "c0", name="handledBy",
                  sourceMultiplicity="0..*", sourceRole="bookingsHandled", sourceNavigable=False,
                  targetMultiplicity="1", targetRole="handler", targetNavigable=True),
        ],
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
    nodes = class_buml_to_json(model)["nodes"]

    enum_node = next(n for n in nodes
                     if n.get("type") == "class"
                     and (n["data"].get("stereotype") or "").lower() == "enumeration"
                     and n["data"]["name"] == "BookingStatus")
    literals = [row["name"] for row in enum_node["data"]["attributes"]]
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
