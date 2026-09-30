"""
Tests for the "did you mean" suggestion appended to dropped OCL invariants
whose ``self.X`` navigation fails to resolve.

Motivating case (a hotel model produced by the modeling assistant): a
Guest<->Booking association rolls its role name "guests" onto the *Booking*
end instead of the Guest end -- the classic UML "role on the wrong end"
mistake. The converter's end-ownership rule (``class_diagram_processor.py``)
is standard and correct: it puts the property on the *opposite* class, so
``Guest.guests`` is created (not ``Booking.guests``) while ``Booking`` gets
an auto-derived singular ``guest``. ``self.guests`` in ``context Booking``
then correctly fails to resolve and the invariant is dropped -- but with no
hint that ``self.guest`` is the property that exists.

Similarly, a Room<->Booking association leaves the Booking end's role blank,
so it falls back to the singular class-name default (``booking``) even
though its multiplicity is 0..* -- ``self.bookings`` in ``context Room``
fails for the same underlying reason: a naming mismatch, not a broken model.

These tests lock in the conservative fix: the OCL type-checker's rejection
is left untouched (a genuinely wrong reference still gets rejected), but the
conversion-issue ``reason`` is enriched with the actual property name when
it can be identified with certainty -- never a guess.
"""

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Multiplicity, Property,
)

from besser.utilities.web_modeling_editor.backend.services.converters.parsers.ocl_parser import (
    _suggest_property_fix,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)


# ---------------------------------------------------------------------------
# _suggest_property_fix — unit tests against a hand-built domain model
# ---------------------------------------------------------------------------

def _hotel_model():
    """Booking/Guest/Room, wired exactly as in the motivating model:
    the "guests" role sits on the Booking end (wrong end) and the Room<->Booking
    association leaves Booking's own end role blank (many-valued but singular).
    """
    booking = Class("Booking")
    guest = Class("Guest")
    room = Class("Room")

    # "guests" association: Booking's own property is "guest" (type=Guest);
    # Guest's own property is "guests" (type=Booking) -- the misplaced role.
    guest_end = Property("guest", type=guest, multiplicity=Multiplicity(1, 9999))
    guests_end = Property("guests", type=booking, multiplicity=Multiplicity(0, 9999))
    BinaryAssociation(name="guests", ends={guest_end, guests_end})

    # "rooms" association: Booking's own property is "rooms" (type=Room);
    # Room's own property is "booking" (type=Booking), blank role -> singular
    # default despite being many-valued.
    rooms_end = Property("rooms", type=room, multiplicity=Multiplicity(1, 9999))
    booking_end = Property("booking", type=booking, multiplicity=Multiplicity(0, 9999))
    BinaryAssociation(name="rooms", ends={rooms_end, booking_end})

    model = DomainModel("Hotel", types={booking, guest, room})
    return model


def test_suggests_sibling_end_misplaced_on_wrong_end():
    """'guests' was rolled onto the Booking end; Booking's real property is 'guest'."""
    model = _hotel_model()
    suggestion = _suggest_property_fix(
        model, "Booking", "Property 'guests' not found in context 'Booking'"
    )
    assert suggestion == "guest"


def test_suggests_singular_property_for_naive_plural():
    """'bookings' was never a role anywhere; Room's real property is the
    many-valued but singularly-named 'booking'."""
    model = _hotel_model()
    suggestion = _suggest_property_fix(
        model, "Room", "Property 'bookings' not found in context 'Room'"
    )
    assert suggestion == "booking"


def test_no_suggestion_for_genuinely_wrong_property_name():
    """A name unrelated to any association end must not get a false suggestion."""
    model = _hotel_model()
    assert _suggest_property_fix(
        model, "Booking", "Property 'nonexistentThing' not found in context 'Booking'"
    ) is None


def test_no_suggestion_when_class_unknown():
    model = _hotel_model()
    assert _suggest_property_fix(
        model, "Nope", "Property 'guests' not found in context 'Nope'"
    ) is None


def test_no_suggestion_for_unrelated_error_text():
    """Only the specific 'Property ... not found in context ...' shape is handled."""
    model = _hotel_model()
    assert _suggest_property_fix(model, "Booking", "missing ')'.") is None


# ---------------------------------------------------------------------------
# End-to-end: process_class_diagram -> conversion_issues["reason"]
# ---------------------------------------------------------------------------

def _hotel_class(node_id, name, x):
    return {"id": node_id, "type": "class",
            "position": {"x": x, "y": 0}, "width": 160, "height": 100,
            "data": {"name": name, "stereotype": None, "attributes": [], "methods": []}}


def _hotel_diagram_json(booking_inv: str, room_inv: str):
    """v4 hotel diagram: free-standing ``ClassOCLConstraint`` nodes carry the
    invariants' full text (the context comes from the text itself)."""
    return {
        "id": "hotel-diagram",
        "title": "HotelTest",
        "model": {
            "version": "4.0.0",
            "type": "ClassDiagram",
            "nodes": [
                _hotel_class("cls-Booking", "Booking", 0),
                _hotel_class("cls-Guest", "Guest", 300),
                _hotel_class("cls-Room", "Room", 600),
                {"id": "ocl-guests", "type": "ClassOCLConstraint",
                 "position": {"x": 0, "y": 200}, "width": 210, "height": 90,
                 "data": {"expression": booking_inv}},
                {"id": "ocl-bookings", "type": "ClassOCLConstraint",
                 "position": {"x": 300, "y": 200}, "width": 210, "height": 90,
                 "data": {"expression": room_inv}},
            ],
            "edges": [
                {"id": "rel-guests", "type": "ClassBidirectional",
                 "source": "cls-Guest", "target": "cls-Booking",
                 "data": {"name": "guests",
                          "sourceRole": "", "sourceMultiplicity": "1..*",
                          "targetRole": "guests", "targetMultiplicity": "0..*",
                          "points": []}},
                {"id": "rel-rooms", "type": "ClassBidirectional",
                 "source": "cls-Booking", "target": "cls-Room",
                 "data": {"name": "rooms",
                          "sourceRole": "", "sourceMultiplicity": "0..*",
                          "targetRole": "rooms", "targetMultiplicity": "1..*",
                          "points": []}},
            ],
        },
    }


def test_conversion_issue_reason_suggests_correct_property_for_guests():
    diagram = _hotel_diagram_json(
        booking_inv="context Booking inv guestsWithinCapacity: self.guests->size() > 0",
        room_inv="context Room inv noOverlappingBookings: self.bookings->size() > 0",
    )
    dm = process_class_diagram(diagram)
    issues = {i["name"]: i for i in dm.conversion_issues}

    assert issues["guestsWithinCapacity"]["code"] == "parse_error"
    assert "Property 'guests' not found in context 'Booking'" in issues["guestsWithinCapacity"]["reason"]
    assert "(did you mean 'self.guest'?)" in issues["guestsWithinCapacity"]["reason"]

    assert issues["noOverlappingBookings"]["code"] == "parse_error"
    assert "Property 'bookings' not found in context 'Room'" in issues["noOverlappingBookings"]["reason"]
    assert "(did you mean 'self.booking'?)" in issues["noOverlappingBookings"]["reason"]

    # The type-checker still rejects both -- this is a better message, not a
    # silent rewrite: neither constraint is present in the executable model.
    assert dm.constraints == set()


def test_conversion_issue_reason_has_no_suggestion_for_unrelated_typo():
    """A genuinely wrong reference on Booking must be rejected with no
    fabricated suggestion -- proves the fix doesn't weaken real errors into
    false positives."""
    diagram = _hotel_diagram_json(
        booking_inv="context Booking inv typo: self.totallyMadeUp->size() > 0",
        room_inv="context Room inv noOverlappingBookings: self.bookings->size() > 0",
    )
    dm = process_class_diagram(diagram)
    issues = {i["name"]: i for i in dm.conversion_issues}
    assert "did you mean" not in issues["typo"]["reason"]
