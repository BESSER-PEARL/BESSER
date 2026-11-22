"""
Investigation tests: can relational OCL be transpiled into Pydantic validators?

Ground truth is a "hotel" model used in real Spec-Driven Agent runs, which has
exactly two OCL invariants that navigate a relationship and are never resolved
by any model:

    context Booking inv guestsWithinCapacity:
      self.guests->size() <= self.rooms->collect(maxOccupancy)->sum()

    context Room inv noOverlappingBookings:
      self.bookings->forAll(b1, b2 | b1 <> b2 implies
        b1.departureDate <= b2.arrivalDate or b2.departureDate <= b1.arrivalDate)

These tests record two findings against that model (vendored as
``fixtures/hotel_class_diagram.json``, layout data stripped) and an inline
reconstruction of its relevant fragment:

1. Neither constraint ever reaches ``ocl_utils.py`` for this model. The web
   editor converter's OCL type-checker
   (``besser/utilities/web_modeling_editor/backend/services/converters/``)
   rejects both before conversion, because each references the WRONG
   association end -- Booking's real property is ``guest`` (singular, but
   multi-valued), Room's is ``booking`` -- and records them as
   ``conversion_issues`` rather than attaching them to
   ``domain_model.constraints``.

2. Even with the property name corrected, ``ocl_utils.py`` is right to keep
   skipping both: neither is expressible from a Booking/Room Create payload
   alone. ``guestsWithinCapacity`` needs ``Room.maxOccupancy`` off related
   rows that a Booking payload never carries (only room ids, or -- with
   ``nested_creations`` -- an ambiguous id/nested-object union the client is
   free to send either way). ``noOverlappingBookings`` needs every existing
   sibling ``Booking`` row for a ``Room``, which by definition includes rows
   outside the current request. Both need a database session, i.e. the
   router handler, not a Pydantic validator.

This records why there is no relational-OCL transpiler. The rejected
constraints are not silently lost: the "still has the three working
validators" test also confirms each gets a TODO comment, sourced from
``conversion_issues``, on the class its OCL ``context`` names.
"""
import json
import os

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, Constraint, DateType, DomainModel,
    IntegerType, Multiplicity, Property,
)
from besser.generators.pydantic_classes import PydanticGenerator
from besser.generators.pydantic_classes.ocl_utils import parse_ocl_constraint

REAL_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "hotel_class_diagram.json",
)

GUESTS_WITHIN_CAPACITY = (
    "context Booking inv guestsWithinCapacity: "
    "self.guests->size() <= self.rooms->collect(maxOccupancy)->sum()"
)
GUESTS_WITHIN_CAPACITY_ROLE_FIXED = (
    "context Booking inv guestsWithinCapacity: "
    "self.guest->size() <= self.rooms->collect(maxOccupancy)->sum()"
)
NO_OVERLAPPING_BOOKINGS = (
    "context Room inv noOverlappingBookings: "
    "self.bookings->forAll(b1, b2 | b1 <> b2 implies "
    "b1.departureDate <= b2.arrivalDate or b2.departureDate <= b1.arrivalDate)"
)
NO_OVERLAPPING_BOOKINGS_ROLE_FIXED = (
    "context Room inv noOverlappingBookings: "
    "self.booking->forAll(b1, b2 | b1 <> b2 implies "
    "b1.departureDate <= b2.arrivalDate or b2.departureDate <= b1.arrivalDate)"
)


@pytest.fixture
def hotel_model():
    """Reconstructs the Booking/Guest/Room fragment with the SAME association
    shape as the real model: Booking's own navigable property toward Guest is
    ``guest`` (not ``guests`` -- that role sits on the Guest end instead), and
    Room's own navigable property toward Booking is ``booking`` (not
    ``bookings``). This mirrors what ``tests/utilities/.../test_ocl_property_
    suggestion.py`` already established about this model's role names.
    """
    booking = Class(name="Booking", attributes={
        Property(name="arrivalDate", type=DateType),
        Property(name="departureDate", type=DateType),
    })
    guest = Class(name="Guest")
    room = Class(name="Room", attributes={
        Property(name="maxOccupancy", type=IntegerType),
    })

    guest_end = Property(name="guest", type=guest, multiplicity=Multiplicity(1, 9999))
    guests_end = Property(name="guests", type=booking, multiplicity=Multiplicity(0, 9999))
    BinaryAssociation(name="guests", ends={guest_end, guests_end})

    rooms_end = Property(name="rooms", type=room, multiplicity=Multiplicity(1, 9999))
    booking_end = Property(name="booking", type=booking, multiplicity=Multiplicity(0, 9999))
    BinaryAssociation(name="rooms", ends={rooms_end, booking_end})

    return DomainModel(name="Hotel", types={booking, guest, room})


def _find(domain_model, class_name):
    return next(c for c in domain_model.get_classes() if c.name == class_name)


# ============================================================================
# Finding 1: the real model never hands these two constraints to ocl_utils.py
# ============================================================================

class TestRealModelNeverReachesOclUtils:
    """The converter rejects both constraints before conversion; ocl_utils.py
    never sees them for this model."""

    def test_real_model_omits_both_relational_constraints(self):
        from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
            process_class_diagram,
        )

        with open(REAL_MODEL_PATH, encoding="utf-8") as handle:
            diagram = json.load(handle)
        domain_model = process_class_diagram(diagram)

        constraint_names = {c.name for c in domain_model.constraints}
        assert "guestsWithinCapacity" not in constraint_names
        assert "noOverlappingBookings" not in constraint_names
        # The other three invariants on this model (non-relational) DO make it through.
        assert {"validEmail", "validPhone", "arrivalBeforeDeparture"} <= constraint_names

        issues_by_name = {i["name"]: i for i in domain_model.conversion_issues}
        assert "guestsWithinCapacity" in issues_by_name
        assert "self.guest" in issues_by_name["guestsWithinCapacity"]["reason"]
        assert "noOverlappingBookings" in issues_by_name
        assert "self.booking" in issues_by_name["noOverlappingBookings"]["reason"]

    def test_real_model_pydantic_output_still_has_the_three_working_validators(self):
        """Confirms the claim that non-relational OCL already works end-to-end
        on this exact model: validEmail/validPhone/arrivalBeforeDeparture are
        real validators in the generated file, with no involvement from an LLM.
        The two relational constraints get a TODO comment instead -- on the
        class their OCL ``context`` names -- handing them to the LLM agent;
        neither gets a validator, since neither is transpilable."""
        from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
            process_class_diagram,
        )

        with open(REAL_MODEL_PATH, encoding="utf-8") as handle:
            diagram = json.load(handle)
        domain_model = process_class_diagram(diagram)

        import py_compile
        import tempfile
        with tempfile.TemporaryDirectory() as out_dir:
            PydanticGenerator(model=domain_model, backend=True, output_dir=out_dir).generate()
            path = os.path.join(out_dir, "pydantic_classes.py")
            py_compile.compile(path, doraise=True)
            with open(path, encoding="utf-8") as handle:
                source = handle.read()

        assert "re.fullmatch(r'^[^\\s@]+@[^\\s@]+\\.[A-Za-z]{2,}$', v) is not None" in source
        assert "re.fullmatch(r'^\\+?[0-9]{7,15}$', v) is not None" in source
        assert "if not (self.arrivalDate <= self.departureDate):" in source

        # Both relational constraints were rejected upstream (wrong role name)
        # and never reached domain_model.constraints -- but each now leaves a
        # TODO comment, sourced from domain_model.conversion_issues, on the
        # class its OCL ``context`` names. It is a comment only: no validator
        # exists for either, so nothing enforces them here.
        booking_block = source.split("class BookingCreate(BaseModel):", 1)[1].split("\nclass ")[0]
        room_block = source.split("class RoomCreate(BaseModel):", 1)[1].split("\nclass ")[0]

        assert "# TODO: OCL constraint 'guestsWithinCapacity' is NOT enforced in this file." in booking_block
        assert "did you mean 'self.guest'?" in booking_block
        assert "# TODO: OCL constraint 'noOverlappingBookings' is NOT enforced in this file." in room_block
        assert "did you mean 'self.booking'?" in room_block

        assert "guestsWithinCapacity" not in room_block
        assert "noOverlappingBookings" not in booking_block


# ============================================================================
# Finding 2: even attached (role name corrected), ocl_utils.py must still skip
# ============================================================================

class TestRelationalConstraintsAreNotPydanticExpressible:
    """Bypasses the converter and hands ocl_utils.py the two expressions
    directly -- both as they are actually written in the model, and with the
    association-end name corrected -- to isolate ocl_utils.py's own behaviour
    from the upstream rejection proven above."""

    @pytest.mark.parametrize("expression", [
        GUESTS_WITHIN_CAPACITY,
        GUESTS_WITHIN_CAPACITY_ROLE_FIXED,
    ], ids=["as-written (self.guests)", "role-corrected (self.guest)"])
    def test_guests_within_capacity_is_always_skipped(self, hotel_model, expression):
        booking = _find(hotel_model, "Booking")
        constraint = Constraint(
            name="guestsWithinCapacity", context=booking, expression=expression, language="OCL",
        )
        hotel_model.constraints = {constraint}

        result = parse_ocl_constraint(constraint, hotel_model)
        assert result == {"skipped": True}

    @pytest.mark.parametrize("expression", [
        NO_OVERLAPPING_BOOKINGS,
        NO_OVERLAPPING_BOOKINGS_ROLE_FIXED,
    ], ids=["as-written (self.bookings)", "role-corrected (self.booking)"])
    def test_no_overlapping_bookings_is_always_skipped(self, hotel_model, expression):
        room = _find(hotel_model, "Room")
        constraint = Constraint(
            name="noOverlappingBookings", context=room, expression=expression, language="OCL",
        )
        hotel_model.constraints = {constraint}

        result = parse_ocl_constraint(constraint, hotel_model)
        assert result == {"skipped": True}

    def test_generated_file_leaves_a_trace_not_broken_code(self, hotel_model, tmpdir):
        """Whatever the upstream converter does with these two constraints
        someday, if they ever DO reach the Pydantic generator, the output
        must still compile and must never assert a check it cannot honour."""
        booking = _find(hotel_model, "Booking")
        room = _find(hotel_model, "Room")
        hotel_model.constraints = {
            Constraint(
                name="guestsWithinCapacity", context=booking,
                expression=GUESTS_WITHIN_CAPACITY_ROLE_FIXED, language="OCL",
            ),
            Constraint(
                name="noOverlappingBookings", context=room,
                expression=NO_OVERLAPPING_BOOKINGS_ROLE_FIXED, language="OCL",
            ),
        }

        out_dir = str(tmpdir.mkdir("output"))
        PydanticGenerator(model=hotel_model, backend=True, output_dir=out_dir).generate()
        path = os.path.join(out_dir, "pydantic_classes.py")

        import py_compile
        py_compile.compile(path, doraise=True)
        with open(path, encoding="utf-8") as handle:
            source = handle.read()

        assert "# TODO: OCL constraint 'guestsWithinCapacity' is NOT enforced in this file." in source
        assert "# TODO: OCL constraint 'noOverlappingBookings' is NOT enforced in this file." in source
        # The only "->" left is the verbatim OCL quoted inside the comment;
        # every such line must itself be a comment, never executable code.
        for line in source.splitlines():
            if "->" in line:
                assert line.lstrip().startswith("#"), line
