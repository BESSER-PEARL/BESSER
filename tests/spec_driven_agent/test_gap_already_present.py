"""Tests for the three "already present" evidence families added to the
Phase-2 gap analyser, extending ``_drop_present_enumerations``'s proven
pattern (match on model evidence, never on task wording) to:

  * ``_drop_present_attributes``       - an attribute the model already
    declares on the named class (drop when the ask is pure declaration,
    annotate when it also asks for derived behaviour the model does not
    encode).
  * ``_note_present_regex_validations`` - a validation the model already
    carries as a successfully-converted OCL ``self.<attr>.matches(regex)``
    invariant (always annotate, never drop).
  * ``_note_present_relationships``     - an association end / relationship
    the model already declares (always annotate, never drop).

A 271-task, 15-run survey found ~25% of LLM-planner tasks re-litigating
work router.py.j2 / sql_alchemy.py.j2 / pydantic_classes_template.py.j2
already produce deterministically from the model. Every family above was
verified against the real generated code before being written (see the
docstrings on each function in gap_analyzer.py for the exact runs/files).

Every one of these tests FAILS on the pre-change gap_analyzer.py: none of
``_drop_present_attributes``, ``_note_present_regex_validations`` or
``_note_present_relationships`` existed there (ImportError at collection),
and ``analyze_gaps_via_llm`` did not call them (the wiring tests would
observe the un-annotated / un-dropped task list).

Live counterexamples this suite pins as regressions, both found while
verifying the recipes of recorded runs:
  * se7k3zbx: "add a computed 'commercialStatus' attribute that is derived
    from the bill's settlement status, transitioning ..." names an
    attribute the model already has, but the transition LOGIC is the real,
    still-missing ask - a naive drop would have lost it.
  * gpt-5.6-terra-dp3trml9: a single task bundles the already-satisfied
    validEmail/validPhone re-ask together with "enforce Person
    identifyingNumber uniqueness", which the model does not carry anywhere
    - a naive whole-task drop on the validEmail/validPhone match would have
    silently discarded that second, genuine requirement.
  * several runs re-describe the REJECTED 'guestsWithinCapacity' /
    'noOverlappingBookings' invariants ("does not exceed the combined
    capacity of all rooms") without naming them - close enough to a
    real association mention (Booking.guests, Room.bookings) that the
    relationship family would otherwise annotate them as "already handled".
"""
from __future__ import annotations

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    Constraint,
    DomainModel,
    Generalization,
    Multiplicity,
    PrimitiveDataType,
    Property,
)
from besser.spec_driven_agent.planning.gap_analyzer import (
    _drop_present_attributes,
    _note_present_regex_validations,
    _note_present_relationships,
    analyze_gaps_via_llm,
)
from besser.spec_driven_agent.providers.llm_client import UsageTracker


def _hotel_model() -> DomainModel:
    """Shaped like the live-verified model: Person carries validEmail /
    validPhone as successfully-converted invariants; Employee and Guest
    generalize Person; Booking has contact/handledBy/guests/rooms
    associations; ReservedRoom carries agreedPrice/extraCharges. One
    REJECTED invariant (guestsWithinCapacity) is attached the way the real
    converter attaches one: on ``conversion_issues``, never on
    ``constraints``.
    """
    str_t = PrimitiveDataType("str")
    float_t = PrimitiveDataType("float")
    int_t = PrimitiveDataType("int")
    bool_t = PrimitiveDataType("bool")

    person = Class(name="Person")
    person.attributes = {
        Property(name="email", type=str_t),
        Property(name="phone", type=str_t),
    }
    employee = Class(name="Employee")
    guest = Class(name="Guest")

    booking = Class(name="Booking")
    booking.attributes = {
        Property(name="arrivalDate", type=str_t),
        Property(name="departureDate", type=str_t),
    }

    room = Class(name="Room")
    room.attributes = {Property(name="maxOccupancy", type=int_t)}

    reserved_room = Class(name="ReservedRoom")
    reserved_room.attributes = {
        Property(name="agreedPrice", type=float_t),
        Property(name="extraCharges", type=float_t),
    }

    bill = Class(name="Bill")
    bill.attributes = {Property(name="settled", type=bool_t)}

    def link(name, a, a_mult, b, b_mult):
        return BinaryAssociation(name=name, ends={
            Property(name=a[0], type=a[1], multiplicity=Multiplicity(*a_mult)),
            Property(name=b[0], type=b[1], multiplicity=Multiplicity(*b_mult)),
        })

    associations = {
        link("contact", ("contact", person), (1, 1), ("booking", booking), (0, "*")),
        link("handledBy", ("employee", employee), (1, 1), ("handledBy", booking), (0, "*")),
        link("guests", ("guest", guest), (1, "*"), ("guests", booking), (0, "*")),
        link("rooms", ("rooms", room), (1, "*"), ("booking", booking), (0, "*")),
    }

    model = DomainModel(
        name="Hotel",
        types={person, employee, guest, booking, room, reserved_room, bill},
        associations=associations,
        generalizations={
            Generalization(general=person, specific=employee),
            Generalization(general=person, specific=guest),
        },
        constraints={
            Constraint(
                name="validEmail", context=person, language="OCL",
                expression=(
                    r"context Person inv validEmail: "
                    r"self.email.matches('^[^\s@]+@[^\s@]+\.[A-Za-z]{2,}$')"
                ),
            ),
            Constraint(
                name="validPhone", context=person, language="OCL",
                expression=r"context Person inv validPhone: self.phone.matches('^\+?[0-9]{7,15}$')",
            ),
        },
    )
    # Attached exactly the way json_to_buml/class_diagram_processor.py
    # attaches a rejected OCL box: a list of dicts on the model, never a
    # Constraint in domain_model.constraints.
    model.conversion_issues = [{
        "id": "ocl-conversion-fake",
        "context": "Booking",
        "name": "guestsWithinCapacity",
        "kind": "invariant",
        "expression": "context Booking inv guestsWithinCapacity: self.guests->size() <= self.rooms->collect(maxOccupancy)->sum()",
        "code": "parse_error",
        "reason": "Warning: Invalid OCL syntax: Property 'guests' not found in context 'Booking' (did you mean 'self.guest'?)",
    }]
    return model


# ==========================================================================
# Family 2: _drop_present_attributes
# ==========================================================================


class TestDropPresentAttributes:

    def test_pure_declaration_of_a_present_attribute_is_dropped(self):
        task = (
            "Add a 'extraCharges' attribute to the ReservedRoom class to "
            "store per-room extra charges, enabling Requirement R79."
        )
        assert _drop_present_attributes([task], _hotel_model()) == []

    def test_two_attributes_named_together_are_both_matched_and_dropped(self):
        task = (
            "In the SQL Alchemy model in web_app/backend/sql_alchemy.py, "
            "add the 'agreedPrice' and 'extraCharges' attributes to the "
            "ReservedRoom class, with type float."
        )
        assert _drop_present_attributes([task], _hotel_model()) == []

    def test_matching_type_word_does_not_block_the_drop(self):
        task = "Add the 'settled' attribute to Bill as a boolean, as required by R50."
        assert _drop_present_attributes([task], _hotel_model()) == []

    def test_conflicting_type_word_blocks_the_drop(self):
        """A type word that disagrees with the model names a real gap
        (the model has agreedPrice as float), not a duplicate."""
        task = "Add the 'agreedPrice' attribute to the ReservedRoom class as an integer, as required by R1."
        assert _drop_present_attributes([task], _hotel_model()) == [task]

    def test_derived_behaviour_language_downgrades_to_annotation_not_drop(self):
        """Live counterexample (se7k3zbx): the attribute already exists,
        but the transition LOGIC - which the model does not encode - is
        the real, still-open ask. The requirement must survive."""
        task = (
            "In the Booking model, add a computed 'commercialStatus' "
            "attribute that is derived from the bill's settlement status, "
            "transitioning from 'awaiting payment' to 'confirmed' when the "
            "bill is settled."
        )
        model = _hotel_model()
        booking = next(c for c in model.get_classes() if c.name == "Booking")
        commercial = PrimitiveDataType("str")
        booking.attributes = set(booking.attributes) | {
            Property(name="commercialStatus", type=commercial, is_derived=True),
        }
        [noted] = _drop_present_attributes([task], model)
        assert task in noted, "the requirement itself must survive"
        assert "already declares this attribute" in noted
        assert "focus on the requested BEHAVIOUR" in noted

    # ---- negative tests -------------------------------------------------

    def test_attribute_the_model_does_not_have_survives(self):
        task = "Add a new attribute 'discountRate' to the ReservedRoom class as required by R1."
        assert _drop_present_attributes([task], _hotel_model()) == [task]

    def test_unresolvable_class_name_survives(self):
        task = "Add a new attribute 'agreedPrice' to the Invoice class as required by R1."
        assert _drop_present_attributes([task], _hotel_model()) == [task]

    def test_attribute_present_on_a_different_class_is_not_a_match(self):
        """'extraCharges' exists on ReservedRoom, not on Room - naming Room
        must not borrow evidence from an unrelated class."""
        task = "Add a new attribute 'extraCharges' to the Room class as required by R1."
        assert _drop_present_attributes([task], _hotel_model()) == [task]

    def test_task_about_the_rejected_constraint_is_never_touched(self):
        """Even if it happened to also use the word 'attribute', a task
        naming the REJECTED constraint must never be modified by this
        family - conversion_issues is never proof of anything implemented."""
        task = (
            "Recover the 'guestsWithinCapacity' invariant attribute logic "
            "rejected during conversion; enforce it in create_booking."
        )
        assert _drop_present_attributes([task], _hotel_model()) == [task]

    def test_no_domain_model_means_no_filtering(self):
        task = "Add a new attribute 'agreedPrice' to the ReservedRoom class."
        assert _drop_present_attributes([task], None) == [task]

    def test_non_matching_tasks_pass_through_unchanged(self):
        tasks = [
            "Add pagination to the Room list endpoint.",
            "Implement Booking.cancel in booking_methods.py.",
        ]
        assert _drop_present_attributes(list(tasks), _hotel_model()) == tasks


# ==========================================================================
# Family 3: _note_present_regex_validations
# ==========================================================================


class TestNotePresentRegexValidations:

    def test_email_validation_task_is_annotated_not_dropped(self):
        task = (
            "In the Person model, ensure that the 'email' attribute is "
            "validated against the regex pattern '^\\S+@\\S+\\.\\S{2,}$' "
            "to enforce valid email format."
        )
        [noted] = _note_present_regex_validations([task], _hotel_model())
        assert task in noted, "the requirement itself must survive"
        assert "validEmail" in noted
        assert "@field_validator" in noted

    def test_direct_constraint_name_mention_matches(self):
        task = "Enforce the `validPhone` constraint on create, update, and bulk-create paths."
        [noted] = _note_present_regex_validations([task], _hotel_model())
        assert "Person.phone" in noted and "validPhone" in noted

    def test_subclass_of_the_constraint_context_still_matches(self):
        """Guest/Employee generalize Person in the fixture; the generated
        GuestCreate/EmployeeCreate schemas subclass PersonCreate in real
        output, so the @field_validator is inherited."""
        task = "Validate Guest email and phone formats on every create/update path."
        [noted] = _note_present_regex_validations([task], _hotel_model())
        assert "validEmail" in noted or "validPhone" in noted

    def test_bundled_requirement_survives_verbatim(self):
        """Live counterexample (gpt-5.6-terra-dp3trml9 task 14): the
        already-satisfied validEmail/validPhone re-ask is bundled with a
        genuinely open 'identifyingNumber uniqueness' requirement in the
        SAME task. Dropping the whole task would silently lose it - this
        family must only ever annotate."""
        task = (
            "Update routers/person.py, routers/guest.py, and "
            "routers/employee.py to enforce Person `identifyingNumber` "
            "uniqueness plus the `validEmail` and `validPhone` constraints "
            "on create, update, and bulk-create paths."
        )
        [noted] = _note_present_regex_validations([task], _hotel_model())
        assert "identifyingNumber" in noted and "uniqueness" in noted
        assert noted.startswith(task)

    # ---- negative tests -------------------------------------------------

    def test_attribute_with_no_matching_constraint_survives(self):
        task = "In the Room model, ensure the 'roomNumber' attribute is validated against a regex."
        assert _note_present_regex_validations([task], _hotel_model()) == [task]

    def test_task_about_the_rejected_constraint_is_never_touched(self):
        task = (
            "Implement the 'guestsWithinCapacity' invariant in the Booking "
            "model to enforce that the number of guests does not exceed "
            "capacity, using a regex-free validation."
        )
        assert _note_present_regex_validations([task], _hotel_model()) == [task]

    def test_no_marker_word_leaves_the_task_alone(self):
        """Naming the attribute is not enough without a validation cue -
        otherwise any task that merely mentions 'email' would be touched."""
        task = "Add an 'email' column index to the Person table for faster lookups."
        assert _note_present_regex_validations([task], _hotel_model()) == [task]

    def test_no_domain_model_returns_tasks_unchanged(self):
        task = "Validate the 'email' attribute against the regex pattern."
        assert _note_present_regex_validations([task], None) == [task]


# ==========================================================================
# Family 1: _note_present_relationships
# ==========================================================================


class TestNotePresentRelationships:

    def test_relationship_validation_task_is_annotated_not_dropped(self):
        """Live shape (run trilraak, x4: contact/guests/handledBy/rooms)."""
        task = (
            "Ensure the 'contact' relationship in Booking enforces that "
            "the contact person is a valid person in the system by "
            "validating the person's ID in the create_booking endpoint."
        )
        [noted] = _note_present_relationships([task], _hotel_model())
        assert task in noted, "the requirement itself must survive"
        assert "'contact' is a relationship the model already declares on Booking" in noted

    def test_bundle_of_several_relationships_is_annotated_once(self):
        task = (
            "Enforce Booking relationship cardinalities and required "
            "references: exactly one contact Person, exactly one "
            "handledBy Employee, at least one Guest, and at least one Room."
        )
        [noted] = _note_present_relationships([task], _hotel_model())
        assert task in noted
        assert noted.count("NOTE:") == 1

    # ---- negative tests -------------------------------------------------

    def test_paraphrased_rejected_constraint_survives_untouched(self):
        """Neither rejected invariant is named, but the CAPACITY/AGGREGATE
        shape of the rule is exactly what both rejected constraints in the
        live dataset look like - this must not be mistaken for a plain
        existence/cardinality ask the router already covers."""
        task = (
            "Add a constraint in the Booking class to enforce that the "
            "total number of guests does not exceed the combined capacity "
            "of all rooms."
        )
        assert _note_present_relationships([task], _hotel_model()) == [task]

    def test_task_naming_the_rejected_constraint_survives_untouched(self):
        task = (
            "Implement the 'guests' association in the Booking class to "
            "support the invariant 'guestsWithinCapacity' as required by R75."
        )
        assert _note_present_relationships([task], _hotel_model()) == [task]

    def test_unresolvable_class_name_survives(self):
        task = "Ensure the 'contact' relationship in Invoice references a valid person."
        assert _note_present_relationships([task], _hotel_model()) == [task]

    def test_relationship_name_not_on_the_model_survives(self):
        task = "Ensure the 'sponsor' relationship in Booking references a valid person."
        assert _note_present_relationships([task], _hotel_model()) == [task]

    def test_no_marker_word_leaves_the_task_alone(self):
        task = "In Booking, store the contact person's phone number for the front desk."
        assert _note_present_relationships([task], _hotel_model()) == [task]

    def test_no_domain_model_returns_tasks_unchanged(self):
        task = "Ensure the 'contact' relationship in Booking is a valid reference."
        assert _note_present_relationships([task], None) == [task]


# ==========================================================================
# Wired end-to-end through analyze_gaps_via_llm
# ==========================================================================


class _Planner:
    """Looks like a real provider (has ``_client``); returns a fixed list."""
    _client = object()
    model = "capture-model"

    def __init__(self, tasks):
        self.usage = UsageTracker("capture-model")
        self._reply = tasks

    def chat(self, system, messages, tools):
        import json
        return {"content": [{"type": "text", "text": json.dumps(self._reply)}]}


class TestWiredIntoTheAnalyser:

    def test_present_evidence_families_are_applied_end_to_end(self):
        already_present_attr = (
            "Add a 'extraCharges' attribute to the ReservedRoom class to "
            "store per-room extra charges, enabling Requirement R79."
        )
        already_present_regex = (
            "In the Person model, ensure that the 'email' attribute is "
            "validated against the regex pattern for a valid email format."
        )
        already_present_rel = (
            "Ensure the 'contact' relationship in Booking enforces a "
            "valid reference by validating the person's ID."
        )
        rejected_constraint_task = (
            "Implement the 'guestsWithinCapacity' invariant in the "
            "Booking model to enforce that the number of guests does not "
            "exceed the combined capacity of all rooms."
        )
        genuine_task = "Add a DELETE /booking/{id}/cancel endpoint."

        tasks = analyze_gaps_via_llm(
            instructions="Hotel bookings.",
            generator_used="generate_web_app",
            domain_model=_hotel_model(),
            inventory="web_app/backend/routers/booking.py 28808",
            llm_client=_Planner([
                already_present_attr, already_present_regex,
                already_present_rel, rejected_constraint_task, genuine_task,
            ]),
        )

        assert already_present_attr not in tasks, "pure attribute declaration must be dropped"
        assert any(genuine_task in t for t in tasks)
        assert any(
            already_present_regex in t and "validEmail" in t for t in tasks
        ), "regex-validation task must survive, annotated"
        assert any(
            already_present_rel in t and "already declares" in t for t in tasks
        ), "relationship task must survive, annotated"
        assert any(
            t == rejected_constraint_task for t in tasks
        ), "the rejected-constraint task must survive completely UNCHANGED"

    def test_rejected_constraint_task_alone_cannot_be_emptied_out(self):
        """A planner list containing ONLY the rejected-constraint task must
        not be reduced to nothing by any of the three new families.

        The harness may PREPEND its own task for a rejected invariant the
        planner never raised (see test_gap_rejected_constraints.py) - that
        is work being added, not this task being removed.
        """
        rejected_constraint_task = (
            "Add a constraint in the Room class to enforce that no two "
            "active bookings overlap in date range."
        )
        tasks = analyze_gaps_via_llm(
            instructions="Hotel bookings.",
            generator_used="generate_web_app",
            domain_model=_hotel_model(),
            inventory="web_app/backend/routers/booking.py 28808",
            llm_client=_Planner([rejected_constraint_task]),
        )
        assert rejected_constraint_task in tasks
        assert all(t == rejected_constraint_task or t.startswith("Enforce the ")
                   for t in tasks), "nothing but harness enforcement tasks may be added"
