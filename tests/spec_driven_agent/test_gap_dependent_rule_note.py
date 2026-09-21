"""A create-time task about rows that cannot exist yet must say where the rule can hold.

Live run 9a6063ed (2026-09-18), task #8 of the gap analyser::

    In web_app/backend/routers/booking.py, add validation in the
    'create_booking' function to enforce that the total number of guests
    does not exceed the sum of room capacities across all BookedRooms.

Phase 2 implemented it word for word, as a check at insert time. But
``BookedRoomCreate.booking`` is mandatory: a BookedRoom cannot exist before
its Booking, so at insert time a Booking has no BookedRooms, capacity was
always 0, and every POST /booking/ was a 400 - with a guest ("exceeds room
capacity (0)") and without one ("At least 1 Guest(s) required"). The
delivered app could not create a Booking by any sequence of requests.

The fact that makes the task unsatisfiable is in the model, so the planner's
task is annotated deterministically: the rule stays, its placement moves to
where the aggregate is complete.
"""
from __future__ import annotations

import json

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DateType,
    DomainModel,
    Generalization,
    IntegerType,
    Multiplicity,
    Property,
    StringType,
)
from besser.spec_driven_agent.planning.gap_analyzer import (
    _note_dependent_rule_placement,
    analyze_gaps_via_llm,
)

TASK_8 = (
    "In web_app/backend/routers/booking.py, add validation in the "
    "'create_booking' function to enforce that the total number of guests "
    "does not exceed the sum of room capacities across all BookedRooms."
)


def _hotel_model() -> DomainModel:
    """The run's model: BookedRoom and Bill require a Booking; Booking
    requires a contact Person, an Employee and at least one Guest."""
    person = Class(name="Person", attributes={Property(name="email", type=StringType)})
    employee = Class(name="Employee")
    guest = Class(name="Guest")
    room = Class(name="Room", attributes={Property(name="capacity", type=IntegerType)})
    booking = Class(name="Booking", attributes={Property(name="arrivalDate", type=DateType)})
    booked_room = Class(name="BookedRoom", attributes={Property(name="agreedPrice", type=IntegerType)})
    bill = Class(name="Bill", attributes={Property(name="billNumber", type=IntegerType)})

    def link(name, a, a_mult, b, b_mult):
        return BinaryAssociation(name=name, ends={
            Property(name=a[0], type=a[1], multiplicity=Multiplicity(*a_mult)),
            Property(name=b[0], type=b[1], multiplicity=Multiplicity(*b_mult)),
        })

    associations = {
        link("contact", ("contact", person), (1, 1), ("booking", booking), (0, "*")),
        link("handledBy", ("employee", employee), (1, 1), ("handledBy", booking), (0, "*")),
        link("guests", ("guest", guest), (1, "*"), ("guests", booking), (0, "*")),
        link("bookedRooms", ("booking", booking), (1, 1), ("bookedRooms", booked_room), (0, "*")),
        link("roomOf", ("room", room), (1, 1), ("bookedroom", booked_room), (0, "*")),
        link("bill", ("billBooking", booking), (1, 1), ("bill", bill), (0, 1)),
    }
    return DomainModel(
        name="Hotel",
        types={person, employee, guest, room, booking, booked_room, bill},
        associations=associations,
        generalizations={
            Generalization(general=person, specific=employee),
            Generalization(general=person, specific=guest),
        },
    )


def test_the_live_task_is_told_where_the_rule_can_hold():
    [task] = _note_dependent_rule_placement([TASK_8], _hotel_model())
    assert task.startswith(TASK_8.rstrip(".") + ". NOTE: ")
    assert "no BookedRoom row can exist before the Booking it requires" in task
    assert "cannot be checked while creating the Booking" in task
    assert "where BookedRoom rows are created, updated or deleted" in task
    assert "create them inline in the same request" in task
    # Bill also requires a Booking, but the task does not mention it.
    assert "Bill" not in task


def test_the_note_reaches_the_task_list_phase_2_receives():
    class _Planner:
        """Looks like a real provider; returns the run's task."""
        _client = object()

        def chat(self, system, messages, tools):
            return {"content": [{"type": "text", "text": json.dumps([TASK_8])}]}

    tasks = analyze_gaps_via_llm(
        instructions="Guests must not exceed the capacity of the booked rooms.",
        generator_used="generate_web_app",
        domain_model=_hotel_model(),
        inventory="web_app/backend/routers/booking.py 28808",
        llm_client=_Planner(),
    )
    assert tasks is not None and len(tasks) == 1
    assert "NOTE: no BookedRoom row can exist before the Booking it requires" in tasks[0]


def test_a_rule_about_a_prerequisite_is_left_alone():
    """A Booking needs a Person; a Person does not need a Booking. Checking
    the contact at create time is exactly right, so nothing is added."""
    task = "In create_booking, validate that the contact Person exists and has a valid email."
    assert _note_dependent_rule_placement([task], _hotel_model()) == [task]


def test_a_create_task_on_the_child_is_left_alone():
    """Creating a BookedRoom with its Booking already there is the normal order."""
    task = "In create_bookedroom, reject a Booking whose commercialStatus is CANCELLED."
    assert _note_dependent_rule_placement([task], _hotel_model()) == [task]


def test_a_create_task_that_names_no_dependent_is_left_alone():
    task = "In create_booking, ensure arrivalDate is before departureDate."
    assert _note_dependent_rule_placement([task], _hotel_model()) == [task]


def test_prose_spellings_of_the_classes_match():
    task = "When creating a booking, check that the guests fit in the booked rooms and add the bill total."
    [noted] = _note_dependent_rule_placement([task], _hotel_model())
    assert "NOTE: no Bill or BookedRoom row can exist before the Booking it requires" in noted


def test_creating_something_else_for_a_booking_is_not_creating_a_booking():
    task = "Add an endpoint that creates a Bill for a Booking and marks the BookedRooms as invoiced."
    assert _note_dependent_rule_placement([task], _hotel_model()) == [task]


def test_without_a_model_tasks_pass_through():
    assert _note_dependent_rule_placement([TASK_8], None) == [TASK_8]


def test_a_model_without_mandatory_ends_adds_nothing():
    a = Class(name="A")
    b = Class(name="B")
    optional = BinaryAssociation(name="ab", ends={
        Property(name="a", type=a, multiplicity=Multiplicity(0, 1)),
        Property(name="bs", type=b, multiplicity=Multiplicity(0, "*")),
    })
    model = DomainModel(name="M", types={a, b}, associations={optional})
    task = "In create_a, validate the attached Bs."
    assert _note_dependent_rule_placement([task], model) == [task]
