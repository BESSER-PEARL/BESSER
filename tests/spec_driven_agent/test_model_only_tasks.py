"""A task cannot ask Phase 2 to edit the B-UML model: no tool does that.

Phase 2's model tools are query-only (``query_class``, ``get_constraints_for``,
``list_classes_with``, ``validate_model``). Live run 7aybctis (Qwen,
2026-09-19) planned three tasks phrased "add an association class ... to the
domain model" and "add a constraint ... to the Booking class in the domain
model". The agent understood the intent and cited ``pydantic_classes.py``, but
had no write evidence for a file it had not changed, so each spent its three
checklist attempts and was recorded BLOCKED — nine of the run's thirty-four
turns on work it structurally could not do.
"""

import pytest

from besser.spec_driven_agent.gap_analyzer import _note_model_only_tasks


FILES = ["web_app/backend/pydantic_classes.py", "web_app/backend/sql_alchemy.py"]

LIVE_TASKS = [
    "Add a new association class 'ReservedRoom' with attributes 'agreedPrice' "
    "and 'extraCharges' to the Booking-Room relationship in the domain model",
    "Add a constraint 'guestsWithinCapacity' to the Booking class in the domain "
    "model: context Booking inv guestsWithinCapacity: self.guests->size() <= 3",
]


@pytest.mark.parametrize("task", LIVE_TASKS)
def test_a_model_mutation_task_is_redirected_into_the_code(task):
    noted = _note_model_only_tasks([task], FILES)[0]

    assert task in noted, "the requirement itself must survive"
    assert "read-only in this phase" in noted
    assert "web_app/backend/pydantic_classes.py" in noted


def test_the_annotation_names_a_real_file_or_stays_generic():
    noted = _note_model_only_tasks([LIVE_TASKS[1]], [])[0]

    assert "generated Pydantic/ORM modules" in noted
    assert "`" not in noted.split("read-only in this phase")[1]


@pytest.mark.parametrize("task", [
    "Implement POST /booking/ exactly as defined in the domain model",
    "Add a create form for Booking in web_app/frontend/src/pages/Booking.tsx",
    "Validate the payload according to the domain model constraints",
    "Add pagination to the Room list endpoint",
])
def test_ordinary_code_tasks_are_left_alone(task):
    assert _note_model_only_tasks([task], FILES) == [task]


def test_non_string_entries_pass_through():
    payload = [{"text": "something structured"}, 7]

    assert _note_model_only_tasks(payload, FILES) == payload


# Live task texts from run n_6i2i5r, where the first version of this helper
# annotated a task that named an editable ORM file and pointed the agent at a
# different one.
NAMES_A_REAL_FILE = [
    "In the model web_app/backend/sql_alchemy.py, add the 'totalPrice' "
    "attribute to Bill as a float",
    "In the model web_app/backend/sql_alchemy.py, add the 'settled' attribute "
    "to Bill as a boolean",
    "Add a unique constraint on roomNumber in the model sql_alchemy.py",
]


@pytest.mark.parametrize("task", NAMES_A_REAL_FILE)
def test_a_task_naming_an_orm_source_file_is_left_alone(task):
    """"the model <path>.py" is a file the agent can edit, not the B-UML model."""
    assert _note_model_only_tasks([task], FILES) == [task]


def test_the_real_model_mutation_is_still_redirected_alongside_it():
    """The fix must not silence the case it was written for."""
    noted = _note_model_only_tasks([LIVE_TASKS[0]], FILES)[0]

    assert "read-only in this phase" in noted


# The live kinie9zr shape: the agent put this in a Pydantic validator, where
# self.rooms holds link objects with no maxOccupancy, and every POST /booking/
# raised AttributeError.
RELATIONAL = (
    "Recover model constraint 'guestsWithinCapacity' rejected during conversion. "
    "Add it to the Booking class in the domain model. Original OCL: context Booking "
    "inv guestsWithinCapacity: self.guests->size() <= self.rooms->collect(maxOccupancy)->sum()"
)
FIELD_SHAPE = (
    "Add a constraint to the Room class in the domain model so that maxOccupancy "
    "is at least 1 and standardPrice is positive"
)


def test_a_rule_that_reads_related_rows_is_sent_to_the_router():
    noted = _note_model_only_tasks([RELATIONAL], FILES)[0]

    assert "CANNOT run in a Pydantic validator" in noted
    assert "database session" in noted
    assert "pydantic_classes.py" not in noted, "named the file that cannot enforce it"


def test_a_field_constraint_still_names_the_validator():
    noted = _note_model_only_tasks([FIELD_SHAPE], FILES)[0]

    assert "pydantic_classes.py" in noted
    assert "CANNOT run in a Pydantic validator" not in noted


@pytest.mark.parametrize("navigation", ["->forAll(", "->exists(", "->select(", "->collect("])
def test_every_collection_navigation_counts_as_relational(navigation):
    task = f"Add a constraint in the domain model: context Room inv x: self.bookings{navigation}b | b.id)"

    assert "database session" in _note_model_only_tasks([task], FILES)[0]
