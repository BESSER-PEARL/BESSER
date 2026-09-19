"""A rejected OCL invariant has to become work, not just a comment.

The Pydantic generator now leaves a TODO where it could not enforce a
constraint. That makes the omission visible; it does not make it anyone's
job. On the hotel model, ``guestsWithinCapacity`` and
``noOverlappingBookings`` are both rejected during conversion, and unless
the planner happens to raise them the run ships them as comments.
"""

import pytest

from besser.generators.llm import gap_analyzer


HOTEL_ISSUES = [
    {
        "context": "Booking",
        "name": "guestsWithinCapacity",
        "expression": "context Booking inv guestsWithinCapacity: self.guests->size() <= 4",
        "reason": ("Warning: Invalid OCL syntax in 'context Booking inv "
                   "guestsWithinCapacity: self.guests->size() <= 4': Property "
                   "'guests' not found in context 'Booking' (did you mean 'self.guest'?)"),
        "code": "parse_error",
    },
    {
        "context": "Room",
        "name": "noOverlappingBookings",
        "expression": "context Room inv noOverlappingBookings: self.bookings->forAll(b1, b2 | true)",
        "reason": "Warning: Property 'bookings' not found in context 'Room'",
        "code": "parse_error",
    },
]


class _Model:
    def __init__(self, issues):
        self.conversion_issues = issues


@pytest.fixture
def model():
    return _Model(HOTEL_ISSUES)


def test_a_rejected_constraint_becomes_a_task(model):
    tasks = gap_analyzer._note_rejected_constraints(model, [])

    assert len(tasks) == 2
    assert any("guestsWithinCapacity" in t for t in tasks)
    assert any("noOverlappingBookings" in t for t in tasks)


def test_the_task_carries_the_ocl_and_where_to_put_it(model):
    task = next(t for t in gap_analyzer._note_rejected_constraints(model, [])
                if "guestsWithinCapacity" in t)

    assert "self.guests->size() <= 4" in task
    assert "router handlers that create or update Booking" in task
    # The rejection reason is the model-vs-spec delta: it names the property
    # that did not resolve, so the agent can write the check against the end
    # the model actually declares.
    assert "did you mean 'self.guest'?" in task
    # ...without the converter quoting the whole constraint back a second time.
    assert task.count("self.guests->size() <= 4") == 1


def test_a_constraint_the_planner_already_raised_is_not_duplicated(model):
    planned = ["Implement noOverlappingBookings in the Room router"]

    tasks = gap_analyzer._note_rejected_constraints(model, planned)

    assert len(tasks) == 1
    assert "guestsWithinCapacity" in tasks[0]


def test_nothing_is_added_when_conversion_rejected_nothing():
    assert gap_analyzer._note_rejected_constraints(_Model([]), []) == []
    assert gap_analyzer._note_rejected_constraints(None, []) == []


def test_an_issue_with_no_name_or_no_expression_is_skipped():
    model = _Model([
        {"context": "Room", "name": "", "expression": "context Room inv x: true"},
        {"context": "Room", "name": "unnamedRule", "expression": ""},
    ])

    assert gap_analyzer._note_rejected_constraints(model, []) == []


def test_the_checklist_is_not_swamped_by_a_model_that_rejects_everything():
    issues = [{"context": "C", "name": f"rule{n}", "expression": f"context C inv rule{n}: true"}
              for n in range(20)]

    tasks = gap_analyzer._note_rejected_constraints(_Model(issues), [])

    assert len(tasks) == gap_analyzer._MAX_REJECTED_CONSTRAINT_TASKS
