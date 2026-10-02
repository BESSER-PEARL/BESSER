"""A clean duplicate refusal must not read as an unverified create route.

The probe creates a parent, which materialises its own association rows, then
POSTs that same pair to the association endpoint. A correct app refuses. That
refusal is what triggers the fresh-reference retry -- without it the finding
is reported as ``create unverified``, which is a blocker.

The old rule matched only raw database errors (409 plus a SQLite/Postgres
constraint string), so it had the incentive backwards: a scaffold leaking
``UNIQUE constraint failed`` got the retry and passed, while one answering
``400 {"error": "Relationship already exists"}`` was condemned. Measured on run
claude-sonnet-5-3s9nd9go, where it held the fix loop open for two attempts
against code that was already correct.
"""
import pytest

from besser.spec_driven_agent.validation.constructibility import _is_duplicate_refusal


@pytest.mark.parametrize("status, text", [
    # The regression: a well-behaved app, which the old rule missed on BOTH
    # arms -- wrong status, and no database string in the body.
    (400, '{"error":"Relationship already exists","message":"Relationship already exists"}'),
    (409, '{"detail":"Booking already linked to this room"}'),
    (422, '{"detail":"Room already booked for these dates"}'),
    (400, '{"detail":"Duplicate reservation"}'),
    (409, '{"detail":"That guest is already assigned to the booking"}'),
    # Still recognised: the raw integrity errors the old rule was built for.
    (409, "sqlite3.IntegrityError: UNIQUE constraint failed: reservedroom.rooms_id"),
    (400, 'duplicate key value violates unique constraint "reservedroom_pkey"'),
])
def test_duplicate_refusals_are_recognised(status, text):
    assert _is_duplicate_refusal(status, text) is True


@pytest.mark.parametrize("status, text", [
    # Other business refusals must NOT be mistaken for duplicates -- retrying
    # against a fresh reference would hide a real finding.
    (400, '{"detail":"Guest count exceeds the combined capacity of the rooms"}'),
    (400, '{"detail":"Arrival date must not fall after departure"}'),
    (422, '{"detail":[{"loc":["body","email"],"msg":"value is not a valid email"}]}'),
    (404, '{"detail":"Booking not found"}'),
    # A server failure is a concrete defect, never a duplicate.
    (500, "sqlalchemy.exc.OperationalError: no such table: reservedroom"),
    # Success is not a refusal.
    (201, '{"id": 3}'),
])
def test_other_outcomes_are_not_duplicates(status, text):
    assert _is_duplicate_refusal(status, text) is False


def test_the_match_is_case_insensitive():
    """Apps capitalise inconsistently; the old rule was case-sensitive."""
    assert _is_duplicate_refusal(400, '{"error":"RELATIONSHIP ALREADY EXISTS"}') is True
    assert _is_duplicate_refusal(409, "Unique Constraint Failed: x") is True


def test_a_bare_400_is_not_enough():
    """The refusal has to name duplication. Treating every 400 as a collision
    would retry past genuine business rules and manufacture false passes."""
    assert _is_duplicate_refusal(400, '{"detail":"Bad Request"}') is False
    assert _is_duplicate_refusal(400, "") is False
