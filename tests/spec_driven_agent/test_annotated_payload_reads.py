"""A handler that annotates its payload was invisible to this check.

``_create_schema_router_mismatches`` recognised a payload only by the
``<entity>_data`` naming convention, so ``async def create(payload:
BookingCreate)`` could read any field it liked and nothing looked. The
annotation is the better signal anyway: it NAMES the schema instead of
guessing it from a variable, so it cannot mis-resolve when the two
disagree.

The census put this class at 7 runs with 3 caught; the residue is handlers
that annotate rather than follow the naming convention.

The risk direction is false positives, because this makes the check see
more code. The last four tests are the boundary.
"""

from besser.spec_driven_agent.validation.python_source import (
    _create_schema_router_mismatches,
)


_SCHEMAS = (
    "from pydantic import BaseModel\n"
    "\n"
    "class BookingCreate(BaseModel):\n"
    "    bookingNumber: str\n"
    "    arrivalDate: date\n"
)


def _write(root, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _app(tmp_path, handler: str, schemas: str = _SCHEMAS):
    _write(tmp_path, "backend/pydantic_classes.py", schemas)
    _write(tmp_path, "backend/routers/booking.py", handler)
    return _create_schema_router_mismatches(str(tmp_path))


def test_an_annotated_payload_reading_an_absent_field_is_caught(tmp_path):
    issues = _app(tmp_path, (
        "from pydantic_classes import BookingCreate\n"
        "\n"
        "async def create_booking(payload: BookingCreate):\n"
        "    return Booking(id=payload.id)\n"
    ))

    assert len(issues) == 1, issues
    assert "`payload.id`" in issues[0]
    assert "BookingCreate does not define `id`" in issues[0]


def test_the_naming_convention_still_works(tmp_path):
    """The unannotated handlers this check was written for must not regress."""
    issues = _app(tmp_path, (
        "async def create_booking(booking_data):\n"
        "    return Booking(id=booking_data.id)\n"
    ))

    assert len(issues) == 1, issues
    assert "`booking_data.id`" in issues[0]


def test_the_annotation_wins_when_it_disagrees_with_the_name(tmp_path):
    """A variable named for one entity but typed as another schema.

    The annotation is authoritative: reading `arrivalDate` is fine because
    BookingCreate declares it, even though the variable says guest.
    """
    issues = _app(tmp_path, (
        "async def create(guest_data: BookingCreate):\n"
        "    return Booking(arrival=guest_data.arrivalDate)\n"
    ))

    assert issues == []


def test_an_annotated_payload_reading_a_declared_field_is_clean(tmp_path):
    assert _app(tmp_path, (
        "async def create_booking(payload: BookingCreate):\n"
        "    return Booking(number=payload.bookingNumber)\n"
    )) == []


def test_a_hasattr_guarded_read_is_still_skipped(tmp_path):
    """The guard that kept a shipped 11/11 app clean must still apply."""
    assert _app(tmp_path, (
        "async def create_booking(payload: BookingCreate):\n"
        "    existing = query.filter(\n"
        "        Booking.id != payload.id"
        " if hasattr(payload, 'id') and payload.id else True\n"
        "    ).all()\n"
        "    return existing\n"
    )) == []


def test_a_parameter_annotated_as_something_else_is_ignored(tmp_path):
    """Only Create schemas are payloads; a Session must not be read as one."""
    assert _app(tmp_path, (
        "async def create_booking(database: Session):\n"
        "    return database.query(Booking).all()\n"
    )) == []


def test_an_unknown_annotation_is_ignored(tmp_path):
    """A type that names nothing in the app cannot be resolved, so stay quiet."""
    assert _app(tmp_path, (
        "async def create_booking(payload: SomethingUnknown):\n"
        "    return payload.whatever\n"
    )) == []


def test_a_local_variable_is_not_treated_as_a_payload(tmp_path):
    """Only function ARGUMENTS are annotated payloads."""
    assert _app(tmp_path, (
        "async def create_booking(payload: BookingCreate):\n"
        "    booking = Booking()\n"
        "    return booking.anything\n"
    )) == []
