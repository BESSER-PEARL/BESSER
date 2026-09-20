"""A derived enum column must not ship NOT NULL with no way to fill it.

Commit 143b7656 removed every ``is_derived`` attribute from the backend
*Create* schema (the server owns them) and gave the derived columns whose
type has an unambiguous zero a server-side default. Enums were excluded on
purpose: the hotel model's ``BookingPhysicalStatus`` would have defaulted to
``CHECKED_IN`` while the spec says a booking "starts out with the guests not
yet arrived", so every new booking would have shipped already checked in.

The consequence was not a wrong initial state, it was no booking at all. The
column stayed ``NOT NULL`` with no default and nothing assigning it, so the
ORM sent ``None`` and SQLite refused the INSERT::

    NOT NULL constraint failed: booking.commercialStatus

measured on POST /booking/ (hotel) and POST /order/ (inventory) on every raw
generated app. The column is now nullable instead: a derived attribute has no
client-supplied value at INSERT time, so NULL is what "not yet computed"
looks like, and it matches the field's absence from the Create schema. It
invents no semantic initial state — that is still the spec's to give, and
the gap analyser still asks the agent for it.
"""

import os

import pytest

from besser.BUML.metamodel.structural import (
    BooleanType, Class, DateTimeType, DomainModel, Enumeration,
    EnumerationLiteral, FloatType, IntegerType, Property, StringType,
)
from besser.generators.sql_alchemy import SQLAlchemyGenerator


def _status_enum() -> Enumeration:
    return Enumeration(name="BookingPhysicalStatus", literals={
        EnumerationLiteral(name="CHECKED_IN"),
        EnumerationLiteral(name="CHECKED_OUT"),
        EnumerationLiteral(name="NOT_YET_ARRIVED"),
    })


def _generate(model: DomainModel, tmp_path) -> str:
    SQLAlchemyGenerator(model=model, output_dir=str(tmp_path)).generate()
    with open(os.path.join(str(tmp_path), "sql_alchemy.py"), encoding="utf-8") as f:
        return f.read()


def _column_line(code: str, attribute: str) -> str:
    for line in code.splitlines():
        if line.strip().startswith(f"{attribute}: Mapped_["):
            return line.strip()
    raise AssertionError(f"no column emitted for {attribute}\n{code}")


def _booking_model(**status_kwargs) -> DomainModel:
    status = _status_enum()
    booking = Class(name="Booking")
    booking.attributes = {
        Property(name="bookingNumber", type=StringType),
        Property(name="physicalStatus", type=status, is_derived=True, **status_kwargs),
    }
    return DomainModel(name="Hotel", types={booking, status})


class TestTheDerivedEnumColumn:

    def test_is_nullable(self, tmp_path):
        """FAILS before the fix: emitted ``mapped_column(Enum(...))`` with no
        nullable and no default, so the INSERT violated NOT NULL."""
        line = _column_line(_generate(_booking_model(), tmp_path), "physicalStatus")
        assert "nullable=True" in line

    def test_is_typed_optional(self, tmp_path):
        """The annotation has to agree with the column, or SQLAlchemy 2.0's
        ``Mapped[]`` and the DDL disagree about the same attribute."""
        line = _column_line(_generate(_booking_model(), tmp_path), "physicalStatus")
        assert "Mapped_[Optional_[BookingPhysicalStatus]]" in line

    def test_is_not_defaulted_to_a_literal(self, tmp_path):
        """The heuristic that was tried and measured: literal order is not a
        decision, and CHECKED_IN is the wrong booking to ship."""
        line = _column_line(_generate(_booking_model(), tmp_path), "physicalStatus")
        assert "default=" not in line

    def test_an_explicit_model_default_still_wins(self, tmp_path):
        """A modeller who set an initial state has already decided; the column
        keeps it and stays NOT NULL."""
        line = _column_line(
            _generate(_booking_model(default_value="NOT_YET_ARRIVED"), tmp_path),
            "physicalStatus")
        assert "default=BookingPhysicalStatus.NOT_YET_ARRIVED" in line
        assert "nullable=True" not in line


class TestTheOtherDerivedTypesAreUnchanged:
    """Only the types with no zero move; 143b7656's defaults stay as they were."""

    @pytest.mark.parametrize("type_obj,expected", [
        (IntegerType, "default=0"),
        (FloatType, "default=0"),
        (StringType, 'default=""'),
        (BooleanType, "default=False"),
    ])
    def test_a_zeroable_derived_attribute_keeps_its_server_default(
            self, tmp_path, type_obj, expected):
        cls = Class(name="Booking")
        cls.attributes = {Property(name="derived", type=type_obj, is_derived=True)}
        line = _column_line(_generate(DomainModel(name="M", types={cls}), tmp_path),
                            "derived")
        assert expected in line
        assert "nullable=True" not in line

    def test_an_optional_derived_attribute_is_not_given_a_zero(self, tmp_path):
        """``is_optional`` already means nullable, and NULL there means unknown
        — filling it with 0 would be a different claim."""
        cls = Class(name="Booking")
        cls.attributes = {
            Property(name="derived", type=IntegerType, is_derived=True, is_optional=True),
        }
        line = _column_line(_generate(DomainModel(name="M", types={cls}), tmp_path),
                            "derived")
        assert "nullable=True" in line
        assert "default=" not in line

    def test_a_derived_audit_timestamp_keeps_its_server_stamp(self, tmp_path):
        """createdAt has its own default; it must not be diverted to nullable."""
        cls = Class(name="Booking")
        cls.attributes = {
            Property(name="createdAt", type=DateTimeType, is_derived=True),
        }
        line = _column_line(_generate(DomainModel(name="M", types={cls}), tmp_path),
                            "createdAt")
        assert "default=dt_datetime.now" in line
        assert "nullable=True" not in line

    def test_a_non_derived_enum_stays_required(self, tmp_path):
        """The client supplies it, so NOT NULL is still the right column."""
        status = _status_enum()
        cls = Class(name="Booking")
        cls.attributes = {Property(name="physicalStatus", type=status)}
        line = _column_line(
            _generate(DomainModel(name="M", types={cls, status}), tmp_path),
            "physicalStatus")
        assert "nullable=True" not in line
        assert "default=" not in line
