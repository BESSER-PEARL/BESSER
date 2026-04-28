"""Tests for ``besser.BUML.notations.kg_to_buml.datatype_mapping``."""

import datetime as dt

import pytest

from besser.BUML.metamodel.structural import (
    BooleanType,
    DateTimeType,
    DateType,
    FloatType,
    IntegerType,
    StringType,
    TimeDeltaType,
    TimeType,
)
from besser.BUML.notations.kg_to_buml.datatype_mapping import parse_literal, xsd_to_primitive


XSD = "http://www.w3.org/2001/XMLSchema#"


@pytest.mark.parametrize(
    "iri, expected_type",
    [
        (XSD + "string", StringType),
        (XSD + "anyURI", StringType),
        (XSD + "token", StringType),
        (XSD + "boolean", BooleanType),
        (XSD + "integer", IntegerType),
        (XSD + "int", IntegerType),
        (XSD + "long", IntegerType),
        (XSD + "short", IntegerType),
        (XSD + "byte", IntegerType),
        (XSD + "nonNegativeInteger", IntegerType),
        (XSD + "positiveInteger", IntegerType),
        (XSD + "negativeInteger", IntegerType),
        (XSD + "nonPositiveInteger", IntegerType),
        (XSD + "unsignedInt", IntegerType),
        (XSD + "decimal", FloatType),
        (XSD + "float", FloatType),
        (XSD + "double", FloatType),
        (XSD + "date", DateType),
        (XSD + "dateTime", DateTimeType),
        (XSD + "dateTimeStamp", DateTimeType),
        (XSD + "time", TimeType),
        (XSD + "duration", TimeDeltaType),
        (XSD + "gYear", StringType),
        (XSD + "hexBinary", StringType),
    ],
)
def test_xsd_to_primitive_maps_known_iris(iri, expected_type):
    primitive, known = xsd_to_primitive(iri)
    assert primitive is expected_type
    assert known is True


def test_xsd_to_primitive_unknown_falls_back_to_string():
    primitive, known = xsd_to_primitive("http://example.org/Unknown")
    assert primitive is StringType
    assert known is False


def test_xsd_to_primitive_empty_returns_string_known():
    primitive, known = xsd_to_primitive(None)
    assert primitive is StringType
    assert known is True


def test_parse_literal_int():
    assert parse_literal("42", XSD + "integer") == 42
    assert parse_literal("-7", XSD + "long") == -7


def test_parse_literal_float():
    assert parse_literal("3.14", XSD + "decimal") == pytest.approx(3.14)


def test_parse_literal_boolean():
    assert parse_literal("true", XSD + "boolean") is True
    assert parse_literal("0", XSD + "boolean") is False


def test_parse_literal_date():
    assert parse_literal("2024-12-31", XSD + "date") == dt.date(2024, 12, 31)


def test_parse_literal_datetime_z_and_offset():
    assert parse_literal("2024-01-15T10:30:00Z", XSD + "dateTime") == dt.datetime(
        2024, 1, 15, 10, 30, tzinfo=dt.timezone.utc
    )


def test_parse_literal_time():
    assert parse_literal("08:15:30", XSD + "time") == dt.time(8, 15, 30)


def test_parse_literal_duration():
    assert parse_literal("P1Y2M3DT4H5M6S", XSD + "duration") == dt.timedelta(
        days=365 + 60 + 3, seconds=4 * 3600 + 5 * 60 + 6
    )


def test_parse_literal_invalid_returns_raw_string():
    # Invalid integer literal — function returns the raw string instead of raising.
    assert parse_literal("not-a-number", XSD + "integer") == "not-a-number"


def test_parse_literal_string_passes_through():
    assert parse_literal("hello", XSD + "string") == "hello"
    assert parse_literal("hello", None) == "hello"
