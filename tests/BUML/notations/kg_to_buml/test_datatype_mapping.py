"""Tests for ``besser.BUML.notations.kg_to_buml.datatype_mapping``."""


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
from besser.BUML.notations.kg_to_buml.datatype_mapping import xsd_to_primitive


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
