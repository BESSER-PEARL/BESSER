"""XSD datatype → BESSER primitive mapping and literal-value parsing."""

from __future__ import annotations

import datetime as _dt
from typing import Any, Optional, Tuple

from besser.BUML.metamodel.structural import (
    AnyType,
    BooleanType,
    DateTimeType,
    DateType,
    FloatType,
    IntegerType,
    PrimitiveDataType,
    StringType,
    TimeDeltaType,
    TimeType,
)


_XSD = "http://www.w3.org/2001/XMLSchema#"

# Map from full XSD datatype IRI to BESSER primitive.
_XSD_TO_PRIMITIVE: dict[str, PrimitiveDataType] = {
    # Strings & string-like
    f"{_XSD}string": StringType,
    f"{_XSD}normalizedString": StringType,
    f"{_XSD}token": StringType,
    f"{_XSD}language": StringType,
    f"{_XSD}Name": StringType,
    f"{_XSD}NCName": StringType,
    f"{_XSD}NMTOKEN": StringType,
    f"{_XSD}QName": StringType,
    f"{_XSD}anyURI": StringType,
    f"{_XSD}hexBinary": StringType,
    f"{_XSD}base64Binary": StringType,
    f"{_XSD}gYear": StringType,
    f"{_XSD}gYearMonth": StringType,
    f"{_XSD}gMonth": StringType,
    f"{_XSD}gDay": StringType,
    f"{_XSD}gMonthDay": StringType,
    # Booleans
    f"{_XSD}boolean": BooleanType,
    # Integers (all integer subtypes collapse to int)
    f"{_XSD}integer": IntegerType,
    f"{_XSD}int": IntegerType,
    f"{_XSD}long": IntegerType,
    f"{_XSD}short": IntegerType,
    f"{_XSD}byte": IntegerType,
    f"{_XSD}nonNegativeInteger": IntegerType,
    f"{_XSD}nonPositiveInteger": IntegerType,
    f"{_XSD}positiveInteger": IntegerType,
    f"{_XSD}negativeInteger": IntegerType,
    f"{_XSD}unsignedInt": IntegerType,
    f"{_XSD}unsignedLong": IntegerType,
    f"{_XSD}unsignedShort": IntegerType,
    f"{_XSD}unsignedByte": IntegerType,
    # Floating-point / decimal
    f"{_XSD}decimal": FloatType,
    f"{_XSD}float": FloatType,
    f"{_XSD}double": FloatType,
    # Temporal
    f"{_XSD}date": DateType,
    f"{_XSD}dateTime": DateTimeType,
    f"{_XSD}dateTimeStamp": DateTimeType,
    f"{_XSD}time": TimeType,
    f"{_XSD}duration": TimeDeltaType,
    # RDF-flavoured "any"
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString": StringType,
    "http://www.w3.org/2000/01/rdf-schema#Literal": StringType,
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#PlainLiteral": StringType,
}


def xsd_to_primitive(datatype_iri: Optional[str]) -> Tuple[PrimitiveDataType, bool]:
    """Map an XSD datatype IRI to a BESSER ``PrimitiveDataType``.

    Returns ``(primitive, is_known)``: when the IRI isn't recognised, falls
    back to :data:`StringType` and ``is_known=False`` so the caller can emit
    a warning.

    A bare/empty datatype yields ``(StringType, True)`` (RDF semantics: an
    untyped literal is a plain string).
    """
    if not datatype_iri:
        return StringType, True
    primitive = _XSD_TO_PRIMITIVE.get(datatype_iri)
    if primitive is not None:
        return primitive, True
    return StringType, False


def _parse_iso_duration(value: str) -> _dt.timedelta:
    """Parse an ISO-8601 duration (``PnYnMnDTnHnMnS``) into a ``timedelta``.

    Years and months don't have a fixed length, so they are approximated
    (year=365 days, month=30 days). This matches the loose semantics expected
    when round-tripping ABox data into a typed BUML model.
    """
    import re

    match = re.match(
        r"^(?P<sign>-)?P"
        r"(?:(?P<years>\d+)Y)?"
        r"(?:(?P<months>\d+)M)?"
        r"(?:(?P<weeks>\d+)W)?"
        r"(?:(?P<days>\d+)D)?"
        r"(?:T"
        r"(?:(?P<hours>\d+)H)?"
        r"(?:(?P<minutes>\d+)M)?"
        r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?"
        r")?$",
        value,
    )
    if not match:
        raise ValueError(f"Invalid ISO-8601 duration: {value!r}")
    parts = match.groupdict()
    sign = -1 if parts["sign"] else 1
    days = (
        int(parts["years"] or 0) * 365
        + int(parts["months"] or 0) * 30
        + int(parts["weeks"] or 0) * 7
        + int(parts["days"] or 0)
    )
    seconds = (
        int(parts["hours"] or 0) * 3600
        + int(parts["minutes"] or 0) * 60
        + float(parts["seconds"] or 0)
    )
    return _dt.timedelta(days=sign * days, seconds=sign * seconds)


def parse_literal(value: str, datatype_iri: Optional[str]) -> Any:
    """Parse a lexical literal into a Python value matching its XSD datatype.

    On parse failure, returns the original string — the caller is responsible
    for downgrading the slot to ``StringType`` and emitting a warning.
    """
    primitive, _known = xsd_to_primitive(datatype_iri)
    if primitive is StringType or primitive is AnyType:
        return value
    name = primitive.name
    try:
        if name == "bool":
            lowered = value.strip().lower()
            if lowered in {"true", "1"}:
                return True
            if lowered in {"false", "0"}:
                return False
            raise ValueError(f"not a boolean: {value!r}")
        if name == "int":
            return int(value)
        if name == "float":
            return float(value)
        if name == "date":
            return _dt.date.fromisoformat(value)
        if name == "datetime":
            # rdflib emits ``2024-01-01T00:00:00+00:00``; ``fromisoformat`` handles it
            # (Python 3.11+ also handles the ``Z`` suffix, but we strip just in case).
            cleaned = value.replace("Z", "+00:00")
            return _dt.datetime.fromisoformat(cleaned)
        if name == "time":
            return _dt.time.fromisoformat(value)
        if name == "timedelta":
            return _parse_iso_duration(value)
    except (ValueError, TypeError):
        return value
    return value


__all__ = ["xsd_to_primitive", "parse_literal"]
