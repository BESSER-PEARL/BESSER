"""XSD datatype → BESSER primitive mapping."""

from __future__ import annotations

from typing import Optional, Tuple

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


#: BUML primitive name → the module-level singleton instance.
#:
#: ``PrimitiveDataType`` whitelists exactly these nine names and rejects
#: anything else, and ``DomainModel.types`` auto-injects the whole set — so a
#: generator must resolve to these singletons rather than constructing its own.
PRIMITIVE_BY_NAME: dict[str, PrimitiveDataType] = {
    p.name: p
    for p in (
        StringType, IntegerType, FloatType, BooleanType,
        TimeType, DateType, DateTimeType, TimeDeltaType, AnyType,
    )
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


__all__ = ["xsd_to_primitive", "PRIMITIVE_BY_NAME"]
