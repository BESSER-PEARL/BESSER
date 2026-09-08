"""Robustness tests for the class-diagram ``buml_to_json`` path used by the
``/get-svg`` (and ``/get-json-model``) endpoints.

Covers two failures that surfaced as opaque ``500``/``400`` responses:

* an attribute whose ``default_value`` is an ``EnumerationLiteral`` (or any
  non-primitive metamodel object) made the diagram JSON un-serialisable, so the
  render POST raised ``TypeError`` -> 500;
* a leading UTF-8 BOM (``\\ufeff``) tripped the sandboxed parser with
  "invalid non-printable character U+FEFF" -> 400.
"""

import json

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Enumeration, EnumerationLiteral,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
    parse_buml_content, class_buml_to_json,
)


def test_enum_literal_default_value_is_json_serialisable():
    """An EnumerationLiteral default must be coerced to its name, not stored raw."""
    low = EnumerationLiteral(name="LOW")
    high = EnumerationLiteral(name="HIGH")
    priority = Enumeration(name="Priority", literals={low, high})
    level = Property(name="level", type=priority, default_value=low)
    task = Class(name="Task", attributes={level})
    model = DomainModel(name="Tracker", types={task, priority})

    diagram_json = class_buml_to_json(model)

    default_value = next(
        e["defaultValue"]
        for e in diagram_json["elements"].values()
        if e.get("name") == "level" and "defaultValue" in e
    )
    assert default_value == "LOW"
    # The whole payload (as POSTed to the render service) must serialise.
    json.dumps({"model": diagram_json, "autoLayout": True})


def test_primitive_default_value_is_preserved():
    """A plain string/number default must pass through unchanged."""
    name = Property(name="name", type="str", default_value="anon")
    person = Class(name="Person", attributes={name})
    model = DomainModel(name="M", types={person})

    diagram_json = class_buml_to_json(model)
    default_value = next(
        e["defaultValue"]
        for e in diagram_json["elements"].values()
        if e.get("name") == "name" and "defaultValue" in e
    )
    assert default_value == "anon"


def test_parse_buml_content_strips_leading_bom():
    """A file saved with a UTF-8 BOM must parse instead of raising a syntax error."""
    body = (
        'name_attr = Property(name="name", type=StringType)\n'
        'Person = Class(name="Person", attributes={name_attr})\n'
        'domain_model = DomainModel(name="M", types={Person})\n'
    )
    with_bom = "﻿" + body

    model = parse_buml_content(with_bom)

    assert model is not None
    assert {c.name for c in model.get_classes()} == {"Person"}
