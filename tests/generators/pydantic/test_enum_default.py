"""Regression: an enum attribute default must not double the enum name.

A model can carry an enum default stored either QUALIFIED ("Priority.MEDIUM",
as the visual editor / an LLM-assembled spec often does) or as a bare member
("MEDIUM"). The Pydantic template used to always prepend the enum name, so a
qualified default produced::

    priority: Priority = Priority.Priority.MEDIUM

which crashed the generated backend on import::

    AttributeError: Priority

The template now prepends the enum name only when it isn't already there.
"""
import os
import tempfile

import pytest

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    Property,
)
from besser.generators.pydantic_classes.pydantic_classes_generator import (
    PydanticGenerator,
)


def _generate(default_value):
    prio = Enumeration(
        name="Priority",
        literals={EnumerationLiteral(name="MEDIUM"), EnumerationLiteral(name="HIGH")},
    )
    attr = Property(name="priority", type=prio, is_id=False)
    attr.default_value = default_value
    cls = Class(name="TodoItem", attributes={attr})
    model = DomainModel(name="m", types={cls, prio})
    out_dir = tempfile.mkdtemp()
    PydanticGenerator(model=model, output_dir=out_dir).generate()
    with open(os.path.join(out_dir, "pydantic_classes.py"), encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("stored", ["Priority.MEDIUM", "MEDIUM"])
def test_enum_default_is_valid_and_not_doubled(stored):
    code = _generate(stored)
    compile(code, "pydantic_classes.py", "exec")           # must import cleanly
    assert "Priority.Priority" not in code                  # not doubled
    assert "= Priority.MEDIUM" in code                      # resolves to the member
