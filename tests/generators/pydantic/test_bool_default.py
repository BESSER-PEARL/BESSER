"""Regression: a boolean attribute default must be emitted as valid Python.

A model can carry a boolean default stored lower-case (``"false"`` / ``"true"``),
e.g. from the visual editor or an LLM-assembled spec. The Pydantic template used
to emit it verbatim (only ``str`` and ``Enumeration`` were special-cased), so
``guaranteeRequired: bool = false`` reached the generated ``pydantic_classes.py``
and crashed the backend on import:

    NameError: name 'false' is not defined. Did you mean: 'False'?

The template now normalises boolean defaults to Python ``True`` / ``False``.
"""
import os
import tempfile

import pytest

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
    PrimitiveDataType,
    Property,
)
from besser.generators.pydantic_classes.pydantic_classes_generator import (
    PydanticGenerator,
)


def _generate(default_value, is_optional=False):
    bool_t = PrimitiveDataType("bool")
    attr = Property(name="guaranteeRequired", type=bool_t, is_id=False)
    attr.default_value = default_value
    attr.is_optional = is_optional
    cls = Class(name="Reservation", attributes={attr})
    model = DomainModel(name="m", types={cls, bool_t})
    out_dir = tempfile.mkdtemp()
    PydanticGenerator(model=model, output_dir=out_dir).generate()
    with open(os.path.join(out_dir, "pydantic_classes.py"), encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ("false", "False"),
        ("true", "True"),
        ("False", "False"),
        ("True", "True"),
    ],
)
def test_bool_default_is_valid_python(stored, expected):
    code = _generate(stored)
    # The whole module must be importable (no `false`/`true` NameErrors).
    compile(code, "pydantic_classes.py", "exec")
    assert f"= {expected}" in code
    assert "= false" not in code and "= true" not in code


def test_optional_bool_default_is_valid_python():
    code = _generate("false", is_optional=True)
    compile(code, "pydantic_classes.py", "exec")
    assert "Optional[bool] = False" in code
