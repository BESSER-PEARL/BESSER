"""Enum literals keep declaration order in every generator.

Django, Pydantic and SQLAlchemy sorted literals by name (LOW/MEDIUM/HIGH became
HIGH/LOW/MEDIUM) while the Python generator kept declaration order.
"""
import os
import re

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Enumeration, EnumerationLiteral, Property,
)
from besser.generators.django import DjangoGenerator
from besser.generators.pydantic_classes import PydanticGenerator
from besser.generators.python_classes import PythonGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator


def _model():
    priority = Enumeration(name="Priority", literals={
        EnumerationLiteral(name=n) for n in ("LOW", "MEDIUM", "HIGH")})
    task = Class(name="Task", attributes={Property(name="priority", type=priority)})
    return DomainModel(name="Tasks", types={task, priority})


def _python(model, out):
    PythonGenerator(model, output_dir=out).generate()
    return os.path.join(out, "classes.py")


def _pydantic(model, out):
    PydanticGenerator(model, output_dir=out).generate()
    return os.path.join(out, "pydantic_classes.py")


def _sqlalchemy(model, out):
    SQLAlchemyGenerator(model, output_dir=out).generate()
    return os.path.join(out, "sql_alchemy.py")


def _django(model, out):
    gen = DjangoGenerator(model, project_name="proj", app_name="app", output_dir=out)
    os.makedirs(gen._app_dir(), exist_ok=True)
    gen.generate_models()
    return os.path.join(gen._app_dir(), "models.py")


@pytest.mark.parametrize("generate", [_python, _pydantic, _sqlalchemy, _django])
def test_literals_in_declaration_order(generate, tmp_path):
    code = open(generate(_model(), str(tmp_path)), encoding="utf-8").read()
    order = list(dict.fromkeys(re.findall(r"\b(LOW|MEDIUM|HIGH)\b", code)))
    assert order == ["LOW", "MEDIUM", "HIGH"], code
