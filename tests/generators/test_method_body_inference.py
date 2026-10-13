"""Method bodies inferred from the class shape (method_body.py.j2).

A modeled method without code gets a minimal correct body when its name says
what it does (getter, setter, predicate, clear, __str__) and an honest
``raise NotImplementedError`` otherwise. The Python and Django copies of the
macro must agree, including on inherited attributes.
"""
import ast

import pytest

from besser.BUML.metamodel.structural import (
    BooleanType, Class, DomainModel, Generalization, Method, Parameter, Property, StringType,
)
from besser.generators.django import DjangoGenerator
from besser.generators.python_classes import PythonGenerator


@pytest.fixture
def model():
    base = Class(name="Base", attributes={Property(name="code", type=StringType)})
    item = Class(name="Item", attributes={
        Property(name="title", type=StringType),
        Property(name="active", type=BooleanType),
    }, methods={
        Method(name="get_title"),
        Method(name="set_title", parameters=[Parameter(name="value", type=StringType)]),
        Method(name="is_active"),
        Method(name="has_title"),
        Method(name="clear_title"),
        Method(name="get_code"),
        Method(name="ship"),
        Method(name="get_missing"),
    })
    empty = Class(name="Empty", methods={Method(name="__str__")})
    return DomainModel(name="Shop", types={base, item, empty},
                       generalizations={Generalization(general=base, specific=item)})


def _method(source: str, name: str) -> str:
    """The body of ``def name(...)``, up to the next def/class."""
    start = source.index(f"def {name}(")
    rest = source[start:].split("\n", 1)[1]
    body = []
    for line in rest.split("\n"):
        if line.strip().startswith(("def ", "class ", "@")):
            break
        body.append(line.strip())
    return "\n".join(b for b in body if b)


@pytest.fixture
def python_source(tmp_path, model):
    PythonGenerator(model, output_dir=str(tmp_path)).generate()
    return (tmp_path / "classes.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("name, body", [
    ("get_title", "return self.title"),
    ("set_title", "self.title = value"),
    ("is_active", "return self.active"),          # bool attribute: returned as-is
    ("has_title", "return bool(self.title)"),     # other types: truthiness
    ("clear_title", "self.title = None"),
    ("get_code", "return self.code"),             # inherited attribute
    ("ship", 'raise NotImplementedError("Implement ship based on your business rules")'),
    ("get_missing", 'raise NotImplementedError("Implement get_missing based on your business rules")'),
])
def test_python_infers_body_from_name(python_source, name, body):
    assert _method(python_source, name) == body


def test_python_str_without_printable_attributes_falls_back_to_id(python_source):
    assert _method(python_source, "__str__") == 'return f"Empty(id={id(self)})"'


def test_django_agrees_with_python_and_falls_back_to_pk(tmp_path, model):
    gen = DjangoGenerator(model, project_name="proj", app_name="app", output_dir=str(tmp_path))
    (tmp_path / "proj" / "app").mkdir(parents=True)
    gen.generate_models()
    source = (tmp_path / "proj" / "app" / "models.py").read_text(encoding="utf-8")
    # Consecutive methods used to be glued onto one line ("return self.x    def y(self):").
    ast.parse(source)
    body = lambda name: _method(source, name).split('"""')[-1].strip()  # drop the docstring
    assert body("get_code") == "return self.code"
    assert body("is_active") == "return self.active"
    assert 'return f"Empty(pk={self.pk})"' in source
