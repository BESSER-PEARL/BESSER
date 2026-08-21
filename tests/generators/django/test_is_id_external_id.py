"""Tests that `is_id` / `is_external_id` flow through the Django generator.

Covers:
- `is_id=True` -> `primary_key=True` in the generated field
- `is_external_id=True` (non-PK) -> `unique=True` in the generated field
- `is_external_id` attributes drive the generated `__str__` (issue #225)
- Composite external id: `__str__` joins all of them

We compile the rendered ``models.py`` to validate syntax; we don't spin up
Django itself because the backend tests should stay lightweight. The field
calls are introspected as plain strings.
"""

import os

import pytest
from jinja2 import Environment, FileSystemLoader

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Enumeration, IntegerType, PrimitiveDataType, Property,
    StringType,
)
from besser.utilities import sort_by_timestamp


@pytest.fixture
def render_models():
    """Return a render() function bound to the Django models template."""
    templates_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))),
        "besser", "generators", "django", "templates",
    )
    env = Environment(
        loader=FileSystemLoader(templates_path),
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.do"],
    )
    env.tests["is_primitive_data_type"] = lambda v: isinstance(v, PrimitiveDataType)
    env.tests["is_enumeration"] = lambda v: isinstance(v, Enumeration)
    env.globals["sort_by_timestamp"] = sort_by_timestamp
    tpl = env.get_template("models.py.j2")

    def _render(model: DomainModel) -> str:
        return tpl.render(
            model=model, one_to_one={}, fkeys={}, many_to_many={},
        )

    return _render


def _book_with_pk_and_external_id() -> DomainModel:
    """Surrogate PK id + `isbn` as external id + plain `title`."""
    book = Class(name="Book")
    book.attributes = {
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="isbn", type=StringType, is_external_id=True),
        Property(name="title", type=StringType),
    }
    model = DomainModel(name="Shop")
    model.add_type(book)
    return model


def test_is_id_emits_primary_key(render_models):
    rendered = render_models(_book_with_pk_and_external_id())
    # Grab the `id = ...` line (surrogate PK)
    id_line = next(
        line for line in rendered.splitlines()
        if line.strip().startswith("id =")
    )
    assert "primary_key=True" in id_line


def test_external_id_emits_unique(render_models):
    rendered = render_models(_book_with_pk_and_external_id())
    isbn_line = next(
        line for line in rendered.splitlines()
        if line.strip().startswith("isbn =")
    )
    title_line = next(
        line for line in rendered.splitlines()
        if line.strip().startswith("title =")
    )
    assert "unique=True" in isbn_line
    # Plain attribute must stay plain.
    assert "unique=True" not in title_line
    assert "primary_key=True" not in title_line


def test_str_uses_external_id(render_models):
    """__str__ returns the external id, not the first primitive attribute."""
    rendered = render_models(_book_with_pk_and_external_id())
    assert "return str(self.isbn)" in rendered
    # It must not fall back to the surrogate PK or the title.
    assert "return str(self.id)" not in rendered
    assert "return str(self.title)" not in rendered


def test_str_composite_external_id(render_models):
    """With multiple external-id attrs, __str__ joins them with ' - '."""
    person = Class(name="Person")
    person.attributes = {
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="first_name", type=StringType, is_external_id=True),
        Property(name="last_name", type=StringType, is_external_id=True),
    }
    model = DomainModel(name="Directory")
    model.add_type(person)

    rendered = render_models(model)
    assert 'str(x) for x in (self.first_name, self.last_name)' in rendered \
        or 'str(x) for x in (self.last_name, self.first_name)' in rendered
    assert '" - ".join' in rendered


def test_str_fallback_without_external_id(render_models):
    """If no external id is declared, __str__ falls back to the first primitive
    attribute — the pre-existing behavior is preserved."""
    cls = Class(name="Widget")
    cls.attributes = {
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="label", type=StringType),
    }
    model = DomainModel(name="WidgetModel")
    model.add_type(cls)

    rendered = render_models(model)
    # Either the PK ('id') or 'label' may come first by timestamp; both are
    # valid fallbacks — the point is that the composite-external-id branch
    # is NOT used.
    assert '" - ".join' not in rendered
    assert "return str(self." in rendered


def test_rendered_models_are_valid_python(render_models):
    """The generated models.py must be syntactically valid Python so that
    Django's runtime loader can import it."""
    rendered = render_models(_book_with_pk_and_external_id())
    # Compile with a fake filename; raises SyntaxError on bad output.
    compile(rendered, "<rendered-models>", "exec")
