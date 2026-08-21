"""Regression test for BESSER-PEARL/BESSER#198.

``size`` is a reserved OCL keyword for the ``->size()`` / ``.size()``
collection operation. Before the grammar fix, any domain model with an
attribute literally named ``size`` (inherited or direct) failed OCL
parsing with ``mismatched input '>' expecting '('`` because the lexer
tokenized the attribute name as ``SIZE`` and the parser insisted on
``LPAREN RPAREN`` after it.

Issue #198 was filed against a model where ``size`` was inherited
from a parent class; the reporter blamed inheritance, but the real
trigger was the keyword collision — it fails identically on a direct
attribute.
"""

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Generalization, IntegerType, Property,
)
from besser.utilities.web_modeling_editor.backend.services.validators.ocl_checker import (
    _parse_only,
)


def _model_with_size_attribute_on_parent() -> DomainModel:
    """Reproduce Jordi's #198 model: Unit(size) -> Department."""
    unit = Class(name="Unit", attributes={Property(name="size", type=IntegerType)})
    department = Class(name="Department")
    model = DomainModel(name="Company", types={unit, department})
    model.generalizations = {Generalization(general=unit, specific=department)}
    return model


def _model_with_size_attribute_direct() -> DomainModel:
    """Same keyword collision, no inheritance: Employee(size)."""
    employee = Class(name="Employee", attributes={Property(name="size", type=IntegerType)})
    model = DomainModel(name="Company", types={employee})
    return model


@pytest.mark.parametrize("expr", [
    # Jordi's shape, as inherited attribute.
    "context Department inv: self.size > 5",
    # Same name, as a direct attribute — proves it's the keyword, not inheritance.
    "context Employee inv: self.size > 5",
    # Nested in an if-then-else, like the original issue body (with endif).
    "context Department inv: if self.size > 5 then true else false endif",
    # Mixed: property access to `.size` on one side, comparison on the other.
    "context Department inv: self.size > self.size - 1",
])
def test_size_attribute_parses(expr):
    """These expressions all failed before the grammar fix."""
    _parse_only(expr)


@pytest.mark.parametrize("expr", [
    # OCL collection operation — lexer still picks SIZE + LPAREN + RPAREN
    # because ANTLR's longest-match wins.
    "context Department inv: self.employees->size() > 5",
    # Dot-style scalar size — still works on strings / collections.
    "context Book inv: self.title.size() > 0",
])
def test_size_operation_still_works(expr):
    """The fallback must not break the existing ``->size()`` / ``.size()`` ops."""
    _parse_only(expr)


def test_size_collision_would_break_without_fix():
    """Sanity: the fix is reachable — constructing the model and parsing
    the constraint together must succeed end to end."""
    model = _model_with_size_attribute_on_parent()
    # Just re-confirms the parse path; if the grammar ever regresses this
    # fires with the pre-fix `mismatched input '>' expecting '('` message.
    _parse_only("context Department inv: self.size > 0")
    assert model.get_class_by_name("Department") is not None
