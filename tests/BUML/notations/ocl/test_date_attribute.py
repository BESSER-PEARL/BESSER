"""Regression test for the ``date`` keyword collision.

``date`` is a reserved B-OCL token (``DATE : 'Date' | 'date'``, it opens
``Date::now()``), so before the grammar fix the lexer never produced an ``ID``
for it and two positions were unreachable:

* ``self.date`` — ``dotNavigation`` expects ``ID``, so any model with an
  attribute named ``date`` failed to parse. Ontologies declare one constantly
  (``dcterms:date``), and the KG importer maps it straight through.
* ``oclIsTypeOf(date)`` — ``typeRef`` accepted the builtin type keywords or an
  ``ID``, and ``date`` is neither. BUML's own ``DateType`` is named ``date``,
  so *every* model with a date-typed attribute could hit this.

This is the same collision, and the same fix, as ``size`` in
:mod:`tests.BUML.metamodel.ocl.test_size_attribute` (BESSER-PEARL/BESSER#198).
"""

import pytest

from besser.BUML.metamodel.ocl.ocl import PropertyCallExpression
from besser.BUML.metamodel.structural import (
    Class, DateType, DomainModel, Generalization, IntegerType, Property, StringType,
)
from besser.BUML.notations.ocl import parse_ocl
from besser.utilities.web_modeling_editor.backend.services.validators.ocl_checker import (
    _parse_only,
)


def _model() -> DomainModel:
    """``Record(date, created, title)`` with ``Archived`` inheriting from it."""
    record = Class(name="Record", attributes={
        Property(name="date", type=DateType),
        Property(name="created", type=DateType),
        Property(name="title", type=StringType),
        Property(name="pages", type=IntegerType),
    })
    archived = Class(name="Archived")
    model = DomainModel(name="Library", types={record, archived})
    model.generalizations = {Generalization(general=record, specific=archived)}
    return model


@pytest.mark.parametrize("expr", [
    # Direct attribute named `date`.
    "context Record inv: self.date > 0",
    # Same name, inherited — proves it is the keyword, not the hierarchy.
    "context Archived inv: self.date > 0",
    # `Date` spelled with a capital, the lexer's other alternative.
    "context Record inv: self.Date > 0",
    # The shape the KG emits for owl:subDataPropertyOf over a `date` property.
    "context Record inv: self.created->asSet()->forAll(v | self.date->asSet()->includes(v))",
    # A `date` type reference, which BUML's DateType makes unavoidable.
    "context Record inv: self.title->forAll(v | v.oclIsTypeOf(date) or v.oclIsTypeOf(datetime))",
    "context Record inv: self.title->forAll(v | v.oclIsKindOf(Date))",
    "context Record inv: self.title->forAll(v | v.oclAsType(date) = v)",
])
def test_date_keyword_positions_parse(expr):
    """Every one of these failed before ``DATE`` was added to the grammar."""
    _parse_only(expr)


@pytest.mark.parametrize("expr", [
    # `Date::now()` / `Date::today()` must keep working: the new alternatives
    # must not shadow `dateFuncExpr`.
    "context Record inv: Date::now()",
    "context Record inv: Date::today()",
])
def test_date_function_calls_still_work(expr):
    _parse_only(expr)


def test_date_navigation_resolves_to_the_property():
    """Parsing is not enough — the fallback must bind the real ``Property``."""
    model = _model()
    constraint = parse_ocl("context Record inv: self.date > 0", model)
    left = constraint.ast.arguments[0]
    assert isinstance(left, PropertyCallExpression)
    assert left.property.name == "date"
    assert left.property.type is DateType
