"""``true`` / ``false`` literals must evaluate.

``BOCLVisitorImpl`` stored the lexeme (the string ``"true"``), which the bocl
evaluator wrote verbatim into the Python expression it evaluates, so
``self.isOpen = true`` and every ``... else true endif`` failed with
"name 'true' is not defined".
"""

import pytest

from besser.BUML.metamodel.object import AttributeLink, DataValue, Object, ObjectModel
from besser.BUML.metamodel.ocl.ocl import BooleanLiteralExpression
from besser.BUML.metamodel.ocl.walk import walk
from besser.BUML.metamodel.structural import (
    BooleanType, Class, Constraint, DomainModel, IntegerType, Property,
)
from besser.BUML.notations.ocl.api import parse_ocl
from besser.utilities.web_modeling_editor.backend.services.validators.ocl_checker import (
    check_ocl_constraint,
)

IS_OPEN = Property("isOpen", BooleanType)
SEATS = Property("seats", IntegerType)
LIBRARY = Class("Library", attributes={IS_OPEN, SEATS})


def _check(expression, is_open, seats):
    model = DomainModel("Libraries", types={LIBRARY}, constraints={
        Constraint(name="c", context=LIBRARY, expression=expression, language="OCL"),
    })
    lib = Object("lib", LIBRARY, slots=[
        AttributeLink(attribute=IS_OPEN, value=DataValue(classifier=BooleanType, value=is_open)),
        AttributeLink(attribute=SEATS, value=DataValue(classifier=IntegerType, value=seats)),
    ])
    return check_ocl_constraint(model, ObjectModel("objs", objects={lib}))


@pytest.mark.parametrize(("expression", "holds", "violated"), [
    ("context Library inv: self.isOpen = true", (True, 10), (False, 10)),
    ("context Library inv: self.isOpen <> false", (True, 10), (False, 10)),
    ("context Library inv: if self.isOpen then self.seats > 0 else true endif", (False, 0), (True, 0)),
])
def test_boolean_literals_evaluate(expression, holds, violated):
    ok = _check(expression, *holds)
    assert ok["success"], ok
    assert "Evaluates to: True" in ok["valid_constraints"][0]

    bad = _check(expression, *violated)
    assert not bad["success"], bad
    assert "Constraint violation" in bad["invalid_constraints"][0]


def test_boolean_literal_value_is_a_bool():
    model = DomainModel("Libraries", types={LIBRARY})
    ast = parse_ocl("context Library inv: self.isOpen = false", model).ast
    literals = [n for n in walk(ast) if isinstance(n, BooleanLiteralExpression)]
    assert [lit.value for lit in literals] == [False]
