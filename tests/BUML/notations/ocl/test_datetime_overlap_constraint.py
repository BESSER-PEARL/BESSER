"""Regression tests for BESSER-PEARL/BESSER#400.

The scheduling invariant from the issue combines several constructs:
``X.allInstances()`` (dot form), a two-iterator ``forAll``, ``implies``,
``<>`` between objects, navigation through an association end and then an
attribute, and comparisons of ``DateTime`` attributes::

    context Session inv NoRoomTimeOverlap:
      Session.allInstances()->forAll(s1, s2 |
        s1 <> s2 implies
          s1.room <> s2.room or
          s1.scheduleSlot.endTime <= s2.scheduleSlot.startTime or
          s1.scheduleSlot.startTime >= s2.scheduleSlot.endTime)

The editor rejected it because the grammar had no ``expr.allInstances()``
alternative (only ``Class::allInstances()``); DateTime itself was never the
problem. These tests pin the parse and the shape of the resulting AST, and
the base visitor that the B-OCL evaluator (``bocl.OCLWrapper``) uses.
"""

import pytest
from antlr4 import CommonTokenStream, InputStream

from besser.BUML.metamodel.object import AttributeLink, DataValue, Object, ObjectModel
from besser.BUML.metamodel.ocl.ocl import (
    LoopExp, OperationCallExpression, PropertyCallExpression, TypeExp, VariableExp,
)
from besser.BUML.metamodel.ocl.predicates import is_comparison, is_implies, is_or
from besser.BUML.metamodel.ocl.walk import walk
from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, Constraint, DateTimeType, DomainModel, Multiplicity,
    Property, StringType,
)
from besser.BUML.notations.ocl.api import parse_ocl
from besser.BUML.notations.ocl.BOCLLexer import BOCLLexer
from besser.BUML.notations.ocl.BOCLParser import BOCLParser
from besser.BUML.notations.ocl.visitor import BOCLVisitorImpl

OVERLAP_OCL = """context Session
inv NoRoomTimeOverlap:
  Session.allInstances()->forAll(s1, s2 |
    s1 <> s2 implies
      s1.room <> s2.room or
      s1.scheduleSlot.endTime <= s2.scheduleSlot.startTime or
      s1.scheduleSlot.startTime >= s2.scheduleSlot.endTime
  )"""


@pytest.fixture
def conference():
    """Event / Session / Room / ScheduleSlot, as in the issue."""
    event = Class("Event", attributes={Property("title", StringType)})
    room = Class("Room", attributes={Property("name", StringType)})
    slot = Class("ScheduleSlot", attributes={
        Property("startTime", DateTimeType), Property("endTime", DateTimeType),
    })
    session = Class("Session", attributes={Property("title", StringType)})
    associations = {
        BinaryAssociation("SessionRoom", ends={
            Property("room", room, multiplicity=Multiplicity(1, 1)),
            Property("sessions", session, multiplicity=Multiplicity(0, "*")),
        }),
        BinaryAssociation("SessionSlot", ends={
            Property("scheduleSlot", slot, multiplicity=Multiplicity(1, 1)),
            Property("slotSessions", session, multiplicity=Multiplicity(0, "*")),
        }),
        BinaryAssociation("EventSession", ends={
            Property("event", event, multiplicity=Multiplicity(1, 1)),
            Property("eventSessions", session, multiplicity=Multiplicity(0, "*")),
        }),
    }
    return DomainModel("Conference", types={event, room, slot, session},
                       associations=associations)


def _cls(model, name):
    return next(t for t in model.types if t.name == name)


def test_issue_constraint_parses(conference):
    constraint = parse_ocl(OVERLAP_OCL, conference)
    assert constraint.context is _cls(conference, "Session")


def test_issue_constraint_ast_shape(conference):
    ast = parse_ocl(OVERLAP_OCL, conference).ast
    session = _cls(conference, "Session")

    # Session.allInstances()->forAll(s1, s2 | ...)
    assert isinstance(ast, LoopExp) and ast.name == "forAll"
    assert isinstance(ast.source, OperationCallExpression)
    assert ast.source.operation == "ALLInstances"
    assert isinstance(ast.source.source, TypeExp) and ast.source.source.name == "Session"
    assert [it.name for it in ast.iterator] == ["s1", "s2"]
    assert all(it.type is session for it in ast.iterator)

    # s1 <> s2 implies (... or ... or ...): implies binds loosest.
    body = ast.body[0]
    assert is_implies(body)
    guard, disjunction = body.arguments
    assert is_comparison(guard) and guard.operation == "<>"
    assert all(isinstance(a, VariableExp) for a in (guard.arguments[0], guard.arguments[2]))
    assert is_or(disjunction)

    # Every DateTime comparison navigates sN.scheduleSlot.<time attribute>.
    comparisons = [n for n in walk(body) if is_comparison(n)]
    assert [c.operation for c in comparisons] == ["<>", "<>", "<=", ">="]
    for comparison in comparisons[2:]:
        for operand in (comparison.arguments[0], comparison.arguments[2]):
            assert isinstance(operand, PropertyCallExpression)
            assert operand.property.type is DateTimeType
            assert operand.source.property.name == "scheduleSlot"
            assert operand.source.source.name in {"s1", "s2"}


def _base_visitor_ast(text, model, context):
    parser = BOCLParser(CommonTokenStream(BOCLLexer(InputStream(text))))
    return BOCLVisitorImpl(model, None, context).visit(parser.oclFile())


@pytest.mark.parametrize("form", ["Session.allInstances()", "Session::allInstances()"])
def test_base_visitor_builds_allinstances_for_both_forms(conference, form):
    """``bocl.OCLWrapper`` builds its AST with ``BOCLVisitorImpl``; the dot
    form used to come back as ``None`` there, so the loop iterated nothing."""
    ast = _base_visitor_ast(
        f"context Session inv: {form}->forAll(s | s.title <> '')",
        conference, _cls(conference, "Session"),
    )
    assert isinstance(ast, LoopExp)
    assert isinstance(ast.source, OperationCallExpression)
    assert ast.source.operation == "ALLInstances"
    assert ast.source.source.name == "Session"


def _session_objects(conference, titles):
    session = _cls(conference, "Session")
    title = next(a for a in session.attributes if a.name == "title")
    return [Object(f"s{i}", session, slots=[AttributeLink(DataValue(StringType, t), title)])
            for i, t in enumerate(titles)]


@pytest.mark.parametrize("titles, expected", [(["a", "b"], True), (["a", ""], False)])
def test_bocl_evaluates_dot_form_allinstances(conference, titles, expected):
    """End to end through the external evaluator: the dot form now iterates."""
    OCLWrapper = pytest.importorskip("bocl.OCLWrapper").OCLWrapper
    om = ObjectModel("om", objects=set(_session_objects(conference, titles)))
    constraint = Constraint(
        "titled", _cls(conference, "Session"),
        "context Session inv titled: Session.allInstances()->forAll(s | s.title <> '')", "OCL",
    )
    assert OCLWrapper(conference, om).evaluate(constraint) is expected

