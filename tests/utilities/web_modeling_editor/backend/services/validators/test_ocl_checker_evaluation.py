"""``check_ocl_constraint`` on invariants the B-OCL evaluator cannot evaluate.

Regression tests for BESSER-PEARL/BESSER#400. The scheduling invariant from
the issue parses, but the external evaluator (``bocl.OCLWrapper``, ``bocl``
1.0.1) cannot evaluate it against an object diagram: it binds both iterators
of ``forAll(s1, s2 | ...)`` to the same element, cannot compare objects, and
splices ``datetime`` values into its ``eval()`` string as
``2026-01-01 09:00:00`` (a Python ``SyntaxError``). Before the fix the checker
reported such constraints either as a bogus ``Evaluates to: ()`` success or
as an invalid constraint with a Python error; now it reports them as
syntactically valid, not evaluated, with the reason as a warning.
"""

from datetime import datetime

import pytest

pytest.importorskip("bocl.OCLWrapper")

from besser.BUML.metamodel.object import (  # noqa: E402
    AttributeLink, DataValue, Link, LinkEnd, Object, ObjectModel,
)
from besser.BUML.metamodel.structural import (  # noqa: E402
    BinaryAssociation, Class, Constraint, DateTimeType, DateType, DomainModel, IntegerType,
    Multiplicity, Property, StringType, TimeType,
)
from besser.BUML.notations.ocl.api import parse_ocl  # noqa: E402
from besser.utilities.web_modeling_editor.backend.services.converters.parsers.ocl_parser import (  # noqa: E402
    process_ocl_constraints,
)
from besser.utilities.web_modeling_editor.backend.services.validators import ocl_checker  # noqa: E402
from besser.utilities.web_modeling_editor.backend.services.validators.ocl_checker import (  # noqa: E402
    _evaluation_limitation, check_ocl_constraint,
)

OVERLAP_OCL = """context Session
inv NoRoomTimeOverlap:
  Session.allInstances()->forAll(s1, s2 |
    s1 <> s2 implies
      s1.room <> s2.room or
      s1.scheduleSlot.endTime <= s2.scheduleSlot.startTime or
      s1.scheduleSlot.startTime >= s2.scheduleSlot.endTime
  )"""


class Conference:
    """Room / ScheduleSlot / Session domain model plus an object-model builder."""

    def __init__(self):
        self.title = Property("title", StringType)
        self.capacity = Property("capacity", IntegerType)
        self.day = Property("day", DateType)
        self.opens = Property("opens", TimeType)
        self.start = Property("startTime", DateTimeType)
        self.end = Property("endTime", DateTimeType)
        self.room = Class("Room", attributes={Property("name", StringType)})
        self.slot = Class("ScheduleSlot", attributes={self.start, self.end, self.day, self.opens})
        self.session = Class("Session", attributes={self.title, self.capacity})
        self.room_end = Property("room", self.room, multiplicity=Multiplicity(1, 1))
        self.sessions_end = Property("sessions", self.session, multiplicity=Multiplicity(0, "*"))
        self.slot_end = Property("scheduleSlot", self.slot, multiplicity=Multiplicity(1, 1))
        self.slot_sessions_end = Property("slotSessions", self.session, multiplicity=Multiplicity(0, "*"))
        self.session_room = BinaryAssociation("SessionRoom", ends={self.room_end, self.sessions_end})
        self.session_slot = BinaryAssociation("SessionSlot", ends={self.slot_end, self.slot_sessions_end})
        self.model = DomainModel(
            "Conference", types={self.room, self.slot, self.session},
            associations={self.session_room, self.session_slot},
        )

    def add_constraint(self, text):
        routing, warnings = process_ocl_constraints(text, self.model, 0)
        assert warnings == [] and len(routing) == 1
        self.model.constraints = self.model.constraints | {routing[0][1]}

    def objects(self, second_start, titles=("Keynote", "Workshop")):
        """Two sessions in one room; slot 1 is 09:00-10:00, slot 2 starts at ``second_start``."""
        room = Object("r1", self.room)
        slots = []
        for i, (start, end) in enumerate([(datetime(2026, 3, 2, 9), datetime(2026, 3, 2, 10)),
                                          (second_start, datetime(2026, 3, 2, 11))]):
            slots.append(Object(f"slot{i}", self.slot, slots=[
                AttributeLink(DataValue(DateTimeType, start), self.start),
                AttributeLink(DataValue(DateTimeType, end), self.end),
            ]))
        sessions = [
            Object(f"session{i}", self.session, slots=[
                AttributeLink(DataValue(StringType, title), self.title),
                AttributeLink(DataValue(IntegerType, 10), self.capacity),
            ])
            for i, title in enumerate(titles)
        ]
        for i, (session, slot) in enumerate(zip(sessions, slots)):
            Link(f"room{i}", self.session_room, [
                LinkEnd("room", self.room_end, room), LinkEnd("sessions", self.sessions_end, session)])
            Link(f"slot{i}", self.session_slot, [
                LinkEnd("scheduleSlot", self.slot_end, slot),
                LinkEnd("slotSessions", self.slot_sessions_end, session)])
        return ObjectModel("conference", objects={room, *slots, *sessions})


@pytest.fixture
def conf():
    return Conference()


def test_issue_constraint_is_valid_without_object_model(conf):
    conf.add_constraint(OVERLAP_OCL)
    result = check_ocl_constraint(conf.model)
    assert result["success"] is True
    assert result["invalid_constraints"] == []
    assert len(result["valid_constraints"]) == 1
    assert "NoRoomTimeOverlap" in result["valid_constraints"][0]
    assert result["warnings"] == []


@pytest.mark.parametrize("second_start", [
    datetime(2026, 3, 2, 10),      # back to back: satisfied
    datetime(2026, 3, 2, 9, 30),   # overlapping: violated
], ids=["no-overlap", "overlap"])
def test_issue_constraint_is_reported_as_not_evaluated(conf, second_start):
    """Neither a false violation (Python SyntaxError) nor a bogus success."""
    conf.add_constraint(OVERLAP_OCL)
    result = check_ocl_constraint(conf.model, conf.objects(second_start))

    assert result["success"] is True
    assert result["invalid_constraints"] == []
    [entry] = result["valid_constraints"]
    assert "Syntax valid; not evaluated against the object diagram" in entry
    assert "Evaluates to" not in entry
    [warning] = result["warnings"]
    assert warning.startswith("OCL [Session inv NoRoomTimeOverlap] not evaluated")
    assert "'forAll' with several iterator variables (s1, s2)" in warning
    assert "comparing objects ('s1 <> s2')" in warning
    assert "comparing datetime values" in warning


@pytest.mark.parametrize("titles, expected_valid", [
    (("Keynote", "Workshop"), True),
    (("Keynote", ""), False),
])
def test_dot_form_allinstances_still_evaluates(conf, titles, expected_valid):
    """Constraints the evaluator supports keep being evaluated (and the dot
    form ``Session.allInstances()`` now iterates the instances)."""
    conf.add_constraint(
        "context Session inv titled: Session.allInstances()->forAll(s | s.title <> '')"
    )
    result = check_ocl_constraint(conf.model, conf.objects(datetime(2026, 3, 2, 10), titles))
    assert result["warnings"] == []
    assert result["success"] is expected_valid
    if expected_valid:
        assert "Evaluates to: True" in result["valid_constraints"][0]
    else:
        assert "Constraint violation" in result["invalid_constraints"][0]


def test_non_boolean_result_is_not_reported_as_valid(conf, monkeypatch):
    """A non-Boolean evaluator result used to be shown as ``Evaluates to: ()``."""
    conf.add_constraint("context Session inv positive: self.capacity > 0")
    monkeypatch.setattr(ocl_checker.OCLWrapper, "evaluate", lambda self, c: ())
    result = check_ocl_constraint(conf.model, conf.objects(datetime(2026, 3, 2, 10)))
    assert result["success"] is True
    assert "Evaluates to" not in result["valid_constraints"][0]
    assert result["warnings"] == [
        "OCL [Session inv positive] not evaluated against the object diagram: "
        "the evaluator returned a non-Boolean value (())"
    ]


def test_evaluator_crash_on_valid_ocl_is_not_a_user_error(conf, monkeypatch):
    """Valid B-OCL the evaluator cannot handle (e.g. ``->isEmpty()`` in bocl 1.0.1,
    which crashed with "invalid syntax (<string>, line 1)") used to be reported as
    an error in the user's constraint."""
    conf.add_constraint("context Session inv positive: self.capacity > 0")

    def crash(self, constraint):
        raise SyntaxError("invalid syntax (<string>, line 1)")

    monkeypatch.setattr(ocl_checker.OCLWrapper, "evaluate", crash)
    result = check_ocl_constraint(conf.model, conf.objects(datetime(2026, 3, 2, 10)))
    assert result["success"] is True
    assert result["invalid_constraints"] == []
    assert result["warnings"] == [
        "OCL [Session inv positive] not evaluated against the object diagram: "
        "the evaluator does not support it yet (invalid syntax (<string>, line 1))"
    ]


def test_unknown_property_is_still_an_error(conf):
    """A constraint that does not parse stays an error (not 'not evaluated')."""
    conf.model.constraints = {Constraint(
        name="typo", context=conf.session, expression="context Session inv typo: self.capacityy > 0",
        language="OCL",
    )}
    result = check_ocl_constraint(conf.model, conf.objects(datetime(2026, 3, 2, 10)))
    assert result["success"] is False
    assert "capacityy" in result["invalid_constraints"][0]


def _limitation(conf, body):
    return _evaluation_limitation(parse_ocl(f"context Session inv: {body}", conf.model).ast)


@pytest.mark.parametrize("body, expected", [
    ("Session.allInstances()->forAll(s | s.room = self.room)", "'self' inside a 'forAll' body"),
    ("Session.allInstances()->forAll(s1 | Session.allInstances()->forAll(s2 | s1.title = s2.title))",
     "a nested 'forAll' that refers to the outer iterator variable 's1'"),
    ("self.room <> self.room", "comparing objects ('self.room <> self.room')"),
    ("self.scheduleSlot.opens < self.scheduleSlot.opens", "comparing time values"),
    ("self.scheduleSlot.startTime < self.scheduleSlot.endTime", "comparing datetime values"),
    ("self.scheduleSlot.day >= Date::today()", "comparing date values"),
])
def test_unsupported_constructs_are_detected(conf, body, expected):
    reason = _limitation(conf, body)
    assert reason is not None and expected in reason
    assert reason.endswith("not supported by the evaluator yet")


@pytest.mark.parametrize("body", [
    "self.capacity > 0",
    "self.title <> ''",
    "Session.allInstances()->forAll(s | s.capacity > 0)",
    "Session.allInstances()->exists(s | s.title = 'x')",
])
def test_supported_constructs_are_not_flagged(conf, body):
    assert _limitation(conf, body) is None
