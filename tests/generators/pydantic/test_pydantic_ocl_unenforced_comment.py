"""
Tests for the unified "unenforced OCL constraint" comment.

Owner's decision: relational OCL is never transpiled into a Pydantic
validator. The deterministic generator's only job for such a constraint is
to leave a comment that hands the rule off to the LLM Spec-Driven Agent,
which implements it in the router handler (after loading related rows
through the database session).

Two, previously very different, situations must both produce that comment:

1. ``skipped`` -- the constraint reaches ``ocl_utils.py`` (it is attached to
   ``domain_model.constraints``) but can't be expressed as a validator on a
   single Create payload (collections, relationships, ...).
2. ``rejected`` -- the constraint never reaches ``ocl_utils.py`` at all: the
   web editor's OCL type-checker rejected it first and recorded it on
   ``domain_model.conversion_issues`` instead. Before this file's fix, this
   case left no trace anywhere in the generated code.

Both comments render from the single ``ocl_utils.SKIP_COMMENT_TEMPLATE``
(the template used to hardcode its own separate copy of the ``skipped``
text and silently ignore this constant; ``conversion_issues`` was not read
at all).
"""

import os
import py_compile

import pytest

from besser.BUML.metamodel.structural import (
    Class, Constraint, DomainModel, IntegerType, Property,
)
from besser.generators.pydantic_classes import PydanticGenerator


def generate(domain_model, output_dir, backend=True):
    """Run the Pydantic generator and return (path, source)."""
    PydanticGenerator(
        model=domain_model, backend=backend, output_dir=str(output_dir)
    ).generate()
    path = os.path.join(str(output_dir), "pydantic_classes.py")
    with open(path, "r", encoding="utf-8") as handle:
        return path, handle.read()


@pytest.fixture
def booking_class():
    return Class(name="Booking", attributes={
        Property(name="nights", type=IntegerType),
    })


@pytest.fixture
def booking_model(booking_class):
    return DomainModel(name="BookingModel", types={booking_class})


# ============================================================================
# The skip comment has one source of truth
# ============================================================================

class TestSkipCommentIsUnified:
    """Before the fix, ``SKIP_COMMENT_TEMPLATE`` was defined but unused: the
    template hardcoded an independent copy of the same string. Proven here by
    changing the constant and checking the generated file picks it up --
    which is only possible if the template actually renders it."""

    def test_template_renders_SKIP_COMMENT_TEMPLATE(
        self, booking_class, booking_model, tmpdir, monkeypatch
    ):
        import besser.generators.pydantic_classes.ocl_utils as ocl_utils

        monkeypatch.setattr(
            ocl_utils, "SKIP_COMMENT_TEMPLATE",
            "SENTINEL-TEXT constraint={name} expr={expression} reason={reason}",
        )
        booking_model.constraints = {
            Constraint(
                name="Capacity", context=booking_class,
                expression="context Booking inv Capacity: self.rooms->size() <= 5",
                language="OCL",
            )
        }
        _, source = generate(booking_model, tmpdir.mkdir("output"))

        assert "SENTINEL-TEXT constraint=Capacity" in source
        # The old hardcoded copy must be gone, not just supplemented.
        assert "NOTE: OCL constraint" not in source


# ============================================================================
# The handoff contract: name, OCL verbatim, why, where -- for BOTH paths
# ============================================================================

class TestSkippedCommentContent:
    """The ``skipped`` path: constraint reached ocl_utils.py, stayed untranslatable."""

    def test_comment_carries_name_ocl_reason_and_where(
        self, booking_class, booking_model, tmpdir
    ):
        expression = (
            "context Booking inv NumberOfGuestsDoesNotExceedRoomCapacity: "
            "self.nights->size() <= self.rooms->collect(r | r.max_people)->sum()"
        )
        booking_model.constraints = {
            Constraint(
                name="NumberOfGuestsDoesNotExceedRoomCapacity", context=booking_class,
                expression=expression, language="OCL",
            )
        }
        path, source = generate(booking_model, tmpdir.mkdir("output"))
        py_compile.compile(path, doraise=True)

        assert "NumberOfGuestsDoesNotExceedRoomCapacity" in source
        assert expression in source  # OCL verbatim, not paraphrased
        assert "not transpilable" in source  # why
        assert "router handler" in source  # where
        assert "database session" in source
        # A comment only -- nothing attempts to enforce it.
        assert "field_validator('nights')" not in source
        assert "model_validator" not in source

    def test_no_ocl_arrow_syntax_leaks_outside_a_comment_line(
        self, booking_class, booking_model, tmpdir
    ):
        """Every physical line containing OCL ``->`` must itself start with
        ``#``, so a multi-line/odd constraint can never break out of comment
        context into executable code."""
        booking_model.constraints = {
            Constraint(
                name="Capacity", context=booking_class,
                expression="context Booking inv Capacity: self.rooms->size() <= 5",
                language="OCL",
            )
        }
        path, source = generate(booking_model, tmpdir.mkdir("output"))
        py_compile.compile(path, doraise=True)

        for line in source.splitlines():
            if "->" in line:
                assert line.lstrip().startswith("#"), line


class TestRejectedCommentContent:
    """The ``rejected`` path: the OCL converter never attached the constraint
    at all; it only exists in ``domain_model.conversion_issues``."""

    def test_comment_is_placed_on_the_context_class(self, tmpdir):
        booking = Class(name="Booking", attributes={
            Property(name="nights", type=IntegerType),
        })
        room = Class(name="Room", attributes={
            Property(name="maxOccupancy", type=IntegerType),
        })
        model = DomainModel(name="Hotel", types={booking, room})
        model.conversion_issues = [{
            "category": "ocl",
            "context": "Booking",
            "name": "guestsWithinCapacity",
            "kind": "invariant",
            "expression": (
                "context Booking inv guestsWithinCapacity: "
                "self.guests->size() <= self.rooms->collect(maxOccupancy)->sum()"
            ),
            "code": "parse_error",
            "reason": (
                "Warning: Invalid OCL syntax in '...': Property 'guests' not "
                "found in context 'Booking' (did you mean 'self.guest'?)"
            ),
        }]

        path, source = generate(model, tmpdir.mkdir("output"))
        py_compile.compile(path, doraise=True)

        def class_block(src, class_header):
            """This class's own body: from its header to the next top-level
            ``class`` line (or end of file), independent of class order."""
            after = src.split(class_header, 1)[1]
            return after.split("\nclass ")[0]

        booking_block = class_block(source, "class BookingCreate(BaseModel):")
        room_block = class_block(source, "class RoomCreate(BaseModel):")

        assert "guestsWithinCapacity" in booking_block
        assert "did you mean 'self.guest'?" in booking_block
        assert "router handler" in booking_block
        # Rejected constraints get no validator either -- comment only.
        assert "field_validator" not in booking_block
        assert "model_validator" not in booking_block
        # Room has no conversion issue of its own: must not gain one.
        assert "guestsWithinCapacity" not in room_block

    def test_unnamed_or_reasonless_issue_does_not_crash(self, tmpdir):
        """Defensive shape: a conversion issue missing optional keys must
        still render *something* sane rather than raising."""
        booking = Class(name="Booking", attributes={
            Property(name="nights", type=IntegerType),
        })
        model = DomainModel(name="Hotel", types={booking})
        model.conversion_issues = [{"category": "ocl", "context": "Booking"}]

        path, source = generate(model, tmpdir.mkdir("output"))
        py_compile.compile(path, doraise=True)
        assert "unnamed" in source

    def test_issue_for_a_different_class_is_not_attached_here(self, tmpdir):
        booking = Class(name="Booking", attributes={
            Property(name="nights", type=IntegerType),
        })
        model = DomainModel(name="Hotel", types={booking})
        model.conversion_issues = [{
            "category": "ocl", "context": "SomeOtherClass",
            "name": "irrelevant", "expression": "context SomeOtherClass inv irrelevant: self.x",
            "reason": "whatever",
        }]

        _, source = generate(model, tmpdir.mkdir("output"))
        assert "irrelevant" not in source


# ============================================================================
# get_rejected_constraints_for_class -- direct unit coverage
# ============================================================================

class TestGetRejectedConstraintsForClass:
    """Imports are inline so a missing symbol fails only this test, not the
    whole module (useful pre-fix, where this function does not exist yet)."""

    def test_empty_or_none_issues_yield_nothing(self):
        from besser.generators.pydantic_classes.ocl_utils import (
            get_rejected_constraints_for_class,
        )
        assert get_rejected_constraints_for_class(None, "Booking") == []
        assert get_rejected_constraints_for_class([], "Booking") == []

    def test_filters_by_context_and_sorts_by_name(self):
        from besser.generators.pydantic_classes.ocl_utils import (
            get_rejected_constraints_for_class,
        )
        issues = [
            {"context": "Booking", "name": "Zeta", "expression": "context Booking inv Zeta: self.x",
             "reason": "r"},
            {"context": "Room", "name": "Other", "expression": "context Room inv Other: self.y",
             "reason": "r"},
            {"context": "Booking", "name": "Alpha", "expression": "context Booking inv Alpha: self.x",
             "reason": "r"},
        ]
        result = get_rejected_constraints_for_class(issues, "Booking")
        assert [r["constraint_name"] for r in result] == ["Alpha", "Zeta"]
        assert all(r["rejected"] is True for r in result)

    def test_leading_warning_prefix_is_dropped_but_suggestion_kept(self):
        from besser.generators.pydantic_classes.ocl_utils import (
            get_rejected_constraints_for_class,
        )
        issues = [{
            "context": "Booking",
            "name": "guestsWithinCapacity",
            "expression": "context Booking inv guestsWithinCapacity: self.guests->size() <= 1",
            "reason": (
                "Warning: Invalid OCL syntax in '...': Property 'guests' not "
                "found in context 'Booking' (did you mean 'self.guest'?)"
            ),
        }]
        result = get_rejected_constraints_for_class(issues, "Booking")
        comment = "\n".join(result[0]["comment_lines"])
        assert not comment.startswith("# Warning: Warning:")
        assert "did you mean 'self.guest'?" in comment


# ============================================================================
# Comment only: never changes runtime behaviour
# ============================================================================

class TestCommentDoesNotEnforceAnything:
    """A skipped/rejected constraint must never raise -- it is a comment,
    not a check. This is the direct behavioural guard for the task rule
    'do NOT emit Python that changes behaviour -- a comment only'."""

    def test_generated_model_accepts_data_that_violates_the_rejected_rule(self, tmpdir):
        booking = Class(name="Booking", attributes={
            Property(name="nights", type=IntegerType),
        })
        model = DomainModel(name="Hotel", types={booking})
        model.conversion_issues = [{
            "category": "ocl", "context": "Booking", "name": "TooManyGuests",
            "expression": "context Booking inv TooManyGuests: self.guests->size() <= 1",
            "reason": "whatever",
        }]

        path, _ = generate(model, tmpdir.mkdir("output"))
        py_compile.compile(path, doraise=True)

        import importlib.util
        import sys
        import uuid
        module_name = f"pydantic_classes_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            # nights=999 would violate "TooManyGuests" if it were enforced;
            # since it's comment-only, construction must simply succeed.
            module.BookingCreate(nights=999)
        finally:
            sys.modules.pop(module_name, None)
