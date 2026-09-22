"""Tests for besser.BUML.notations.ocl.pretty_printer.pretty_print.

Round-trip identity: ``pretty_print(parse_ocl(s)) == s`` for every input that's
already in canonical form (no extra parens, canonical operator spelling).
"""

import pytest

from besser.BUML.notations.ocl import parse_ocl, pretty_print


@pytest.mark.parametrize("source", [
    "context Employee inv: self.age > 16",
    "context Employee inv: self.salary <= 50000.0",
    "context Employee inv: self.age >= 18 or self.salary >= 1000.0",
    "context Employee inv: not self.age > 16",
    "context Employee inv: self.age > 16 and self.salary > 0.0",
    "context Department inv: self.employee->forAll(e | e.age > 16)",
    "context Employee inv: self.employer.minSalary <= self.salary",
])
def test_round_trip_identity(source, model):
    constraint = parse_ocl(source, model)
    rendered = pretty_print(constraint)
    assert rendered == source


def test_pretty_print_handles_raw_expression(model):
    constraint = parse_ocl("context Employee inv: self.age > 16", model)
    # Pass just the expression, not the whole OCLConstraint.
    rendered = pretty_print(constraint.ast)
    assert rendered == "self.age > 16"


def test_pretty_print_emits_context_prologue_for_constraint(model):
    constraint = parse_ocl("context Employee inv: self.age > 16", model)
    assert pretty_print(constraint).startswith("context Employee inv: ")


def test_pretty_print_boolean_literal_python_bool(model):
    constraint = parse_ocl(
        "context Employee inv: self.age > 16 and true",
        model,
    )
    rendered = pretty_print(constraint)
    assert rendered == "context Employee inv: self.age > 16 and true"


def test_pretty_print_isEmpty(model):
    constraint = parse_ocl(
        "context Department inv: self.employee->isEmpty()",
        model,
    )
    assert pretty_print(constraint) == "context Department inv: self.employee->isEmpty()"


def test_pretty_print_size(model):
    constraint = parse_ocl(
        "context Department inv: self.employee->size() > 0",
        model,
    )
    assert pretty_print(constraint) == "context Department inv: self.employee->size() > 0"
