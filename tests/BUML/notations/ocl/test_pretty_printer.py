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


@pytest.mark.parametrize("source", [
    # Collection operations reachable from the KG converter. Each of these used
    # to come back as a method call named after the AST's internal operation
    # constant — `self.employee.ASSET()`, `.INCLUDES(...)` — which re-parses as
    # something entirely different instead of failing loudly.
    "context Department inv: self.employee->asSet()->size() > 0",
    "context Department inv: self.employee->includes(self)",
    "context Department inv: self.employee->excludes(self)",
    "context Department inv: self.employee->isUnique(name)",
    "context Department inv: self.employee->intersection(self.employee)->isEmpty()",
    "context Department inv: self.employee->union(self.employee)->size() > 0",
    "context Department inv: self.employee->first()",
    "context Department inv: self.employee->last()",
    "context Department inv: self.employee->sum() > 0",
    "context Department inv: self.employee->asSet()->forAll(e | e.age > 16)",
    "context Employee inv: Employee::allInstances()->size() > 0",
])
def test_collection_operations_round_trip(source, model):
    """``pretty_print`` must spell every arrow operation the way OCL does."""
    assert pretty_print(parse_ocl(source, model)) == source


def test_collection_operation_output_reparses(model):
    """The printed form has to survive a second parse unchanged — a fixed point,
    not merely something the grammar happens to accept."""
    source = "context Department inv: self.employee->asSet()->forAll(e | self.employee->includes(e))"
    once = pretty_print(parse_ocl(source, model))
    twice = pretty_print(parse_ocl(once, model))
    assert once == source
    assert twice == once
