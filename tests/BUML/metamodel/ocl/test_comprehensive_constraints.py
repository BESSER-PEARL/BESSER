"""
Comprehensive OCL constraint tests on the University model.

Tests 25 constraints expected True and 15 expected False,
covering: comparisons, and/or/not/implies, forAll, exists,
select, collect, size, oclIsTypeOf, if/then/else, arithmetic.
"""

import pytest
from bocl.OCLWrapper import OCLWrapper
from tests.BUML.metamodel.ocl.university_model import (
    domain_model, object_model,
    true_constraints, false_constraints,
)


@pytest.fixture
def wrapper():
    return OCLWrapper(domain_model, object_model)


# ----------------------------------------------------------------
# TRUE constraints
# ----------------------------------------------------------------
TRUE_CASES = sorted(true_constraints, key=lambda c: c.name)


@pytest.mark.parametrize(
    "constraint",
    TRUE_CASES,
    ids=[c.name for c in TRUE_CASES],
)
def test_constraint_true(wrapper, constraint):
    result = wrapper.evaluate(constraint)
    print(f"  {constraint.expression}  =>  {result}")
    assert result is True, (
        f"Expected True for '{constraint.name}': {constraint.expression}"
    )


# ----------------------------------------------------------------
# FALSE constraints
# ----------------------------------------------------------------
FALSE_CASES = sorted(false_constraints, key=lambda c: c.name)


@pytest.mark.parametrize(
    "constraint",
    FALSE_CASES,
    ids=[c.name for c in FALSE_CASES],
)
def test_constraint_false(wrapper, constraint):
    result = wrapper.evaluate(constraint)
    print(f"  {constraint.expression}  =>  {result}")
    assert result is False, (
        f"Expected False for '{constraint.name}': {constraint.expression}"
    )
