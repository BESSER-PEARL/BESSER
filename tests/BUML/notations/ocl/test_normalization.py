"""Integration tests for B-OCL normalization (PRD §6.3).

Each test parses an OCL string, normalizes it, pretty-prints the result, and
checks identity against the expected normal form. Idempotence is asserted on
every case: ``normalize(normalize(c)) == normalize(c)``.

Test inputs trace the worked examples in Cabot & Teniente, Sci. Comput.
Program. 68 (2007). T7's input uses ``->size() > 0``, the form ``->notEmpty()``
normalizes to (the rewrite system itself never produces ``notEmpty``).
"""

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, FloatType, IntegerType, Multiplicity,
    Property,
)
from besser.BUML.notations.ocl import normalize, parse_ocl, pretty_print


@pytest.fixture(scope="module")
def product_model() -> DomainModel:
    """The thesis ecommerce fragment used by CorrectProduct'."""
    product = Class("Product", attributes={
        Property("price", FloatType),
        Property("maxDiscount", IntegerType),
    })
    return DomainModel("Shop", types={product}, associations=set())


@pytest.fixture(scope="module")
def staff_model() -> DomainModel:
    """Two collections of employees, so ``union`` has two different sources."""
    department = Class("Department")
    employee = Class("Employee", attributes={Property("age", IntegerType)})
    staff = BinaryAssociation("StaffOf", ends={
        Property("staffDept", department, multiplicity=Multiplicity(1, 1)),
        Property("staff", employee, multiplicity=Multiplicity(0, "*")),
    })
    interns = BinaryAssociation("InternsOf", ends={
        Property("internDept", department, multiplicity=Multiplicity(1, 1)),
        Property("interns", employee, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel(
        "StaffModel",
        types={department, employee},
        associations={staff, interns},
    )


def _normalize_to_string(input_str, model):
    parsed = parse_ocl(input_str, model)
    normalized = normalize(parsed, model)
    return pretty_print(normalized), normalized


def assert_normalizes_to(input_str, expected_str, model):
    rendered, normalized = _normalize_to_string(input_str, model)
    assert rendered == expected_str, (
        f"\n  got:      {rendered}\n  expected: {expected_str}"
    )
    # Idempotence — normalizing again must be a no-op.
    twice = pretty_print(normalize(normalized, model))
    assert twice == expected_str, f"not idempotent: {twice!r}"


# ---------------------------------------------------------------------------
# PRD §6.3 — T1..T8
# ---------------------------------------------------------------------------

def test_T1_boolean_constant(model):
    assert_normalizes_to(
        "context Employee inv: self.age > 16 and true",
        "context Employee inv: self.age > 16",
        model,
    )


def test_T2_double_negation(model):
    assert_normalizes_to(
        "context Employee inv: not (not (self.age > 16))",
        "context Employee inv: self.age > 16",
        model,
    )


def test_T3_demorgan_with_negated_comparisons(model):
    assert_normalizes_to(
        "context Employee inv: not (self.age < 18 and self.salary < 1000.0)",
        "context Employee inv: self.age >= 18 or self.salary >= 1000.0",
        model,
    )


def test_T4_implies_elimination(model):
    assert_normalizes_to(
        "context Employee inv: self.age < 25 implies self.salary <= 50000.0",
        "context Employee inv: self.age >= 25 or self.salary <= 50000.0",
        model,
    )


def test_T5_exists_to_forall(model):
    assert_normalizes_to(
        "context Department inv: not self.employee->exists(e | e.salary > 1000000.0)",
        "context Department inv: self.employee->forAll(e | e.salary <= 1000000.0)",
        model,
    )


def test_T6_allinstances_removal(model):
    assert_normalizes_to(
        "context Employee inv: Employee::allInstances()->forAll(e | e.age > 16)",
        "context Employee inv: self.age > 16",
        model,
    )


def test_not_empty_becomes_size_greater_than_zero(model):
    assert_normalizes_to(
        "context Department inv: self.employee->notEmpty()",
        "context Department inv: self.employee->size() > 0",
        model,
    )


def test_T7_multiplicity_aware_simplification(model):
    # `self.employer` has multiplicity 1..1, so size() > 0 → true,
    # then `true implies X` → X.
    assert_normalizes_to(
        "context Employee inv: self.employer->size() > 0 implies "
        "self.employer.minSalary <= self.salary",
        "context Employee inv: self.employer.minSalary <= self.salary",
        model,
    )


def test_T8_paper_maxsalary_integration(model):
    """Paper §2.4, steps 1–6. End-to-end pipeline test."""
    assert_normalizes_to(
        "context Department inv: Department::allInstances()->forAll("
        "d | not d.employee->select(e | e.age < 25)->exists("
        "e | e.salary > d.maxJuniorSal))",
        "context Department inv: self.employee->forAll("
        "e | e.age >= 25 or e.salary <= self.maxJuniorSal)",
        model,
    )


# ---------------------------------------------------------------------------
# Property tests — PRD §6.4
# ---------------------------------------------------------------------------

from besser.BUML.metamodel.ocl import (
    is_implies, is_xor, is_isempty, is_loop, walk,
)
from besser.BUML.metamodel.ocl.ocl import IfExp


@pytest.mark.parametrize("ocl_input", [
    "context Employee inv: self.age > 16 and true",
    "context Employee inv: not (not (self.age > 16))",
    "context Employee inv: self.age < 25 implies self.salary <= 50000.0",
    "context Department inv: not self.employee->exists(e | e.salary > 1000000.0)",
    "context Department inv: Department::allInstances()->forAll("
    "d | not d.employee->select(e | e.age < 25)->exists("
    "e | e.salary > d.maxJuniorSal))",
])
def test_operator_surface_invariant(ocl_input, model):
    """Normal form contains no implies / xor / exists / reject / isEmpty / IfExp."""
    parsed = parse_ocl(ocl_input, model)
    normalized = normalize(parsed, model)
    for node in walk(normalized.ast):
        assert not is_implies(node), pretty_print(normalized)
        assert not is_xor(node), pretty_print(normalized)
        assert not is_isempty(node), pretty_print(normalized)
        assert not is_loop(node, "exists"), pretty_print(normalized)
        assert not is_loop(node, "reject"), pretty_print(normalized)
        assert not isinstance(node, IfExp), pretty_print(normalized)


@pytest.mark.parametrize("ocl_input", [
    "context Employee inv: self.age > 16",
    "context Employee inv: self.age < 25 implies self.salary <= 50000.0",
    "context Department inv: self.employee->forAll(e | e.age >= 25)",
])
def test_idempotence(ocl_input, model):
    parsed = parse_ocl(ocl_input, model)
    once = normalize(parsed, model)
    twice = normalize(once, model)
    assert pretty_print(once) == pretty_print(twice)


def test_if_with_non_boolean_branches_is_left_alone(model):
    """``if C then <int> else <int> endif`` must not be flattened.

    Rewriting it would emit ``and`` / ``or`` over integers, which is not
    OCL-valid. The IfExp must survive normalization unchanged.
    """
    parsed = parse_ocl(
        "context Employee inv: "
        "if self.age > 16 then 5 else 10 endif = 5",
        model,
    )
    normalized = normalize(parsed, model)

    if_nodes = [n for n in walk(normalized.ast) if isinstance(n, IfExp)]
    assert len(if_nodes) == 1, (
        f"IfExp with non-boolean branches should survive; got: "
        f"{pretty_print(normalized)}"
    )


# ---------------------------------------------------------------------------
# Chapter 3 — Cabot, Incremental Integrity Checking, tables 3.1–3.3 and §3.2–3.3
# ---------------------------------------------------------------------------

def test_chapter_correct_product(product_model):
    """CorrectProduct' simplifies to the form used in the thesis."""
    assert_normalizes_to(
        "context Product inv: not Product.allInstances()->exists("
        "p | p.price <= 0 or p.maxDiscount > 60)",
        "context Product inv: self.price > 0 and self.maxDiscount <= 60",
        product_model,
    )


def test_chapter_unique_name_replaces_only_the_first_iterator(model):
    """§3.2 multi-iterator allInstances, then implies is eliminated."""
    assert_normalizes_to(
        "context Employee inv: Employee.allInstances()->forAll("
        "e1, e2 | e1 <> e2 implies e1.name <> e2.name)",
        "context Employee inv: Employee::allInstances()->forAll("
        "e2 | self = e2 or self.name <> e2.name)",
        model,
    )


def test_allinstances_is_kept_when_the_body_already_uses_self(model):
    assert_normalizes_to(
        "context Employee inv: Employee.allInstances()->forAll("
        "e | e.salary <= self.salary)",
        "context Employee inv: Employee::allInstances()->forAll("
        "e | e.salary <= self.salary)",
        model,
    )


def test_allinstances_is_kept_when_the_body_uses_an_implicit_self_attribute(model):
    assert_normalizes_to(
        "context Employee inv: Employee.allInstances()->forAll(e | e.age > age)",
        "context Employee inv: Employee::allInstances()->forAll(e | e.age > age)",
        model,
    )


def test_table_boolean_bounds_and_equality(model):
    assert_normalizes_to(
        "context Employee inv: self.age > 16 and self.age <= 16",
        "context Employee inv: false",
        model,
    )
    assert_normalizes_to(
        "context Employee inv: self.age > 16 or self.age <= 16",
        "context Employee inv: true",
        model,
    )
    assert_normalizes_to(
        "context Employee inv: self.age > 16 or self.age < 16",
        "context Employee inv: self.age <> 16",
        model,
    )
    assert_normalizes_to(
        "context Employee inv: (self.age > 16) = true",
        "context Employee inv: self.age > 16",
        model,
    )
    assert_normalizes_to(
        "context Employee inv: (self.age > 16) = false",
        "context Employee inv: self.age <= 16",
        model,
    )


def test_table_boolean_equality_reaches_conjunctive_form(model):
    # Table 3.1 turns X = Y into (X and Y) or (not X and not Y). Distribution
    # then yields the two clauses of (X iff Y). Negated comparisons land on
    # the left of each disjunction, so the text is the chapter form with
    # clauses and disjuncts swapped.
    assert_normalizes_to(
        "context Employee inv: (self.age > 16) = (self.salary > 0.0)",
        "context Employee inv: (self.age <= 16 or self.salary > 0.0) "
        "and (self.salary <= 0.0 or self.age > 16)",
        model,
    )


def test_table_size_or_forall_and_negated_emptiness(model):
    assert_normalizes_to(
        "context Department inv: self.employee->size() <= 0 or "
        "self.employee->forAll(e | e.age > 16)",
        "context Department inv: self.employee->forAll(e | e.age > 16)",
        model,
    )
    assert_normalizes_to(
        "context Department inv: not self.employee->isEmpty()",
        "context Department inv: self.employee->size() > 0",
        model,
    )
    assert_normalizes_to(
        "context Department inv: not self.employee->notEmpty()",
        "context Department inv: self.employee->size() = 0",
        model,
    )


def test_table_iterator_fusions(model):
    assert_normalizes_to(
        "context Department inv: self.employee->select(e | e.age > 16)->size() "
        "= self.employee->size()",
        "context Department inv: self.employee->forAll(e | e.age > 16)",
        model,
    )
    assert_normalizes_to(
        "context Department inv: self.employee->select(e | e.age > 25)->exists("
        "e | e.salary > 1.0)",
        "context Department inv: not self.employee->forAll("
        "e | e.age <= 25 or e.salary <= 1.0)",
        model,
    )
    assert_normalizes_to(
        "context Department inv: self.employee->forAll(e | e.age > 16) and "
        "self.employee->forAll(e | e.salary > 0.0)",
        "context Department inv: self.employee->forAll("
        "e | e.age > 16 and e.salary > 0.0)",
        model,
    )
    assert_normalizes_to(
        "context Department inv: self.employee->forAll("
        "e | e.age > 16 and self.employee->forAll(f | f.salary > 0.0))",
        "context Department inv: self.employee->forAll("
        "e, f | e.age > 16 and f.salary > 0.0)",
        model,
    )


def test_table_is_unique(model):
    assert_normalizes_to(
        "context Department inv: self.employee->isUnique(age)",
        "context Department inv: self.employee->forAll("
        "x1, x2 | x1 = x2 or x1.age <> x2.age)",
        model,
    )


def test_table_union_forall(staff_model):
    assert_normalizes_to(
        "context Department inv: self.staff->union(self.interns)->forAll("
        "e | e.age > 16)",
        "context Department inv: self.staff->forAll(e | e.age > 16) and "
        "self.interns->forAll(e | e.age > 16)",
        staff_model,
    )
    assert_normalizes_to(
        "context Department inv: self.staff->union(self.interns).age->forAll("
        "a | a > 16)",
        "context Department inv: self.staff.age->forAll(a | a > 16) and "
        "self.interns.age->forAll(a | a > 16)",
        staff_model,
    )


def test_chapter_xor_and_if_and_distribution(model):
    assert_normalizes_to(
        "context Employee inv: self.age > 16 xor self.salary > 0.0",
        "context Employee inv: (self.age > 16 or self.salary > 0.0) and "
        "(self.age <= 16 or self.salary <= 0.0)",
        model,
    )
    assert_normalizes_to(
        "context Employee inv: if self.age > 16 then self.salary > 0.0 "
        "else self.salary < 0.0 endif",
        "context Employee inv: (self.age <= 16 or self.salary > 0.0) and "
        "(self.age > 16 or self.salary < 0.0)",
        model,
    )
    assert_normalizes_to(
        "context Employee inv: self.age > 16 or self.salary > 0.0 and self.name = 'a'",
        "context Employee inv: (self.age > 16 or self.salary > 0.0) and "
        "(self.age > 16 or self.name = 'a')",
        model,
    )


def test_if_with_boolean_branches_is_eliminated(model):
    """``if C then <bool-expr> else <bool-expr> endif`` is still folded."""
    parsed = parse_ocl(
        "context Employee inv: "
        "if self.age > 16 then self.salary > 0.0 else self.salary < 0.0 endif",
        model,
    )
    normalized = normalize(parsed, model)
    for node in walk(normalized.ast):
        assert not isinstance(node, IfExp), (
            f"boolean-branched IfExp must be eliminated; got: "
            f"{pretty_print(normalized)}"
        )
