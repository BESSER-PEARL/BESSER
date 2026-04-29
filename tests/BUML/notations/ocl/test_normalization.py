"""Integration tests for B-OCL normalization (PRD §6.3).

Each test parses an OCL string, normalizes it, pretty-prints the result, and
checks identity against the expected normal form. Idempotence is asserted on
every case: ``normalize(normalize(c)) == normalize(c)``.

Test inputs trace the worked examples in Cabot & Teniente, Sci. Comput.
Program. 68 (2007). T7's input uses ``->size() > 0`` instead of ``->notEmpty()``
because ``notEmpty`` is not in the BESSER grammar today (the rewrite system
itself never produces it; the substitution is purely a test-input choice).
"""

import pytest

from besser.BUML.notations.ocl import normalize, parse_ocl, pretty_print


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
    is_implies, is_xor, is_isempty, is_loop, is_atomic_type_test, walk,
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


def test_normalization_preserves_source_location(model):
    """Rewritten root nodes inherit line/col from the original (Pre-work C)."""
    parsed = parse_ocl(
        "context Employee inv: self.age < 25 implies self.salary <= 50000.0",
        model,
    )
    assert parsed.ast.line is not None
    original_line = parsed.ast.line

    normalized = normalize(parsed, model)
    # ``implies`` is rewritten to ``not e1 or e2``; the new top-level OR
    # node should carry the original line.
    assert normalized.ast.line == original_line


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
