"""Utility constants and helper functions for the Alloy generator."""

import re

from besser.BUML.metamodel.structural import Enumeration
from besser.generators.alloy_generator.translate_ocl_alloy import (
    TranslatorState,
    generate_dates_and_order,
)

ALLOY_IDENTIFIER_REGEX = re.compile(r"[^A-Za-z0-9_]")

ALLOY_KEYWORDS = {
    "abstract", "all", "and", "as", "assert", "but", "check", "disj",
    "else", "enum", "exactly", "expt", "fact", "for", "fun", "iden",
    "iff", "implies", "in", "int", "Int", "let", "lone", "module",
    "no", "none", "not", "one", "open", "or", "pred", "run", "seq",
    "set", "sig", "some", "sum", "univ"
}

MULTIPLICITY_LIMIT = 9999


def sanitize_alloy_name(name: str) -> str:
    """Returns a valid Alloy identifier derived from *name*.

    Args:
        name: Raw identifier to sanitize.

    Returns:
        A non-empty string that is a legal Alloy identifier.
    """
    sanitized = ALLOY_IDENTIFIER_REGEX.sub("", name)
    if not sanitized:
        return "_unnamed"
    if sanitized[0].isdigit() or sanitized in ALLOY_KEYWORDS:
        sanitized = "_" + sanitized
    return sanitized


def build_consistency_rule(
    class_a: str,
    rel_a_b: str,
    mult_b: list,
    class_b: str,
    rel_b_a: str,
    mult_a: list,
    arrow_a_b: bool,
    arrow_b_a: bool,
) -> str:
    """Builds the Alloy cardinality-consistency facts for one association end.

    Emits ``fact`` blocks when the multiplicity differs from the implicit
    ``1..1`` default.  When the navigation direction of an end is enabled, its
    facts navigate the field directly (``a.<A>_<rel>``); when the direction is
    not navigable but the opposite end is, the facts navigate the opposite
    field in reverse (``<B>_<rel>.a``) so the multiplicity still holds.  If
    neither end is navigable there is no field to express the relation and no
    fact is emitted.

    Args:
        class_a:   Name of class A (source side).
        rel_a_b:   Role name navigating from A to B.
        mult_b:    ``[min, max]`` multiplicity for the B side.
        class_b:   Name of class B (target side).
        rel_b_a:   Role name navigating from B to A.
        mult_a:    ``[min, max]`` multiplicity for the A side.
        arrow_a_b: ``True`` when A→B is navigable.
        arrow_b_a: ``True`` when B→A is navigable.

    Returns:
        A string containing zero or more Alloy ``fact`` declarations.
    """
    res = "\n"
    # B-side multiplicity: how many B instances each A is related to.
    if not (mult_b[0] == 1 and mult_b[1] == 1):
        if arrow_a_b:
            nav = f"a.{class_a}_{rel_a_b}"
        elif arrow_b_a:
            nav = f"{class_b}_{rel_b_a}.a"
        else:
            nav = None
        if nav:
            if mult_b[0] >= 1 and mult_b[0] < MULTIPLICITY_LIMIT:
                res += f"\nfact{{all a: {class_a} | #({nav})>={mult_b[0]} }}"
            if mult_b[1] >= 1 and mult_b[1] < MULTIPLICITY_LIMIT:
                res += f"\nfact{{all a: {class_a} | #({nav})<={mult_b[1]} }}"

    # A-side multiplicity: how many A instances each B is related to.
    if not (mult_a[0] == 1 and mult_a[1] == 1):
        if arrow_b_a:
            nav = f"b.{class_b}_{rel_b_a}"
        elif arrow_a_b:
            nav = f"{class_a}_{rel_a_b}.b"
        else:
            nav = None
        if nav:
            if mult_a[0] >= 1 and mult_a[0] < MULTIPLICITY_LIMIT:
                res += f"\nfact{{all b: {class_b} | #({nav})>={mult_a[0]} }}"
            if mult_a[1] >= 1 and mult_a[1] < MULTIPLICITY_LIMIT:
                res += f"\nfact{{all b: {class_b} | #({nav})<={mult_a[1]} }}"

    return res


def collect_enumerations(model) -> dict:
    """Returns a mapping of enumeration names to their literal name sets.

    Args:
        model: A ``DomainModel`` whose elements may include enumerations.

    Returns:
        A dictionary mapping each enumeration name to the set of its literal
        names.
    """
    return {
        enum_obj.name: {lit.name for lit in (enum_obj.literals or set())}
        for enum_obj in model.elements
        if isinstance(enum_obj, Enumeration)
    }


def generate_date_block(
    state: TranslatorState, basic_signatures: set, scope: int
) -> str:
    """Generate the date universe and ordering block when needed.

    Args:
        state:          Translator state carrying discovered date literals.
        basic_signatures: Set of basic type names used in the model.
        scope:           Alloy scope (max atoms per signature).

    Returns:
        A string with the date ``one sig`` declarations and ordering fact,
        or an empty string when no date support is required.
    """
    if state.dates or "date" in basic_signatures:
        return generate_dates_and_order(state.dates, scope)
    return ""
