"""
Utility functions for Alloy code generation.

This module contains the helper functions used by :class:`AlloyGenerator` to
translate B-UML domain models into Alloy specifications, as well as name
sanitization utilities for Alloy-compatible identifiers.
"""

import re
from collections import defaultdict
from pathlib import Path

from besser.BUML.metamodel.structural import DomainModel, Enumeration
from besser.generators.alloy.translate_ocl_alloy import (
    TranslatorState,
    ocl_to_alloy,
)

# ----------------------------------------------------------------------
# Name sanitization
# ----------------------------------------------------------------------

ALLOY_IDENTIFIER_REGEX = re.compile(r"[^A-Za-z0-9_]")

ALLOY_KEYWORDS = {
    "abstract", "all", "and", "as", "assert", "but", "check", "disj",
    "else", "enum", "exactly", "expt", "fact", "for", "fun", "iden",
    "iff", "implies", "in", "int", "Int", "let", "lone", "module",
    "no", "none", "not", "one", "open", "or", "pred", "run", "seq",
    "set", "sig", "some", "sum", "univ"
}


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


def sanitize_model_names(model: DomainModel) -> None:
    """Sanitizes class and attribute names in *model* for Alloy compatibility.

    Modifies the model **in-place** so that every class and attribute name is
    a legal Alloy identifier.
    """
    for class_obj in model.classes_sorted_by_inheritance():
        class_obj.name = sanitize_alloy_name(class_obj.name)
        for attr in class_obj.attributes:
            attr.name = sanitize_alloy_name(attr.name)


# ----------------------------------------------------------------------
# BUML -> Alloy translation helpers
# ----------------------------------------------------------------------

MULTIPLICITY_LIMIT = 9999


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
    if not (mult_b[0] == 1 and mult_b[1] == 1):
        if arrow_a_b:
            nav = f"a.{class_a}_{rel_a_b}"
        elif arrow_b_a:
            nav = f"{class_b}_{rel_b_a}.a"
        else:
            nav = None
        if nav:
            if mult_b[0] >= 1 and mult_b[0] < MULTIPLICITY_LIMIT:
                res += f"\nfact{{ all a: {class_a} | #({nav}) >= {mult_b[0]} }}"
            if mult_b[1] >= 1 and mult_b[1] < MULTIPLICITY_LIMIT:
                res += f"\nfact{{ all a: {class_a} | #({nav})<={mult_b[1]} }}"

    if not (mult_a[0] == 1 and mult_a[1] == 1):
        if arrow_b_a:
            nav = f"b.{class_b}_{rel_b_a}"
        elif arrow_a_b:
            nav = f"{class_a}_{rel_a_b}.b"
        else:
            nav = None
        if nav:
            if mult_a[0] >= 1 and mult_a[0] < MULTIPLICITY_LIMIT:
                res += f"\nfact{{ all b: {class_b} | #({nav}) >= {mult_a[0]} }}"
            if mult_a[1] >= 1 and mult_a[1] < MULTIPLICITY_LIMIT:
                res += f"\nfact{{ all b: {class_b} | #({nav}) <= {mult_a[1]} }}"

    return res


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
        return state.date_ops.generate_date_block(state.dates, scope)
    return ""


def build_inheritance_and_attribute_maps(
    model: DomainModel,
) -> tuple[dict, dict, set, list[str]]:
    """Builds inheritance, attribute, and signature maps from *model*.

    Returns:
        A tuple ``(inherits_from, data, basic_signatures, sigs_nv)``.
    """
    inherits_from: dict = defaultdict(list)
    data: dict = defaultdict(list)
    basic_signatures: set = set()
    sigs_nv: list[str] = []

    for class_obj in model.classes_sorted_by_inheritance():
        sigs_nv.append(class_obj.name)

        if len(class_obj.parents()) == 0:
            inherits_from[class_obj.name].append("_")
        else:
            for parent in class_obj.parents():
                inherits_from[class_obj.name].append(parent.name)

        for attr in class_obj.attributes:
            attr_type = "date" if attr.type.name in ("date", "datetime", "time", "timedelta") else attr.type.name
            data[class_obj.name].append(f"{attr.name}:{attr_type}")
            if not isinstance(attr.type, Enumeration):
                basic_signatures.add(attr_type)
                sigs_nv.append(attr_type)

    return inherits_from, data, basic_signatures, sigs_nv


def process_associations(model: DomainModel, data: dict) -> list[str]:
    """Processes associations, building consistency facts and updating *data*.

    Returns:
        A list of Alloy fact strings for associations.
    """
    facts_rules: list[str] = []

    for assoc in model.associations:
        d, h = assoc.ends
        mult_b = [h.multiplicity.min, h.multiplicity.max]
        mult_a = [d.multiplicity.min, d.multiplicity.max]
        arrow_a_b = bool(h.is_navigable)
        arrow_b_a = bool(d.is_navigable)

        facts_rules.append(
            build_consistency_rule(
                d.type.name, h.name, mult_b,
                h.type.name, d.name, mult_a,
                arrow_a_b, arrow_b_a,
            )
        )
        data[h.type.name].append(f"{d.name}:{d.type.name}")
        data[d.type.name].append(f"{h.name}:{h.type.name}")

        if arrow_a_b and arrow_b_a:
            facts_rules.append(
                f"fact{{ {d.type.name}_{h.name} = ~{h.type.name}_{d.name} }}"
            )

    return facts_rules


def translate_constraints(
    model: DomainModel, inherits_from: dict, data: dict, enums: dict
) -> TranslatorState:
    """Translates OCL constraints to Alloy facts in-place.

    Returns:
        A :class:`TranslatorState` object, carrying accumulated state (e.g. date
        literals discovered during translation).
    """
    state = TranslatorState()
    for constraint in sorted(model.constraints, key=lambda c: c.name):
        context = constraint.context.name
        ocl_str = constraint.expression.split(":", 1)[1]
        constraint.expression = ocl_to_alloy(
            inherits_from, data, ocl_str, context, state, enums
        )
    return state


_UTILS_SNIPPETS = (
    "fun image [s: univ -> univ]: set univ { { f: univ | some i: univ | i -> f in s } }\n"
    "fun toSeq [a: set univ, rel: univ -> univ]: univ -> univ { a <: rel }\n"
    "fun collect [s: univ -> univ, r: univ -> univ]: univ -> univ { s.r }\n"
)


def generate_utils_module(output_dir: str | Path) -> Path:
    """Writes ``utils.als`` in *output_dir* with the shared Alloy helper functions.

    ``model.als`` opens this module via ``open utils``, so it must live in the
    same directory as the generated specification.
    """
    content = "module utils\n\n" + _UTILS_SNIPPETS
    path = Path(output_dir) / "utils.als"
    path.write_text(content, encoding="utf-8")
    return path
