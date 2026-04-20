"""
OCL constraint parsing utilities for converting WME JSON to BUML.

The canonical wire shape is **full OCL text** in the ``ClassOCLConstraint``
element's ``constraint`` field — e.g.

    context Book inv pages_positive: self.pages > 0
    context Book::decrease_stock(qty: int) pre: qty > 0
    context Book::decrease_stock(qty: int) post: self.stock >= 0

This file extracts the routing kind (invariant / precondition / postcondition)
and target method from the parsed OCL header, then defers the actual lex/parse
work to :func:`besser.BUML.notations.ocl.api.parse_ocl`. The caller in
``class_diagram_processor.py`` uses the returned routing tuple to decide
whether the constraint lands on ``domain_model.constraints`` or on a
``Method.pre`` / ``Method.post`` list.

Malformed OCL is skipped with a warning rather than aborting the whole
conversion (matches the project's "skip-with-warning" policy).

A small back-compat helper :func:`legacy_body_only_to_text` is kept for
JSON files that were produced by an earlier intermediate iteration that
stored body-only OCL alongside ``kind`` / ``targetMethodId`` /
``constraintName`` metadata. It synthesises the canonical full-text form
so the rest of the pipeline only ever sees one shape. New code should not
call it.
"""

import re
from typing import Optional

from besser.BUML.metamodel.structural import Class, DomainModel, Method
from besser.BUML.metamodel.ocl.ocl import OCLConstraint
from besser.BUML.notations.ocl.api import parse_ocl
from besser.BUML.notations.ocl.error_handling import BOCLSyntaxError


_KIND_FROM_KW = {"inv": "invariant", "pre": "precondition", "post": "postcondition"}
_KW_FROM_KIND = {v: k for k, v in _KIND_FROM_KW.items()}

# Match the routing prefix of an OCL constraint:
#   context X inv [name]:
#   context X::method(params) pre|post:
# The method-signature segment is optional (only present for pre/post).
# The constraint name is optional; only invariants accept one in BOCL.
_HEADER_RE = re.compile(
    r"\bcontext\s+(?P<class>\w+)"
    r"(?:::(?P<method>\w+)\s*\([^)]*\))?"
    r"\s+(?P<kw>inv|pre|post)\b"
    r"(?:\s+(?P<name>\w+))?"
    r"\s*:",
    re.IGNORECASE,
)


def _typeref_name(t) -> str:
    """Best-effort BOCL type name for a Property/Parameter type."""
    if t is None:
        return "any"
    if hasattr(t, "name") and t.name:
        return t.name
    return str(t)


def _extract_inline_description(block: str) -> tuple[str, str]:
    """Split an OCL constraint block into its expression and an inline description.

    OCL supports ``--`` single-line comments. We treat any text following ``--``
    on a line as a natural-language description of the constraint. The returned
    expression has the comment stripped so it can be fed to the OCL parser
    without relying on comment-handling behaviour.

    Returns:
        tuple[str, str]: ``(expression_without_comments, description_or_empty_string)``.
    """
    description_parts: list[str] = []
    cleaned_lines: list[str] = []
    for raw_line in block.split('\n'):
        match = re.search(r'--\s*(.*?)\s*$', raw_line)
        if match and match.group(1):
            description_parts.append(match.group(1))
            raw_line = raw_line[:match.start()].rstrip()
        cleaned_lines.append(raw_line)
    cleaned = '\n'.join(cleaned_lines)
    description = ' '.join(description_parts).strip()
    return cleaned, description


def parse_constraint_text(
    text: str,
    domain_model: DomainModel,
) -> tuple[str, OCLConstraint, Optional[str], Optional[str]]:
    """Parse one OCL block and report enough routing context to attach it.

    Returns a 4-tuple:

    * ``kind`` — ``"invariant"`` / ``"precondition"`` / ``"postcondition"``.
    * ``constraint`` — the parsed :class:`OCLConstraint`. If the OCL header
      carried a name (only valid for invariants), it is set on
      ``constraint.name``; otherwise ``constraint.name`` is left at the
      ``"parsed"`` default for the caller to rename.
    * ``class_name`` — the context class name as written in the header.
    * ``method_name`` — for pre/post, the target method name; ``None`` for
      invariants.

    Raises:
        ValueError: when the text doesn't contain a recognisable OCL header.
        BOCLSyntaxError: when the body fails to parse against the BOCL grammar.
    """
    match = _HEADER_RE.search(text)
    if match is None:
        raise ValueError(
            "OCL text does not contain a recognisable "
            "'context <Class> inv|pre|post' header"
        )
    class_name = match.group("class")
    method_name = match.group("method")
    kw = match.group("kw").lower()
    name = match.group("name")

    kind = _KIND_FROM_KW[kw]

    # Resolve the context class ourselves and pass it to parse_ocl.
    # parse_ocl's auto-detection regex assumes `context X inv|pre|post|init`
    # and does not handle the method-signature segment in
    # `context X::method(params) pre:`, so passing context_class explicitly
    # bypasses the upstream regex entirely.
    context_class = next(
        (t for t in domain_model.types if isinstance(t, Class) and t.name == class_name),
        None,
    )
    if context_class is None:
        raise ValueError(
            f"OCL header references unknown class {class_name!r}"
        )

    constraint = parse_ocl(text, domain_model, context_class=context_class)
    if name:
        # Only invariants carry a name in BOCL; preserve it verbatim.
        constraint.name = name
    return kind, constraint, class_name, method_name


def process_ocl_constraints(
    ocl_text: str,
    domain_model: DomainModel,
    counter: int,
    default_description: Optional[str] = None,
) -> tuple[list[tuple[str, OCLConstraint, Optional[str], Optional[str]]], list[str]]:
    """Split a textarea blob on ``context`` boundaries and parse each block.

    The WME's OCL constraint element holds free text that may contain
    several `context X (inv|pre|post) ...` blocks pasted together. Each is
    parsed independently; failures are reported as warnings so one bad
    block does not lose the rest of the document.

    Natural-language explanations can be attached to a constraint by either
    passing ``default_description`` (applied to every constraint extracted
    from this block) or by embedding an OCL ``--`` comment inline (per
    constraint). An inline comment always wins over ``default_description``.

    Returns:
        ``(routing_tuples, warnings)`` where each routing tuple has the
        same shape as :func:`parse_constraint_text`'s return value.
    """
    if not ocl_text:
        return [], []

    routing: list[tuple[str, OCLConstraint, Optional[str], Optional[str]]] = []
    warnings: list[str] = []

    blocks = re.split(r"(?i)(?=\bcontext\b)", ocl_text)
    block_idx = 0

    for block in blocks:
        block = block.strip()
        if not block or not block.lower().startswith("context"):
            continue
        # Extract inline ``--`` description per block before collapsing
        # newlines: OCL comments terminate at end-of-line, so newlines must
        # still be present when we scan for them.
        cleaned_block, inline_description = _extract_inline_description(block)
        line = cleaned_block.replace("\n", " ").strip()
        block_idx += 1

        try:
            kind, constraint, class_name, method_name = parse_constraint_text(line, domain_model)
        except BOCLSyntaxError as e:
            warnings.append(f"Warning: Invalid OCL syntax in '{line}': {e}")
            continue
        except ValueError as e:
            warnings.append(f"Warning: Could not parse OCL constraint '{line}': {e}")
            continue

        # Auto-generate a fallback name only when the user didn't supply one.
        # ``parse_ocl`` initialises constraints with ``name='parsed'``; if it's
        # still that value we know no header name was captured.
        if constraint.name in (None, "", "parsed"):
            if kind == "invariant":
                constraint.name = f"{class_name}_inv_{counter}_{block_idx}"
            else:
                base_method = method_name or class_name
                constraint.name = f"{base_method}_{_KW_FROM_KIND[kind]}_{counter}_{block_idx}"

        # Inline ``--`` description always wins over the per-element default.
        description = inline_description or (default_description or None)
        if description:
            constraint.description = description

        routing.append((kind, constraint, class_name, method_name))

    return routing, warnings


def legacy_body_only_to_text(
    body: str,
    kind: str,
    name: Optional[str],
    context_class: Class,
    method: Optional[Method] = None,
) -> str:
    """Reconstruct full-text OCL from the body-only intermediate JSON shape.

    The WME briefly stored constraints in a body-only form during an
    earlier iteration (textarea = ``self.pages > 0``, with ``kind`` /
    ``targetMethodId`` / ``constraintName`` siblings). To keep those JSON
    files loadable without a manual migration, the ingest path detects
    them and synthesises the canonical full-text form here. The result
    is then fed through the normal :func:`parse_constraint_text` path.

    New code should not call this; the canonical wire shape is full text.

    Raises:
        ValueError: invalid ``kind``, or pre/post without a ``method``.
    """
    if kind not in _KIND_FROM_KW.values():
        raise ValueError(
            f"legacy_body_only_to_text: unknown kind {kind!r}; "
            f"expected one of {tuple(_KIND_FROM_KW.values())}"
        )
    safe_name = (name or "").strip()
    if kind == "invariant":
        if safe_name:
            return f"context {context_class.name} inv {safe_name}: {body}"
        return f"context {context_class.name} inv: {body}"
    if method is None:
        raise ValueError(
            f"legacy_body_only_to_text: method is required when kind={kind!r}"
        )
    params = ", ".join(
        f"{p.name}: {_typeref_name(p.type)}" for p in method.parameters
    )
    return f"context {context_class.name}::{method.name}({params}) {_KW_FROM_KIND[kind]}: {body}"
