"""On-demand chapter-3 simplification for a class diagram sent by the editor."""

import re
from typing import Any

from besser.BUML.notations.ocl import simplify_constraint
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_class_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
)
from besser.utilities.web_modeling_editor.backend.services.ocl.incremental_preanalysis import (
    _without_stored_events,
)


_BLOCK_SPLIT = re.compile(r"(?i)(?=\bcontext\b)")
_HEADER_RE = re.compile(
    r"\bcontext\s+\w+"
    r"(?:::\w+\s*\([^)]*\))?"
    r"\s+(?P<kw>inv|pre|post)\b"
    r"(?:\s+(?P<name>\w+))?"
    r"\s*:",
    re.IGNORECASE,
)


def constraint_simplification(json_data: dict[str, Any]) -> dict[str, Any]:
    """Simplify every invariant in a class-diagram payload.

    Each OCL box is rewritten only when a chapter-3 rule changes its body.
    The response then carries the normalized text and the original box text,
    so the editor can show the normalized constraint and attach the original
    as a comment. A box that is already in normal form is reported as
    unchanged and gets no comment.

    Raises:
        ConversionError: if ``json_data`` is not a class diagram.
    """
    model = json_data.get("model") or {}
    if model.get("type") != "ClassDiagram":
        raise ConversionError("Constraint simplification only applies to class diagrams.")

    domain = process_class_diagram(_without_stored_events(json_data))
    by_element = getattr(domain, "ocl_constraints_by_element", {}) or {}
    box_text = getattr(domain, "ocl_box_text", {}) or {}
    warnings = list(getattr(domain, "ocl_warnings", []) or [])
    items = []
    for element_id, constraints in sorted(
        by_element.items(),
        key=lambda pair: min(constraint.name for constraint in pair[1]),
    ):
        shown = ""
        for constraint in constraints:
            shown = box_text.get(constraint.name, shown)
        expression, original, box_warnings = _simplify_box(shown, domain, constraints)
        warnings.extend(box_warnings)
        item = {"elementId": element_id, "changed": original is not None}
        if original is not None:
            item["expression"] = expression
            item["original"] = original
        items.append(item)
    return {"constraints": items, "warnings": warnings}


def _simplify_box(text: str, model, constraints: list) -> tuple[str, str | None, list[str]]:
    """Return ``(new_text, original_or_none, warnings)`` for one OCL box."""
    if not text or not constraints:
        return text, None, []
    pending = list(constraints)
    parts = _BLOCK_SPLIT.split(text)
    changed = False
    warnings: list[str] = []
    new_parts: list[str] = []
    for part in parts:
        stripped = part.strip()
        if not stripped.lower().startswith("context"):
            new_parts.append(part)
            continue
        header = _HEADER_RE.search(part)
        if header is None or header.group("kw").lower() != "inv":
            new_parts.append(part)
            continue
        constraint = _take_constraint(pending, header.group("name"))
        if constraint is None:
            new_parts.append(part)
            continue
        try:
            original = simplify_constraint(constraint, model, source_text=stripped)
        except (RuntimeError, ValueError) as error:
            warnings.append(
                f"Warning: could not simplify '{constraint.name}': {error}"
            )
            new_parts.append(part)
            continue
        if original is None:
            new_parts.append(part)
            continue
        changed = True
        prefix = part[:part.lower().find("context")]
        new_parts.append(prefix + constraint.expression)
    if not changed:
        return text, None, warnings
    return "".join(new_parts), text, warnings


def _take_constraint(pending: list, name: str | None):
    """Remove and return the invariant this block belongs to."""
    if name:
        for index, constraint in enumerate(pending):
            if constraint.name == name:
                return pending.pop(index)
    if len(pending) == 1 or (pending and not name):
        return pending.pop(0)
    return None
