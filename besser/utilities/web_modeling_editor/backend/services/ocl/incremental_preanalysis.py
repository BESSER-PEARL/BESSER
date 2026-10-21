"""On-demand chapter-4 pre-analysis for a class diagram sent by the editor."""

from typing import Any

from besser.BUML.notations.ocl.incremental import analyze_invariants
from besser.BUML.metamodel.structural_event import event_payloads
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_class_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
)


_STORED_EVENT_FIELDS = ("violatingEvents", "violatingEventsSource")


def _without_stored_events(json_data: dict[str, Any]) -> dict[str, Any]:
    """Return a diagram payload that does not carry a previous analysis.

    The editor keeps the last run on each OCL box. Rebuilding those links
    while starting a new run turns a stale or unmatched event into a warning,
    and the new analysis replaces the links anyway. The computation reads the
    constraint text only.
    """
    model = json_data.get("model") or {}
    elements = model.get("elements") or {}
    cleaned: dict[str, Any] = {}
    changed = False
    for element_id, element in elements.items():
        if (
            isinstance(element, dict)
            and element.get("type") == "ClassOCLConstraint"
            and any(field in element for field in _STORED_EVENT_FIELDS)
        ):
            element = {
                key: value for key, value in element.items() if key not in _STORED_EVENT_FIELDS
            }
            changed = True
        cleaned[element_id] = element
    if not changed:
        return json_data
    return {**json_data, "model": {**model, "elements": cleaned}}


def incremental_preanalysis(json_data: dict[str, Any]) -> dict[str, Any]:
    """Analyze every invariant in a class-diagram payload.

    The returned ``source`` is the constraint expression that was analyzed,
    so the editor can tell a later edit apart from this result. Events are
    ``ViolatingEvent`` links on the temporary domain model; the editor
    copies them onto the constraint boxes. Events already stored on the
    boxes are ignored, so a second run starts from the current text.

    Raises:
        ConversionError: if ``json_data`` is not a class diagram.
    """
    model = json_data.get("model") or {}
    if model.get("type") != "ClassDiagram":
        raise ConversionError("Incremental pre-analysis only applies to class diagrams.")

    domain = process_class_diagram(_without_stored_events(json_data))
    results = analyze_invariants(domain)
    by_name = {constraint.name: constraint for constraint in domain.constraints}
    element_ids = getattr(domain, "ocl_element_ids", {})
    box_text = getattr(domain, "ocl_box_text", {})
    warnings = list(getattr(domain, "ocl_warnings", []) or [])
    items = []
    for name, result in results.items():
        constraint = by_name[name]
        warnings.extend(result.warnings)
        items.append({
            "name": name,
            "elementId": element_ids.get(name),
            "source": box_text.get(name, constraint.expression),
            "warnings": list(result.warnings),
            "events": event_payloads(constraint),
        })
    return {"constraints": items, "warnings": warnings}
