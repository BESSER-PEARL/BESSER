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


def incremental_preanalysis(json_data: dict[str, Any]) -> dict[str, Any]:
    """Analyze every invariant in a class-diagram payload.

    The returned ``source`` is the constraint expression that was analyzed,
    so the editor can tell a later edit apart from this result. Events are
    ``ViolatingEvent`` links on the temporary domain model; the editor
    copies them onto the constraint boxes.

    Raises:
        ConversionError: if ``json_data`` is not a class diagram.
    """
    model = json_data.get("model") or {}
    if model.get("type") != "ClassDiagram":
        raise ConversionError("Incremental pre-analysis only applies to class diagrams.")

    domain = process_class_diagram(json_data)
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
