"""Component diagram conversion: ComponentModel -> WME JSON.

Implements 02-... §5. Pure metamodel-object → JSON walk; the file-import
``component_buml_to_json(content)`` wrapper is gated on
``03-component-deployment-code-builders-guide.md`` (which lands the
``component_model_to_code`` builder this wrapper exec()'s).
"""

import logging
import uuid
from typing import Optional

from besser.BUML.metamodel.uml_component import (
    AgenticEdge,
    Component,
    ComponentDependency,
    ComponentModel,
    Interface,
    InterfaceProvided,
    InterfaceRequired,
    Subsystem,
)
from besser.utilities.utils import sort_by_timestamp
from besser.utilities.web_modeling_editor.backend.services.converters.stereotype_tokens import (
    format_agentic_edge_stereotype,
    format_component_stereotype,
    format_component_subtype_stereotype,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
)

logger = logging.getLogger(__name__)


def component_object_to_json(model: ComponentModel) -> dict:
    """Convert a ``ComponentModel`` into a WME Component diagram (JSON dict).

    Args:
        model: The ``ComponentModel`` to convert.

    Returns:
        dict: WME ``UMLModel`` envelope with ``elements`` and ``relationships``.
    """
    if not isinstance(model, ComponentModel):
        raise ConversionError(
            f"component_object_to_json expects a ComponentModel, "
            f"got {type(model).__name__}."
        )

    elements: dict = {}
    relationships: dict = {}
    id_map: dict = {}

    def id_for(obj) -> str:
        if obj not in id_map:
            stashed = (obj.layout or {}).get("id") if obj.layout else None
            id_map[obj] = stashed or str(uuid.uuid4())
        return id_map[obj]

    # Pre-mint ids in deterministic order so element / relationship ordering
    # in the output is stable.
    for c in sort_by_timestamp(model.components):
        id_for(c)
    for iface in sort_by_timestamp(model.interfaces):
        id_for(iface)
    for rel in sort_by_timestamp(model.relationships):
        id_for(rel)

    # Components
    for component in sort_by_timestamp(model.components):
        try:
            entry = _emit_component_entry(component, id_for)
            elements[id_for(component)] = entry
        except Exception as exc:
            logger.error(
                "Failed to emit Component '%s': %s", component.name, exc,
            )

    # Interfaces (free-standing — Component-diagram model holds them at root)
    for interface in sort_by_timestamp(model.interfaces):
        try:
            entry = _emit_interface_entry(interface)
            elements[id_for(interface)] = entry
        except Exception as exc:
            logger.error("Failed to emit Interface '%s': %s", interface.name, exc)

    # Relationships
    for rel in sort_by_timestamp(model.relationships):
        try:
            entry = _emit_relationship_entry(rel, id_for)
            if entry is not None:
                relationships[id_for(rel)] = entry
        except Exception as exc:
            logger.error(
                "Failed to emit Component relationship %s: %s",
                type(rel).__name__, exc,
            )

    size = _compute_size(elements)

    return {
        "version": "3.0.0",
        "type": "ComponentDiagram",
        "size": size,
        "interactive": {"elements": {}, "relationships": {}},
        "elements": elements,
        "relationships": relationships,
        "assessments": {},
    }


def _emit_component_entry(component: Component, id_for) -> dict:
    """Emit a Component / Subsystem / Skill / Tool node entry."""
    layout = component.layout or {}
    if isinstance(component, Subsystem):
        wme_type = layout.get("wme_type") or "Subsystem"
    else:
        wme_type = layout.get("wme_type") or "Component"
    bounds = layout.get("bounds") or _default_component_bounds(wme_type)
    entry = {
        "id": id_for(component),
        "name": component.name,
        "type": wme_type,
        "owner": id_for(component.parent) if component.parent is not None else None,
        "bounds": bounds,
    }
    # Subsystem keeps the literal "subsystem" stereotype; Skill/Tool get the
    # subtype-promotion prefix on the stereotype string.
    if isinstance(component, Subsystem):
        base_stereotype = format_component_stereotype(component)
        stereotype = "subsystem" if not base_stereotype else f"subsystem {base_stereotype}"
    else:
        stereotype = format_component_subtype_stereotype(component)
    if stereotype:
        entry["stereotype"] = stereotype
        entry["displayStereotype"] = bool(layout.get("displayStereotype", True))
    return entry


def _emit_interface_entry(interface: Interface) -> dict:
    """Emit a ComponentInterface entry."""
    layout = interface.layout or {}
    wme_type = layout.get("wme_type") or "ComponentInterface"
    bounds = layout.get("bounds") or {"x": 0, "y": 0, "width": 20, "height": 20}
    return {
        "id": (layout.get("id") or str(uuid.uuid4())),
        "name": interface.name,
        "type": wme_type,
        "owner": layout.get("owner"),
        "bounds": bounds,
    }


def _emit_relationship_entry(rel, id_for) -> Optional[dict]:
    """Emit one relationship entry."""
    layout = rel.layout or {}
    bounds = layout.get("bounds") or {"x": 0, "y": 0, "width": 1, "height": 1}
    path = layout.get("path") or [{"x": 0, "y": 0}, {"x": 1, "y": 0}]
    source = {
        "element": id_for(rel.source),
        "direction": layout.get("source_direction") or "Right",
    }
    target = {
        "element": id_for(rel.target),
        "direction": layout.get("target_direction") or "Left",
    }
    is_manually_layouted = bool(layout.get("isManuallyLayouted", False))

    if isinstance(rel, InterfaceProvided):
        wme_type = "ComponentInterfaceProvided"
        stereotype = " ".join(rel.stereotypes)
    elif isinstance(rel, InterfaceRequired):
        wme_type = "ComponentInterfaceRequired"
        stereotype = " ".join(rel.stereotypes)
    elif isinstance(rel, AgenticEdge):
        wme_type = "ComponentDependency"
        stereotype = format_agentic_edge_stereotype(
            rel.kind, rel.permissions, list(rel.stereotypes),
        )
    elif isinstance(rel, ComponentDependency):
        wme_type = "ComponentDependency"
        stereotype = " ".join(rel.stereotypes)
    else:
        logger.warning(
            "Unknown Component relationship class %s; skipping.", type(rel).__name__,
        )
        return None

    entry = {
        "id": id_for(rel),
        "name": rel.name or "",
        "type": wme_type,
        "owner": layout.get("owner"),
        "bounds": bounds,
        "path": path,
        "source": source,
        "target": target,
        "isManuallyLayouted": is_manually_layouted,
    }
    if stereotype:
        entry["stereotype"] = stereotype
    return entry


def _default_component_bounds(wme_type: str) -> dict:
    """Return a placeholder bounds dict for freshly-built (no-layout) elements."""
    if wme_type == "Subsystem":
        return {"x": 0, "y": 0, "width": 200, "height": 120}
    return {"x": 0, "y": 0, "width": 160, "height": 100}


def _compute_size(elements: dict) -> dict:
    """Compute the diagram bounding box (min 800x600) from all element bounds."""
    max_x = 800
    max_y = 600
    for entry in elements.values():
        bounds = entry.get("bounds") or {}
        right = (bounds.get("x") or 0) + (bounds.get("width") or 0)
        bottom = (bounds.get("y") or 0) + (bounds.get("height") or 0)
        if right > max_x:
            max_x = right
        if bottom > max_y:
            max_y = bottom
    return {"width": int(max_x), "height": int(max_y)}


def component_buml_to_json(content: str) -> dict:
    """Convert a Component BUML ``.py`` file's source text into a WME
    Component diagram (JSON).

    Execs ``content`` in a fresh namespace, finds the resulting
    ``ComponentModel``, and delegates to ``component_object_to_json``.
    Wraps the four expected exec failure modes (``SyntaxError``,
    ``NameError``, ``TypeError``, ``ValueError``) into ``ConversionError``
    so ``@handle_endpoint_errors`` maps them to a 400. Other exception
    types propagate as 500s — that's the load-bearing distinction
    between bad-upload and backend-broken (see 03-... §7 / BPMN 04- §5
    for the full reasoning).
    """
    namespace: dict = {}
    try:
        exec(content, namespace)
    except (SyntaxError, NameError, TypeError, ValueError) as exc:
        raise ConversionError(
            f"Component BUML file failed to execute: {exc}"
        ) from exc

    model = _find_component_model(namespace)
    if model is None:
        raise ConversionError(
            "Component BUML file produced no ComponentModel — expected "
            "a top-level variable (`component_model = ComponentModel(...)` "
            "is the convention emitted by `component_model_to_code`)."
        )
    return component_object_to_json(model)


def _find_component_model(namespace: dict):
    """Return the ComponentModel from the exec'd namespace, preferring
    the conventional ``component_model`` variable name; fall back to any
    ComponentModel instance in the namespace."""
    candidate = namespace.get("component_model")
    if isinstance(candidate, ComponentModel):
        return candidate
    for value in namespace.values():
        if isinstance(value, ComponentModel):
            return value
    return None
