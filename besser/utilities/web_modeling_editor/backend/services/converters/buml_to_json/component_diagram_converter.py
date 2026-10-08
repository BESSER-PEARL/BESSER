"""Component diagram conversion: ComponentModel -> WME JSON.

``component_object_to_json`` is a pure metamodel-object -> JSON walk, the
inverse of ``json_to_buml.component_diagram_processor``.
``component_buml_to_json(content)`` loads a BUML file produced by
``component_model_to_code`` and converts the model it defines.
"""

import uuid

from besser.BUML.metamodel.uml_component import (
    AgentCategory,
    AgenticComponent,
    AgenticComponentModel,
    AgenticEdge,
    AgenticEdgeKind,
    Component,
    ComponentDependency,
    ComponentModel,
    Database,
    Interface,
    InterfaceProvided,
    InterfaceRequired,
    LLM,
    Locality,
    Permission,
    RAG,
    Skill,
    Subsystem,
    Tool,
)
from besser.utilities.utils import sort_by_timestamp
from besser.utilities.web_modeling_editor.backend.services.converters.stereotype_tokens import (
    format_agentic_edge_stereotype,
    format_component_stereotype,
    format_component_subtype_stereotype,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json._safe_buml_loader import (
    safe_load_buml,
    strip_buml_imports,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
)

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

    # Every component, including those reachable only through a Subsystem.
    components = sort_by_timestamp(model.all_components())
    interfaces = sort_by_timestamp(model.interfaces)
    model_relationships = sort_by_timestamp(model.relationships)

    # Pre-mint ids in deterministic order so element / relationship ordering
    # in the output is stable.
    for c in components:
        id_for(c)
    for iface in interfaces:
        id_for(iface)
    for rel in model_relationships:
        id_for(rel)

    for component in components:
        elements[id_for(component)] = _emit_component_entry(component, id_for)

    # Interfaces (free-standing — Component-diagram model holds them at root)
    for interface in interfaces:
        elements[id_for(interface)] = _emit_interface_entry(interface, id_for)

    for rel in model_relationships:
        relationships[id_for(rel)] = _emit_relationship_entry(rel, id_for)

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
    # Skill/Tool/LLM/Database/RAG get the subtype-promotion token in front;
    # the editor's «subsystem» / «component» defaults ride along as free-form
    # stereotypes.
    if isinstance(component, Subsystem):
        stereotype = format_component_stereotype(component)
    else:
        stereotype = format_component_subtype_stereotype(component)
    if stereotype:
        entry["stereotype"] = stereotype
        entry["displayStereotype"] = bool(layout.get("displayStereotype", True))
    if wme_type == "Component":
        # The editor's Component element carries the cross-diagram links.
        entry["realizes"] = list(component.realizes)
        entry["processModelRefs"] = list(component.process_model_refs)
        if component.agent_model_ref is not None:
            entry["agentModelRef"] = component.agent_model_ref
    return entry


def _emit_interface_entry(interface: Interface, id_for) -> dict:
    """Emit a ComponentInterface entry."""
    layout = interface.layout or {}
    wme_type = layout.get("wme_type") or "ComponentInterface"
    bounds = layout.get("bounds") or {"x": 0, "y": 0, "width": 20, "height": 20}
    return {
        "id": id_for(interface),
        "name": interface.name,
        "type": wme_type,
        "owner": layout.get("owner"),
        "bounds": bounds,
    }


def _emit_relationship_entry(rel, id_for) -> dict:
    """Emit one relationship entry.

    Raises:
        ConversionError: for a relationship class with no editor mapping.
    """
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
        raise ConversionError(
            f"Component relationship class {type(rel).__name__} has no editor mapping."
        )

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
    """Convert Component BUML source into a WME Component diagram (JSON).

    Strip generated imports, then load the source with the restricted BUML
    loader and an explicit allowlist of Component metamodel names. Find the
    resulting ``ComponentModel`` and delegate to ``component_object_to_json``.

    Syntax, name, type, and value errors, including rejected AST constructs,
    become ``ConversionError``. Unexpected exceptions propagate to the
    endpoint's error handler.
    """
    allowed_names = {
        "AgentCategory": AgentCategory,
        "AgenticComponent": AgenticComponent,
        "AgenticComponentModel": AgenticComponentModel,
        "AgenticEdge": AgenticEdge,
        "AgenticEdgeKind": AgenticEdgeKind,
        "Component": Component,
        "ComponentDependency": ComponentDependency,
        "ComponentModel": ComponentModel,
        "Database": Database,
        "Interface": Interface,
        "InterfaceProvided": InterfaceProvided,
        "InterfaceRequired": InterfaceRequired,
        "LLM": LLM,
        "Locality": Locality,
        "Permission": Permission,
        "RAG": RAG,
        "Skill": Skill,
        "Subsystem": Subsystem,
        "Tool": Tool,
    }
    try:
        namespace = safe_load_buml(
            strip_buml_imports(content),
            allowed_names,
        )
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
