"""Component diagram conversion: ComponentModel -> v4 WME JSON (``{nodes, edges}``).

``component_object_to_json`` is a pure metamodel-object -> JSON walk, the
inverse of ``json_to_buml.component_diagram_processor``.
``component_buml_to_json(content)`` loads a BUML file produced by
``component_model_to_code`` and converts the model it defines.
Wire shape: ``docs/source/migrations/uml-v4-shape.md`` (ComponentDiagram).
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
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    COMPONENT_REQUIRED_INTERFACE_EDGE_TYPES,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json._node_builders import (
    envelope_size,
    layout_unplaced_nodes,
    make_positions_parent_relative,
    node_from_layout,
    parents_first,
    stereotyped_edge,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json._safe_buml_loader import (
    safe_load_buml,
    strip_buml_imports,
)
from besser.utilities.web_modeling_editor.backend.services.converters.stereotype_tokens import (
    format_agentic_edge_stereotype,
    format_component_stereotype,
    format_component_subtype_stereotype,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
)

_DEFAULT_SIZE = {
    "component": (160, 100),
    "componentSubsystem": (200, 120),
    "componentInterface": (20, 20),
}


def component_object_to_json(model: ComponentModel) -> dict:
    """Convert a ``ComponentModel`` into a v4 WME Component diagram (JSON dict).

    Args:
        model: The ``ComponentModel`` to convert.

    Returns:
        dict: v4 ``UMLModel`` envelope with ``nodes`` and ``edges``.
    """
    if not isinstance(model, ComponentModel):
        raise ConversionError(
            f"component_object_to_json expects a ComponentModel, "
            f"got {type(model).__name__}."
        )

    id_map: dict = {}

    def id_for(obj) -> str:
        if obj not in id_map:
            stashed = (obj.layout or {}).get("id") if obj.layout else None
            id_map[obj] = stashed or str(uuid.uuid4())
        return id_map[obj]

    # Every component, including those reachable only through a Subsystem.
    components = sort_by_timestamp(model.all_components())
    interfaces = sort_by_timestamp(model.interfaces)
    for element in (*components, *interfaces):
        id_for(element)

    nodes = [_component_node(c, id_for) for c in components]
    emitted_ids = {node["id"] for node in nodes}
    nodes += [_interface_node(i, id_for, emitted_ids) for i in interfaces]
    nodes = parents_first(nodes)

    placed_ids = {id_for(e) for e in (*components, *interfaces) if (e.layout or {}).get("bounds")}
    make_positions_parent_relative(nodes, placed_ids)
    layout_unplaced_nodes(nodes, placed_ids)

    edges = [_relationship_edge(rel, id_for) for rel in sort_by_timestamp(model.relationships)]

    return {
        "version": "4.0.0",
        "type": "ComponentDiagram",
        "title": model.name or "",
        "size": envelope_size(nodes),
        "nodes": nodes,
        "edges": edges,
        "interactive": {"elements": {}, "relationships": {}},
        "assessments": {},
    }


def _component_node(component: Component, id_for) -> dict:
    """Emit a ``component`` / ``componentSubsystem`` node."""
    header = bool((component.layout or {}).get("displayStereotype", True))
    parent_id = id_for(component.parent) if component.parent is not None else None
    if isinstance(component, Subsystem):
        data = {
            "name": component.name,
            "stereotype": format_component_stereotype(component),
            "isComponentSubsystemHeaderShown": header,
        }
        return node_from_layout(component, id_for(component), "componentSubsystem", data, parent_id,
                                _DEFAULT_SIZE["componentSubsystem"])
    # Skill/Tool/LLM/Database/RAG get the subtype-promotion token in front.
    data = {
        "name": component.name,
        "stereotype": format_component_subtype_stereotype(component),
        "isComponentHeaderShown": header,
        "realizes": list(component.realizes),
        "processModelRefs": list(component.process_model_refs),
    }
    if component.agent_model_ref is not None:
        data["agentModelRef"] = component.agent_model_ref
    return node_from_layout(component, id_for(component), "component", data, parent_id,
                            _DEFAULT_SIZE["component"])


def _interface_node(interface: Interface, id_for, emitted_ids: set) -> dict:
    """Emit a ``componentInterface`` node; its drawn parent comes from the layout."""
    layout = interface.layout or {}
    parent_id = layout.get("parentId", layout.get("owner"))
    return node_from_layout(
        interface, id_for(interface), "componentInterface", {"name": interface.name},
        parent_id if parent_id in emitted_ids else None, _DEFAULT_SIZE["componentInterface"],
    )


def _relationship_edge(rel, id_for) -> dict:
    """Emit one edge.

    Raises:
        ConversionError: for a relationship class with no editor mapping.
    """
    layout = rel.layout or {}
    if isinstance(rel, InterfaceProvided):
        edge_type = "ComponentProvidedInterface"
        stereotype = " ".join(rel.stereotypes)
    elif isinstance(rel, InterfaceRequired):
        stashed = layout.get("type")
        edge_type = stashed if stashed in COMPONENT_REQUIRED_INTERFACE_EDGE_TYPES else "ComponentRequiredInterface"
        stereotype = " ".join(rel.stereotypes)
    elif isinstance(rel, AgenticEdge):
        edge_type = "ComponentDependency"
        stereotype = format_agentic_edge_stereotype(rel.kind, rel.permissions, list(rel.stereotypes))
    elif isinstance(rel, ComponentDependency):
        edge_type = "ComponentDependency"
        stereotype = " ".join(rel.stereotypes)
    else:
        raise ConversionError(
            f"Component relationship class {type(rel).__name__} has no editor mapping."
        )
    return stereotyped_edge(rel, id_for, edge_type, stereotype)


def component_buml_to_json(content: str) -> dict:
    """Convert Component BUML source into a v4 WME Component diagram (JSON).

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
