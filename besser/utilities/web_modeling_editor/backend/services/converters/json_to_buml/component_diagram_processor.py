"""Component diagram processing: v4 WME JSON (``{nodes, edges}``) -> ComponentModel.

Three-pass walk (nodes, containment via ``parentId``, edges) with a layout side
channel, agentic-profile token resolution, and ``AgenticEdge`` dispatch on the
``ComponentDependency`` edge type. Wire shape: ``docs/source/migrations/uml-v4-shape.md``
(ComponentDiagram).

Element or relationship data the metamodel rejects raises ``ConversionError``;
unknown element types and dangling relationship endpoints are logged and
skipped, as in the BPMN processor.
"""

import logging
from typing import Optional

from besser.BUML.metamodel.uml_component import (
    AgenticComponent,
    AgenticComponentModel,
    AgenticEdge,
    Component,
    ComponentDependency,
    ComponentElement,
    ComponentModel,
    ComponentRelationship,
    Database,
    Interface,
    InterfaceProvided,
    InterfaceRequired,
    LLM,
    Permission,
    RAG,
    Skill,
    Subsystem,
    Tool,
)
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    COMPONENT_REQUIRED_INTERFACE_EDGE_TYPES,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml._node_helpers import (
    edge_layout,
    node_data,
    node_layout,
)
from besser.utilities.web_modeling_editor.backend.services.converters.stereotype_tokens import (
    agentic_edge_extra_tokens,
    apply_component_stereotype_tokens,
    extract_permission_scopes,
    parse_agentic_edge_kind,
    parse_component_node_subtype,
    stereotype_has_agentic_tokens,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
)

logger = logging.getLogger(__name__)

# Component subtype class-name (from parse_component_node_subtype) → class.
# Mirrors COMPONENT_SUBTYPE_TOKENS values; llm/db/rag are Tool subclasses.
_COMPONENT_SUBTYPE_CLASSES = {
    "Skill": Skill,
    "Tool": Tool,
    "LLM": LLM,
    "Database": Database,
    "RAG": RAG,
}



def process_component_diagram(json_data: dict) -> ComponentModel:
    """Convert a v4 WME Component diagram into a ``ComponentModel``.

    Args:
        json_data: ``{"title": <str>, "model": {"nodes": [...], "edges": [...]}}``.

    Returns:
        ComponentModel: an ``AgenticComponentModel`` when the diagram carries any
        agentic element, a plain ``ComponentModel`` otherwise.

    Raises:
        ConversionError: structural failures (missing ``model``, an element or
            edge the metamodel rejects, a containment cycle).
    """
    if not isinstance(json_data, dict):
        raise ConversionError(
            f"Component diagram payload must be a dict, "
            f"got {type(json_data).__name__}."
        )
    model_data = json_data.get("model")
    if not isinstance(model_data, dict):
        raise ConversionError("Component diagram JSON is missing the 'model' key.")
    nodes = [n for n in (model_data.get("nodes") or []) if isinstance(n, dict)]
    edges = [e for e in (model_data.get("edges") or []) if isinstance(e, dict)]
    name = json_data.get("title") or "Generated_Component_Model"

    permissions_by_scope: dict = {}
    objects_by_id = _build_nodes(nodes)
    _reconstruct_containment(nodes, objects_by_id)
    relationship_objects = _build_relationships(edges, objects_by_id, permissions_by_scope)

    components = {obj for obj in objects_by_id.values() if isinstance(obj, Component)}
    interfaces = {obj for obj in objects_by_id.values() if isinstance(obj, Interface)}
    permissions = set(permissions_by_scope.values())
    relationships_set = set(relationship_objects)

    # Skill / Tool count as agentic-notation vocabulary.
    is_agentic = (
        bool(permissions)
        or any(isinstance(c, (AgenticComponent, Skill, Tool)) for c in components)
        or any(isinstance(r, AgenticEdge) for r in relationships_set)
    )
    if is_agentic:
        return AgenticComponentModel(
            name=name,
            components=components,
            interfaces=interfaces,
            permissions=permissions,
            relationships=relationships_set,
        )
    return ComponentModel(
        name=name,
        components=components,
        interfaces=interfaces,
        relationships=relationships_set,
    )


def _build_nodes(nodes: list) -> dict:
    """Pass 1 — create every element object, indexed by node id."""
    nodes_by_id = {n.get("id"): n for n in nodes if n.get("id")}
    objects_by_id: dict = {}
    for node in nodes:
        node_id = node.get("id")
        try:
            obj = _build_component_node(node_id, node.get("type"), node_data(node))
        except (TypeError, ValueError) as exc:
            raise ConversionError(
                f"Could not build Component element '{node_id}': {exc}"
            ) from exc
        if obj is None:
            continue
        obj.layout = node_layout(node, nodes_by_id)
        objects_by_id[node_id] = obj
    return objects_by_id


def _build_component_node(node_id: str, node_type: str, data: dict) -> Optional[ComponentElement]:
    """Build a Component / Subsystem / Skill / Tool / Interface from a v4 node.

    Returns ``None`` when the node type is unknown (logs a warning).
    """
    name = data.get("name") or ""
    stereotype = data.get("stereotype") or ""

    if node_type == "componentInterface":
        return Interface(name=name)

    if node_type == "componentSubsystem":
        subsystem = Subsystem(name=name)
        apply_component_stereotype_tokens(subsystem, stereotype)
        return subsystem

    if node_type == "component":
        subtype = parse_component_node_subtype(stereotype)
        if subtype is not None:
            # skill/tool/llm/db/rag → the matching Component subclass.
            obj = _COMPONENT_SUBTYPE_CLASSES[subtype](name=name)
        elif stereotype_has_agentic_tokens(stereotype):
            # An agent-category token promotes a bare Component to AgenticComponent.
            obj = AgenticComponent(name=name)
        else:
            obj = Component(name=name)
        apply_component_stereotype_tokens(obj, stereotype)
        # Cross-diagram refs, read as the editor wrote them: realizes -> Class
        # ids, processModelRefs -> BPMN diagram ids, agentModelRef -> the Agent
        # diagram id. Resolved at project level.
        obj.realizes = data.get("realizes") or []
        obj.process_model_refs = data.get("processModelRefs") or []
        obj.agent_model_ref = data.get("agentModelRef") or None
        return obj

    logger.warning(
        "Component element %s has unknown type %r; skipping.", node_id, node_type,
    )
    return None


def _reconstruct_containment(nodes: list, objects_by_id: dict) -> None:
    """Pass 2 — turn ``parentId`` pointers into ``Subsystem.add_child`` calls.

    Valid containment is *Component -> Subsystem*; any other shape (an
    Interface drawn inside a Subsystem, a dangling parent) is logged and the
    element stays top-level. A containment cycle raises ``ConversionError``.
    """
    for node in nodes:
        node_id = node.get("id")
        child = objects_by_id.get(node_id)
        parent_id = node.get("parentId")
        if child is None or not parent_id:
            continue
        parent = objects_by_id.get(parent_id)
        if parent is None:
            logger.warning(
                "Component element %s references unknown parent %s; treating as root.",
                node_id, parent_id,
            )
            continue
        if not isinstance(parent, Subsystem):
            logger.warning(
                "Component element %s parent %s is %s, not Subsystem; treating as root.",
                node_id, parent_id, type(parent).__name__,
            )
            continue
        if not isinstance(child, Component):
            # The metamodel places Interfaces on the model root; the drawn
            # parent survives in the layout only.
            logger.warning(
                "Component element %s of type %s cannot be a child of "
                "Subsystem %s; staying at root.",
                node_id, type(child).__name__, parent_id,
            )
            continue
        try:
            parent.add_child(child)
        except ValueError as exc:
            raise ConversionError(
                f"Component element '{node_id}' cannot be nested in '{parent_id}': {exc}"
            ) from exc


def _build_relationships(edges: list, objects_by_id: dict, permissions_by_scope: dict) -> list:
    """Pass 3 — build relationship objects from the ``edges`` list.

    AgenticEdge dispatch and permission-suffix parsing happen here.
    """
    out: list = []
    for edge in edges:
        edge_id = edge.get("id")
        source = objects_by_id.get(edge.get("source"))
        target = objects_by_id.get(edge.get("target"))
        if source is None or target is None:
            logger.warning(
                "Component relationship %s has dangling endpoint(s) "
                "(source=%s, target=%s); skipping.",
                edge_id, edge.get("source"), edge.get("target"),
            )
            continue
        try:
            obj = _build_component_relationship(edge, source, target, permissions_by_scope)
        except (TypeError, ValueError) as exc:
            raise ConversionError(
                f"Could not build Component relationship '{edge_id}': {exc}"
            ) from exc
        if obj is None:
            continue
        layout = edge_layout(edge)
        layout["type"] = edge.get("type")
        obj.layout = layout
        out.append(obj)
    return out


def _build_component_relationship(
    edge: dict, source, target, permissions_by_scope: dict,
) -> Optional[ComponentRelationship]:
    """Build one relationship object from a v4 edge.

    Branches on the edge ``type`` and (for ``ComponentDependency``) on the
    agentic-edge kind in ``data.stereotype``. A plain relationship keeps its
    stereotype (e.g. «use») verbatim in ``stereotypes``; an AgenticEdge keeps
    the tokens besides its kind and permission suffix.
    """
    edge_type = edge.get("type")
    data = edge.get("data") if isinstance(edge.get("data"), dict) else {}
    name = data.get("name") or data.get("label") or ""
    stereotype = data.get("stereotype") or ""
    free_form = stereotype.split()

    if edge_type == "ComponentProvidedInterface":
        return InterfaceProvided(source=source, target=target, name=name,
                                 stereotypes=free_form)
    if edge_type in COMPONENT_REQUIRED_INTERFACE_EDGE_TYPES:
        return InterfaceRequired(source=source, target=target, name=name,
                                 stereotypes=free_form)
    if edge_type == "ComponentDependency":
        kind = parse_agentic_edge_kind(stereotype)
        if kind is None:
            return ComponentDependency(source=source, target=target, name=name,
                                       stereotypes=free_form)
        permissions = _resolve_permissions(stereotype, permissions_by_scope)
        return AgenticEdge(
            source=source, target=target, kind=kind,
            permissions=permissions, name=name,
            stereotypes=agentic_edge_extra_tokens(stereotype, kind),
        )

    logger.warning(
        "Component relationship %s has unknown type %r; skipping.",
        edge.get("id"), edge_type,
    )
    return None


def _resolve_permissions(stereotype: str, permissions_by_scope: dict) -> list:
    """Parse a ``{permission: scope1, scope2}`` suffix from an agentic-edge
    stereotype string and resolve each scope to a deduped ``Permission``
    instance. The dedup map is shared across the whole diagram.
    """
    out = []
    for scope in extract_permission_scopes(stereotype):
        permission = permissions_by_scope.get(scope)
        if permission is None:
            permission = Permission(name=scope, scope=scope)
            permissions_by_scope[scope] = permission
        out.append(permission)
    return out
