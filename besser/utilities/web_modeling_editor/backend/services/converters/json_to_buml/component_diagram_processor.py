"""Component diagram processing: WME JSON -> ComponentModel.

Implements 02-... §4. Three-pass walk (nodes, containment, relationships)
with layout-side-channel passthrough, agentic-profile token resolution, and
``AgenticEdge`` dispatch on the ``ComponentDependency`` wire type.
"""

import logging
from typing import Optional

from besser.BUML.metamodel.uml_component import (
    AgenticEdge,
    Component,
    ComponentDependency,
    ComponentElement,
    ComponentModel,
    ComponentRelationship,
    Interface,
    InterfaceProvided,
    InterfaceRequired,
    Permission,
    Skill,
    Subsystem,
    Tool,
)
from besser.utilities.web_modeling_editor.backend.services.converters.stereotype_tokens import (
    apply_component_stereotype_tokens,
    extract_permission_scopes,
    parse_agentic_edge_kind,
    parse_component_node_subtype,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
)

logger = logging.getLogger(__name__)


def process_component_diagram(json_data: dict) -> ComponentModel:
    """Convert a WME Component diagram (JSON) into a ``ComponentModel``.

    Args:
        json_data: ``{"title": <str>, "model": {...}}`` envelope.

    Returns:
        ComponentModel: assembled metamodel instance.

    Raises:
        ConversionError: structural failures (missing ``model``, unrecoverable
            element shape).
    """
    if not isinstance(json_data, dict):
        raise ConversionError(
            f"Component diagram payload must be a dict, "
            f"got {type(json_data).__name__}."
        )
    model_data = json_data.get("model")
    if not isinstance(model_data, dict):
        raise ConversionError("Component diagram JSON is missing the 'model' key.")
    elements = model_data.get("elements") or {}
    relationships = model_data.get("relationships") or {}
    name = json_data.get("title") or "Generated_Component_Model"

    permissions_by_scope: dict = {}
    nodes_by_id: dict = _build_nodes(elements)
    _reconstruct_containment(elements, nodes_by_id)
    relationship_objects = _build_relationships(
        relationships, nodes_by_id, permissions_by_scope,
    )

    components = {obj for obj in nodes_by_id.values() if isinstance(obj, Component)}
    interfaces = {obj for obj in nodes_by_id.values() if isinstance(obj, Interface)}

    return ComponentModel(
        name=name,
        components=components,
        interfaces=interfaces,
        permissions=set(permissions_by_scope.values()),
        relationships=set(relationship_objects),
    )


def _build_nodes(elements: dict) -> dict:
    """Pass 1 — create every element object, indexed by WME id."""
    nodes_by_id: dict = {}
    for elem_id, elem in elements.items():
        if not isinstance(elem, dict):
            logger.warning(
                "Component element %s is not a dict (got %s); skipping.",
                elem_id, type(elem).__name__,
            )
            continue
        obj = _build_component_node(elem_id, elem)
        if obj is None:
            continue
        # Stash the layout side-channel (D10).
        obj.layout = {
            "id": elem_id,
            "owner": elem.get("owner"),
            "bounds": elem.get("bounds"),
            "displayStereotype": elem.get("displayStereotype", True),
            "wme_type": elem.get("type"),
        }
        nodes_by_id[elem_id] = obj
    return nodes_by_id


def _build_component_node(elem_id: str, elem: dict) -> Optional[ComponentElement]:
    """Build a single Component / Subsystem / Skill / Tool / Interface from
    a WME element dict.

    Returns ``None`` when the element type is unknown (logs a warning;
    house-pattern: collect-don't-throw on a single bad element).
    """
    elem_type = elem.get("type")
    name = elem.get("name") or ""
    stereotype = elem.get("stereotype") or ""

    if elem_type == "ComponentInterface":
        return Interface(name=name)

    if elem_type == "Subsystem":
        subsystem = Subsystem(name=name)
        apply_component_stereotype_tokens(subsystem, stereotype)
        return subsystem

    if elem_type == "Component":
        subtype = parse_component_node_subtype(stereotype)
        if subtype == "Skill":
            obj = Skill(name=name)
        elif subtype == "Tool":
            obj = Tool(name=name)
        else:
            obj = Component(name=name)
        apply_component_stereotype_tokens(obj, stereotype)
        return obj

    logger.warning(
        "Component element %s has unknown type %r; skipping.", elem_id, elem_type,
    )
    return None


def _reconstruct_containment(elements: dict, nodes_by_id: dict) -> None:
    """Pass 2 — turn WME ``owner`` pointers into ``Subsystem.add_child`` calls.

    A WME element's ``owner`` references another element's id. For
    Component-diagram nodes, valid containment is *child -> Subsystem*; any
    other shape (component pointing to interface, dangling owner) is
    log-and-skip (the node stays top-level).
    """
    for elem_id, elem in elements.items():
        if not isinstance(elem, dict):
            continue
        child = nodes_by_id.get(elem_id)
        if child is None:
            continue
        owner_id = elem.get("owner")
        if not owner_id:
            continue
        parent = nodes_by_id.get(owner_id)
        if parent is None:
            logger.warning(
                "Component element %s references unknown owner %s; treating as root.",
                elem_id, owner_id,
            )
            continue
        if not isinstance(parent, Subsystem):
            logger.warning(
                "Component element %s owner %s is %s, not Subsystem; "
                "treating as root.",
                elem_id, owner_id, type(parent).__name__,
            )
            continue
        if not isinstance(child, Component):
            # Interfaces inside a Subsystem are not modelled (the metamodel
            # places Interfaces on the model root). Drop the containment
            # link; the Interface stays at root level.
            logger.warning(
                "Component element %s of type %s cannot be a child of "
                "Subsystem %s; staying at root.",
                elem_id, type(child).__name__, owner_id,
            )
            continue
        parent.add_child(child)


def _build_relationships(relationships: dict, nodes_by_id: dict,
                         permissions_by_scope: dict) -> list:
    """Pass 3 — build relationship objects from the ``relationships`` map.

    AgenticEdge dispatch and permission-suffix parsing happen here. Returns
    a list of relationship objects (in iteration order); the caller wraps
    them into the model's relationship set.
    """
    out: list = []
    for rel_id, rel in relationships.items():
        if not isinstance(rel, dict):
            logger.warning(
                "Component relationship %s is not a dict; skipping.", rel_id,
            )
            continue
        source_id = (rel.get("source") or {}).get("element")
        target_id = (rel.get("target") or {}).get("element")
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(target_id)
        if source is None or target is None:
            logger.warning(
                "Component relationship %s has dangling endpoint(s) "
                "(source=%s, target=%s); skipping.",
                rel_id, source_id, target_id,
            )
            continue
        try:
            obj = _build_component_relationship(
                rel_id, rel, source, target, permissions_by_scope,
            )
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Component relationship %s failed to construct (%s); skipping.",
                rel_id, exc,
            )
            continue
        if obj is None:
            continue
        obj.layout = {
            "id": rel_id,
            "owner": rel.get("owner"),
            "bounds": rel.get("bounds"),
            "path": rel.get("path"),
            "source_direction": (rel.get("source") or {}).get("direction"),
            "target_direction": (rel.get("target") or {}).get("direction"),
            "isManuallyLayouted": rel.get("isManuallyLayouted", False),
            "wme_type": rel.get("type"),
        }
        out.append(obj)
    return out


def _build_component_relationship(
    rel_id: str, rel: dict, source, target, permissions_by_scope: dict,
) -> Optional[ComponentRelationship]:
    """Build one relationship object from a WME edge dict.

    Branches on ``rel["type"]`` and (for ``ComponentDependency``) on the
    agentic-edge kind extracted from the stereotype.
    """
    rel_type = rel.get("type")
    name = rel.get("name") or ""
    stereotype = rel.get("stereotype") or ""

    if rel_type == "ComponentInterfaceProvided":
        return InterfaceProvided(source=source, target=target, name=name)
    if rel_type == "ComponentInterfaceRequired":
        return InterfaceRequired(source=source, target=target, name=name)
    if rel_type == "ComponentDependency":
        kind = parse_agentic_edge_kind(stereotype)
        if kind is None:
            return ComponentDependency(source=source, target=target, name=name)
        permissions = _resolve_permissions(stereotype, permissions_by_scope)
        return AgenticEdge(
            source=source, target=target, kind=kind,
            permissions=permissions, name=name,
        )

    logger.warning(
        "Component relationship %s has unknown type %r; skipping.",
        rel_id, rel_type,
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
