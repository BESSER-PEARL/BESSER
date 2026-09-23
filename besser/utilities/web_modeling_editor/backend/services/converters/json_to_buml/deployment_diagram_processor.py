"""Deployment diagram processing: WME JSON -> DeploymentModel.

Implements 02-... §6. Three-pass walk (nodes / artifacts, containment via
``owner`` chain, relationships) with the synthetic-Artifact mapping for
Apollon's ``DeploymentComponent`` (D12), multiplicity parsing from the
artifact name, and ``DeploymentAssociation`` discrimination on endpoint
types.
"""

import logging
from typing import Optional

from besser.BUML.metamodel.uml_deployment import (
    Artifact,
    CommunicationPath,
    DeploymentDependency,
    DeploymentModel,
    DeploymentRelation,
    DeploymentRelationship,
    Interface,
    InterfaceProvided,
    InterfaceRequired,
    Node,
)
from besser.utilities.web_modeling_editor.backend.services.converters.multiplicity_format import (
    parse_from_name,
)
from besser.utilities.web_modeling_editor.backend.services.converters.stereotype_tokens import (
    apply_artifact_stereotype_tokens,
    apply_node_stereotype_tokens,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
)

logger = logging.getLogger(__name__)


def process_deployment_diagram(json_data: dict) -> DeploymentModel:
    """Convert a WME Deployment diagram (JSON) into a ``DeploymentModel``.

    Args:
        json_data: ``{"title": <str>, "model": {...}}`` envelope.

    Returns:
        DeploymentModel: assembled metamodel instance.

    Raises:
        ConversionError: structural failures.
    """
    if not isinstance(json_data, dict):
        raise ConversionError(
            f"Deployment diagram payload must be a dict, "
            f"got {type(json_data).__name__}."
        )
    model_data = json_data.get("model")
    if not isinstance(model_data, dict):
        raise ConversionError("Deployment diagram JSON is missing the 'model' key.")
    elements = model_data.get("elements") or {}
    relationships = model_data.get("relationships") or {}
    name = json_data.get("title") or "Generated_Deployment_Model"

    nodes_by_id, parsed_multiplicities = _build_nodes(elements)
    owner_links = _reconstruct_containment(elements, nodes_by_id)
    relationship_objects = _build_relationships(
        relationships, nodes_by_id, parsed_multiplicities, owner_links,
    )

    root_nodes = {obj for nid, obj in nodes_by_id.items()
                  if isinstance(obj, Node) and obj.parent is None}
    root_artifacts = {obj for nid, obj in nodes_by_id.items()
                      if isinstance(obj, Artifact) and obj.parent is None}
    interfaces = {obj for obj in nodes_by_id.values() if isinstance(obj, Interface)}

    return DeploymentModel(
        name=name,
        nodes=root_nodes,
        artifacts=root_artifacts,
        interfaces=interfaces,
        relationships=set(relationship_objects),
    )


def _build_nodes(elements: dict) -> tuple:
    """Pass 1 — create every element object, indexed by WME id.

    Returns ``(nodes_by_id, parsed_multiplicities)`` — the second map keys
    artifact WME-ids to the ``Multiplicity`` parsed from their names, so
    Pass 3 can apply it on ``DeploymentRelation`` construction.
    """
    nodes_by_id: dict = {}
    parsed_multiplicities: dict = {}
    for elem_id, elem in elements.items():
        if not isinstance(elem, dict):
            logger.warning(
                "Deployment element %s is not a dict (got %s); skipping.",
                elem_id, type(elem).__name__,
            )
            continue
        obj, parsed_mult = _build_deployment_node(elem_id, elem)
        if obj is None:
            continue
        if parsed_mult is not None:
            parsed_multiplicities[elem_id] = parsed_mult
        layout = {
            "id": elem_id,
            "owner": elem.get("owner"),
            "bounds": elem.get("bounds"),
            "displayStereotype": elem.get("displayStereotype", True),
            "wme_type": elem.get("type"),
        }
        # Stash the raw name so the emitter can restore exact pre-import string
        # when multiplicity is the default — Q4 belt-and-suspenders.
        raw_name = elem.get("name") or ""
        if parsed_mult is not None and raw_name != obj.name:
            layout["original_name"] = raw_name
        obj.layout = layout
        nodes_by_id[elem_id] = obj
    return nodes_by_id, parsed_multiplicities


def _build_deployment_node(elem_id: str, elem: dict) -> tuple:
    """Build a single Node / Artifact / Interface from a WME element dict.

    Returns ``(obj, parsed_multiplicity)``. ``DeploymentComponent`` becomes
    a synthetic ``Artifact`` per D12; its WME id rides in ``manifests`` so
    the cross-diagram reference round-trips.
    """
    elem_type = elem.get("type")
    raw_name = elem.get("name") or ""
    stereotype = elem.get("stereotype") or ""

    if elem_type == "DeploymentNode":
        node = Node(name=raw_name)
        apply_node_stereotype_tokens(node, stereotype)
        return node, None

    if elem_type == "DeploymentInterface":
        return Interface(name=raw_name), None

    if elem_type == "DeploymentArtifact":
        clean_name, mult = parse_from_name(raw_name)
        # 6b-2 — WME stamps the Agent-diagram UUID as `agentModelRef`
        # (guide 33 / full-via-Artifact). Absent on non-agent artifacts.
        agent_model_ref = elem.get("agentModelRef")
        artifact = Artifact(name=clean_name, agent_model_ref=agent_model_ref)
        apply_artifact_stereotype_tokens(artifact, stereotype)
        return artifact, mult

    if elem_type == "DeploymentComponent":
        # D12: synthetic Artifact representing the Component, the
        # original Component id lives in manifests (the cross-diagram link).
        clean_name, mult = parse_from_name(raw_name)
        agent_model_ref = elem.get("agentModelRef")
        artifact = Artifact(name=clean_name, manifests=[elem_id],
                            agent_model_ref=agent_model_ref)
        apply_artifact_stereotype_tokens(artifact, stereotype)
        return artifact, mult

    logger.warning(
        "Deployment element %s has unknown type %r; skipping.", elem_id, elem_type,
    )
    return None, None


def _reconstruct_containment(elements: dict, nodes_by_id: dict) -> dict:
    """Pass 2 — turn WME ``owner`` pointers into nested object containment.

    Returns ``{artifact_id: node_id}`` for every artifact-on-node owner link,
    so Pass 3 can deduplicate against explicit ``DeploymentAssociation``
    edges (Appendix B.2 rule: owner-link and explicit edge express the
    same deployment; emit only one ``DeploymentRelation``).
    """
    owner_links: dict = {}
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
                "Deployment element %s references unknown owner %s; "
                "treating as root.",
                elem_id, owner_id,
            )
            continue
        if isinstance(child, Artifact) and isinstance(parent, Node):
            parent.add_artifact(child)
            owner_links[elem_id] = owner_id
        elif isinstance(child, Node) and isinstance(parent, Node):
            if child is parent:
                logger.warning(
                    "Deployment node %s claims itself as owner; treating as root.",
                    elem_id,
                )
                continue
            parent.add_nested_node(child)
        elif isinstance(child, Interface):
            # Interfaces are model-root in the metamodel — drop the
            # containment link silently.
            continue
        else:
            logger.warning(
                "Deployment element %s of type %s cannot be a child of "
                "type %s; staying at root.",
                elem_id, type(child).__name__, type(parent).__name__,
            )
    return owner_links


def _build_relationships(relationships: dict, nodes_by_id: dict,
                         parsed_multiplicities: dict, owner_links: dict) -> list:
    """Pass 3 — build relationship objects, discriminating
    ``DeploymentAssociation`` by endpoint type, applying parsed multiplicity
    on ``DeploymentRelation``, and dedup'ing against owner-link synthesis.
    """
    out: list = []
    explicit_artifact_on_node: set = set()  # (artifact_id, node_id) pairs

    for rel_id, rel in relationships.items():
        if not isinstance(rel, dict):
            logger.warning(
                "Deployment relationship %s is not a dict; skipping.", rel_id,
            )
            continue
        source_id = (rel.get("source") or {}).get("element")
        target_id = (rel.get("target") or {}).get("element")
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(target_id)
        if source is None or target is None:
            logger.warning(
                "Deployment relationship %s has dangling endpoint(s) "
                "(source=%s, target=%s); skipping.",
                rel_id, source_id, target_id,
            )
            continue
        try:
            obj = _build_deployment_relationship(
                rel_id, rel, source, target, source_id, target_id,
                parsed_multiplicities,
            )
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Deployment relationship %s failed to construct (%s); skipping.",
                rel_id, exc,
            )
            continue
        if obj is None:
            continue
        if isinstance(obj, DeploymentRelation):
            explicit_artifact_on_node.add((source_id, target_id))
        obj.layout = {
            "id": rel_id,
            "owner": rel.get("owner"),
            "bounds": rel.get("bounds"),
            "path": rel.get("path"),
            "source_direction": (rel.get("source") or {}).get("direction"),
            "target_direction": (rel.get("target") or {}).get("direction"),
            "isManuallyLayouted": rel.get("isManuallyLayouted", False),
            "wme_type": rel.get("type"),
            "wme_origin": "association",
        }
        out.append(obj)

    # Synthesise DeploymentRelation for owner-link artifact-on-node pairs
    # that did NOT also have an explicit DeploymentAssociation.
    for artifact_id, node_id in owner_links.items():
        if (artifact_id, node_id) in explicit_artifact_on_node:
            continue
        artifact = nodes_by_id.get(artifact_id)
        node = nodes_by_id.get(node_id)
        if not isinstance(artifact, Artifact) or not isinstance(node, Node):
            continue
        mult = parsed_multiplicities.get(artifact_id)
        try:
            rel = DeploymentRelation(
                source=artifact, target=node, multiplicity=mult,
            )
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Synthesised DeploymentRelation for artifact %s on node %s "
                "failed (%s); skipping.",
                artifact_id, node_id, exc,
            )
            continue
        rel.layout = {"wme_origin": "owner"}
        out.append(rel)

    return out


def _build_deployment_relationship(
    rel_id: str, rel: dict, source, target,
    source_id: str, target_id: str, parsed_multiplicities: dict,
) -> Optional[DeploymentRelationship]:
    """Build one relationship object from a WME edge dict.

    Branches on ``rel["type"]`` and (for ``DeploymentAssociation``) on the
    endpoint types per §3.6.1.
    """
    rel_type = rel.get("type")
    name = rel.get("name") or ""

    if rel_type == "DeploymentDependency":
        return DeploymentDependency(source=source, target=target, name=name)
    if rel_type == "DeploymentInterfaceProvided":
        return InterfaceProvided(source=source, target=target, name=name)
    if rel_type == "DeploymentInterfaceRequired":
        return InterfaceRequired(source=source, target=target, name=name)
    if rel_type == "DeploymentAssociation":
        if isinstance(source, Artifact) and isinstance(target, Node):
            mult = parsed_multiplicities.get(source_id)
            return DeploymentRelation(
                source=source, target=target, multiplicity=mult, name=name,
            )
        if isinstance(source, Node) and isinstance(target, Node):
            return CommunicationPath(source=source, target=target, name=name)
        logger.warning(
            "DeploymentAssociation %s has unsupported endpoint types: "
            "%s -> %s; skipping.",
            rel_id, type(source).__name__, type(target).__name__,
        )
        return None

    logger.warning(
        "Deployment relationship %s has unknown type %r; skipping.",
        rel_id, rel_type,
    )
    return None
