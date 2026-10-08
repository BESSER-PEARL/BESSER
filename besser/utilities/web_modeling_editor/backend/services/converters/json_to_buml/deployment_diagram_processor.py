"""Deployment diagram processing: v4 WME JSON (``{nodes, edges}``) -> DeploymentModel.

Three-pass walk: build nodes / artifacts / interfaces, rebuild containment from
the ``parentId`` chain, then build relationships. The editor's
``deploymentComponent`` node becomes a ``DeploymentComponent`` (an ``Artifact``
subclass the generators never deploy), the multiplicity is parsed from the
artifact name (``"Coder [3]"``), and ``DeploymentAssociation`` is discriminated
on its endpoint types (Artifact -> Node is a ``DeploymentRelation``,
Node <-> Node a ``CommunicationPath``). Wire shape:
``docs/source/migrations/uml-v4-shape.md`` (DeploymentDiagram).

Element or relationship data the metamodel rejects raises ``ConversionError``;
unknown element types and dangling relationship endpoints are logged and
skipped, as in the BPMN processor.
"""

import logging
from typing import Optional

from besser.BUML.metamodel.uml_deployment import (
    Artifact,
    CommunicationPath,
    DeploymentComponent,
    DeploymentDependency,
    DeploymentModel,
    DeploymentRelation,
    DeploymentRelationship,
    Interface,
    InterfaceProvided,
    InterfaceRequired,
    Node,
)
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    DEPLOYMENT_REQUIRED_INTERFACE_EDGE_TYPES,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml._node_helpers import (
    edge_layout,
    node_data,
    node_layout,
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
    """Convert a v4 WME Deployment diagram into a ``DeploymentModel``.

    Args:
        json_data: ``{"title": <str>, "model": {"nodes": [...], "edges": [...]}}``.

    Returns:
        DeploymentModel: assembled metamodel instance.

    Raises:
        ConversionError: structural failures (missing ``model``, an element or
            relationship the metamodel rejects, a containment cycle).
    """
    if not isinstance(json_data, dict):
        raise ConversionError(
            f"Deployment diagram payload must be a dict, "
            f"got {type(json_data).__name__}."
        )
    model_data = json_data.get("model")
    if not isinstance(model_data, dict):
        raise ConversionError("Deployment diagram JSON is missing the 'model' key.")
    nodes = [n for n in (model_data.get("nodes") or []) if isinstance(n, dict)]
    edges = [e for e in (model_data.get("edges") or []) if isinstance(e, dict)]
    name = json_data.get("title") or "Generated_Deployment_Model"

    objects_by_id, parsed_multiplicities = _build_nodes(nodes)
    owner_links = _reconstruct_containment(nodes, objects_by_id)
    relationship_objects = _build_relationships(
        edges, objects_by_id, parsed_multiplicities, owner_links,
    )

    root_nodes = {obj for obj in objects_by_id.values()
                  if isinstance(obj, Node) and obj.parent is None}
    root_artifacts = {obj for obj in objects_by_id.values()
                      if isinstance(obj, Artifact) and obj.parent is None}
    interfaces = {obj for obj in objects_by_id.values() if isinstance(obj, Interface)}

    return DeploymentModel(
        name=name,
        nodes=root_nodes,
        artifacts=root_artifacts,
        interfaces=interfaces,
        relationships=set(relationship_objects),
    )


def _build_nodes(nodes: list) -> tuple:
    """Pass 1 — create every element object, indexed by node id.

    Returns ``(objects_by_id, parsed_multiplicities)`` — the second map keys
    artifact node ids to the ``Multiplicity`` parsed from their names, so
    Pass 3 can apply it on ``DeploymentRelation`` construction.
    """
    nodes_by_id = {n.get("id"): n for n in nodes if n.get("id")}
    objects_by_id: dict = {}
    parsed_multiplicities: dict = {}
    for node in nodes:
        node_id = node.get("id")
        data = node_data(node)
        try:
            obj, parsed_mult = _build_deployment_node(node_id, node.get("type"), data)
        except (TypeError, ValueError) as exc:
            raise ConversionError(
                f"Could not build Deployment element '{node_id}': {exc}"
            ) from exc
        if obj is None:
            continue
        if parsed_mult is not None:
            parsed_multiplicities[node_id] = parsed_mult
        layout = node_layout(node, nodes_by_id)
        # Keep the raw name so the converter restores the exact string when
        # the parsed multiplicity is the default (e.g. "Coder [1]").
        raw_name = data.get("name") or ""
        if parsed_mult is not None and raw_name != obj.name:
            layout["original_name"] = raw_name
        obj.layout = layout
        objects_by_id[node_id] = obj
    return objects_by_id, parsed_multiplicities


def _build_deployment_node(node_id: str, node_type: str, data: dict) -> tuple:
    """Build a single Node / Artifact / DeploymentComponent / Interface.

    Returns ``(obj, parsed_multiplicity)``. ``manifests`` (ids of the
    Components an Artifact manifests) and ``agentModelRef`` (the Agent diagram
    an Artifact deploys) are read from ``data`` as the editor wrote them.
    """
    raw_name = data.get("name") or ""
    stereotype = data.get("stereotype") or ""

    if node_type == "deploymentNode":
        node = Node(name=raw_name)
        apply_node_stereotype_tokens(node, stereotype)
        return node, None

    if node_type == "deploymentInterface":
        return Interface(name=raw_name), None

    if node_type in ("deploymentArtifact", "deploymentComponent"):
        artifact_class = Artifact if node_type == "deploymentArtifact" else DeploymentComponent
        clean_name, mult = parse_from_name(raw_name)
        artifact = artifact_class(
            name=clean_name,
            manifests=data.get("manifests") or [],
            agent_model_ref=data.get("agentModelRef") or None,
        )
        apply_artifact_stereotype_tokens(artifact, stereotype)
        return artifact, mult

    logger.warning(
        "Deployment element %s has unknown type %r; skipping.", node_id, node_type,
    )
    return None, None


def _reconstruct_containment(nodes: list, objects_by_id: dict) -> dict:
    """Pass 2 — turn ``parentId`` pointers into nested object containment.

    Returns ``{artifact_id: node_id}`` for every artifact-on-node parent link,
    so Pass 3 can deduplicate against explicit ``DeploymentAssociation``
    edges (a parent link and an explicit edge express the same deployment;
    only one ``DeploymentRelation`` is emitted).
    """
    owner_links: dict = {}
    for node in nodes:
        node_id = node.get("id")
        child = objects_by_id.get(node_id)
        parent_id = node.get("parentId")
        if child is None or not parent_id:
            continue
        parent = objects_by_id.get(parent_id)
        if parent is None:
            logger.warning(
                "Deployment element %s references unknown parent %s; treating as root.",
                node_id, parent_id,
            )
            continue
        if isinstance(child, Artifact) and isinstance(parent, Node):
            parent.add_artifact(child)
            owner_links[node_id] = parent_id
        elif isinstance(child, Node) and isinstance(parent, Node):
            try:
                parent.add_nested_node(child)
            except ValueError as exc:
                raise ConversionError(
                    f"Deployment node '{node_id}' cannot be nested in '{parent_id}': {exc}"
                ) from exc
        elif isinstance(child, Interface):
            # Interfaces live on the model root; the drawn parent is layout only
            # (kept in ``layout["parentId"]``).
            continue
        else:
            logger.warning(
                "Deployment element %s of type %s cannot be a child of "
                "type %s; staying at root.",
                node_id, type(child).__name__, type(parent).__name__,
            )
    return owner_links


def _build_relationships(edges: list, objects_by_id: dict,
                         parsed_multiplicities: dict, owner_links: dict) -> list:
    """Pass 3 — build relationship objects, discriminating
    ``DeploymentAssociation`` by endpoint type, applying parsed multiplicity
    on ``DeploymentRelation``, and deduplicating against parent links.
    """
    out: list = []
    explicit_artifact_on_node: set = set()  # (artifact_id, node_id) pairs

    for edge in edges:
        edge_id = edge.get("id")
        source_id = edge.get("source")
        target_id = edge.get("target")
        source = objects_by_id.get(source_id)
        target = objects_by_id.get(target_id)
        if source is None or target is None:
            logger.warning(
                "Deployment relationship %s has dangling endpoint(s) "
                "(source=%s, target=%s); skipping.",
                edge_id, source_id, target_id,
            )
            continue
        try:
            obj = _build_deployment_relationship(
                edge, source, target, parsed_multiplicities.get(source_id),
            )
        except (TypeError, ValueError) as exc:
            raise ConversionError(
                f"Could not build Deployment relationship '{edge_id}': {exc}"
            ) from exc
        if obj is None:
            continue
        if isinstance(obj, DeploymentRelation):
            explicit_artifact_on_node.add((source_id, target_id))
        layout = edge_layout(edge)
        layout["type"] = edge.get("type")
        layout["wme_origin"] = "association"
        obj.layout = layout
        out.append(obj)

    # Synthesise DeploymentRelation for parent-link artifact-on-node pairs
    # that did NOT also have an explicit DeploymentAssociation.
    for artifact_id, node_id in owner_links.items():
        if (artifact_id, node_id) in explicit_artifact_on_node:
            continue
        rel = DeploymentRelation(
            source=objects_by_id[artifact_id], target=objects_by_id[node_id],
            multiplicity=parsed_multiplicities.get(artifact_id),
        )
        rel.layout = {"wme_origin": "owner"}
        out.append(rel)

    return out


def _build_deployment_relationship(
    edge: dict, source, target, source_multiplicity,
) -> Optional[DeploymentRelationship]:
    """Build one relationship object from a v4 edge.

    Branches on the edge ``type`` and, for ``DeploymentAssociation``, on the
    endpoint types. The free-form ``data.stereotype`` string is kept as the
    relationship's ``stereotypes`` (whitespace-separated) so it round-trips.
    """
    edge_type = edge.get("type")
    data = edge.get("data") if isinstance(edge.get("data"), dict) else {}
    name = data.get("name") or data.get("label") or ""
    stereotypes = (data.get("stereotype") or "").split()

    if edge_type == "DeploymentDependency":
        return DeploymentDependency(source=source, target=target, name=name,
                                    stereotypes=stereotypes)
    if edge_type == "DeploymentProvidedInterface":
        return InterfaceProvided(source=source, target=target, name=name,
                                 stereotypes=stereotypes)
    if edge_type in DEPLOYMENT_REQUIRED_INTERFACE_EDGE_TYPES:
        return InterfaceRequired(source=source, target=target, name=name,
                                 stereotypes=stereotypes)
    if edge_type == "DeploymentAssociation":
        if isinstance(source, Artifact) and isinstance(target, Node):
            return DeploymentRelation(
                source=source, target=target, name=name, stereotypes=stereotypes,
                multiplicity=source_multiplicity,
            )
        if isinstance(source, Node) and isinstance(target, Node):
            return CommunicationPath(source=source, target=target, name=name,
                                     stereotypes=stereotypes)
        raise ConversionError(
            f"DeploymentAssociation '{edge.get('id')}' must connect Artifact -> Node or "
            f"Node -> Node, got {type(source).__name__} -> {type(target).__name__}."
        )

    logger.warning(
        "Deployment relationship %s has unknown type %r; skipping.",
        edge.get("id"), edge_type,
    )
    return None
