"""Deployment diagram conversion: DeploymentModel -> v4 WME JSON (``{nodes, edges}``).

``deployment_object_to_json`` is a pure metamodel-object -> JSON walk, the
inverse of ``json_to_buml.deployment_diagram_processor``. The multiplicity of
an artifact's first ``DeploymentRelation`` goes into the artifact's name
suffix (``"Coder [3]"``) and a ``DeploymentComponent`` is emitted as the
editor's ``deploymentComponent`` node. ``deployment_buml_to_json(content)``
loads a BUML file produced by ``deployment_model_to_code``.
Wire shape: ``docs/source/migrations/uml-v4-shape.md`` (DeploymentDiagram).
"""

import logging
import uuid

from besser.BUML.metamodel.structural import (
    Multiplicity,
    UNLIMITED_MAX_MULTIPLICITY,
)
from besser.BUML.metamodel.uml_deployment import (
    Artifact,
    CommunicationPath,
    DeploymentComponent,
    DeploymentDependency,
    DeploymentModel,
    DeploymentRelation,
    Interface,
    InterfaceProvided,
    InterfaceRequired,
    Locality,
    Node,
    NodeKind,
)
from besser.utilities.utils import sort_by_timestamp
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    DEPLOYMENT_REQUIRED_INTERFACE_EDGE_TYPES,
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
from besser.utilities.web_modeling_editor.backend.services.converters.multiplicity_format import (
    format_to_name,
)
from besser.utilities.web_modeling_editor.backend.services.converters.stereotype_tokens import (
    format_artifact_stereotype,
    format_node_stereotype,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
)

logger = logging.getLogger(__name__)

_DEFAULT_SIZE = {
    "deploymentNode": (200, 140),
    "deploymentArtifact": (160, 40),
    "deploymentComponent": (160, 100),
    "deploymentInterface": (20, 20),
}


def deployment_object_to_json(model: DeploymentModel) -> dict:
    """Convert a ``DeploymentModel`` into a v4 WME Deployment diagram (JSON dict).

    Args:
        model: The ``DeploymentModel`` to convert.

    Returns:
        dict: v4 ``UMLModel`` envelope with ``nodes`` and ``edges``.
    """
    if not isinstance(model, DeploymentModel):
        raise ConversionError(
            f"deployment_object_to_json expects a DeploymentModel, "
            f"got {type(model).__name__}."
        )

    id_map: dict = {}

    def id_for(obj) -> str:
        if obj not in id_map:
            stashed = (obj.layout or {}).get("id") if obj.layout else None
            id_map[obj] = stashed or str(uuid.uuid4())
        return id_map[obj]

    all_nodes = sort_by_timestamp(model.all_nodes())
    all_artifacts = sort_by_timestamp(model.all_artifacts())
    all_interfaces = sort_by_timestamp(model.interfaces)
    all_relationships = sort_by_timestamp(model.relationships)
    elements = (*all_nodes, *all_artifacts, *all_interfaces)
    for element in elements:
        id_for(element)

    # Artifact -> deterministic DeploymentRelation for the multiplicity
    # round-trip (the first relation by timestamp wins on divergence).
    artifact_to_relation = _index_artifact_to_relation(all_relationships)

    nodes = [_node_node(n, id_for) for n in all_nodes]
    nodes += [_artifact_node(a, id_for, artifact_to_relation) for a in all_artifacts]
    emitted_ids = {node["id"] for node in nodes}
    nodes += [_interface_node(i, id_for, emitted_ids) for i in all_interfaces]
    nodes = parents_first(nodes)

    placed_ids = {id_for(e) for e in elements if (e.layout or {}).get("bounds")}
    make_positions_parent_relative(nodes, placed_ids)
    layout_unplaced_nodes(nodes, placed_ids)

    # Parent-link-only relations are already encoded by the artifact's
    # ``parentId`` and are not emitted as edges (see the processor's dedup).
    edges = [
        _relationship_edge(rel, id_for) for rel in all_relationships
        if (rel.layout or {}).get("wme_origin") != "owner"
    ]

    return {
        "version": "4.0.0",
        "type": "DeploymentDiagram",
        "title": model.name or "",
        "size": envelope_size(nodes),
        "nodes": nodes,
        "edges": edges,
        "interactive": {"elements": {}, "relationships": {}},
        "assessments": {},
    }


def _index_artifact_to_relation(relationships: list) -> dict:
    """Return ``{artifact: first DeploymentRelation by timestamp}`` so the
    artifact-name multiplicity suffix uses a deterministic choice when an
    artifact has multiple DeploymentRelations.

    Logs a warning when an artifact's relations carry *different*
    multiplicities (the first relation by timestamp is used).
    """
    out: dict = {}
    for rel in relationships:
        if not isinstance(rel, DeploymentRelation):
            continue
        artifact = rel.source
        if artifact not in out:
            out[artifact] = rel
            continue
        first = out[artifact]
        if (first.multiplicity.min, first.multiplicity.max) != (
                rel.multiplicity.min, rel.multiplicity.max):
            logger.warning(
                "Artifact '%s' has divergent DeploymentRelation multiplicities "
                "(first: %s..%s vs %s..%s); using the first by timestamp.",
                artifact.name,
                first.multiplicity.min, first.multiplicity.max,
                rel.multiplicity.min, rel.multiplicity.max,
            )
    return out


def _node_node(node: Node, id_for) -> dict:
    """Emit a ``deploymentNode`` node."""
    data = {
        "name": node.name,
        "stereotype": format_node_stereotype(node),
        "isComponentHeaderShown": bool((node.layout or {}).get("displayStereotype", True)),
    }
    parent_id = id_for(node.parent) if node.parent is not None else None
    return node_from_layout(node, id_for(node), "deploymentNode", data, parent_id,
                            _DEFAULT_SIZE["deploymentNode"])


def _artifact_node(artifact: Artifact, id_for, artifact_to_relation: dict) -> dict:
    """Emit a ``deploymentArtifact`` or ``deploymentComponent`` node.

    Suffixes the artifact's name with the multiplicity of its first
    DeploymentRelation (when non-default). Falls back to
    ``layout["original_name"]`` when the multiplicity is the default and the
    imported name carried an explicit suffix (e.g. ``"Coder [1]"``).
    """
    layout = artifact.layout or {}
    is_component = isinstance(artifact, DeploymentComponent)
    node_type = "deploymentComponent" if is_component else "deploymentArtifact"

    rel = artifact_to_relation.get(artifact)
    mult = rel.multiplicity if rel is not None else None
    is_default = mult is None or (mult.min == 1 and mult.max == 1)
    if is_default and layout.get("original_name"):
        name = layout["original_name"]
    else:
        name = format_to_name(artifact.name, mult)

    data = {"name": name}
    stereotype = format_artifact_stereotype(artifact)
    if is_component:
        data["stereotype"] = stereotype
        data["isComponentHeaderShown"] = bool(layout.get("displayStereotype", True))
    elif stereotype:
        data["stereotype"] = stereotype
    # A deploymentArtifact always carries ``manifests`` and, for an agent,
    # ``agentModelRef``; a deploymentComponent only when set.
    if not is_component or artifact.manifests:
        data["manifests"] = list(artifact.manifests)
    if artifact.agent_model_ref is not None:
        data["agentModelRef"] = artifact.agent_model_ref

    parent_id = id_for(artifact.parent) if artifact.parent is not None else None
    return node_from_layout(artifact, id_for(artifact), node_type, data, parent_id,
                            _DEFAULT_SIZE[node_type])


def _interface_node(interface: Interface, id_for, emitted_ids: set) -> dict:
    """Emit a ``deploymentInterface`` node; its drawn parent comes from the layout."""
    layout = interface.layout or {}
    parent_id = layout.get("parentId", layout.get("owner"))
    return node_from_layout(
        interface, id_for(interface), "deploymentInterface", {"name": interface.name},
        parent_id if parent_id in emitted_ids else None, _DEFAULT_SIZE["deploymentInterface"],
    )


def _relationship_edge(rel, id_for) -> dict:
    """Emit one edge.

    Raises:
        ConversionError: for a relationship class with no editor mapping.
    """
    if isinstance(rel, (DeploymentRelation, CommunicationPath)):
        edge_type = "DeploymentAssociation"
    elif isinstance(rel, DeploymentDependency):
        edge_type = "DeploymentDependency"
    elif isinstance(rel, InterfaceProvided):
        edge_type = "DeploymentProvidedInterface"
    elif isinstance(rel, InterfaceRequired):
        stashed = (rel.layout or {}).get("type")
        edge_type = (stashed if stashed in DEPLOYMENT_REQUIRED_INTERFACE_EDGE_TYPES
                     else "DeploymentRequiredInterface")
    else:
        raise ConversionError(
            f"Deployment relationship class {type(rel).__name__} has no editor mapping."
        )
    return stereotyped_edge(rel, id_for, edge_type, " ".join(rel.stereotypes))


def deployment_buml_to_json(content: str) -> dict:
    """Convert Deployment BUML source into a v4 WME Deployment diagram (JSON).

    Strip generated imports, then load the source with the restricted BUML
    loader and an explicit allowlist of Deployment metamodel names. Find the
    resulting ``DeploymentModel`` and delegate to ``deployment_object_to_json``.

    Syntax, name, type, and value errors, including rejected AST constructs,
    become ``ConversionError``. Unexpected exceptions propagate to the
    endpoint's error handler.
    """
    allowed_names = {
        "Artifact": Artifact,
        "CommunicationPath": CommunicationPath,
        "DeploymentComponent": DeploymentComponent,
        "DeploymentDependency": DeploymentDependency,
        "DeploymentModel": DeploymentModel,
        "DeploymentRelation": DeploymentRelation,
        "Interface": Interface,
        "InterfaceProvided": InterfaceProvided,
        "InterfaceRequired": InterfaceRequired,
        "Locality": Locality,
        "Multiplicity": Multiplicity,
        "Node": Node,
        "NodeKind": NodeKind,
        "UNLIMITED_MAX_MULTIPLICITY": UNLIMITED_MAX_MULTIPLICITY,
    }
    try:
        namespace = safe_load_buml(
            strip_buml_imports(content),
            allowed_names,
        )
    except (SyntaxError, NameError, TypeError, ValueError) as exc:
        raise ConversionError(
            f"Deployment BUML file failed to execute: {exc}"
        ) from exc

    model = _find_deployment_model(namespace)
    if model is None:
        raise ConversionError(
            "Deployment BUML file produced no DeploymentModel — expected "
            "a top-level variable (`deployment_model = DeploymentModel(...)` "
            "is the convention emitted by `deployment_model_to_code`)."
        )
    return deployment_object_to_json(model)


def _find_deployment_model(namespace: dict):
    """Return the DeploymentModel from the exec'd namespace, preferring
    the conventional ``deployment_model`` variable name; fall back to any
    DeploymentModel instance."""
    candidate = namespace.get("deployment_model")
    if isinstance(candidate, DeploymentModel):
        return candidate
    for value in namespace.values():
        if isinstance(value, DeploymentModel):
            return value
    return None
