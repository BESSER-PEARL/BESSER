"""Deployment diagram conversion: DeploymentModel -> WME JSON.

``deployment_object_to_json`` is a pure metamodel-object -> JSON walk, the
inverse of ``json_to_buml.deployment_diagram_processor``. The multiplicity of
an artifact's first ``DeploymentRelation`` goes into the artifact's name
suffix (``"Coder [3]"``) and a ``DeploymentComponent`` is emitted as the
editor's ``DeploymentComponent`` element. ``deployment_buml_to_json(content)``
loads a BUML file produced by ``deployment_model_to_code``.
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
from besser.utilities.web_modeling_editor.backend.services.converters.multiplicity_format import (
    format_to_name,
)
from besser.utilities.web_modeling_editor.backend.services.converters.stereotype_tokens import (
    format_artifact_stereotype,
    format_node_stereotype,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json._safe_buml_loader import (
    safe_load_buml,
    strip_buml_imports,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
)

logger = logging.getLogger(__name__)


def deployment_object_to_json(model: DeploymentModel) -> dict:
    """Convert a ``DeploymentModel`` into a WME Deployment diagram (JSON dict).

    Args:
        model: The ``DeploymentModel`` to convert.

    Returns:
        dict: WME ``UMLModel`` envelope with ``elements`` and ``relationships``.
    """
    if not isinstance(model, DeploymentModel):
        raise ConversionError(
            f"deployment_object_to_json expects a DeploymentModel, "
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

    all_nodes = sort_by_timestamp(model.all_nodes())
    all_artifacts = sort_by_timestamp(model.all_artifacts())
    all_interfaces = sort_by_timestamp(model.interfaces)
    all_relationships = sort_by_timestamp(model.relationships)

    # Pre-mint ids for stable ordering.
    for n in all_nodes:
        id_for(n)
    for a in all_artifacts:
        id_for(a)
    for iface in all_interfaces:
        id_for(iface)
    for rel in all_relationships:
        id_for(rel)

    # Index artifact -> deterministic DeploymentRelation for the multiplicity
    # round-trip (the first relation by timestamp wins on divergence).
    artifact_to_relation = _index_artifact_to_relation(all_relationships)

    for node in all_nodes:
        elements[id_for(node)] = _emit_node_entry(node, id_for)

    # Artifacts (including nested ones)
    for artifact in all_artifacts:
        elements[id_for(artifact)] = _emit_artifact_entry(
            artifact, id_for, artifact_to_relation,
        )

    for interface in all_interfaces:
        elements[id_for(interface)] = _emit_interface_entry(interface, id_for)

    # Relationships — emit explicit edges; owner-link-only relations are
    # already encoded via the artifact's `owner` field and are skipped here
    # to avoid double-encoding (see processor dedup logic).
    for rel in all_relationships:
        if (rel.layout or {}).get("wme_origin") == "owner":
            continue
        relationships[id_for(rel)] = _emit_relationship_entry(rel, id_for)

    size = _compute_size(elements)

    return {
        "version": "3.0.0",
        "type": "DeploymentDiagram",
        "size": size,
        "interactive": {"elements": {}, "relationships": {}},
        "elements": elements,
        "relationships": relationships,
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


def _emit_node_entry(node: Node, id_for) -> dict:
    """Emit a DeploymentNode entry."""
    layout = node.layout or {}
    wme_type = layout.get("wme_type") or "DeploymentNode"
    bounds = layout.get("bounds") or {"x": 0, "y": 0, "width": 200, "height": 140}
    entry = {
        "id": id_for(node),
        "name": node.name,
        "type": wme_type,
        "owner": id_for(node.parent) if node.parent is not None else None,
        "bounds": bounds,
    }
    stereotype = format_node_stereotype(node)
    if stereotype:
        entry["stereotype"] = stereotype
        entry["displayStereotype"] = bool(layout.get("displayStereotype", True))
    return entry


def _emit_artifact_entry(artifact: Artifact, id_for,
                         artifact_to_relation: dict) -> dict:
    """Emit a DeploymentArtifact or DeploymentComponent entry.

    Suffixes the artifact's name with the multiplicity of its first
    DeploymentRelation (when non-default). Falls back to
    ``layout["original_name"]`` when the multiplicity is the default and the
    imported name carried an explicit suffix (e.g. ``"Coder [1]"``).
    """
    layout = artifact.layout or {}
    is_component = isinstance(artifact, DeploymentComponent)
    wme_type = "DeploymentComponent" if is_component else "DeploymentArtifact"
    bounds = layout.get("bounds") or {"x": 0, "y": 0, "width": 160, "height": 40}

    rel = artifact_to_relation.get(artifact)
    mult = rel.multiplicity if rel is not None else None
    is_default = mult is None or (mult.min == 1 and mult.max == 1)
    if is_default and layout.get("original_name"):
        name = layout["original_name"]
    else:
        name = format_to_name(artifact.name, mult)

    # The artifact's WME owner: its parent Node if nested, else None.
    owner = id_for(artifact.parent) if artifact.parent is not None else None

    entry = {
        "id": id_for(artifact),
        "name": name,
        "type": wme_type,
        "owner": owner,
        "bounds": bounds,
    }
    stereotype = format_artifact_stereotype(artifact)
    if stereotype:
        entry["stereotype"] = stereotype
        entry["displayStereotype"] = bool(layout.get("displayStereotype", True))
    # The editor's DeploymentArtifact always carries ``manifests`` and, for an
    # agent, ``agentModelRef``; a DeploymentComponent only when set.
    if not is_component or artifact.manifests:
        entry["manifests"] = list(artifact.manifests)
    if artifact.agent_model_ref is not None:
        entry["agentModelRef"] = artifact.agent_model_ref
    return entry


def _emit_interface_entry(interface: Interface, id_for) -> dict:
    """Emit a DeploymentInterface entry."""
    layout = interface.layout or {}
    wme_type = layout.get("wme_type") or "DeploymentInterface"
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

    if isinstance(rel, DeploymentRelation):
        wme_type = "DeploymentAssociation"
    elif isinstance(rel, CommunicationPath):
        wme_type = "DeploymentAssociation"
    elif isinstance(rel, DeploymentDependency):
        wme_type = "DeploymentDependency"
    elif isinstance(rel, InterfaceProvided):
        wme_type = "DeploymentInterfaceProvided"
    elif isinstance(rel, InterfaceRequired):
        wme_type = "DeploymentInterfaceRequired"
    else:
        raise ConversionError(
            f"Deployment relationship class {type(rel).__name__} has no editor mapping."
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
    stereotype_parts = list(rel.stereotypes)
    if stereotype_parts:
        entry["stereotype"] = " ".join(stereotype_parts)
    return entry


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


def deployment_buml_to_json(content: str) -> dict:
    """Convert Deployment BUML source into a WME Deployment diagram (JSON).

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
