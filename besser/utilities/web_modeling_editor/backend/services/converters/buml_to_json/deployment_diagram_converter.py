"""Deployment diagram conversion: DeploymentModel -> WME JSON.

Implements 02-... §7. Pure metamodel-object → JSON walk. Multiplicity
emission goes into the artifact's name suffix per §3.6.2; the synthetic-
Artifact-as-``DeploymentComponent`` mapping reverses via
``layout["wme_type"]``. ``deployment_buml_to_json(content)`` is gated on
03-.
"""

import logging
import uuid
from typing import Optional

from besser.BUML.metamodel.uml_deployment import (
    Artifact,
    CommunicationPath,
    DeploymentDependency,
    DeploymentModel,
    DeploymentRelation,
    Interface,
    InterfaceProvided,
    InterfaceRequired,
    Node,
)
from besser.utilities.utils import sort_by_timestamp
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

    # Index artifact -> deterministic DeploymentRelation for multiplicity
    # round-trip (Q3 — first-relation wins on divergence).
    artifact_to_relation = _index_artifact_to_relation(all_relationships)

    # Nodes
    for node in all_nodes:
        try:
            elements[id_for(node)] = _emit_node_entry(node, id_for)
        except Exception as exc:
            logger.error("Failed to emit Node '%s': %s", node.name, exc)

    # Artifacts (including nested ones)
    for artifact in all_artifacts:
        try:
            elements[id_for(artifact)] = _emit_artifact_entry(
                artifact, id_for, artifact_to_relation,
            )
        except Exception as exc:
            logger.error("Failed to emit Artifact '%s': %s", artifact.name, exc)

    # Interfaces
    for interface in all_interfaces:
        try:
            elements[id_for(interface)] = _emit_interface_entry(interface)
        except Exception as exc:
            logger.error("Failed to emit Interface '%s': %s", interface.name, exc)

    # Relationships — emit explicit edges; owner-link-only relations are
    # already encoded via the artifact's `owner` field and are skipped here
    # to avoid double-encoding (see processor dedup logic).
    for rel in all_relationships:
        if (rel.layout or {}).get("wme_origin") == "owner":
            continue
        try:
            entry = _emit_relationship_entry(rel, id_for)
            if entry is not None:
                relationships[id_for(rel)] = entry
        except Exception as exc:
            logger.error(
                "Failed to emit Deployment relationship %s: %s",
                type(rel).__name__, exc,
            )

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
    multiplicities (Q3 — first-relation-deterministic).
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
    """Emit a DeploymentArtifact or (synthetic) DeploymentComponent entry.

    Suffixes the artifact's name with the parsed multiplicity from its
    first DeploymentRelation (when non-default), per §3.6.2 / §7.3. Falls
    back to ``layout["original_name"]`` when the parsed multiplicity is the
    default and an original name was stashed (Q4 belt-and-suspenders).
    """
    layout = artifact.layout or {}
    wme_type = layout.get("wme_type") or "DeploymentArtifact"
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
    return entry


def _emit_interface_entry(interface: Interface) -> dict:
    """Emit a DeploymentInterface entry."""
    layout = interface.layout or {}
    wme_type = layout.get("wme_type") or "DeploymentInterface"
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
        logger.warning(
            "Unknown Deployment relationship class %s; skipping.",
            type(rel).__name__,
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
    """Convert a Deployment BUML ``.py`` file's source text into a WME
    Deployment diagram (JSON).

    **Gated on 03-** — needs ``deployment_model_to_code`` from
    ``buml_code_builder``. Raises ``ConversionError`` until then.
    """
    raise ConversionError(
        "deployment_buml_to_json is gated on the 03- code-builder guide; "
        "the file-import path for DeploymentDiagram is not wired yet."
    )
