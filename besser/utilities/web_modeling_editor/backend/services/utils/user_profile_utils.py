"""Utilities for rendering a UserDiagram into a normalized user-profile JSON document.

These helpers were previously colocated in ``generation_router.py`` but are
needed from both the router and ``services/deployment/github_deploy_api.py``.
Keeping them here avoids the circular import between router and deployment.

Pattern: this module is specific to the FastAPI backend context, mirroring
``agent_generation_utils.py`` — it is therefore fine to raise ``HTTPException``
directly so callers do not need to translate domain errors.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import uuid
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set

from fastapi import HTTPException

from besser.utilities.web_modeling_editor.backend.config import get_generator_info
from besser.utilities.web_modeling_editor.backend.constants.user_buml_model import (
    domain_model as user_reference_domain_model,
)
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_object_diagram,
)

logger = logging.getLogger(__name__)


def safe_path(base_dir: str, user_filename: str) -> str:
    """Resolve a user-provided filename safely within base_dir."""
    safe_name = os.path.basename(user_filename)
    full_path = os.path.realpath(os.path.join(base_dir, safe_name))
    if not full_path.startswith(os.path.realpath(base_dir)):
        raise ValueError("Invalid path")
    return full_path


def sanitize_object_model_filename(name: Optional[str]) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', (name or "object_model").strip())
    return cleaned or "object_model"


def build_user_model_node(
    object_id: str,
    objects_by_id: Dict[str, Dict[str, Any]],
    include_identity: bool,
    path: Set[str],
) -> Optional[Dict[str, Any]]:
    if object_id in path:
        return None

    obj = objects_by_id.get(object_id)
    if not obj:
        return None

    path.add(object_id)
    try:
        node: Dict[str, Any] = {}
        if include_identity:
            node["id"] = obj.get("id")
            node["class"] = obj.get("class")

        attributes = obj.get("attributes")
        if isinstance(attributes, dict):
            node.update(attributes)

        child_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        relationships = obj.get("relationships")
        if isinstance(relationships, dict):
            for target_ids in relationships.values():
                if not isinstance(target_ids, list):
                    continue
                for target_id in target_ids:
                    child_obj = objects_by_id.get(target_id)
                    if not child_obj:
                        continue
                    child_node = build_user_model_node(
                        target_id,
                        objects_by_id,
                        include_identity=False,
                        path=path,
                    )
                    if child_node is None:
                        continue
                    key = child_obj.get("class") or child_obj.get("id")
                    if not key:
                        continue
                    child_groups[key].append(child_node)

        for child_key, children in child_groups.items():
            if not children:
                continue
            if len(children) == 1:
                node[child_key] = children[0]
            else:
                node[child_key] = children

        return node
    finally:
        path.remove(object_id)


def build_user_model_hierarchy(document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    objects = document.get("objects")
    if not isinstance(objects, list):
        return None

    objects_by_id: Dict[str, Dict[str, Any]] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        object_id = obj.get("id")
        if object_id:
            objects_by_id[object_id] = obj

    if not objects_by_id:
        return None

    root_id = next(
        (obj_id for obj_id, obj in objects_by_id.items() if obj.get("class") == "User"),
        None,
    )
    if not root_id:
        return None

    root_model = build_user_model_node(root_id, objects_by_id, include_identity=True, path=set())
    if root_model is None:
        return None

    normalized_document = {key: value for key, value in document.items() if key != "objects"}
    normalized_document["model"] = root_model
    return normalized_document


def normalize_user_model_output(object_model, temp_dir: str) -> None:
    """Rewrite the generated JSON so the ``objects`` list is folded into a
    hierarchical ``model`` tree rooted on the ``User`` instance."""
    file_name = sanitize_object_model_filename(getattr(object_model, "name", None))
    json_path = safe_path(temp_dir, f"{file_name}.json")
    if not os.path.isfile(json_path):
        return

    try:
        with open(json_path, "r", encoding="utf-8") as source:
            document = json.load(source)
    except (OSError, json.JSONDecodeError):
        return

    normalized_document = build_user_model_hierarchy(document)
    if not normalized_document:
        return

    with open(json_path, "w", encoding="utf-8") as target:
        json.dump(normalized_document, target, indent=2, ensure_ascii=False)


def generate_user_profile_document(user_profile_model: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the normalized JSON document for a stored user profile diagram.

    Raises:
        HTTPException(400): if the input payload is not a valid UserDiagram.
        HTTPException(500): if JSONObject generator is not configured or fails
            to render the user profile document.
    """
    if not isinstance(user_profile_model, dict):
        raise HTTPException(
            status_code=400,
            detail="userProfileModel must contain a serialized UserDiagram",
        )

    diagram_title = (
        user_profile_model.get("title")
        or user_profile_model.get("name")
        or user_profile_model.get("id")
        or "UserProfile"
    )
    prepared_payload = {
        "title": diagram_title,
        "diagramType": "UserDiagram",
        "model": deepcopy(user_profile_model),
        "generator": "jsonobject",
    }

    model_section = prepared_payload["model"]
    if isinstance(model_section, dict):
        model_section.setdefault("type", "UserDiagram")

    try:
        with tempfile.TemporaryDirectory(prefix=f"user_profile_{uuid.uuid4().hex}_") as temp_dir:
            object_model = process_object_diagram(prepared_payload, user_reference_domain_model)
            generator_info = get_generator_info("jsonobject")
            if not generator_info:
                raise HTTPException(status_code=500, detail="JSONObject generator is not configured")
            generator_class = generator_info.generator_class
            generator_instance = generator_class(object_model, output_dir=temp_dir)
            generator_instance.generate()

            normalize_user_model_output(object_model, temp_dir)

            file_name = sanitize_object_model_filename(getattr(object_model, "name", None))
            json_path = safe_path(temp_dir, f"{file_name}.json")
            if not os.path.isfile(json_path):
                raise HTTPException(
                    status_code=500,
                    detail="Failed to render user profile JSON document",
                )

            with open(json_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to convert user profile model")
        raise HTTPException(
            status_code=400,
            detail="Failed to convert user profile model. Please check the input data.",
        ) from exc
