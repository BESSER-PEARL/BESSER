"""Utilities for rendering a UserDiagram into a normalized user-profile JSON document.

These helpers were previously colocated in ``generation_router.py`` but are
needed from both the router and ``services/deployment/github_deploy_api.py``.
Keeping them here avoids the circular import between router and deployment.

This module is service-layer code: it raises BESSER's custom exceptions
(``ValidationError``, ``GenerationError``) rather than ``HTTPException`` so
the same helpers can be reused outside the FastAPI request lifecycle, and
router callers translate to HTTP status codes via ``@handle_endpoint_errors``.
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

from besser.utilities.web_modeling_editor.backend.constants.user_buml_model import (
    domain_model as user_reference_domain_model,
)
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_object_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    GenerationError,
    ValidationError,
)

logger = logging.getLogger(__name__)


def safe_path(base_dir: str, user_filename: str) -> str:
    """Resolve a user-provided filename safely within base_dir."""
    safe_name = os.path.basename(user_filename)
    full_path = os.path.realpath(os.path.join(base_dir, safe_name))
    real_base = os.path.realpath(base_dir)
    try:
        if os.path.commonpath([full_path, real_base]) != real_base:
            raise ValueError("Invalid path")
    except ValueError as exc:
        # commonpath raises ValueError when paths are on different drives
        # (Windows) or otherwise incomparable — treat as not contained.
        # Re-raise the explicit traversal-rejection with its real cause; only
        # the cross-drive fallback should swallow the original exception.
        if "Invalid path" in str(exc):
            raise
        raise ValueError("Invalid path") from None
    return full_path


def sanitize_object_model_filename(name: Optional[str]) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', (name or "object_model").strip())
    return cleaned or "object_model"


# Root class name of the user metamodel — the box that heads a profile.
_ROOT_CLASS_NAME = "User"
# Identity attribute on the root `User` box that names the profile.
_USER_NAME_ATTRIBUTE = "name"
# Mirror of the frontend `criterionValue` regex (user-profile-graph.ts): splits
# a criterion string "attr <op> value" into (attr, value) on the first operator.
_CRITERION_RE = re.compile(r"^(.*?)(?:<=|>=|==|=|<|>)(.*)$")


def _criterion_value(raw: str, attribute_name: str) -> Optional[str]:
    """Return the value of a named criterion (e.g. ``name = Frenchguy`` → ``Frenchguy``).

    Returns ``None`` when the criterion is for a different attribute, matching
    the frontend reader so backend file names track what the user typed.
    """
    text = raw or ""
    match = _CRITERION_RE.match(text)
    name = (match.group(1) if match else text).strip()
    if name != attribute_name:
        return None
    return (match.group(2) if match else "").strip()


def extract_user_profile_names(json_data: Dict[str, Any]) -> Dict[str, str]:
    """Map each ``User`` box's object name to its human profile name.

    The profile name is carried by a ``name`` ``UserModelAttribute`` on the
    ``User`` box (e.g. "Frenchguy"). It is not a domain property, so
    object-model conversion drops it — we read it straight from the raw diagram
    so output files can be named after the profile ("Frenchguy.json") instead
    of the auto-generated box id ("user_1.json").

    Keyed by the box ``name``, which becomes the generated object's ``id`` and
    thus the ``model.id`` that ``normalize_user_model_output`` matches on. Boxes
    without a filled-in ``name`` attribute are omitted (callers fall back).
    """
    names: Dict[str, str] = {}
    if not isinstance(json_data, dict):
        return names
    model_data = json_data.get("model", {})
    if not isinstance(model_data, dict):
        return names
    elements = model_data.get("elements", {})
    if not elements and isinstance(model_data.get("model"), dict):
        elements = model_data["model"].get("elements", {})
    if not isinstance(elements, dict):
        return names

    for element in elements.values():
        if not isinstance(element, dict):
            continue
        if (
            element.get("type") != "UserModelName"
            or element.get("className") != _ROOT_CLASS_NAME
        ):
            continue
        object_name = element.get("name")
        if not object_name:
            continue
        attribute_ids = element.get("attributes")
        if not isinstance(attribute_ids, list):
            continue
        for attr_id in attribute_ids:
            attr = elements.get(attr_id)
            if not isinstance(attr, dict) or attr.get("type") != "UserModelAttribute":
                continue
            value = _criterion_value(attr.get("name", ""), _USER_NAME_ATTRIBUTE)
            if value:
                names[object_name] = value
                break
    return names


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


def build_user_model_hierarchies(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build one normalized model tree per ``User`` instance in the document.

    A single UserDiagram canvas can hold several user profiles — one ``User``
    object each. Each becomes its own normalized document (the shared metadata
    of ``document`` minus its flat ``objects`` list, plus a ``model`` tree
    rooted on that ``User``). Callers emit one file per entry, zipping when
    there is more than one.
    """
    objects = document.get("objects")
    if not isinstance(objects, list):
        return []

    objects_by_id: Dict[str, Dict[str, Any]] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        object_id = obj.get("id")
        if object_id:
            objects_by_id[object_id] = obj

    if not objects_by_id:
        return []

    user_ids = [
        obj_id for obj_id, obj in objects_by_id.items() if obj.get("class") == "User"
    ]
    if not user_ids:
        return []

    hierarchies: List[Dict[str, Any]] = []
    for root_id in user_ids:
        root_model = build_user_model_node(
            root_id, objects_by_id, include_identity=True, path=set()
        )
        if root_model is None:
            continue
        normalized_document = {
            key: value for key, value in document.items() if key != "objects"
        }
        normalized_document["model"] = root_model
        hierarchies.append(normalized_document)
    return hierarchies


def build_user_model_hierarchy(document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Backward-compatible single-root variant: the first ``User`` hierarchy."""
    hierarchies = build_user_model_hierarchies(document)
    return hierarchies[0] if hierarchies else None


def normalize_user_model_output(
    object_model,
    temp_dir: str,
    profile_names: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Rewrite the generated JSON so the flat ``objects`` list is folded into a
    hierarchical ``model`` tree rooted on each ``User`` instance.

    Returns the list of JSON basenames written to ``temp_dir``:

    * A single ``User`` (the common case, including every stored single-profile
      diagram) rewrites the generated file in place and returns ``[basename]``.
    * Multiple ``User`` instances (several profiles on one canvas) replace the
      combined file with one file per profile, named after the profile — its
      ``name`` attribute value when known (via ``profile_names``, keyed by the
      root ``model.id``), else the ``model`` name/id — sanitized and then
      de-duplicated with ``_2``, ``_3`` …; returns all basenames so callers zip
      when there is >1.

    Returns ``[]`` when there is nothing to fold (missing/unreadable file, or no
    ``User`` root), leaving whatever the generator wrote untouched.
    """
    profile_names = profile_names or {}
    file_name = sanitize_object_model_filename(getattr(object_model, "name", None))
    json_path = safe_path(temp_dir, f"{file_name}.json")
    if not os.path.isfile(json_path):
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as source:
            document = json.load(source)
    except (OSError, json.JSONDecodeError):
        return []

    hierarchies = build_user_model_hierarchies(document)
    if not hierarchies:
        return []

    # Single profile: keep the original filename and rewrite in place so stored
    # single-User diagrams and existing callers see no behavioral change.
    if len(hierarchies) == 1:
        with open(json_path, "w", encoding="utf-8") as target:
            json.dump(hierarchies[0], target, indent=2, ensure_ascii=False)
        return [f"{file_name}.json"]

    # Multiple profiles: drop the combined file and emit one file per profile,
    # named by the profile's own root name/id so downloads are self-describing.
    try:
        os.remove(json_path)
    except OSError:
        pass

    written: List[str] = []
    used_names: Set[str] = set()
    for index, hierarchy in enumerate(hierarchies):
        model = hierarchy.get("model") if isinstance(hierarchy, dict) else None
        raw_name = None
        if isinstance(model, dict):
            # Prefer the profile's `name` attribute (e.g. "Frenchguy"), read
            # from the raw diagram since conversion drops it; then the model's
            # own name/id (the auto-generated box id, e.g. "user_1").
            raw_name = (
                profile_names.get(model.get("id"))
                or model.get("name")
                or model.get("id")
            )
        base = sanitize_object_model_filename(raw_name or f"user_{index + 1}")

        # De-duplicate names so two profiles never overwrite the same file.
        candidate = base
        suffix = 2
        while candidate in used_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used_names.add(candidate)

        basename = f"{candidate}.json"
        out_path = safe_path(temp_dir, basename)
        with open(out_path, "w", encoding="utf-8") as target:
            json.dump(hierarchy, target, indent=2, ensure_ascii=False)
        written.append(basename)

    return written


def generate_user_profile_document(user_profile_model: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the normalized JSON document for a stored user profile diagram.

    Raises:
        ValidationError: if the input payload is not a valid UserDiagram or
            cannot be converted from UML JSON.
        GenerationError: if the JSONObject generator is not configured or
            fails to render the user profile document.
    """
    # Local import to avoid a circular import at package load time:
    # ``backend.config`` -> ``BAFGenerator`` -> ``services.converters`` ->
    # ``services.utils``. Importing here keeps this module safe to eagerly
    # re-export from ``services.utils.__init__``.
    from besser.utilities.web_modeling_editor.backend.config import get_generator_info

    if not isinstance(user_profile_model, dict):
        raise ValidationError("userProfileModel must contain a serialized UserDiagram")

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
                raise GenerationError("JSONObject generator is not configured")
            generator_class = generator_info.generator_class
            generator_instance = generator_class(object_model, output_dir=temp_dir)
            generator_instance.generate()

            normalize_user_model_output(object_model, temp_dir)

            file_name = sanitize_object_model_filename(getattr(object_model, "name", None))
            json_path = safe_path(temp_dir, f"{file_name}.json")
            if not os.path.isfile(json_path):
                raise GenerationError("Failed to render user profile JSON document")

            with open(json_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    except (ValidationError, GenerationError):
        raise
    except Exception as exc:
        logger.exception("Failed to convert user profile model")
        raise ValidationError(
            "Failed to convert user profile model. Please check the input data."
        ) from exc
