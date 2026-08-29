"""
Object converter module for BUML to JSON conversion.
Handles object diagram processing and attribute mapping.
"""

import ast
import logging
import uuid
from typing import Any

from besser.utilities.web_modeling_editor.backend.services.utils import (
    calculate_connection_points,
    calculate_path_points,
    calculate_relationship_bounds,
    determine_connection_direction,
)

logger = logging.getLogger(__name__)


def object_buml_to_json(content: str, domain_json: dict[str, Any]) -> dict[str, Any]:
    """
    Convert an object model Python file content to JSON format matching the frontend structure.

    Args:
        content: Object model Python code as string
        domain_json: Reference domain model JSON for class mapping

    Returns:
        Dictionary representing the object diagram in JSON format
    """
    elements = {}
    relationships = {}

    # Default diagram size
    default_size = {"width": 960, "height": 670}

    # Grid layout configuration for positioning objects
    grid_size = {
        "x_spacing": 250,
        "y_spacing": 180,
        "max_columns": 4,
    }

    current_column = 0
    current_row = 0

    def extract_attributes_from_call(attr_call: ast.Call) -> dict[str, Any]:
        """Extract attribute name/value pairs from an ``attributes(...)`` fluent call."""
        attributes: dict[str, Any] = {}

        for kw in attr_call.keywords:
            if kw.arg is not None and isinstance(kw.value, ast.Constant):
                attributes[kw.arg] = kw.value.value
                continue

            if kw.arg is None and isinstance(kw.value, ast.Dict):
                for key_node, value_node in zip(kw.value.keys, kw.value.values):
                    if key_node is None:
                        continue
                    try:
                        key = ast.literal_eval(key_node)
                        value = ast.literal_eval(value_node)
                    except (ValueError, SyntaxError):
                        continue
                    if isinstance(key, str):
                        attributes[key] = value

        return attributes

    def extract_relationship_assignment(node: ast.AST) -> list[tuple[str, str, str]]:
        """Extract (source_var, relation_name, target_var) triples.

        Supports both single-target assignments (``setattr(obj, 'rel', t)``
        and ``obj.rel = t``) and the set/list literal form emitted for
        many-valued roles (``setattr(obj, 'rel', {t1, t2})``), which yields
        one triple per target so every link survives the JSON conversion.
        """
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Attribute):
                    continue
                if not isinstance(target.value, ast.Name):
                    continue
                if not isinstance(node.value, ast.Name):
                    continue
                return [(target.value.id, target.attr, node.value.id)]

        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if not isinstance(call.func, ast.Name) or call.func.id != "setattr":
                return []
            if len(call.args) != 3:
                return []

            source_node, relation_node, target_node = call.args
            if not isinstance(source_node, ast.Name):
                return []
            try:
                relation_name = ast.literal_eval(relation_node)
            except (ValueError, SyntaxError):
                return []
            if not isinstance(relation_name, str):
                return []

            if isinstance(target_node, ast.Name):
                return [(source_node.id, relation_name, target_node.id)]

            if isinstance(target_node, (ast.Set, ast.List, ast.Tuple)):
                return [
                    (source_node.id, relation_name, elt.id)
                    for elt in target_node.elts
                    if isinstance(elt, ast.Name)
                ]

        return []

    def get_position() -> tuple[int, int]:
        """Return the (x, y) coordinates for the next object following the grid layout."""
        nonlocal current_column, current_row
        x = -460 + (current_column * grid_size["x_spacing"])
        y = -300 + (current_row * grid_size["y_spacing"])

        current_column += 1
        if current_column >= grid_size["max_columns"]:
            current_column = 0
            current_row += 1

        return x, y

    try:
        reference_diagram_json = domain_json

        # Create mapping from class names to class IDs in reference diagram
        class_name_to_id = {}
        class_id_to_attributes = {}

        for elem_id, elem in reference_diagram_json["elements"].items():
            if elem["type"] in ("Class", "AbstractClass"):
                class_name_to_id[elem["name"]] = elem_id
                # Store class attributes for object attribute mapping
                class_attributes = {}
                for attr_id in elem.get("attributes", []):
                    if attr_id in reference_diagram_json["elements"]:
                        attr_elem = reference_diagram_json["elements"][attr_id]
                        # Extract attribute name (remove visibility and type info)
                        attr_name = attr_elem["name"].split(":")[0].strip().lstrip("+-#~")
                        # Get the type: new format has attributeType, legacy has it in the name
                        attr_type = attr_elem.get("attributeType")
                        if not attr_type:
                            # Legacy format: parse type from name like "+ name: type"
                            parts = attr_elem["name"].split(":")
                            attr_type = parts[1].strip() if len(parts) > 1 else "str"
                        attr_default = attr_elem.get("defaultValue")
                        attr_visibility = attr_elem.get("visibility", "public")
                        class_attributes[attr_name] = {
                            "id": attr_id,
                            "type": attr_type,
                            "defaultValue": attr_default,
                            "visibility": attr_visibility,
                        }
                class_id_to_attributes[elem_id] = class_attributes

        # Parse the Python code to extract object instances
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            raise ValueError(
                f"Failed to parse object model Python code: syntax error at line {e.lineno}: {e.msg}"
            ) from e

        # Track objects and their information
        objects_by_name = {}
        object_class_mapping = {}
        object_comments = {}  # object_var -> comment_text
        om_comment = None  # ObjectModel metadata comment

        # Extract object instantiations using fluent API
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id

                        # Check if this is an object instantiation using fluent API
                        if isinstance(node.value, ast.Call):
                            # Look for pattern: ClassName("ObjectName").attributes(...).build()
                            call_chain = []
                            current = node.value

                            # Walk back through the method chain
                            while isinstance(current, ast.Call):
                                if isinstance(current.func, ast.Attribute):
                                    call_chain.append(current.func.attr)
                                    current = current.func.value
                                elif isinstance(current.func, ast.Name):
                                    call_chain.append(current.func.id)
                                    break
                                else:
                                    break

                            # Reverse to get the correct order
                            call_chain.reverse()

                            # Accept both patterns:
                            # 1) ClassName(...).attributes(...).build()
                            # 2) ClassName(...).build()
                            if len(call_chain) >= 2 and call_chain[-1] == "build":
                                class_name = call_chain[0]
                                if class_name in class_name_to_id:
                                    # Extract object name and attributes
                                    object_name = var_name
                                    object_instance_name = None
                                    attributes = {}

                                    # Get the initial call arguments for object name
                                    init_call = node.value
                                    while isinstance(init_call, ast.Call) and isinstance(init_call.func, ast.Attribute):
                                        init_call = init_call.func.value

                                    if (
                                        isinstance(init_call, ast.Call)
                                        and init_call.args
                                        and isinstance(init_call.args[0], ast.Constant)
                                    ):
                                        object_instance_name = init_call.args[0].value

                                    # Find attributes call in the chain
                                    attr_call = node.value
                                    while isinstance(attr_call, ast.Call):
                                        func = attr_call.func
                                        if isinstance(func, ast.Attribute) and func.attr == "attributes":
                                            attributes = extract_attributes_from_call(attr_call)
                                            break
                                        attr_call = func.value if isinstance(func, ast.Attribute) else None
                                        if not attr_call:
                                            break

                                    objects_by_name[object_name] = {
                                        "class_name": class_name,
                                        "class_id": class_name_to_id[class_name],
                                        "instance_name": object_instance_name or object_name,
                                        "attributes": attributes
                                    }
                                    object_class_mapping[object_name] = class_name

                            # Check for ObjectModel instantiation with metadata
                            elif call_chain and call_chain[0] == "ObjectModel":
                                # Extract ObjectModel metadata
                                for kw in node.value.keywords:
                                    if kw.arg == "metadata" and isinstance(kw.value, ast.Call):
                                        for meta_kw in kw.value.keywords:
                                            if meta_kw.arg == "description":
                                                try:
                                                    om_comment = ast.literal_eval(meta_kw.value)
                                                except (ValueError, TypeError) as e:
                                                    logger.warning(
                                                        "Could not evaluate ObjectModel metadata description: %s", e
                                                    )

                # Check for object.classifier.metadata = Metadata(...) patterns
                target = node.targets[0]
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "metadata"
                    # This could be: obj.classifier.metadata = ...
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr == "classifier"
                ):
                    obj_var = target.value.value.id if isinstance(target.value.value, ast.Name) else None
                    if obj_var and isinstance(node.value, ast.Call):
                        for kw in node.value.keywords:
                            if kw.arg == "description":
                                try:
                                    object_comments[obj_var] = ast.literal_eval(kw.value)
                                except (ValueError, TypeError) as e:
                                    logger.warning(
                                        "Could not evaluate object metadata description for '%s': %s",
                                        obj_var, e
                                    )

        # Create object elements in JSON format
        for obj_name, obj_info in objects_by_name.items():
            x, y = get_position()
            object_id = str(uuid.uuid4())

            # Create object attribute elements
            object_attribute_ids = []
            class_id = obj_info["class_id"]
            class_attributes = class_id_to_attributes.get(class_id, {})

            attr_y_offset = 30
            for attr_name, attr_value in obj_info["attributes"].items():
                attr_id = str(uuid.uuid4())
                object_attribute_ids.append(attr_id)

                # Format the value appropriately
                if isinstance(attr_value, str):
                    formatted_value = attr_value
                else:
                    formatted_value = str(attr_value)

                # Get the corresponding class attribute info if available
                class_attr_info = class_attributes.get(attr_name)
                class_attr_id = class_attr_info["id"] if class_attr_info else None
                attr_type = class_attr_info["type"] if class_attr_info else "str"

                attr_element = {
                    "id": attr_id,
                    "name": f"{attr_name} = {formatted_value}",
                    "type": "ObjectAttribute",
                    "owner": object_id,
                    "bounds": {
                        "x": x + 0.5,
                        "y": y + attr_y_offset - 0.5,
                        "width": 199,
                        "height": 30
                    },
                    "attributeId": class_attr_id,
                    "attributeType": attr_type,
                }
                if class_attr_info and class_attr_info.get("defaultValue") is not None:
                    attr_element["defaultValue"] = class_attr_info["defaultValue"]
                elements[attr_id] = attr_element
                attr_y_offset += 30

            # Calculate object height based on attributes
            object_height = max(70, 40 + len(object_attribute_ids) * 30)

            elements[object_id] = {
                "id": object_id,
                "name": obj_info["instance_name"],
                "type": "ObjectName",
                "owner": None,
                "bounds": {
                    "x": x,
                    "y": y,
                    "width": 200,
                    "height": object_height
                },
                "attributes": object_attribute_ids,
                "methods": [],
                "classId": class_id
            }

        # Look for object links/relationships
        # Map each association end (class + role) to its relationship id using
        # exact matches.  Class-based keys avoid substring collisions between
        # roles shared by several associations (e.g. role "subject" in two
        # different relationships).  A link generated from an Alloy field
        # ``<Class>_<role>`` starts at an object of class *Class* using the role
        # of the *other* end, so the key is (source class, target role) and
        # (target class, source role).
        association_end_map: dict[tuple, str] = {}
        for rel_id, rel in reference_diagram_json["relationships"].items():
            if rel["type"] not in ("ClassBidirectional", "ClassUnidirectional"):
                continue
            src_elem_id = rel.get("source", {}).get("element")
            tgt_elem_id = rel.get("target", {}).get("element")
            src_class = (
                reference_diagram_json["elements"].get(src_elem_id, {}).get("name")
                if src_elem_id else None
            )
            tgt_class = (
                reference_diagram_json["elements"].get(tgt_elem_id, {}).get("name")
                if tgt_elem_id else None
            )
            src_role = rel.get("source", {}).get("role", "")
            tgt_role = rel.get("target", {}).get("role", "")

            for key in ((src_class, tgt_role), (tgt_class, src_role)):
                if key[0] and key[1]:
                    if key in association_end_map and association_end_map[key] != rel_id:
                        # Ambiguous role reused by another association: leave
                        # unmapped so the dedup below never collapses unrelated links.
                        association_end_map[key] = None
                    else:
                        association_end_map[key] = rel_id

        # Map each class to its direct parent via ClassInheritance (source is
        # the subclass, target the superclass).  Links produced from an
        # inherited association field are emitted against the leaf class, so
        # association-id resolution below walks up this chain to find the
        # declaring ancestor.
        class_parents: dict[str, str] = {}
        for rel in reference_diagram_json["relationships"].values():
            if rel.get("type") != "ClassInheritance":
                continue
            child_id = rel.get("source", {}).get("element")
            parent_id = rel.get("target", {}).get("element")
            child_name = (
                reference_diagram_json["elements"].get(child_id, {}).get("name")
                if child_id else None
            )
            parent_name = (
                reference_diagram_json["elements"].get(parent_id, {}).get("name")
                if parent_id else None
            )
            if child_name and parent_name:
                class_parents[child_name] = parent_name

        # Deduplicate object links.  A bidirectional association appears in the
        # Alloy XML as two fields (one per role), which the step-3 converter
        # turns into two setattr lines for the same association.  Collapse them
        # by (association, unordered object pair) so each association yields a
        # single link, while distinct associations between the same objects keep
        # one link each.
        link_keys: set[tuple] = set()

        for node in ast.walk(tree):
            for obj_name, relationship_name, target_obj in extract_relationship_assignment(node):
                if obj_name not in objects_by_name or target_obj not in objects_by_name:
                    continue

                rel_id = str(uuid.uuid4())

                source_id = None
                target_id = None

                for elem_id, elem in elements.items():
                    if elem["type"] == "ObjectName" and elem["name"] == objects_by_name[obj_name]["instance_name"]:
                        source_id = elem_id
                    if elem["type"] == "ObjectName" and elem["name"] == objects_by_name[target_obj]["instance_name"]:
                        target_id = elem_id

                if source_id and target_id:
                    source_class = object_class_mapping.get(obj_name)
                    assoc_id = association_end_map.get((source_class, relationship_name))
                    if assoc_id is None:
                        # The field is declared on an ancestor class; follow the
                        # inheritance chain to resolve its association id.
                        ancestor = class_parents.get(source_class) if source_class else None
                        while ancestor:
                            assoc_id = association_end_map.get((ancestor, relationship_name))
                            if assoc_id is not None:
                                break
                            ancestor = class_parents.get(ancestor)

                    # Canonical key: association (when known) + unordered object pair
                    canonical = (
                        (assoc_id, relationship_name) if assoc_id is None else (assoc_id,),
                        frozenset([source_id, target_id]),
                    )
                    if canonical in link_keys:
                        continue
                    link_keys.add(canonical)

                    relationships[rel_id] = {
                        "id": rel_id,
                        "name": f"{relationship_name}",
                        "type": "ObjectLink",
                        "owner": None,
                        "bounds": {
                            "x": -260,
                            "y": -315,
                            "width": 300,
                            "height": 80
                        },
                        "path": [
                            {"x": 0, "y": 80},
                            {"x": 40, "y": 80},
                            {"x": 40, "y": 0},
                            {"x": 300, "y": 0},
                            {"x": 300, "y": 65}
                        ],
                        "source": {
                            "direction": "Right",
                            "element": source_id
                        },
                        "target": {
                            "direction": "Topleft",
                            "element": target_id
                        },
                        "isManuallyLayouted": False,
                        "associationId": assoc_id
                    }

        # Position for comments
        comment_x = -970
        comment_y = -300

        # Create comment elements from metadata
        # 1. ObjectModel comment (unlinked)
        if om_comment:
            comment_id = str(uuid.uuid4())
            elements[comment_id] = {
                "id": comment_id,
                "name": om_comment,
                "type": "Comments",
                "owner": None,
                "bounds": {
                    "x": comment_x,
                    "y": comment_y,
                    "width": 200,
                    "height": 100,
                },
            }
            comment_y += 130

        # 2. Object comments (linked to objects)
        object_var_to_id = {}  # Map object variable names to their element IDs
        for obj_name, obj_info in objects_by_name.items():
            # Find the object ID for this object
            for elem_id, elem in elements.items():
                if elem.get("type") == "ObjectName" and elem.get("name") == obj_info["instance_name"]:
                    object_var_to_id[obj_name] = elem_id
                    break

        for obj_var, comment_text in object_comments.items():
            if obj_var in object_var_to_id:
                comment_id = str(uuid.uuid4())
                object_id = object_var_to_id[obj_var]

                elements[comment_id] = {
                    "id": comment_id,
                    "name": comment_text,
                    "type": "Comments",
                    "owner": None,
                    "bounds": {
                        "x": comment_x,
                        "y": comment_y,
                        "width": 200,
                        "height": 100,
                    },
                }

                # Create Link relationship
                link_id = str(uuid.uuid4())
                source_element = elements[comment_id]
                target_element = elements[object_id]

                source_dir, target_dir = determine_connection_direction(
                    source_element["bounds"], target_element["bounds"]
                )

                source_point = calculate_connection_points(
                    source_element["bounds"], source_dir
                )
                target_point = calculate_connection_points(
                    target_element["bounds"], target_dir
                )

                path_points = calculate_path_points(
                    source_point, target_point, source_dir, target_dir
                )
                rel_bounds = calculate_relationship_bounds(path_points)

                relationships[link_id] = {
                    "id": link_id,
                    "name": "",
                    "type": "Link",
                    "owner": None,
                    "bounds": rel_bounds,
                    "path": path_points,
                    "source": {
                        "direction": source_dir,
                        "element": comment_id,
                        "bounds": {
                            "x": source_point["x"],
                            "y": source_point["y"],
                            "width": 0,
                            "height": 0,
                        },
                    },
                    "target": {
                        "direction": target_dir,
                        "element": object_id,
                        "bounds": {
                            "x": target_point["x"],
                            "y": target_point["y"],
                            "width": 0,
                            "height": 0,
                        },
                    },
                    "isManuallyLayouted": False,
                }

                comment_y += 130

        return {
            "version": "3.0.0",
            "type": "ObjectDiagram",
            "size": default_size,
            "interactive": {"elements": {}, "relationships": {}},
            "elements": elements,
            "relationships": relationships,
            "assessments": {},
            "referenceDiagramData": reference_diagram_json
        }

    except Exception as e:
        logger.exception("Error parsing object BUML content")
        raise ValueError(f"Failed to convert object BUML to JSON: {e!s}") from e
