"""
Object converter module for BUML to JSON conversion.
Handles object diagram processing and attribute mapping.
"""

import ast
import logging
import uuid
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

from besser.utilities.web_modeling_editor.backend.services.utils import (
    determine_connection_direction, calculate_connection_points,
    calculate_path_points, calculate_relationship_bounds
)


#: Element types per diagram type: (object element, attribute element).
#: A UserDiagram is an object diagram over the user reference domain model; the
#: editor gives its objects and criteria their own element types, which
#: ``json_to_buml.object_diagram_processor`` reads back the same way.
_ELEMENT_TYPES = {
    "ObjectDiagram": ("ObjectName", "ObjectAttribute"),
    "UserDiagram": ("UserModelName", "UserModelAttribute"),
}


def _attribute_value(node: ast.AST):
    """Return the value an ``.attributes(name=<node>)`` keyword carries, or None.

    Mirrors what ``object_model_to_code`` writes: literals (including negative
    numbers), enumeration literals as ``Enum.LITERAL`` (read back as the literal
    name) and datetimes as ``datetime.datetime.fromisoformat("...")`` (read
    back as the ISO string).
    """
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        pass
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.attr
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fromisoformat"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return node.args[0].value
    return None


def object_buml_to_json(
    content: str,
    domain_json: Dict[str, Any],
    diagram_type: str = "ObjectDiagram",
) -> Dict[str, Any]:
    """
    Convert an object model Python file content to JSON format matching the frontend structure.

    Args:
        content: Object model Python code as string
        domain_json: Reference domain model JSON for class mapping
        diagram_type: ``"ObjectDiagram"`` (default) or ``"UserDiagram"``. A
            UserDiagram uses ``UserModelName`` / ``UserModelAttribute``
            elements (equality criteria, ``attributeOperator`` ``"=="``) and
            names each object's class in ``className``; it carries no
            ``referenceDiagramData``, since the editor supplies the user
            reference model itself.

    Returns:
        Dictionary representing the object diagram in JSON format
    """
    if diagram_type not in _ELEMENT_TYPES:
        raise ValueError(f"Unsupported object diagram type: {diagram_type}")
    is_user_diagram = diagram_type == "UserDiagram"
    object_element_type, attribute_element_type = _ELEMENT_TYPES[diagram_type]

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

    def get_position():
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

        # Superclasses per class (ClassInheritance: source = specific, target = general)
        parents_of: Dict[str, List[str]] = {}
        for rel in reference_diagram_json["relationships"].values():
            if rel.get("type") == "ClassInheritance":
                parents_of.setdefault(rel["source"]["element"], []).append(rel["target"]["element"])

        def lineage(class_id: str) -> List[str]:
            """The class followed by all its ancestors, nearest first."""
            seen, order, queue = set(), [], [class_id]
            while queue:
                current = queue.pop(0)
                if current in seen:
                    continue
                seen.add(current)
                order.append(current)
                queue.extend(parents_of.get(current, []))
            return order

        # Inherited attributes resolve too; a class's own attribute wins over an ancestor's.
        class_id_to_attributes = {
            class_id: {
                name: info
                for ancestor in reversed(lineage(class_id))
                for name, info in class_id_to_attributes.get(ancestor, {}).items()
            }
            for class_id in class_id_to_attributes
        }

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

                            # Check if this matches the fluent pattern; an object
                            # without slots is just ClassName("ObjectName").build()
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

                                    if isinstance(init_call, ast.Call) and len(init_call.args) > 0:
                                        if isinstance(init_call.args[0], ast.Constant):
                                            object_instance_name = init_call.args[0].value

                                    # Find attributes call in the chain
                                    attr_call = node.value
                                    while isinstance(attr_call, ast.Call):
                                        if isinstance(attr_call.func, ast.Attribute) and attr_call.func.attr == "attributes":
                                            # Extract attributes from keyword arguments
                                            for kw in attr_call.keywords:
                                                value = _attribute_value(kw.value)
                                                if value is not None:
                                                    attributes[kw.arg] = value
                                            break
                                        attr_call = attr_call.func.value if isinstance(attr_call.func, ast.Attribute) else None
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
                                    if kw.arg == "metadata":
                                        if isinstance(kw.value, ast.Call):
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
                if isinstance(target, ast.Attribute) and target.attr == "metadata":
                    # This could be: obj.classifier.metadata = ...
                    if isinstance(target.value, ast.Attribute) and target.value.attr == "classifier":
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
        object_var_to_id = {}  # object variable name -> element ID
        for obj_name, obj_info in objects_by_name.items():
            x, y = get_position()
            object_id = str(uuid.uuid4())
            object_var_to_id[obj_name] = object_id

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
                    "type": attribute_element_type,
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
                if is_user_diagram:
                    # A BUML slot is an equality criterion; the comparator of a
                    # non-equality criterion ("age < 18") is not in the metamodel.
                    attr_element["attributeOperator"] = "=="
                if class_attr_info and class_attr_info.get("defaultValue") is not None:
                    attr_element["defaultValue"] = class_attr_info["defaultValue"]
                elements[attr_id] = attr_element
                attr_y_offset += 30

            # Calculate object height based on attributes
            object_height = max(70, 40 + len(object_attribute_ids) * 30)

            object_element = {
                "id": object_id,
                "name": obj_info["instance_name"],
                "type": object_element_type,
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
            if is_user_diagram:
                # The editor resolves user-model classes from its own copy of
                # the user metamodel, whose class IDs differ from the ones
                # generated here; the class name is the stable key.
                object_element["classId"] = None
                object_element["className"] = obj_info["class_name"]
            elements[object_id] = object_element

        # Look for object links/relationships
        associations = [
            (rel_id, rel.get("source", {}), rel.get("target", {}))
            for rel_id, rel in reference_diagram_json["relationships"].items()
            if rel["type"] in ["ClassBidirectional", "ClassUnidirectional"]
        ]

        def find_association(class_id: str, role: str):
            """The association whose far end is ``role`` and whose near end is the
            object's class or an ancestor; else any association with that end name."""
            own = set(lineage(class_id)) if class_id else set()
            fallback = None
            for rel_id, source, target in associations:
                if (target.get("role") == role and source.get("element") in own) or (
                    source.get("role") == role and target.get("element") in own
                ):
                    return rel_id
                if fallback is None and role in (source.get("role"), target.get("role")):
                    fallback = rel_id
            return fallback

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            # Look for assignments like: obj1.end = obj2 (or {obj2, obj3} for a
            # to-many end, as object_model_to_code writes them)
            if isinstance(node.value, ast.Name):
                target_vars = [node.value.id]
            elif isinstance(node.value, (ast.Set, ast.List, ast.Tuple)):
                target_vars = [elt.id for elt in node.value.elts if isinstance(elt, ast.Name)]
            else:
                continue
            for target in node.targets:
                if not isinstance(target, ast.Attribute):
                    continue
                obj_name = target.value.id if isinstance(target.value, ast.Name) else None
                relationship_name = target.attr
                source_id = object_var_to_id.get(obj_name)
                if not source_id:
                    continue

                assoc_id = find_association(objects_by_name[obj_name]["class_id"], relationship_name)

                for target_obj in target_vars:
                    target_id = object_var_to_id.get(target_obj)
                    if not target_id:
                        continue
                    rel_id = str(uuid.uuid4())
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

        result = {
            "version": "3.0.0",
            "type": diagram_type,
            "size": default_size,
            "interactive": {"elements": {}, "relationships": {}},
            "elements": elements,
            "relationships": relationships,
            "assessments": {},
        }
        if not is_user_diagram:
            result["referenceDiagramData"] = reference_diagram_json
        return result

    except Exception as e:
        logger.error("Error parsing object BUML content: %s", e, exc_info=True)
        raise ValueError(f"Failed to convert object BUML to JSON: {str(e)}") from e
