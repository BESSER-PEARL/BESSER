"""
Object converter module for BUML to v4 JSON.

Emits the v4 wire shape (``{nodes, edges}``) directly.
"""

import ast
import logging
import uuid
from typing import Dict, Any

from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json._node_builders import (
    make_node, make_edge,
)

logger = logging.getLogger(__name__)


def _resolve_class_index(domain_json: Dict[str, Any]):
    """Build {class_name: node_id} and {node_id: {attr_name: meta}} from a v4 reference diagram."""
    class_name_to_id: dict = {}
    class_id_to_attributes: dict = {}
    for node in (domain_json or {}).get("nodes") or []:
        if node.get("type") != "class":
            continue
        data = node.get("data") or {}
        stereotype = (data.get("stereotype") or "").strip().lower()
        if stereotype in ("interface", "enumeration", "oclconstraint"):
            continue
        class_name_to_id[data.get("name", "")] = node.get("id")
        attrs: dict = {}
        for attr in data.get("attributes") or []:
            attrs[attr.get("name", "")] = {
                "id": attr.get("id"),
                "type": attr.get("attributeType", "str"),
                "defaultValue": attr.get("defaultValue"),
                "visibility": attr.get("visibility", "public"),
            }
        class_id_to_attributes[node.get("id")] = attrs
    return class_name_to_id, class_id_to_attributes


_CLASS_ASSOCIATION_EDGE_TYPES = (
    "ClassBidirectional",
    "ClassUnidirectional",
    "ClassAggregation",
    "ClassComposition",
)


def _resolve_association_index(domain_json: Dict[str, Any]) -> dict:
    """Build {role_pair: edge_id} from v4 reference diagram association edges.

    Covers every class-association edge type that can host roles
    (Bi/Uni/Aggregation/Composition) so ObjectLinks emitted by this
    converter carry the ``associationId`` of the originating
    association regardless of its v4 edge type.
    """
    out: dict = {}
    for edge in (domain_json or {}).get("edges") or []:
        if edge.get("type") not in _CLASS_ASSOCIATION_EDGE_TYPES:
            continue
        d = edge.get("data") or {}
        source_role = d.get("sourceRole", "") or ""
        target_role = d.get("targetRole", "") or ""
        if source_role or target_role:
            out[f"{source_role}-{target_role}"] = edge.get("id")
    return out


def object_buml_to_json(content: str, domain_json: Dict[str, Any]) -> Dict[str, Any]:
    """Convert object model Python source to a v4 ``{nodes, edges}`` payload."""
    nodes: list = []
    edges: list = []

    grid_size = {"x_spacing": 250, "y_spacing": 180, "max_columns": 4}
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
        reference_diagram_json = domain_json or {}
        class_name_to_id, class_id_to_attributes = _resolve_class_index(reference_diagram_json)

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            raise ValueError(
                f"Failed to parse object model Python code: syntax error at line {e.lineno}: {e.msg}"
            ) from e

        objects_by_name: dict = {}
        object_class_mapping: dict = {}
        object_comments: dict = {}
        om_comment = None

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        if isinstance(node.value, ast.Call):
                            call_chain = []
                            current = node.value
                            while isinstance(current, ast.Call):
                                if isinstance(current.func, ast.Attribute):
                                    call_chain.append(current.func.attr)
                                    current = current.func.value
                                elif isinstance(current.func, ast.Name):
                                    call_chain.append(current.func.id)
                                    break
                                else:
                                    break
                            call_chain.reverse()

                            if len(call_chain) >= 3 and call_chain[-1] == "build":
                                class_name = call_chain[0]
                                if class_name in class_name_to_id:
                                    object_name = var_name
                                    object_instance_name = None
                                    attributes = {}
                                    init_call = node.value
                                    while isinstance(init_call, ast.Call) and isinstance(init_call.func, ast.Attribute):
                                        init_call = init_call.func.value
                                    if isinstance(init_call, ast.Call) and len(init_call.args) > 0:
                                        if isinstance(init_call.args[0], ast.Constant):
                                            object_instance_name = init_call.args[0].value
                                    attr_call = node.value
                                    while isinstance(attr_call, ast.Call):
                                        if isinstance(attr_call.func, ast.Attribute) and attr_call.func.attr == "attributes":
                                            for kw in attr_call.keywords:
                                                if isinstance(kw.value, ast.Constant):
                                                    attributes[kw.arg] = kw.value.value
                                            break
                                        attr_call = attr_call.func.value if isinstance(attr_call.func, ast.Attribute) else None
                                        if not attr_call:
                                            break
                                    objects_by_name[object_name] = {
                                        "class_name": class_name,
                                        "class_id": class_name_to_id[class_name],
                                        "instance_name": object_instance_name or object_name,
                                        "attributes": attributes,
                                    }
                                    object_class_mapping[object_name] = class_name
                            elif call_chain and call_chain[0] == "ObjectModel":
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

                target = node.targets[0]
                if isinstance(target, ast.Attribute) and target.attr == "metadata":
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
                                            obj_var, e,
                                        )

        # Emit object nodes.
        object_var_to_node_id: dict = {}
        for obj_name, obj_info in objects_by_name.items():
            x, y = get_position()
            object_id = str(uuid.uuid4())
            class_id = obj_info["class_id"]
            class_attributes = class_id_to_attributes.get(class_id, {})
            attribute_rows: list = []
            for attr_name, attr_value in obj_info["attributes"].items():
                if isinstance(attr_value, str):
                    formatted_value = attr_value
                else:
                    formatted_value = str(attr_value)
                class_attr_info = class_attributes.get(attr_name)
                attr_type = class_attr_info["type"] if class_attr_info else "str"
                row: dict = {
                    "id": str(uuid.uuid4()),
                    "name": f"{attr_name} = {formatted_value}",
                    "attributeType": attr_type,
                }
                if class_attr_info and class_attr_info.get("id"):
                    row["attributeId"] = class_attr_info["id"]
                if class_attr_info and class_attr_info.get("defaultValue") is not None:
                    row["defaultValue"] = class_attr_info["defaultValue"]
                attribute_rows.append(row)
            object_height = max(70, 40 + len(attribute_rows) * 30)
            nodes.append(make_node(
                node_id=object_id,
                type_="objectName",
                data={
                    "name": obj_info["instance_name"],
                    "classId": class_id,
                    "attributes": attribute_rows,
                    "methods": [],
                },
                position={"x": x, "y": y},
                width=200,
                height=object_height,
            ))
            object_var_to_node_id[obj_name] = object_id

        # Object links.
        association_id_map = _resolve_association_index(reference_diagram_json)

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        obj_name = target.value.id if isinstance(target.value, ast.Name) else None
                        relationship_name = target.attr
                        if obj_name in objects_by_name:
                            if isinstance(node.value, ast.Name) and node.value.id in objects_by_name:
                                target_obj = node.value.id
                                source_id = object_var_to_node_id.get(obj_name)
                                target_id = object_var_to_node_id.get(target_obj)
                                if source_id and target_id:
                                    assoc_id = None
                                    for key, aid in association_id_map.items():
                                        if relationship_name in key:
                                            assoc_id = aid
                                            break
                                    edges.append(make_edge(
                                        edge_id=str(uuid.uuid4()),
                                        source=source_id,
                                        target=target_id,
                                        type_="ObjectLink",
                                        data={
                                            "name": relationship_name,
                                            "associationId": assoc_id,
                                            "points": [],
                                        },
                                        source_handle="Right",
                                        target_handle="Topleft",
                                    ))

        # Comments.
        comment_x = -970
        comment_y = -300
        if om_comment:
            nodes.append(make_node(
                node_id=str(uuid.uuid4()),
                type_="Comments",
                data={"name": om_comment},
                position={"x": comment_x, "y": comment_y},
                width=200,
                height=100,
            ))
            comment_y += 130

        for obj_var, comment_text in object_comments.items():
            if obj_var in object_var_to_node_id:
                comment_id = str(uuid.uuid4())
                nodes.append(make_node(
                    node_id=comment_id,
                    type_="Comments",
                    data={"name": comment_text},
                    position={"x": comment_x, "y": comment_y},
                    width=200,
                    height=100,
                ))
                edges.append(make_edge(
                    edge_id=str(uuid.uuid4()),
                    source=comment_id,
                    target=object_var_to_node_id[obj_var],
                    type_="Link",
                    data={"points": []},
                ))
                comment_y += 130

        return {
            "version": "4.0.0",
            "type": "ObjectDiagram",
            "title": "",
            "size": {"width": 960, "height": 670},
            "nodes": nodes,
            "edges": edges,
            "interactive": {"elements": {}, "relationships": {}},
            "assessments": {},
            "referenceDiagramData": reference_diagram_json,
        }

    except Exception as e:
        logger.error("Error parsing object BUML content: %s", e, exc_info=True)
        raise ValueError(f"Failed to convert object BUML to JSON: {str(e)}") from e
