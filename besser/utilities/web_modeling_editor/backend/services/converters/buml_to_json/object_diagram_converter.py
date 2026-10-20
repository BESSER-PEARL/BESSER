"""
Object converter module for BUML to v4 JSON.

Emits the v4 wire shape (``{nodes, edges}``) directly.
"""

import ast
import logging
import uuid
from typing import Dict, Any, List

from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json._node_builders import (
    make_node, make_edge, grid_layout,
)

logger = logging.getLogger(__name__)

#: (object node type, link edge type) per diagram type. A UserDiagram is an
#: object diagram over the user reference domain model;
#: ``json_to_buml.object_diagram_processor`` reads both back the same way.
_NODE_TYPES = {
    "ObjectDiagram": ("objectName", "ObjectLink"),
    "UserDiagram": ("UserModelName", "UserModelLink"),
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


def _resolve_association_index(domain_json: Dict[str, Any]) -> list:
    """List ``(edge_id, source_node_id, source_role, target_node_id, target_role)``
    for every v4 reference association edge that can host roles
    (Bi/Uni/Aggregation/Composition)."""
    out: list = []
    for edge in (domain_json or {}).get("edges") or []:
        if edge.get("type") not in _CLASS_ASSOCIATION_EDGE_TYPES:
            continue
        d = edge.get("data") or {}
        out.append((
            edge.get("id"),
            edge.get("source"), d.get("sourceRole", "") or "",
            edge.get("target"), d.get("targetRole", "") or "",
        ))
    return out


def object_buml_to_json(
    content: str,
    domain_json: Dict[str, Any],
    diagram_type: str = "ObjectDiagram",
) -> Dict[str, Any]:
    """Convert object model Python source to a v4 ``{nodes, edges}`` payload.

    ``diagram_type`` is ``"ObjectDiagram"`` (default) or ``"UserDiagram"``. A
    UserDiagram uses ``UserModelName`` nodes whose rows are equality criteria
    (``attributeOperator`` ``"=="``) and names each object's class in
    ``className``; it carries no ``referenceDiagramData``, since the editor
    supplies the user reference model itself.
    """
    if diagram_type not in _NODE_TYPES:
        raise ValueError(f"Unsupported object diagram type: {diagram_type}")
    is_user_diagram = diagram_type == "UserDiagram"
    object_node_type, link_edge_type = _NODE_TYPES[diagram_type]
    nodes: list = []
    edges: list = []

    try:
        reference_diagram_json = domain_json or {}
        class_name_to_id, class_id_to_attributes = _resolve_class_index(reference_diagram_json)

        # Superclasses per class (ClassInheritance: source = specific, target = general).
        parents_of: Dict[str, List[str]] = {}
        for edge in reference_diagram_json.get("edges") or []:
            if edge.get("type") == "ClassInheritance":
                parents_of.setdefault(edge.get("source"), []).append(edge.get("target"))

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

                            # An object without slots is ClassName("name").build().
                            if len(call_chain) >= 2 and call_chain[-1] == "build":
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
        object_nodes: list = []
        for obj_name, obj_info in objects_by_name.items():
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
                # v4 object rows keep the attribute name and runtime value
                # in separate fields (``ObjectNodeAttribute.value``); the
                # frontend renders "name = value" from the split fields.
                # ``visibility`` + ``attributeType`` together keep the
                # frontend's v3 round-trip from re-parsing the row and
                # dropping the type.
                row: dict = {
                    "id": str(uuid.uuid4()),
                    "name": attr_name,
                    "value": formatted_value,
                    "attributeType": attr_type,
                    "visibility": (class_attr_info or {}).get("visibility") or "public",
                }
                if class_attr_info and class_attr_info.get("id"):
                    row["attributeId"] = class_attr_info["id"]
                if class_attr_info and class_attr_info.get("defaultValue") is not None:
                    row["defaultValue"] = class_attr_info["defaultValue"]
                if is_user_diagram:
                    # A BUML slot is an equality criterion; the comparator of a
                    # non-equality criterion ("age < 18") is not in the metamodel.
                    row["attributeOperator"] = "=="
                attribute_rows.append(row)
            object_height = max(70, 40 + len(attribute_rows) * 30)
            node_payload: dict = {
                "name": obj_info["instance_name"],
                "classId": class_id,
                "attributes": attribute_rows,
                "methods": [],
            }
            if is_user_diagram:
                # The editor resolves user-model classes from its own copy of
                # the user metamodel, whose class ids differ from the ones
                # generated here; the class name is the stable key.
                node_payload["classId"] = None
                node_payload["className"] = obj_info["class_name"]
            object_nodes.append(make_node(
                node_id=object_id,
                type_=object_node_type,
                data=node_payload,
                position={"x": 0, "y": 0},
                width=200,
                height=object_height,
            ))
            object_var_to_node_id[obj_name] = object_id
        grid_layout(object_nodes, origin=(-460, -300), columns=4, gap_x=50, gap_y=80)
        nodes.extend(object_nodes)

        # Object links.
        associations = _resolve_association_index(reference_diagram_json)

        def find_association(class_id: str, role: str):
            """The association whose far end is ``role`` and whose near end is the
            object's class or an ancestor; else any association with that end name."""
            own = set(lineage(class_id)) if class_id else set()
            fallback = None
            for edge_id, source_node, source_role, target_node, target_role in associations:
                if (target_role == role and source_node in own) or (
                    source_role == role and target_node in own
                ):
                    return edge_id
                if fallback is None and role in (source_role, target_role):
                    fallback = edge_id
            return fallback

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            # obj1.end = obj2, or {obj2, obj3} for a to-many end (as
            # object_model_to_code writes them).
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
                source_id = object_var_to_node_id.get(obj_name)
                if not source_id:
                    continue
                relationship_name = target.attr
                assoc_id = find_association(objects_by_name[obj_name]["class_id"], relationship_name)
                for target_obj in target_vars:
                    target_id = object_var_to_node_id.get(target_obj)
                    if not target_id:
                        continue
                    edges.append(make_edge(
                        edge_id=str(uuid.uuid4()),
                        source=source_id,
                        target=target_id,
                        type_=link_edge_type,
                        data={
                            "name": relationship_name,
                            "associationId": assoc_id,
                            "points": [],
                        },
                        source_handle="right",
                        target_handle="top-left",
                    ))

        # Comments (``comment`` nodes + ``CommentLink`` edges — the types
        # the React Flow frontend registers).
        comment_x = -970
        comment_y = -300
        if om_comment:
            nodes.append(make_node(
                node_id=str(uuid.uuid4()),
                type_="comment",
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
                    type_="comment",
                    data={"name": comment_text},
                    position={"x": comment_x, "y": comment_y},
                    width=200,
                    height=100,
                ))
                edges.append(make_edge(
                    edge_id=str(uuid.uuid4()),
                    source=comment_id,
                    target=object_var_to_node_id[obj_var],
                    type_="CommentLink",
                    data={"points": []},
                ))
                comment_y += 130

        result = {
            "version": "4.0.0",
            "type": diagram_type,
            "title": "",
            "size": {"width": 960, "height": 670},
            "nodes": nodes,
            "edges": edges,
            "interactive": {"elements": {}, "relationships": {}},
            "assessments": {},
        }
        if not is_user_diagram:
            result["referenceDiagramData"] = reference_diagram_json
        return result

    except Exception as e:
        logger.error("Error parsing object BUML content: %s", e, exc_info=True)
        raise ValueError(f"Failed to convert object BUML to JSON: {str(e)}") from e
