"""Pure shape transforms between v3 ``{elements, relationships}`` and v4
``{nodes[], edges[]}``.

This module is **internal** to the converter package. It is not a
compatibility adapter at the public API surface — the public entry
points (``process_class_diagram``, ``class_buml_to_json``, etc.) are
v4-native after Phase 6 of the React-Flow migration. This module exists
only because the per-diagram business logic that walks BUML <-> JSON
data structures is well-tested and stable, and the structural
transformation between v3 and v4 is a flat, mechanical mapping (parent
id moves, attributes/methods collapse onto data, etc.).

By centralising the shape transform here, every diagram processor
shares one understanding of the v4 layout. The v4-native entry points
import these helpers, normalise inbound v4 to a v3-shaped intermediate
that the per-diagram routines consume unchanged, and inversely lift v3
output back to v4 before returning. The result is identical to a hand
rewrite for the consumer; the difference is that the code path is
testable in isolation.

Reference: ``docs/source/migrations/uml-v4-shape.md``.
"""

from __future__ import annotations

from typing import Any, Optional


# Node types whose payloads collapse into a parent's ``data`` arrays in
# v4. They never appear as standalone v4 nodes; they are restored as v3
# child elements during inbound normalisation only.
_CLASS_MEMBER_NODE_KEYS = {"attributes", "methods", "oclConstraints"}
_OBJECT_MEMBER_NODE_KEYS = {"attributes", "methods"}
_STATE_MEMBER_NODE_KEYS = {"bodies", "fallbackBodies"}

# Mapping from v4 ``data.stereotype`` to a v3 ``type`` string for class
# nodes. Anything not listed here falls through to ``Class``.
_CLASS_STEREOTYPE_TO_V3_TYPE = {
    "abstract": "AbstractClass",
    "interface": "Interface",
    "enumeration": "Enumeration",
}


def v4_to_v3_model(model: dict, diagram_type: str = "ClassDiagram") -> dict:
    """Transform a v4 ``{nodes, edges}`` payload to a v3-shaped payload.

    Returns a NEW dict — input is not mutated. Children whose payload
    collapses into v4 ``data`` arrays (class members, state bodies, ...)
    are re-expanded into separate v3 elements with synthetic ids and
    ``owner`` references so the legacy per-diagram processors can walk
    them unchanged.
    """
    nodes = model.get("nodes") or []
    edges = model.get("edges") or []
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []

    elements: dict[str, dict] = {}
    relationships: dict[str, dict] = {}

    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        v3_elements = _explode_node(node, diagram_type)
        for v3 in v3_elements:
            elements[v3["id"]] = v3

    for edge in edges:
        edge_id = edge.get("id")
        if not edge_id:
            continue
        relationships[edge_id] = _edge_to_relationship(edge)

    out: dict = dict(model)
    out["elements"] = elements
    out["relationships"] = relationships
    out.setdefault("interactive", {"elements": {}, "relationships": {}})
    out.setdefault("assessments", {})
    # Preserve ``referenceDiagramData`` if the caller passed one through.
    return out


def _explode_node(node: dict, diagram_type: str) -> list[dict]:
    """Return the v3 elements that this v4 node represents.

    For most nodes the result is a single-element list (a 1:1 lift). For
    class/object/state/agent nodes whose v4 ``data`` carries collapsed
    member arrays, the result includes additional synthetic v3 child
    elements (one per collapsed member) so that the legacy processors
    can walk them via ``element['attributes']`` / ``element['bodies']``
    / etc.
    """
    node_id = node.get("id")
    node_type = node.get("type") or ""
    data = node.get("data") or {}
    parent_id = node.get("parentId")
    bounds = _node_to_bounds(node)

    primary: dict = {
        "id": node_id,
        "type": _v4_type_to_v3_type(node_type, data),
        "name": data.get("name", ""),
        "owner": parent_id,
        "bounds": bounds,
    }
    # Carry through optional metadata fields that several legacy
    # processors look up (description / uri / icon / classId / ...).
    for key in (
        "description",
        "uri",
        "icon",
        "assessmentNote",
        "fillColor",
        "strokeColor",
        "textColor",
        "stereotype",
        "classId",
        "associationId",
        "intent_description",
        "replyType",
        "ragDatabaseName",
        "dbSelectionType",
        "dbCustomName",
        "dbQueryMode",
        "dbOperation",
        "dbSqlQuery",
        "italic",
        "underline",
        "deviderPosition",
        "hasBody",
        "hasFallbackBody",
        "code",
        "language",
        "constraint",
        "kind",
        "constraintName",
        "targetMethodId",
        "referenceTarget",
    ):
        if key in data:
            primary[key] = data[key]

    elements: list[dict] = [primary]

    # Class / interface / enum / abstract: collapse attributes/methods/
    # oclConstraints into separate v3 ClassAttribute / ClassMethod /
    # ClassOCLConstraint elements so the legacy class processor can walk
    # them via ``element['attributes']`` / ``element['methods']``.
    if node_type == "class":
        # Re-derive v3 class type from stereotype. Enumeration handling
        # below replaces ``ClassAttribute`` ids with literal ids.
        primary["type"] = _CLASS_STEREOTYPE_TO_V3_TYPE.get(
            (data.get("stereotype") or "").strip().lower(), "Class"
        )
        if primary["type"] == "Enumeration":
            literals: list[dict] = []
            for lit in data.get("attributes") or []:
                lit_id = lit.get("id") or f"{node_id}__lit__{lit.get('name', '')}"
                literals.append({
                    "id": lit_id,
                    "name": lit.get("name", ""),
                    "type": "ClassAttribute",
                    "owner": node_id,
                    "bounds": dict(bounds),
                })
            primary["attributes"] = [lit["id"] for lit in literals]
            elements.extend(literals)
        else:
            attribute_ids: list[str] = []
            for attr in data.get("attributes") or []:
                attr_id = attr.get("id") or f"{node_id}__attr__{attr.get('name', '')}"
                v3_attr = {
                    "id": attr_id,
                    "name": attr.get("name", ""),
                    "type": "ClassAttribute",
                    "owner": node_id,
                    "bounds": dict(bounds),
                    "visibility": attr.get("visibility", "public"),
                    "attributeType": attr.get("attributeType", "str"),
                    "isOptional": attr.get("isOptional", False),
                    "isId": attr.get("isId", False),
                    "isExternalId": attr.get("isExternalId", False),
                    "isDerived": attr.get("isDerived", False),
                }
                if attr.get("defaultValue") is not None:
                    v3_attr["defaultValue"] = attr["defaultValue"]
                elements.append(v3_attr)
                attribute_ids.append(attr_id)
            method_ids: list[str] = []
            for method in data.get("methods") or []:
                method_id = method.get("id") or f"{node_id}__m__{method.get('name', '')}"
                v3_method = {
                    "id": method_id,
                    "name": method.get("name", ""),
                    "type": "ClassMethod",
                    "owner": node_id,
                    "bounds": dict(bounds),
                }
                # Optional fields used by the class processor.
                for key in (
                    "code", "implementationType", "stateMachineId",
                    "quantumCircuitId", "visibility",
                ):
                    if method.get(key) is not None:
                        v3_method[key] = method[key]
                elements.append(v3_method)
                method_ids.append(method_id)
            primary["attributes"] = attribute_ids
            primary["methods"] = method_ids

            # Class-level OCL constraints collapse onto data; re-expand
            # them into v3 ClassOCLConstraint elements with linking
            # relationships handled at the edge level.
            for ocl in data.get("oclConstraints") or []:
                ocl_id = ocl.get("id") or f"{node_id}__ocl__{ocl.get('name', '')}"
                elements.append({
                    "id": ocl_id,
                    "type": "ClassOCLConstraint",
                    "name": ocl.get("name", ""),
                    "owner": node_id,
                    "bounds": dict(bounds),
                    "constraint": ocl.get("expression", ""),
                    **(
                        {"description": ocl.get("description")}
                        if ocl.get("description")
                        else {}
                    ),
                })

    elif node_type == "package":
        primary["type"] = "Package"

    elif node_type == "objectName":
        primary["type"] = "ObjectName"
        attribute_ids = []
        for attr in data.get("attributes") or []:
            attr_id = attr.get("id") or f"{node_id}__attr__{attr.get('name', '')}"
            elements.append({
                "id": attr_id,
                "name": attr.get("name", ""),
                "type": "ObjectAttribute",
                "owner": node_id,
                "bounds": dict(bounds),
                **(
                    {"attributeId": attr["attributeId"]}
                    if attr.get("attributeId")
                    else {}
                ),
            })
            attribute_ids.append(attr_id)
        method_ids = []
        for method in data.get("methods") or []:
            method_id = method.get("id") or f"{node_id}__m__{method.get('name', '')}"
            elements.append({
                "id": method_id,
                "name": method.get("name", ""),
                "type": "ObjectMethod",
                "owner": node_id,
                "bounds": dict(bounds),
            })
            method_ids.append(method_id)
        primary["attributes"] = attribute_ids
        primary["methods"] = method_ids

    elif node_type == "UserModelName":
        primary["type"] = "UserModelName"
        attribute_ids = []
        for attr in data.get("attributes") or []:
            attr_id = attr.get("id") or f"{node_id}__attr__{attr.get('name', '')}"
            v3 = {
                "id": attr_id,
                "name": attr.get("name", ""),
                "type": "UserModelAttribute",
                "owner": node_id,
                "bounds": dict(bounds),
            }
            if attr.get("attributeOperator"):
                v3["attributeOperator"] = attr["attributeOperator"]
            if attr.get("attributeValue") is not None:
                v3["attributeValue"] = attr["attributeValue"]
            elements.append(v3)
            attribute_ids.append(attr_id)
        primary["attributes"] = attribute_ids

    elif node_type in ("State", "AgentState"):
        body_ids: list[str] = []
        for body in data.get("bodies") or []:
            body_id = body.get("id") or f"{node_id}__b__{body.get('name', '')}"
            body_type = (
                "AgentStateBody" if node_type == "AgentState" else "StateBody"
            )
            body_elem = {
                "id": body_id,
                "name": body.get("name", ""),
                "type": body_type,
                "owner": node_id,
                "bounds": dict(bounds),
            }
            for key in (
                "replyType",
                "ragDatabaseName",
                "dbSelectionType",
                "dbCustomName",
                "dbQueryMode",
                "dbOperation",
                "dbSqlQuery",
            ):
                if body.get(key) is not None:
                    body_elem[key] = body[key]
            elements.append(body_elem)
            body_ids.append(body_id)
        fallback_ids: list[str] = []
        for body in data.get("fallbackBodies") or []:
            body_id = body.get("id") or f"{node_id}__fb__{body.get('name', '')}"
            body_type = (
                "AgentStateFallbackBody"
                if node_type == "AgentState"
                else "StateFallbackBody"
            )
            elements.append({
                "id": body_id,
                "name": body.get("name", ""),
                "type": body_type,
                "owner": node_id,
                "bounds": dict(bounds),
            })
            fallback_ids.append(body_id)
        primary["bodies"] = body_ids
        primary["fallbackBodies"] = fallback_ids

    elif node_type == "AgentIntent":
        body_ids = []
        for body in data.get("bodies") or []:
            body_id = body.get("id") or f"{node_id}__ib__{body.get('name', '')}"
            elements.append({
                "id": body_id,
                "name": body.get("name", ""),
                "type": "AgentIntentBody",
                "owner": node_id,
                "bounds": dict(bounds),
            })
            body_ids.append(body_id)
        primary["bodies"] = body_ids

    elif node_type == "AgentRagElement":
        primary["type"] = "AgentRagElement"

    elif node_type == "StateCodeBlock":
        primary["type"] = "StateCodeBlock"

    # NN diagram layer / container / reference — attributes collapse to
    # ``data.attributes: dict``. Re-expand them into one v3 attribute
    # element per key so the legacy NN processor walks them unchanged.
    if _is_nn_layer_type(node_type):
        primary["type"] = node_type  # 1:1 type passthrough for NN nodes
        attribute_ids: list[str] = []
        attrs = data.get("attributes") or {}
        if isinstance(attrs, dict):
            for key, value in attrs.items():
                attr_id = f"{node_id}__nnattr__{key}"
                v3_attr_type = _nn_attr_v3_type(node_type, key)
                if v3_attr_type is None:
                    continue
                elements.append({
                    "id": attr_id,
                    "name": key,
                    "attributeName": key,
                    "type": v3_attr_type,
                    "owner": node_id,
                    "bounds": dict(bounds),
                    "value": _stringify(value),
                })
                attribute_ids.append(attr_id)
        primary["attributes"] = attribute_ids
    elif node_type in ("NNContainer", "NNReference"):
        primary["type"] = node_type

    return elements


def _node_to_bounds(node: dict) -> dict:
    pos = node.get("position") or {}
    width = node.get("width", 0) or (node.get("measured") or {}).get("width", 0)
    height = node.get("height", 0) or (node.get("measured") or {}).get("height", 0)
    return {
        "x": pos.get("x", 0),
        "y": pos.get("y", 0),
        "width": width,
        "height": height,
    }


def _v4_type_to_v3_type(v4_type: str, data: dict) -> str:
    """Map a v4 node type to a v3 element type string.

    Class nodes are special: their stereotype determines the v3 type.
    Most other nodes pass through 1:1.
    """
    if v4_type == "class":
        return _CLASS_STEREOTYPE_TO_V3_TYPE.get(
            (data.get("stereotype") or "").strip().lower(), "Class"
        )
    if v4_type == "package":
        return "Package"
    if v4_type == "objectName":
        return "ObjectName"
    return v4_type


def _edge_to_relationship(edge: dict) -> dict:
    edge_data = edge.get("data") or {}
    rel: dict = {
        "id": edge.get("id"),
        "type": edge.get("type"),
        "name": edge_data.get("name", ""),
        "source": {
            "element": edge.get("source"),
            "direction": edge.get("sourceHandle", "Right"),
        },
        "target": {
            "element": edge.get("target"),
            "direction": edge.get("targetHandle", "Left"),
        },
        "path": edge_data.get("points", []),
    }
    # Class associations carry roles + multiplicities on edge.data.
    if edge_data.get("sourceRole") is not None:
        rel["source"]["role"] = edge_data["sourceRole"]
    if edge_data.get("sourceMultiplicity") is not None:
        rel["source"]["multiplicity"] = edge_data["sourceMultiplicity"]
    if edge_data.get("targetRole") is not None:
        rel["target"]["role"] = edge_data["targetRole"]
    if edge_data.get("targetMultiplicity") is not None:
        rel["target"]["multiplicity"] = edge_data["targetMultiplicity"]
    if edge_data.get("isManuallyLayouted") is not None:
        rel["isManuallyLayouted"] = edge_data["isManuallyLayouted"]
    # Object link: associationId.
    if edge_data.get("associationId") is not None:
        rel["associationId"] = edge_data["associationId"]
    # State / agent transition extras.
    for key in (
        "guard", "params",
        "transitionType", "predefined", "custom",
        # Legacy flat shapes kept for AgentStateTransition round-trip
        "predefinedType", "intentName", "fileType",
        "conditionValue", "variable", "operator", "targetValue",
        "event", "customEvent", "customConditions", "condition", "conditions",
    ):
        if key in edge_data:
            rel[key] = edge_data[key]
    return rel


def v3_to_v4_model(
    v3_payload: dict,
    diagram_type: str,
    title: str = "",
) -> dict:
    """Transform a v3 ``{elements, relationships}`` payload to v4.

    Inverse of ``v4_to_v3_model`` — collapses child member elements onto
    their parent node's ``data`` and rewrites relationships to edges.
    Preserves ``id`` values where possible for stable round-trips.
    """
    elements = v3_payload.get("elements") or {}
    relationships = v3_payload.get("relationships") or {}

    # Build set of "child" element ids that collapse into parent data
    # arrays so we don't emit them as standalone v4 nodes.
    collapsed_child_ids: set[str] = set()
    nn_attr_ids: set[str] = set()
    ocl_collapsed_ids: set[str] = set()

    for elem_id, elem in elements.items():
        etype = elem.get("type") or ""
        if etype in (
            "ClassAttribute", "ClassMethod",
            "ObjectAttribute", "ObjectMethod",
            "UserModelAttribute", "UserModelIcon",
            "StateBody", "StateFallbackBody",
            "AgentStateBody", "AgentStateFallbackBody",
            "AgentIntentBody", "AgentIntentDescription",
            "AgentIntentObjectComponent", "ObjectIcon",
        ):
            collapsed_child_ids.add(elem_id)
        elif _is_nn_attr_type(etype):
            nn_attr_ids.add(elem_id)
            collapsed_child_ids.add(elem_id)

    # Determine which OCL constraints can be collapsed onto an owning class.
    # Per spec, ClassOCLConstraint v3 elements collapse onto their owner
    # class as a row in data.oclConstraints when the owner is a class
    # (resolved either via element.owner or via a ClassOCLLink edge).
    ocl_owner_by_id: dict[str, str] = {}
    for elem_id, elem in elements.items():
        if elem.get("type") != "ClassOCLConstraint":
            continue
        owner = elem.get("owner")
        if owner and elements.get(owner, {}).get("type") in (
            "Class", "AbstractClass", "Interface",
        ):
            ocl_owner_by_id[elem_id] = owner
            collapsed_child_ids.add(elem_id)
            ocl_collapsed_ids.add(elem_id)
            continue
        for rel in relationships.values():
            if rel.get("type") != "ClassOCLLink":
                continue
            src = (rel.get("source") or {}).get("element")
            tgt = (rel.get("target") or {}).get("element")
            other = None
            if src == elem_id:
                other = tgt
            elif tgt == elem_id:
                other = src
            if other and elements.get(other, {}).get("type") in (
                "Class", "AbstractClass", "Interface",
            ):
                ocl_owner_by_id[elem_id] = other
                collapsed_child_ids.add(elem_id)
                ocl_collapsed_ids.add(elem_id)
                break

    nodes: list[dict] = []
    for elem_id, elem in elements.items():
        if elem_id in collapsed_child_ids:
            continue
        nodes.append(_v3_element_to_v4_node(
            elem, elements, ocl_owner_by_id, ocl_collapsed_ids,
        ))

    edges: list[dict] = []
    for rel_id, rel in relationships.items():
        if _skip_v3_relationship(rel, ocl_collapsed_ids):
            continue
        edges.append(_v3_rel_to_v4_edge(rel))

    return {
        "version": "4.0.0",
        "type": diagram_type,
        "title": title,
        "size": v3_payload.get("size", {"width": 0, "height": 0}),
        "nodes": nodes,
        "edges": edges,
        "interactive": v3_payload.get("interactive", {"elements": {}, "relationships": {}}),
        "assessments": v3_payload.get("assessments", {}),
        # Mirror keys for legacy test ergonomics. The canonical v4 wire
        # shape is ``nodes`` / ``edges``; these keys are read-only views
        # used by older tests that index ``result["elements"][id]``.
        # New code MUST NOT rely on them — they are intentionally not
        # part of the spec at ``docs/source/migrations/uml-v4-shape.md``.
        "elements": v3_payload.get("elements", {}),
        "relationships": v3_payload.get("relationships", {}),
    }


def _skip_v3_relationship(rel: dict, ocl_collapsed_ids: set[str]) -> bool:
    """Return True if a v3 relationship has no v4 equivalent and should be dropped."""
    rel_type = rel.get("type")
    # Comment links and class-link rels have no direct v4 edge
    # equivalent (they encode parent/membership rather than a real edge).
    if rel_type in ("Link",):
        return True
    # If a ClassOCLLink connects a collapsed constraint, drop it (the
    # constraint is inlined onto the class data).
    if rel_type == "ClassOCLLink":
        src = (rel.get("source") or {}).get("element")
        tgt = (rel.get("target") or {}).get("element")
        if src in ocl_collapsed_ids or tgt in ocl_collapsed_ids:
            return True
    return False


def _v3_element_to_v4_node(
    elem: dict,
    all_elements: dict,
    ocl_owner_by_id: dict[str, str],
    ocl_collapsed_ids: set[str],
) -> dict:
    """Translate a single v3 element into a v4 node, collapsing children."""
    etype = elem.get("type") or ""
    elem_id = elem.get("id")
    bounds = elem.get("bounds") or {"x": 0, "y": 0, "width": 0, "height": 0}
    data: dict = {"name": elem.get("name", "")}
    for key in (
        "description", "uri", "icon", "assessmentNote",
        "fillColor", "strokeColor", "textColor",
        "classId", "associationId",
        "intent_description", "replyType",
        "ragDatabaseName", "dbSelectionType", "dbCustomName",
        "dbQueryMode", "dbOperation", "dbSqlQuery",
        "italic", "underline", "code", "language",
        "constraint", "kind", "constraintName", "targetMethodId",
        "referenceTarget",
    ):
        if elem.get(key) is not None:
            data[key] = elem[key]

    parent_id = elem.get("owner")
    if parent_id is None or parent_id == "":
        parent_id = None
    pos = {"x": bounds.get("x", 0), "y": bounds.get("y", 0)}
    width = bounds.get("width", _DEFAULT_W)
    height = bounds.get("height", _DEFAULT_H)
    measured = {"width": width, "height": height}

    v4_type, data = _v3_type_to_v4(etype, elem, data, all_elements,
                                   ocl_owner_by_id, ocl_collapsed_ids)

    node: dict = {
        "id": elem_id,
        "type": v4_type,
        "position": pos,
        "width": width,
        "height": height,
        "measured": measured,
        "data": data,
    }
    if parent_id:
        node["parentId"] = parent_id
    return node


_DEFAULT_W = 160
_DEFAULT_H = 100


def _v3_type_to_v4(
    etype: str,
    elem: dict,
    data: dict,
    all_elements: dict,
    ocl_owner_by_id: dict[str, str],
    ocl_collapsed_ids: set[str],
) -> tuple[str, dict]:
    """Return ``(v4_type, data)`` for the given v3 element."""
    if etype in ("Class", "AbstractClass", "Interface", "Enumeration"):
        v4_type = "class"
        if etype == "AbstractClass":
            data["stereotype"] = "abstract"
        elif etype == "Interface":
            data["stereotype"] = "interface"
        elif etype == "Enumeration":
            data["stereotype"] = "enumeration"
        else:
            data.setdefault("stereotype", None)

        if etype == "Enumeration":
            literals = []
            for lit_id in elem.get("attributes") or []:
                lit = all_elements.get(lit_id)
                if not lit:
                    continue
                literals.append({
                    "id": lit_id,
                    "name": lit.get("name", ""),
                    "attributeType": "str",
                    "visibility": "public",
                })
            data["attributes"] = literals
            data["methods"] = []
        else:
            attrs = []
            for attr_id in elem.get("attributes") or []:
                a = all_elements.get(attr_id)
                if not a:
                    continue
                row = {
                    "id": attr_id,
                    "name": a.get("name", ""),
                    "attributeType": a.get("attributeType", "str"),
                    "visibility": a.get("visibility", "public"),
                }
                for k in ("isOptional", "isId", "isExternalId", "isDerived"):
                    if k in a:
                        row[k] = a[k]
                if a.get("defaultValue") is not None:
                    row["defaultValue"] = a["defaultValue"]
                attrs.append(row)
            data["attributes"] = attrs
            methods = []
            for m_id in elem.get("methods") or []:
                m = all_elements.get(m_id)
                if not m:
                    continue
                row = {
                    "id": m_id,
                    "name": m.get("name", ""),
                    "visibility": "public",
                    "attributeType": "any",
                }
                for k in (
                    "code", "implementationType", "stateMachineId",
                    "quantumCircuitId",
                ):
                    if m.get(k) is not None:
                        row[k] = m[k]
                methods.append(row)
            data["methods"] = methods

            ocl_rows = []
            for ocl_id, owner_id in ocl_owner_by_id.items():
                if owner_id != elem.get("id"):
                    continue
                ocl = all_elements.get(ocl_id) or {}
                ocl_rows.append({
                    "id": ocl_id,
                    "name": ocl.get("constraintName") or ocl.get("name", "") or "",
                    "expression": ocl.get("constraint", ""),
                    **(
                        {"description": ocl.get("description")}
                        if ocl.get("description")
                        else {}
                    ),
                })
            if ocl_rows:
                data["oclConstraints"] = ocl_rows
        return v4_type, data

    if etype == "Package":
        return "package", data

    if etype == "ObjectName":
        attrs = []
        for attr_id in elem.get("attributes") or []:
            a = all_elements.get(attr_id)
            if not a:
                continue
            row = {"id": attr_id, "name": a.get("name", "")}
            if a.get("attributeId"):
                row["attributeId"] = a["attributeId"]
            if a.get("attributeType"):
                row["attributeType"] = a["attributeType"]
            if a.get("defaultValue") is not None:
                row["defaultValue"] = a["defaultValue"]
            attrs.append(row)
        data["attributes"] = attrs
        methods = []
        for m_id in elem.get("methods") or []:
            m = all_elements.get(m_id)
            if not m:
                continue
            methods.append({"id": m_id, "name": m.get("name", "")})
        data["methods"] = methods
        return "objectName", data

    if etype == "UserModelName":
        attrs = []
        for attr_id in elem.get("attributes") or []:
            a = all_elements.get(attr_id)
            if not a:
                continue
            row = {"id": attr_id, "name": a.get("name", "")}
            if a.get("attributeOperator"):
                row["attributeOperator"] = a["attributeOperator"]
            if a.get("attributeValue") is not None:
                row["attributeValue"] = a["attributeValue"]
            attrs.append(row)
        data["attributes"] = attrs
        return "UserModelName", data

    if etype == "State":
        bodies = []
        for b_id in elem.get("bodies") or []:
            b = all_elements.get(b_id)
            if not b:
                continue
            bodies.append({"id": b_id, "name": b.get("name", "")})
        fbs = []
        for b_id in elem.get("fallbackBodies") or []:
            b = all_elements.get(b_id)
            if not b:
                continue
            fbs.append({"id": b_id, "name": b.get("name", "")})
        data["bodies"] = bodies
        data["fallbackBodies"] = fbs
        return "State", data

    if etype == "AgentState":
        bodies = []
        for b_id in elem.get("bodies") or []:
            b = all_elements.get(b_id)
            if not b:
                continue
            row = {"id": b_id, "name": b.get("name", "")}
            for key in (
                "replyType", "ragDatabaseName", "dbSelectionType",
                "dbCustomName", "dbQueryMode", "dbOperation", "dbSqlQuery",
            ):
                if b.get(key) is not None:
                    row[key] = b[key]
            bodies.append(row)
        fbs = []
        for b_id in elem.get("fallbackBodies") or []:
            b = all_elements.get(b_id)
            if not b:
                continue
            row = {"id": b_id, "name": b.get("name", "")}
            for key in (
                "replyType", "ragDatabaseName", "dbSelectionType",
                "dbCustomName", "dbQueryMode", "dbOperation", "dbSqlQuery",
            ):
                if b.get(key) is not None:
                    row[key] = b[key]
            fbs.append(row)
        data["bodies"] = bodies
        data["fallbackBodies"] = fbs
        data.setdefault("replyType", elem.get("replyType", "text"))
        return "AgentState", data

    if etype == "AgentIntent":
        bodies = []
        for b_id in elem.get("bodies") or []:
            b = all_elements.get(b_id)
            if not b:
                continue
            bodies.append({"id": b_id, "name": b.get("name", "")})
        data["bodies"] = bodies
        data.setdefault("intent_description", elem.get("intent_description", ""))
        return "AgentIntent", data

    if etype == "AgentRagElement":
        return "AgentRagElement", data

    # NN: re-collapse attributes onto data.attributes (dict).
    if _is_nn_layer_type(etype):
        attrs: dict = {}
        for attr_id in elem.get("attributes") or []:
            a = all_elements.get(attr_id)
            if not a:
                continue
            key = _nn_v3_attr_to_key(a.get("type", ""), etype)
            if key is None:
                continue
            attrs[key] = a.get("value", "")
        data["attributes"] = attrs
        return etype, data

    if etype in ("NNContainer", "NNReference"):
        return etype, data

    # ClassOCLConstraint that wasn't collapsed remains a class-typed node
    # with stereotype 'oclConstraint' (rare fallback, see spec).
    if etype == "ClassOCLConstraint":
        data["stereotype"] = "oclConstraint"
        return "class", data

    # Default: passthrough (Comments, marker nodes, fork nodes, etc.).
    return etype, data


def _v3_rel_to_v4_edge(rel: dict) -> dict:
    edge_data: dict = {}
    src = rel.get("source") or {}
    tgt = rel.get("target") or {}
    if rel.get("name"):
        edge_data["name"] = rel["name"]
    if src.get("role") is not None:
        edge_data["sourceRole"] = src["role"]
    if src.get("multiplicity") is not None:
        edge_data["sourceMultiplicity"] = src["multiplicity"]
    if tgt.get("role") is not None:
        edge_data["targetRole"] = tgt["role"]
    if tgt.get("multiplicity") is not None:
        edge_data["targetMultiplicity"] = tgt["multiplicity"]
    if rel.get("isManuallyLayouted") is not None:
        edge_data["isManuallyLayouted"] = rel["isManuallyLayouted"]
    if rel.get("path"):
        edge_data["points"] = rel["path"]
    else:
        edge_data["points"] = []
    if rel.get("associationId") is not None:
        edge_data["associationId"] = rel["associationId"]
    for key in (
        "guard", "params",
        "transitionType", "predefined", "custom",
        "predefinedType", "intentName", "fileType",
        "conditionValue", "variable", "operator", "targetValue",
        "event", "customEvent", "customConditions",
    ):
        if rel.get(key) is not None:
            edge_data[key] = rel[key]
    return {
        "id": rel.get("id"),
        "source": src.get("element"),
        "target": tgt.get("element"),
        "type": rel.get("type"),
        "sourceHandle": src.get("direction", "Right"),
        "targetHandle": tgt.get("direction", "Left"),
        "data": edge_data,
    }


# ---------------------------------------------------------------------------
# NN attribute mapping
# ---------------------------------------------------------------------------

# Layer types whose attributes collapse onto data.attributes in v4.
_NN_LAYER_TYPES = frozenset({
    "Conv1DLayer", "Conv2DLayer", "Conv3DLayer", "PoolingLayer",
    "RNNLayer", "LSTMLayer", "GRULayer", "LinearLayer",
    "FlattenLayer", "EmbeddingLayer", "DropoutLayer",
    "LayerNormalizationLayer", "BatchNormalizationLayer",
    "TensorOp", "Configuration", "TrainingDataset", "TestDataset",
})


# Mapping from v3 attribute-element type prefix (e.g. "OutChannelsAttribute")
# to v4 snake_case key (e.g. "out_channels").  Suffix is the layer slug; we
# strip it dynamically in ``_nn_v3_attr_to_key``.
_NN_ATTR_PREFIX_TO_KEY: dict[str, str] = {
    "Name": "name",
    "KernelDim": "kernel_dim",
    "OutChannels": "out_channels",
    "StrideDim": "stride_dim",
    "InChannels": "in_channels",
    "PaddingAmount": "padding_amount",
    "PaddingType": "padding_type",
    "ActvFunc": "actv_func",
    "NameModuleInput": "name_module_input",
    "InputReused": "input_reused",
    "PermuteIn": "permute_in",
    "PermuteOut": "permute_out",
    "HiddenSize": "hidden_size",
    "ReturnType": "return_type",
    "InputSize": "input_size",
    "Bidirectional": "bidirectional",
    "Dropout": "dropout",
    "BatchFirst": "batch_first",
    "Rate": "rate",
    "OutFeatures": "out_features",
    "InFeatures": "in_features",
    "StartDim": "start_dim",
    "EndDim": "end_dim",
    "NumEmbeddings": "num_embeddings",
    "EmbeddingDim": "embedding_dim",
    "NormalizedShape": "normalized_shape",
    "NumFeatures": "num_features",
    "Dimension": "dimension",
    "PoolingType": "pooling_type",
    "OutputDim": "output_dim",
    "TnsType": "tns_type",
    "ConcatenateDim": "concatenate_dim",
    "LayersOfTensors": "layers_of_tensors",
    "ReshapeDim": "reshape_dim",
    "TransposeDim": "transpose_dim",
    "PermuteDim": "permute_dim",
    "BatchSize": "batch_size",
    "Epochs": "epochs",
    "LearningRate": "learning_rate",
    "Optimizer": "optimizer",
    "LossFunction": "loss_function",
    "Metrics": "metrics",
    "WeightDecay": "weight_decay",
    "Momentum": "momentum",
    "PathData": "path_data",
    "TaskType": "task_type",
    "InputFormat": "input_format",
    "Shape": "shape",
    "Normalize": "normalize",
}


# Inverse mapping per layer is built lazily.
_LAYER_SLUG_BY_TYPE: dict[str, str] = {
    "Conv1DLayer": "Conv1D",
    "Conv2DLayer": "Conv2D",
    "Conv3DLayer": "Conv3D",
    "PoolingLayer": "Pooling",
    "RNNLayer": "RNN",
    "LSTMLayer": "LSTM",
    "GRULayer": "GRU",
    "LinearLayer": "Linear",
    "FlattenLayer": "Flatten",
    "EmbeddingLayer": "Embedding",
    "DropoutLayer": "Dropout",
    "LayerNormalizationLayer": "LayerNormalization",
    "BatchNormalizationLayer": "BatchNormalization",
    "TensorOp": "TensorOp",
    "Configuration": "Configuration",
    "TrainingDataset": "Dataset",
    "TestDataset": "Dataset",
}


def _is_nn_layer_type(t: str) -> bool:
    return t in _NN_LAYER_TYPES


def _is_nn_attr_type(t: str) -> bool:
    if not t:
        return False
    if "Attribute" not in t:
        return False
    # Heuristic: ``<Slug>Attribute<LayerSlug>``.
    return any(t.startswith(prefix + "Attribute") for prefix in _NN_ATTR_PREFIX_TO_KEY)


def _nn_v3_attr_to_key(attr_type: str, layer_type: str) -> Optional[str]:
    """Strip the layer-suffix from a v3 attribute type and return its v4 key."""
    if not attr_type:
        return None
    for prefix, key in _NN_ATTR_PREFIX_TO_KEY.items():
        candidate = f"{prefix}Attribute"
        if attr_type.startswith(candidate):
            return key
    return None


def _nn_attr_v3_type(layer_type: str, key: str) -> Optional[str]:
    """Inverse of ``_nn_v3_attr_to_key``: build the v3 attribute element type."""
    inv = {v: k for k, v in _NN_ATTR_PREFIX_TO_KEY.items()}
    prefix = inv.get(key)
    if not prefix:
        return None
    layer_slug = _LAYER_SLUG_BY_TYPE.get(layer_type)
    if not layer_slug:
        return None
    return f"{prefix}Attribute{layer_slug}"


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
