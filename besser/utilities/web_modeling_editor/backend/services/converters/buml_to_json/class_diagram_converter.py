"""
Domain model conversion from BUML to JSON format.

Emits the v4 wire shape (``{nodes, edges}``) directly — see
``docs/source/migrations/uml-v4-shape.md``. There is no v3-shape
intermediate; every node is built via ``make_node`` and every edge via
``make_edge``.
"""

import logging
import uuid
from besser.BUML.metamodel.structural import (
    Class, Property, Method, Parameter as StructuralParameter, DomainModel,
    PrimitiveDataType, Enumeration,
    EnumerationLiteral, BinaryAssociation, Generalization, Multiplicity,
    UNLIMITED_MAX_MULTIPLICITY, Constraint, AssociationClass, Metadata,
    MethodImplementationType,
)

from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json._node_builders import (
    make_node, make_edge,
)

# Layout constants for auto-grid positioning
LAYOUT_GRID_WIDTH = 1200
LAYOUT_GRID_HEIGHT = 800
LAYOUT_X_SPACING = 300
LAYOUT_Y_SPACING = 200
LAYOUT_MAX_COLUMNS = 3

logger = logging.getLogger(__name__)
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    VISIBILITY_MAP, RELATIONSHIP_TYPES,
)


def parse_buml_content(content: str) -> DomainModel:
    """Parse B-UML content from a Python file and return a DomainModel."""
    try:
        if isinstance(content, DomainModel):
            return content

        safe_globals = {
            "__builtins__": {
                "set": set,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "len": len,
                "range": range,
                "True": True,
                "False": False,
                "None": None,
                "print": lambda *a, **kw: None,
            },
            "Class": Class,
            "Property": Property,
            "Method": Method,
            "Parameter": StructuralParameter,
            "PrimitiveDataType": PrimitiveDataType,
            "BinaryAssociation": BinaryAssociation,
            "Constraint": Constraint,
            "Multiplicity": Multiplicity,
            "UNLIMITED_MAX_MULTIPLICITY": UNLIMITED_MAX_MULTIPLICITY,
            "Generalization": Generalization,
            "Enumeration": Enumeration,
            "EnumerationLiteral": EnumerationLiteral,
            "DomainModel": DomainModel,
            "AssociationClass": AssociationClass,
            "Metadata": Metadata,
            "MethodImplementationType": MethodImplementationType,
            "set": set,
            "StringType": PrimitiveDataType("str"),
            "IntegerType": PrimitiveDataType("int"),
            "FloatType": PrimitiveDataType("float"),
            "BooleanType": PrimitiveDataType("bool"),
            "TimeType": PrimitiveDataType("time"),
            "DateType": PrimitiveDataType("date"),
            "DateTimeType": PrimitiveDataType("datetime"),
            "TimeDeltaType": PrimitiveDataType("timedelta"),
            "AnyType": PrimitiveDataType("any"),
        }

        if not isinstance(content, str):
            raise TypeError(f"Expected B-UML content as str or DomainModel, got {type(content)!r}")

        cleaned_lines = []
        in_import_block = False
        for line in content.splitlines():
            stripped = line.lstrip()
            if in_import_block:
                if ")" in line:
                    in_import_block = False
                continue
            if stripped.startswith(("import ", "from ")):
                if "(" in line and ")" not in line:
                    in_import_block = True
                continue
            if any(gen in line for gen in ["Generator(", ".generate("]):
                continue
            cleaned_lines.append(line)
        cleaned_content = "\n".join(cleaned_lines)

        local_vars = {}
        exec(cleaned_content, safe_globals, local_vars)

        domain_name = "Imported_Domain_Model"
        for var_name, var_value in local_vars.items():
            if isinstance(var_value, DomainModel):
                domain_name = var_value.name

        domain_model = DomainModel(domain_name)
        for var_name, var_value in local_vars.items():
            if isinstance(var_value, (Class, Enumeration)):
                domain_model.types.add(var_value)
            elif isinstance(var_value, Constraint):
                domain_model.constraints.add(var_value)

        for var_name, var_value in local_vars.items():
            if isinstance(var_value, BinaryAssociation):
                domain_model.associations.add(var_value)
            elif isinstance(var_value, Generalization):
                domain_model.generalizations.add(var_value)

        return domain_model

    except Exception as e:
        logger.error("Error parsing B-UML content: %s", e)
        raise ValueError(f"Failed to parse B-UML content: {str(e)}")


def _multiplicity_str(prop: Property) -> str:
    return f"{prop.multiplicity.min}..{'*' if prop.multiplicity.max == UNLIMITED_MAX_MULTIPLICITY else prop.multiplicity.max}"


def _attr_row(attr: Property) -> dict:
    """Build a v4 ``ClassifierMember`` row for an attribute."""
    attr_type = attr.type.name if hasattr(attr.type, "name") else str(attr.type)
    row = {
        "id": str(uuid.uuid4()),
        "name": attr.name,
        "attributeType": attr_type,
        "visibility": attr.visibility,
        "isOptional": attr.is_optional,
        "isId": attr.is_id,
        "isExternalId": attr.is_external_id,
        "isDerived": attr.is_derived,
    }
    if attr.default_value is not None:
        row["defaultValue"] = attr.default_value
    return row


def _method_row(method: Method, type_obj: Class, method_diagram_refs: dict) -> dict:
    """Build a v4 ``ClassifierMember`` row for a method."""
    visibility_symbol = next(
        k for k, v in VISIBILITY_MAP.items() if v == method.visibility
    )
    param_str = []
    for param in method.parameters:
        param_type = param.type.name if hasattr(param.type, "name") else str(param.type)
        param_signature = f"{param.name}: {param_type}"
        if hasattr(param, "default_value") and param.default_value is not None:
            param_signature += f" = {param.default_value}"
        param_str.append(param_signature)
    method_signature = f"{visibility_symbol} {method.name}({', '.join(param_str)})"
    if hasattr(method, "type") and method.type:
        return_type = method.type.name if hasattr(method.type, "name") else str(method.type)
        method_signature += f": {return_type}"

    row: dict = {
        "id": str(uuid.uuid4()),
        "name": method_signature,
        "visibility": method.visibility,
        "attributeType": "any",
    }
    if hasattr(method, "code") and method.code:
        row["code"] = method.code

    if hasattr(method, "implementation_type") and method.implementation_type:
        impl_type_map = {
            MethodImplementationType.NONE: "none",
            MethodImplementationType.CODE: "code",
            MethodImplementationType.BAL: "bal",
            MethodImplementationType.STATE_MACHINE: "state_machine",
            MethodImplementationType.QUANTUM_CIRCUIT: "quantum_circuit",
        }
        impl_type = method.implementation_type
        if isinstance(impl_type, str):
            normalized_impl_type = impl_type.strip()
            if normalized_impl_type.startswith("MethodImplementationType."):
                normalized_impl_type = normalized_impl_type.split(".", maxsplit=1)[1]
            mapped = normalized_impl_type.lower()
            if mapped in {"none", "code", "bal", "state_machine", "quantum_circuit"}:
                row["implementationType"] = mapped
            else:
                row["implementationType"] = "none"
        else:
            row["implementationType"] = impl_type_map.get(impl_type, "none")

    refs = method_diagram_refs.get((type_obj.name, method.name), {})
    state_machine_id = refs.get("stateMachineId") or None
    if not state_machine_id and hasattr(method, "state_machine") and method.state_machine:
        state_machine_id = method.state_machine.name
    if state_machine_id:
        row["stateMachineId"] = state_machine_id
    quantum_circuit_id = refs.get("quantumCircuitId") or None
    if not quantum_circuit_id and hasattr(method, "quantum_circuit") and method.quantum_circuit:
        quantum_circuit_id = method.quantum_circuit.name
    if quantum_circuit_id:
        row["quantumCircuitId"] = quantum_circuit_id

    if not row.get("implementationType"):
        row.pop("implementationType", None)
    if not row.get("stateMachineId"):
        row.pop("stateMachineId", None)
    if not row.get("quantumCircuitId"):
        row.pop("quantumCircuitId", None)
    return row


def class_buml_to_json(domain_model):
    """Convert a B-UML DomainModel to the v4 ``{nodes, edges}`` wire shape."""
    nodes: list = []
    edges: list = []
    method_diagram_refs = getattr(domain_model, 'method_diagram_refs', {})
    layout_positions = getattr(domain_model, '_layout_positions', {})

    grid_size = {
        "x_spacing": LAYOUT_X_SPACING,
        "y_spacing": LAYOUT_Y_SPACING,
        "max_columns": LAYOUT_MAX_COLUMNS,
    }
    current_column = 0
    current_row = 0
    comments_to_create: list = []  # [(text, linked_class_node_id)]

    def get_position():
        nonlocal current_column, current_row
        x = -600 + (current_column * grid_size["x_spacing"])
        y = -300 + (current_row * grid_size["y_spacing"])
        current_column += 1
        if current_column >= grid_size["max_columns"]:
            current_column = 0
            current_row += 1
        return x, y

    class_id_map: dict = {}  # type_obj -> node id

    # Pre-compute OCL constraints to inline per class.
    # ``Constraint.context`` is a Class; collapse the constraint onto that
    # class as a row in ``data.oclConstraints``. Free-standing fallback
    # (no class context) becomes its own ``class`` node with stereotype
    # ``oclConstraint``.
    constraints_by_class: dict = {}
    standalone_constraints: list = []
    for c in domain_model.constraints:
        if isinstance(c, Constraint) and c.context is not None:
            constraints_by_class.setdefault(c.context, []).append(c)
        else:
            standalone_constraints.append(c)

    # Emit class / abstract / interface / enumeration nodes.
    for type_obj in sorted(
        (t for t in domain_model.types if isinstance(t, (Class, Enumeration))),
        key=lambda t: t.name,
    ):
        node_id = str(uuid.uuid4())
        class_id_map[type_obj] = node_id

        saved_bounds = layout_positions.get(type_obj.name)
        if saved_bounds:
            x = saved_bounds["x"]
            y = saved_bounds["y"]
            saved_width = saved_bounds.get("width", 160)
            saved_height = saved_bounds.get("height", 100)
        else:
            x, y = get_position()
            saved_width = None
            saved_height = None

        attribute_rows: list = []
        method_rows: list = []
        ocl_rows: list = []

        if isinstance(type_obj, Class):
            stereotype = "abstract" if type_obj.is_abstract else None
            for attr in sorted(type_obj.attributes, key=lambda a: a.name):
                attribute_rows.append(_attr_row(attr))
            for method in sorted(type_obj.methods, key=lambda m: m.name):
                method_rows.append(_method_row(method, type_obj, method_diagram_refs))

            # OCL constraints anchored on this class.
            for c in sorted(constraints_by_class.get(type_obj, []), key=lambda c: c.name):
                row = {
                    "id": str(uuid.uuid4()),
                    "name": c.name,
                    "expression": c.expression,
                }
                if getattr(c, "description", None):
                    row["description"] = c.description
                ocl_rows.append(row)

            # Method-level pre/post become ocl rows on the owning class too.
            for method in sorted(type_obj.methods, key=lambda m: m.name):
                for kind, constraints in (("precondition", getattr(method, "pre", []) or []),
                                          ("postcondition", getattr(method, "post", []) or [])):
                    for c in constraints:
                        row = {
                            "id": str(uuid.uuid4()),
                            "name": c.name,
                            "expression": c.expression,
                        }
                        if getattr(c, "description", None):
                            row["description"] = c.description
                        ocl_rows.append(row)
        else:
            stereotype = "enumeration"
            ordered_literals = getattr(type_obj, '_ordered_literals', None)
            if ordered_literals is not None:
                literals_iter = ordered_literals
            else:
                literals_iter = sorted(type_obj.literals, key=lambda lit: lit.name)
            for literal in literals_iter:
                attribute_rows.append({
                    "id": str(uuid.uuid4()),
                    "name": literal.name,
                    "attributeType": "str",
                    "visibility": "public",
                })

        data: dict = {"name": type_obj.name, "stereotype": stereotype}
        data["attributes"] = attribute_rows
        data["methods"] = method_rows
        if ocl_rows:
            data["oclConstraints"] = ocl_rows

        if isinstance(type_obj, Class) and getattr(type_obj, 'metadata', None):
            md = type_obj.metadata
            if md.description:
                data["description"] = md.description
                comments_to_create.append((md.description, node_id))
            if md.uri:
                data["uri"] = md.uri
            if md.icon:
                data["icon"] = md.icon

        computed_height = max(100, 30 * (len(attribute_rows) + len(method_rows) + 1))
        node = make_node(
            node_id=node_id,
            type_="class",
            data=data,
            position={"x": x, "y": y},
            width=saved_width if saved_width is not None else 160,
            height=saved_height if saved_height is not None else computed_height,
        )
        nodes.append(node)

    # Free-standing OCL constraints (rare fallback per the spec).
    for c in sorted(standalone_constraints, key=lambda c: c.name):
        x, y = get_position()
        node_id = str(uuid.uuid4())
        data = {
            "name": c.name,
            "stereotype": "oclConstraint",
            "attributes": [],
            "methods": [],
            "constraint": c.expression,
        }
        if getattr(c, "description", None):
            data["description"] = c.description
        nodes.append(make_node(
            node_id=node_id,
            type_="class",
            data=data,
            position={"x": x, "y": y},
            width=200,
            height=100,
        ))

    # Associations.
    for association in sorted(domain_model.associations, key=lambda a: a.name):
        try:
            name = association.name or ""
            ends = sorted(association.ends, key=lambda e: e.name)
            if len(ends) != 2:
                continue
            source_prop, target_prop = ends

            if source_prop.is_composite and not target_prop.is_composite:
                source_prop, target_prop = target_prop, source_prop
            elif not source_prop.is_composite and not target_prop.is_composite:
                if not source_prop.is_navigable and target_prop.is_navigable:
                    pass
                elif source_prop.is_navigable and not target_prop.is_navigable:
                    source_prop, target_prop = target_prop, source_prop
                elif not source_prop.is_navigable and not target_prop.is_navigable:
                    logger.warning("Both ends of association %s are not navigable. Skipping.", name)
                    continue

            source_class = source_prop.type
            target_class = target_prop.type
            if source_class not in class_id_map or target_class not in class_id_map:
                continue

            rel_type = (
                RELATIONSHIP_TYPES["composition"]
                if target_prop.is_composite
                else (
                    RELATIONSHIP_TYPES["bidirectional"]
                    if source_prop.is_navigable and target_prop.is_navigable
                    else RELATIONSHIP_TYPES["unidirectional"]
                )
            )

            saved_rel = layout_positions.get(f"rel_{name}") or {}
            edge_data: dict = {
                "name": name,
                "sourceRole": source_prop.name,
                "sourceMultiplicity": _multiplicity_str(source_prop),
                "targetRole": target_prop.name,
                "targetMultiplicity": _multiplicity_str(target_prop),
                "points": saved_rel.get("path", []),
            }
            if "isManuallyLayouted" in saved_rel:
                edge_data["isManuallyLayouted"] = saved_rel["isManuallyLayouted"]

            edge = make_edge(
                edge_id=str(uuid.uuid4()),
                source=class_id_map[source_class],
                target=class_id_map[target_class],
                type_=rel_type,
                data=edge_data,
                source_handle=saved_rel.get("source_direction", "Right"),
                target_handle=saved_rel.get("target_direction", "Left"),
            )
            edges.append(edge)
        except Exception as e:
            logger.error("Error converting relationship to JSON: %s", e, exc_info=True)
            continue

    # Generalizations.
    for generalization in sorted(
        domain_model.generalizations,
        key=lambda g: (g.specific.name, g.general.name),
    ):
        if (
            generalization.general not in class_id_map
            or generalization.specific not in class_id_map
        ):
            continue
        gen_key = f"gen_{generalization.specific.name}_{generalization.general.name}"
        saved_gen = layout_positions.get(gen_key) or {}
        edge = make_edge(
            edge_id=str(uuid.uuid4()),
            source=class_id_map[generalization.specific],
            target=class_id_map[generalization.general],
            type_="ClassInheritance",
            data={"points": saved_gen.get("path", [])},
        )
        edges.append(edge)

    # Association classes -> ClassLinkRel edges.
    # Note: the AssociationClass has an embedded ``association``; find the
    # association edge we just emitted whose name matches and link it.
    edge_by_name: dict = {}
    for edge in edges:
        edge_data = edge.get("data") or {}
        if edge.get("type") in (
            RELATIONSHIP_TYPES["bidirectional"],
            RELATIONSHIP_TYPES["unidirectional"],
            RELATIONSHIP_TYPES["composition"],
        ) and edge_data.get("name"):
            edge_by_name[edge_data["name"]] = edge["id"]
    for type_obj in sorted(domain_model.types, key=lambda t: t.name):
        if isinstance(type_obj, AssociationClass) and type_obj in class_id_map:
            assoc_edge_id = edge_by_name.get(type_obj.association.name)
            if assoc_edge_id:
                edges.append(make_edge(
                    edge_id=str(uuid.uuid4()),
                    source=assoc_edge_id,
                    target=class_id_map[type_obj],
                    type_="ClassLinkRel",
                    data={"points": []},
                    source_handle="Center",
                    target_handle="Up",
                ))

    # Comments (top-level Comments nodes + Link edges).
    for comment_text, linked_class_id in comments_to_create:
        x, y = get_position()
        comment_id = str(uuid.uuid4())
        nodes.append(make_node(
            node_id=comment_id,
            type_="Comments",
            data={"name": comment_text},
            position={"x": x, "y": y},
            width=160,
            height=100,
        ))
        edges.append(make_edge(
            edge_id=str(uuid.uuid4()),
            source=comment_id,
            target=linked_class_id,
            type_="Link",
            data={"points": []},
        ))

    if hasattr(domain_model, 'metadata') and domain_model.metadata and domain_model.metadata.description:
        x, y = get_position()
        nodes.append(make_node(
            node_id=str(uuid.uuid4()),
            type_="Comments",
            data={"name": domain_model.metadata.description},
            position={"x": x, "y": y},
            width=160,
            height=100,
        ))

    default_size = {"width": LAYOUT_GRID_WIDTH, "height": LAYOUT_GRID_HEIGHT}
    return {
        "version": "4.0.0",
        "type": "ClassDiagram",
        "title": getattr(domain_model, "name", "") or "",
        "size": default_size,
        "nodes": nodes,
        "edges": edges,
        "interactive": {"elements": {}, "relationships": {}},
        "assessments": {},
    }
