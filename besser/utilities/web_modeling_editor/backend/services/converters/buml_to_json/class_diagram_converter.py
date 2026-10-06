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
    make_node, make_edge, grid_layout, snap_up,
)

# Layout constants for auto-grid positioning
LAYOUT_GRID_WIDTH = 1200
LAYOUT_GRID_HEIGHT = 800
LAYOUT_ORIGIN = (-600, -300)
LAYOUT_MAX_COLUMNS = 3
LAYOUT_GAP_Y = 100

# Mirrors the editor's class node metrics (``LAYOUT`` in the library's
# constants.ts and ``calculateMinWidth`` / ``calculateMinHeight``).
CLASS_HEADER_HEIGHT = 40
CLASS_HEADER_HEIGHT_WITH_STEREOTYPE = 50
CLASS_ROW_HEIGHT = 30
CLASS_PADDING = 10
CLASS_MIN_WIDTH = 160
CHAR_WIDTH_PX = 9  # rough average glyph width of the editor's 16px Inter
_VISIBILITY_SYMBOLS = {"public": "+", "private": "-", "protected": "#", "package": "~"}

logger = logging.getLogger(__name__)
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    RELATIONSHIP_TYPES,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json._safe_buml_loader import (
    safe_load_buml,
)


def parse_buml_content(content: str) -> DomainModel:
    """Parse B-UML content from a Python file and return a DomainModel."""
    try:
        if isinstance(content, DomainModel):
            return content

        # Allowlist of names that may appear in the BUML source. Everything
        # else is rejected by the safe AST-based loader (which also skips a
        # trailing ``if __name__ == "__main__":`` block).
        allowed_names = {
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
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "StringType": PrimitiveDataType("str"),
            "IntegerType": PrimitiveDataType("int"),
            "FloatType": PrimitiveDataType("float"),
            "BooleanType": PrimitiveDataType("bool"),
            "TimeType": PrimitiveDataType("time"),
            "DateType": PrimitiveDataType("date"),
            "DateTimeType": PrimitiveDataType("datetime"),
            "TimeDeltaType": PrimitiveDataType("timedelta"),
            "AnyType": PrimitiveDataType("any"),
            # No-op stub: project-exported files often end with a
            # `project = Project(name=..., models=[domain_model], ...)` tail.
            # The class converter only cares about the DomainModel; swallowing
            # the Project(...) call with a stub keeps the sandbox import-tolerant.
            "Project": lambda *args, **kwargs: None,
        }

        if not isinstance(content, str):
            raise TypeError(f"Expected B-UML content as str or DomainModel, got {type(content)!r}")

        # Strip a leading UTF-8 BOM so the sandboxed exec does not fail with
        # "invalid non-printable character U+FEFF" for files saved with a BOM
        # (common from Windows editors).
        if content.startswith("﻿"):
            content = content[1:]

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

        # Execute the cleaned B-UML content through the safe AST-based loader.
        local_vars = safe_load_buml(cleaned_content, allowed_names)

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


def _format_multiplicity_label(multiplicity) -> str:
    """Render a ``Multiplicity`` as its UML association-end label.

    Collapses an exact multiplicity (``min == max``) to a single value
    (``1..1`` -> ``1``, ``5..5`` -> ``5``); an unbounded upper bound renders as
    ``*`` (``0..*``, ``1..*``). The result round-trips through
    ``parse_multiplicity`` (a bare ``N`` is read back as ``N..N``).
    """
    min_val = multiplicity.min
    max_val = multiplicity.max
    if max_val == UNLIMITED_MAX_MULTIPLICITY:
        return f"{min_val}..*"
    if min_val == max_val:
        return f"{min_val}"
    return f"{min_val}..{max_val}"


def _multiplicity_str(prop: Property) -> str:
    """Render a property's multiplicity as its UML association-end label."""
    return _format_multiplicity_label(prop.multiplicity)


def _json_safe_default(value):
    """Coerce a default value to a JSON-serialisable form.

    ``default_value`` may be an ``EnumerationLiteral`` (or other metamodel
    object); use its ``name`` (else its string form) so the diagram JSON can
    be serialised when sent to the render service. Primitives pass through.
    """
    if isinstance(value, (str, int, float, bool)):
        return value
    return getattr(value, "name", None) or str(value)


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
        row["defaultValue"] = _json_safe_default(attr.default_value)
    return row


def _method_row(method: Method, type_obj: Class, method_diagram_refs: dict) -> dict:
    """Build a v4 ``ClassifierMember`` row for a method.

    Emits the canonical inspector-authored shape (``ClassNodeElement`` in
    ``packages/library/lib/types/nodes/NodeProps.ts``): a *bare* ``name``,
    structured ``parameters`` rows, and the return type on ``returnType``
    mirrored onto ``attributeType`` (``any`` when absent) — exactly what
    the frontend's add-method inspector writes, so the canvas renderer
    (``formatDisplayName``) never double-decorates a fused signature.
    """
    return_type = None
    if hasattr(method, "type") and method.type:
        return_type = method.type.name if hasattr(method.type, "name") else str(method.type)

    structured_parameters: list = []
    for param in method.parameters:
        param_type = param.type.name if hasattr(param.type, "name") else str(param.type)
        param_row: dict = {
            "id": str(uuid.uuid4()),
            "name": param.name,
            "parameterType": param_type,
        }
        if hasattr(param, "default_value") and param.default_value is not None:
            param_row["defaultValue"] = _json_safe_default(param.default_value)
        structured_parameters.append(param_row)

    row: dict = {
        "id": str(uuid.uuid4()),
        "name": method.name,
        "visibility": method.visibility,
        "attributeType": return_type or "any",
        "returnType": return_type or "any",
        "parameters": structured_parameters,
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
            MethodImplementationType.NEURAL_NETWORK: "neural_network",
        }
        impl_type = method.implementation_type
        if isinstance(impl_type, str):
            normalized_impl_type = impl_type.strip()
            if normalized_impl_type.startswith("MethodImplementationType."):
                normalized_impl_type = normalized_impl_type.split(".", maxsplit=1)[1]
            mapped = normalized_impl_type.lower()
            if mapped in {"none", "code", "bal", "state_machine", "quantum_circuit", "neural_network"}:
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
    neural_network_id = refs.get("neuralNetworkId") or None
    if not neural_network_id and getattr(method, "neural_network", None):
        # Without a saved diagram id, reference the network by name
        # (``method_nn_linker`` resolves ids and titles alike).
        neural_network_id = method.neural_network.name
    if neural_network_id:
        row["neuralNetworkId"] = neural_network_id

    if not row.get("implementationType"):
        row.pop("implementationType", None)
    if not row.get("stateMachineId"):
        row.pop("stateMachineId", None)
    if not row.get("quantumCircuitId"):
        row.pop("quantumCircuitId", None)
    if not row.get("neuralNetworkId"):
        row.pop("neuralNetworkId", None)
    return row


def _member_label(row: dict, is_method: bool, stereotype) -> str:
    """Approximate the canvas label of a member row (``formatDisplayName``)."""
    if stereotype == "Enumeration":
        return row.get("name", "")
    vis = _VISIBILITY_SYMBOLS.get(row.get("visibility") or "public", "+")
    if is_method:
        params = ", ".join(f"{p['name']}: {p.get('parameterType', '')}" for p in row.get("parameters") or [])
        return f"{vis} {row.get('name', '')}({params}): {row.get('returnType', '')}"
    label = f"{vis} {row.get('name', '')}: {row.get('attributeType', '')}"
    if row.get("defaultValue") not in (None, ""):
        label += f" = {row['defaultValue']}"
    if row.get("isId"):
        label += " {id}"
    return label


def _class_node_size(name: str, stereotype, attribute_rows: list, method_rows: list) -> tuple:
    """Content-based (width, height) of a class node, as the editor sizes it."""
    header = CLASS_HEADER_HEIGHT_WITH_STEREOTYPE if stereotype else CLASS_HEADER_HEIGHT
    labels = [name, f"\u00ab{stereotype}\u00bb" if stereotype else ""]
    labels += [_member_label(r, False, stereotype) for r in attribute_rows]
    labels += [_member_label(r, True, stereotype) for r in method_rows]
    text_width = max(len(label) for label in labels) * CHAR_WIDTH_PX
    width = max(CLASS_MIN_WIDTH, snap_up(text_width + 2 * CLASS_PADDING))
    height = snap_up(header + CLASS_ROW_HEIGHT * (len(attribute_rows) + len(method_rows)))
    return width, height


def class_buml_to_json(domain_model):
    """Convert a B-UML DomainModel to the v4 ``{nodes, edges}`` wire shape."""
    nodes: list = []
    edges: list = []
    method_diagram_refs = getattr(domain_model, 'method_diagram_refs', {})
    layout_positions = getattr(domain_model, '_layout_positions', {})

    comments_to_create: list = []  # [(text, linked_class_node_id)]
    # Nodes without a saved position; placed on a size-aware grid at the end.
    auto_placed: list = []

    class_id_map: dict = {}  # type_obj -> node id

    # Emit class / abstract / interface / enumeration nodes.
    for type_obj in sorted(
        (t for t in domain_model.types if isinstance(t, (Class, Enumeration))),
        key=lambda t: t.name,
    ):
        node_id = str(uuid.uuid4())
        class_id_map[type_obj] = node_id

        saved_bounds = layout_positions.get(type_obj.name)

        attribute_rows: list = []
        method_rows: list = []

        if isinstance(type_obj, Class):
            # The frontend compares stereotypes case-sensitively against
            # the capitalized canonical forms ('Abstract' / 'Interface' /
            # 'Enumeration'); the processor lowercases on read, so the
            # capitalized emit is backward-safe.
            stereotype = "Abstract" if type_obj.is_abstract else None
            for attr in sorted(type_obj.attributes, key=lambda a: a.name):
                attribute_rows.append(_attr_row(attr))
            for method in sorted(type_obj.methods, key=lambda m: m.name):
                method_rows.append(_method_row(method, type_obj, method_diagram_refs))
        else:
            stereotype = "Enumeration"
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

        if isinstance(type_obj, Class) and getattr(type_obj, 'metadata', None):
            md = type_obj.metadata
            if md.description:
                data["description"] = md.description
                comments_to_create.append((md.description, node_id))
            if md.uri:
                data["uri"] = md.uri
            if md.icon:
                data["icon"] = md.icon

        width, height = _class_node_size(type_obj.name, stereotype, attribute_rows, method_rows)
        if saved_bounds:
            position = {"x": saved_bounds["x"], "y": saved_bounds["y"]}
            width = saved_bounds.get("width", width)
            height = saved_bounds.get("height", height)
        else:
            position = {"x": 0, "y": 0}
        node = make_node(
            node_id=node_id,
            type_="class",
            data=data,
            position=position,
            width=width,
            height=height,
        )
        nodes.append(node)
        if not saved_bounds:
            auto_placed.append(node)

    # OCL constraints — every constraint is emitted as its own
    # ``ClassOCLConstraint`` node (the sticky-note shape the frontend
    # renders and edits via ``ClassOCLConstraintEditPanel``), tethered to
    # its anchoring class by a visual ``ClassOCLLink`` edge when one
    # resolves. Mirrors the develop baseline, where constraints are always
    # visible boxes; the canonical full ``context ...`` text rides on
    # ``data.expression`` so the ingest side re-derives the context class
    # from the text itself (the link is purely visual).
    def _emit_constraint_node(c, anchor_class) -> None:
        node_id = str(uuid.uuid4())
        data = {
            "name": c.name,
            "expression": c.expression,
        }
        if getattr(c, "description", None):
            data["description"] = c.description
        node = make_node(
            node_id=node_id,
            type_="ClassOCLConstraint",
            data=data,
            position={"x": 0, "y": 0},
            width=210,
            height=90,
        )
        nodes.append(node)
        auto_placed.append(node)
        if anchor_class is not None and anchor_class in class_id_map:
            edges.append(make_edge(
                edge_id=str(uuid.uuid4()),
                source=node_id,
                target=class_id_map[anchor_class],
                type_="ClassOCLLink",
                data={"points": []},
            ))

    # Class-level invariants (and ownerless free-standing constraints).
    for c in sorted(domain_model.constraints, key=lambda c: c.name):
        anchor = c.context if isinstance(c, Constraint) else None
        _emit_constraint_node(c, anchor)

    # Method-level pre/post conditions, anchored on the owning class.
    # These live on ``Method.pre`` / ``Method.post`` (not in
    # ``domain_model.constraints``), so they need their own pass.
    for type_obj in sorted(
        (t for t in domain_model.types if isinstance(t, Class)),
        key=lambda t: t.name,
    ):
        for method in sorted(type_obj.methods, key=lambda m: m.name):
            for constraints in (getattr(method, "pre", []) or [],
                                getattr(method, "post", []) or []):
                for c in constraints:
                    _emit_constraint_node(c, type_obj)

    # Associations.
    for association in sorted(domain_model.associations, key=lambda a: a.name):
        try:
            name = association.name or ""
            ends = sorted(association.ends, key=lambda e: e.name)
            if len(ends) != 2:
                continue
            source_prop, target_prop = ends
            saved_rel = layout_positions.get(f"rel_{name}") or {}

            # Keep the orientation the diagram was drawn with (recorded by
            # the JSON -> BUML processor as ``source_role``), so the saved
            # handles / points are applied to the right classes. Navigability
            # does not affect orientation: it is emitted explicitly per end.
            if saved_rel.get("source_role") == target_prop.name:
                source_prop, target_prop = target_prop, source_prop

            # The composite (whole) end of a composition is always the target.
            if source_prop.is_composite and not target_prop.is_composite:
                source_prop, target_prop = target_prop, source_prop

            source_class = source_prop.type
            target_class = target_prop.type
            if source_class not in class_id_map or target_class not in class_id_map:
                continue

            # Plain associations are always emitted as ClassBidirectional;
            # which ends are navigable rides on the explicit per-end
            # ``sourceNavigable`` / ``targetNavigable`` flags. (The metamodel
            # has no aggregation flag, so aggregations round-trip as
            # ClassBidirectional too.)
            rel_type = (
                RELATIONSHIP_TYPES["composition"]
                if target_prop.is_composite
                else RELATIONSHIP_TYPES["bidirectional"]
            )

            edge_data: dict = {
                "name": name,
                "sourceRole": source_prop.name,
                "sourceMultiplicity": _multiplicity_str(source_prop),
                "sourceNavigable": bool(source_prop.is_navigable),
                "targetRole": target_prop.name,
                "targetMultiplicity": _multiplicity_str(target_prop),
                "targetNavigable": bool(target_prop.is_navigable),
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
                source_handle=saved_rel.get("source_direction", "right"),
                target_handle=saved_rel.get("target_direction", "left"),
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

    # Comments (top-level ``comment`` nodes + ``CommentLink`` edges — the
    # node / edge types the React Flow frontend registers).
    for comment_text, linked_class_id in comments_to_create:
        comment_id = str(uuid.uuid4())
        node = make_node(
            node_id=comment_id,
            type_="comment",
            data={"name": comment_text},
            position={"x": 0, "y": 0},
            width=160,
            height=100,
        )
        nodes.append(node)
        auto_placed.append(node)
        edges.append(make_edge(
            edge_id=str(uuid.uuid4()),
            source=comment_id,
            target=linked_class_id,
            type_="CommentLink",
            data={"points": []},
        ))

    if hasattr(domain_model, 'metadata') and domain_model.metadata and domain_model.metadata.description:
        node = make_node(
            node_id=str(uuid.uuid4()),
            type_="comment",
            data={"name": domain_model.metadata.description},
            position={"x": 0, "y": 0},
            width=160,
            height=100,
        )
        nodes.append(node)
        auto_placed.append(node)

    # Auto-placed nodes go on the grid, below any nodes with a saved position.
    auto_ids = {n["id"] for n in auto_placed}
    placed = [n for n in nodes if n["id"] not in auto_ids]
    origin = LAYOUT_ORIGIN
    if placed:
        origin = (
            min(n["position"]["x"] for n in placed),
            max(n["position"]["y"] + n["height"] for n in placed) + LAYOUT_GAP_Y,
        )
    grid_layout(auto_placed, origin=origin, columns=LAYOUT_MAX_COLUMNS, gap_y=LAYOUT_GAP_Y)

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
