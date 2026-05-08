"""
Class diagram processing for converting JSON to BUML format.

Reads the v4 wire shape (``{nodes, edges}``) natively. See
``docs/source/migrations/uml-v4-shape.md`` for the spec.
"""

import logging
from typing import Any, Optional, Union

from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError

logger = logging.getLogger(__name__)

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Enumeration, Property, Method, BinaryAssociation,
    Generalization, PrimitiveDataType, EnumerationLiteral, AssociationClass,
    Metadata, Parameter, MethodImplementationType, Type
)
from besser.utilities.web_modeling_editor.backend.services.converters.parsers import (
    parse_attribute, parse_method, parse_multiplicity,
    legacy_body_only_to_text, process_ocl_constraints,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml._node_helpers import (
    node_data, node_bounds,
)
from besser.BUML.notations.ocl.error_handling import BOCLSyntaxError


def parse_method_signature_from_code(
    method_code: str,
    domain_model: DomainModel,
    type_lookup: Optional[dict[str, Type]] = None,
) -> Optional[tuple[str, list[dict[str, str]], Optional[str]]]:
    """Extract method signature from code text when diagram name/signature is malformed."""
    if not isinstance(method_code, str) or not method_code.strip():
        return None

    # String-based parsing to avoid ReDoS on untrusted input
    def_idx = method_code.find("def ")
    if def_idx == -1:
        return None
    after_def = method_code[def_idx + 4:].lstrip()
    paren_open = after_def.find("(")
    if paren_open == -1:
        return None
    method_name = after_def[:paren_open].strip()
    if not method_name or not method_name.replace("_", "").isalnum():
        return None
    paren_close = after_def.find(")", paren_open)
    if paren_close == -1:
        return None
    params_text = after_def[paren_open + 1:paren_close]
    after_paren = after_def[paren_close + 1:].lstrip()
    return_type = None
    if after_paren.startswith("->"):
        ret_text = after_paren[2:]
        end = len(ret_text)
        for ch in (":", "{", "\n"):
            pos = ret_text.find(ch)
            if pos != -1 and pos < end:
                end = pos
        return_type = ret_text[:end].strip() or None
    signature = f"{method_name.strip()}({(params_text or '').strip()})"
    if return_type:
        signature_with_return = f"{signature}: {return_type.strip()}"
    else:
        signature_with_return = signature

    try:
        _, parsed_name, parsed_parameters, parsed_return_type = parse_method(signature_with_return, domain_model, type_lookup=type_lookup)
    except ValueError:
        # Some BAL snippets use return markers that are not BUML type names (e.g., "nothing").
        _, parsed_name, parsed_parameters, parsed_return_type = parse_method(signature, domain_model, type_lookup=type_lookup)

    parsed_parameters = [
        param for param in parsed_parameters if param.get("name") not in {"self", "cls", "this"}
    ]
    return parsed_name, parsed_parameters, parsed_return_type


def _build_type_lookup(domain_model: DomainModel) -> dict[str, Union[Class, Enumeration]]:
    """Build a name-to-type lookup dict for O(1) type resolution."""
    return {
        t.name: t
        for t in domain_model.types
        if isinstance(t, (Enumeration, Class))
    }


def _resolve_type(type_name: str, type_lookup: dict[str, Union[Class, Enumeration]]) -> Type:
    """Resolve a type name to a metamodel type object."""
    resolved = type_lookup.get(type_name)
    if resolved is not None:
        return resolved
    try:
        return PrimitiveDataType(type_name)
    except ValueError:
        raise ConversionError(
            f"Unknown type '{type_name}'. It is not a class, enumeration, "
            f"or valid primitive type (str, int, float, bool, date, datetime, time, any). "
            f"Make sure the referenced type exists in the diagram."
        )


def _class_v4_kind(node: dict) -> str:
    """Return the v3-flavoured kind for a v4 ``class`` node.

    v4 collapses Class / AbstractClass / Interface / Enumeration to a single
    ``class`` node, with ``data.stereotype`` carrying the discriminator. This
    helper returns one of ``Class`` / ``AbstractClass`` / ``Interface`` /
    ``Enumeration`` so the rest of the processor reads naturally.
    """
    data = node_data(node)
    stereotype = (data.get("stereotype") or "").strip().lower()
    if stereotype == "abstract":
        return "AbstractClass"
    if stereotype == "interface":
        return "Interface"
    if stereotype == "enumeration":
        return "Enumeration"
    if stereotype == "oclconstraint":
        return "ClassOCLConstraint"
    return "Class"


def _process_enumerations(
    nodes: list[dict],
    domain_model: DomainModel,
    layout_positions: dict[str, Any],
    comment_elements: dict[str, str],
) -> None:
    """Create ``Enumeration`` instances for every enumeration class node and
    record their layout bounds. Also collects ``Comments`` nodes for later
    processing.
    """
    for node in nodes:
        node_type = node.get("type") or ""
        node_id = node.get("id")
        data = node_data(node)
        # Comments collapse to a top-level ``Comments`` node in v4 (same name).
        if node_type == "Comments":
            comment_text = (data.get("name") or "").strip()
            comment_elements[node_id] = comment_text
            continue

        if node_type != "class":
            continue
        if _class_v4_kind(node) != "Enumeration":
            continue

        element_name = (data.get("name") or "").strip()
        if not element_name or any(char.isspace() for char in element_name):
            raise ConversionError(
                f"Invalid enumeration name: '{element_name}'. Names cannot contain whitespace or be empty."
            )

        literals = []
        seen_literal_names = set()
        for literal in data.get("attributes") or []:
            literal_name = (literal.get("name") or "").strip()
            if not literal_name:
                raise ConversionError(
                    f"Empty enumeration literal name in '{element_name}'."
                )
            if literal_name in seen_literal_names:
                raise ConversionError(
                    f"Duplicate enumeration literal '{literal_name}' in '{element_name}'."
                )
            seen_literal_names.add(literal_name)
            literals.append(EnumerationLiteral(name=literal_name))

        enum = Enumeration(name=element_name, literals=set(literals))
        # Preserve insertion order from the JSON for round-trip fidelity.
        enum._ordered_literals = list(literals)
        try:
            domain_model.add_type(enum)
        except ValueError as e:
            raise ConversionError(str(e))

        layout_positions[element_name] = node_bounds(node)


def _process_classes(
    nodes: list[dict],
    domain_model: DomainModel,
    type_lookup: dict[str, Union[Class, Enumeration]],
    layout_positions: dict[str, Any],
    method_diagram_refs: dict[tuple[str, str], dict[str, str]],
) -> tuple[dict[str, Class], dict[str, Method]]:
    """Create classes with their attributes and methods.

    Two-pass: shells first (so cross-references resolve), then attributes
    and methods.
    """
    class_id_to_class: dict[str, Class] = {}
    method_id_to_method: dict[str, Method] = {}

    # Pass 1: class shells.
    for node in nodes:
        if node.get("type") != "class":
            continue
        kind = _class_v4_kind(node)
        if kind not in ("Class", "AbstractClass"):
            continue

        node_id = node.get("id")
        data = node_data(node)
        class_name = (data.get("name") or "").strip()
        if not class_name or any(char.isspace() for char in class_name):
            raise ConversionError(
                f"Invalid class name: '{class_name}'. Names cannot contain whitespace or be empty."
            )

        is_abstract = kind == "AbstractClass"
        metadata = None
        description = data.get("description")
        uri = data.get("uri")
        icon = data.get("icon")
        if description or uri or icon:
            metadata = Metadata(description=description, uri=uri, icon=icon)
        try:
            cls = Class(name=class_name, is_abstract=is_abstract, metadata=metadata)
            domain_model.add_type(cls)
        except ValueError as e:
            raise ConversionError(str(e))
        class_id_to_class[node_id] = cls
        layout_positions[class_name] = node_bounds(node)

    # Rebuild lookup so class-typed cross-references resolve in pass 2.
    type_lookup = _build_type_lookup(domain_model)

    # Pass 2: attributes and methods.
    for node in nodes:
        if node.get("type") != "class":
            continue
        kind = _class_v4_kind(node)
        if kind not in ("Class", "AbstractClass"):
            continue

        data = node_data(node)
        class_name = (data.get("name") or "").strip()
        cls = domain_model.get_class_by_name(class_name)
        if not cls:
            continue

        attribute_names: set[str] = set()
        for attr in data.get("attributes") or []:
            # v4 attribute rows always carry visibility + attributeType
            # explicitly (see uml-v4-shape.md). Legacy "+ name: type"
            # inline parsing is no longer needed at this layer; the
            # editor migrator normalises it.
            visibility = attr.get("visibility", "public")
            name = (attr.get("name") or "").strip()
            attr_type = attr.get("attributeType")
            if attr_type is None:
                # Legacy fallback: parse from name string. Mirrors what the
                # editor migrator does for very old fixtures, but emit a
                # warning-free path for callers that already produce v4
                # rows without ``attributeType``.
                visibility, name, attr_type = parse_attribute(
                    attr.get("name", ""), domain_model, type_lookup=type_lookup,
                )
            is_optional = attr.get("isOptional", False)
            is_id = attr.get("isId", False)
            is_external_id = attr.get("isExternalId", False)
            is_derived = attr.get("isDerived", False)
            default_value = attr.get("defaultValue")

            if not name:
                continue
            if name in attribute_names:
                raise ConversionError(
                    f"Duplicate attribute name '{name}' found in class '{class_name}'"
                )
            attribute_names.add(name)

            type_obj = _resolve_type(attr_type, type_lookup)
            property_ = Property(
                name=name, type=type_obj, visibility=visibility,
                is_optional=is_optional, is_id=is_id,
                is_external_id=is_external_id, is_derived=is_derived,
                default_value=default_value,
            )
            cls.add_attribute(property_)

        for method in data.get("methods") or []:
            visibility, name, parameters, return_type = parse_method(
                method.get("name", ""), domain_model, type_lookup=type_lookup,
            )

            method_code = method.get("code", "")
            method_name_is_malformed = (
                isinstance(name, str) and name.count("(") != name.count(")")
            )
            if method_name_is_malformed or (not parameters and method_code):
                parsed_from_code = parse_method_signature_from_code(
                    method_code, domain_model, type_lookup=type_lookup,
                )
                if parsed_from_code:
                    parsed_name, parsed_parameters, parsed_return_type = parsed_from_code
                    if method_name_is_malformed and parsed_name:
                        name = parsed_name
                    if not parameters and parsed_parameters:
                        parameters = parsed_parameters
                    if not return_type and parsed_return_type:
                        return_type = parsed_return_type

            impl_type_str = method.get("implementationType", "none")
            if isinstance(impl_type_str, str):
                impl_type_str = impl_type_str.strip().lower()
            state_machine_id = method.get("stateMachineId", "")
            quantum_circuit_id = method.get("quantumCircuitId", "")

            impl_type_map = {
                "none": MethodImplementationType.NONE,
                "code": MethodImplementationType.CODE,
                "bal": MethodImplementationType.BAL,
                "state_machine": MethodImplementationType.STATE_MACHINE,
                "quantum_circuit": MethodImplementationType.QUANTUM_CIRCUIT,
            }
            implementation_type = impl_type_map.get(impl_type_str, MethodImplementationType.NONE)
            if implementation_type == MethodImplementationType.NONE and method_code:
                implementation_type = MethodImplementationType.CODE

            method_params = []
            for param in parameters:
                param_type_name = param.get('type', 'any')
                param_name = param.get('name')
                if not param_name:
                    logger.warning(
                        "Skipping parameter with missing name in method '%s' of class '%s'.",
                        name, class_name,
                    )
                    continue
                param_type_obj = _resolve_type(param_type_name, type_lookup)
                param_obj = Parameter(name=param_name, type=param_type_obj)
                if 'default' in param:
                    param_obj.default_value = param['default']
                method_params.append(param_obj)

            method_obj = Method(
                name=name,
                visibility=visibility,
                parameters=method_params,
                code=method_code,
                implementation_type=implementation_type,
            )

            if state_machine_id or quantum_circuit_id:
                method_diagram_refs[(class_name, name)] = {
                    "stateMachineId": state_machine_id or "",
                    "quantumCircuitId": quantum_circuit_id or "",
                }

            if return_type:
                method_obj.type = _resolve_type(return_type, type_lookup)

            cls.add_method(method_obj)

            method_id = method.get("id")
            if method_id:
                method_id_to_method[method_id] = method_obj

    return class_id_to_class, method_id_to_method


def _process_relationships(
    nodes: list[dict],
    edges: list[dict],
    domain_model: DomainModel,
    layout_positions: dict[str, Any],
    comment_elements: dict[str, str],
    comment_links: dict[str, list[str]],
    all_warnings: list[str],
) -> tuple[dict[str, set[str]], dict[str, BinaryAssociation]]:
    """Process associations, generalizations, and link edges."""
    association_class_candidates: dict[str, set[str]] = {}
    association_by_id: dict[str, BinaryAssociation] = {}

    # Index nodes by id once for O(1) lookups.
    nodes_by_id = {n.get("id"): n for n in nodes if n.get("id")}
    edge_ids: set[str] = {e.get("id") for e in edges if e.get("id")}

    for edge in edges:
        rel_type = edge.get("type")
        source_id = edge.get("source")
        target_id = edge.get("target")
        rel_id = edge.get("id")
        edge_data = edge.get("data") or {}

        if not rel_type or not source_id or not target_id:
            logger.warning("Skipping relationship %s due to missing data.", rel_id)
            all_warnings.append(f"Skipped relationship '{rel_id}': missing type, source, or target.")
            continue

        # OCL link is visual-only; constraint data lives on the class node.
        if rel_type == "ClassOCLLink":
            continue

        if rel_type == "Link":
            comment_id = None
            target = None
            if source_id in comment_elements:
                comment_id = source_id
                target = target_id
            elif target_id in comment_elements:
                comment_id = target_id
                target = source_id
            if comment_id and target:
                comment_links.setdefault(comment_id, []).append(target)
            continue

        if rel_type == "ClassLinkRel":
            # ClassLinkRel connects a class node to an association edge.
            if source_id in nodes_by_id and target_id in edge_ids:
                association_class_candidates.setdefault(source_id, set()).add(target_id)
            elif target_id in nodes_by_id and source_id in edge_ids:
                association_class_candidates.setdefault(target_id, set()).add(source_id)
            continue

        source_node = nodes_by_id.get(source_id)
        target_node = nodes_by_id.get(target_id)
        if not source_node or not target_node:
            logger.warning("Skipping relationship %s due to missing nodes.", rel_id)
            all_warnings.append(f"Skipped relationship '{rel_id}': source or target node not found.")
            continue

        source_class = domain_model.get_class_by_name(node_data(source_node).get("name", ""))
        target_class = domain_model.get_class_by_name(node_data(target_node).get("name", ""))
        if not source_class or not target_class:
            logger.warning("Skipping relationship %s: classes not found in domain model.", rel_id)
            all_warnings.append(f"Skipped relationship '{rel_id}': source or target class not in domain model.")
            continue

        if rel_type in ("ClassBidirectional", "ClassUnidirectional", "ClassComposition", "ClassAggregation"):
            is_composite = rel_type == "ClassComposition"
            source_navigable = rel_type != "ClassUnidirectional"
            target_navigable = True

            source_multiplicity = parse_multiplicity(edge_data.get("sourceMultiplicity", "1"))
            target_multiplicity = parse_multiplicity(edge_data.get("targetMultiplicity", "1"))

            source_role = edge_data.get("sourceRole")
            if not source_role:
                source_role = source_class.name.lower()
                existing_roles = {end.name for assoc in domain_model.associations for end in assoc.ends}
                if source_role in existing_roles:
                    counter = 1
                    while f"{source_role}_{counter}" in existing_roles:
                        counter += 1
                    source_role = f"{source_role}_{counter}"

            source_property = Property(
                name=source_role,
                type=source_class,
                multiplicity=source_multiplicity,
                is_navigable=source_navigable,
            )

            target_role = edge_data.get("targetRole")
            if not target_role:
                target_role = target_class.name.lower()
                existing_roles = {end.name for assoc in domain_model.associations for end in assoc.ends}
                if target_role in existing_roles:
                    counter = 1
                    while f"{target_role}_{counter}" in existing_roles:
                        counter += 1
                    target_role = f"{target_role}_{counter}"

            target_property = Property(
                name=target_role,
                type=target_class,
                multiplicity=target_multiplicity,
                is_navigable=target_navigable,
                is_composite=is_composite,
            )

            association_name = edge_data.get("name") or f"{source_class.name}_{target_class.name}"
            existing_assoc_names = {assoc.name for assoc in domain_model.associations}
            if association_name in existing_assoc_names:
                counter = 1
                while f"{association_name}_{counter}" in existing_assoc_names:
                    counter += 1
                association_name = f"{association_name}_{counter}"

            association = BinaryAssociation(
                name=association_name,
                ends={source_property, target_property},
            )
            domain_model.associations.add(association)
            association_by_id[rel_id] = association

            rel_layout: dict[str, Any] = {}
            if edge_data.get("points"):
                rel_layout["path"] = edge_data["points"]
            if edge_data.get("isManuallyLayouted") is not None:
                rel_layout["isManuallyLayouted"] = edge_data["isManuallyLayouted"]
            if edge.get("sourceHandle"):
                rel_layout["source_direction"] = edge["sourceHandle"]
            if edge.get("targetHandle"):
                rel_layout["target_direction"] = edge["targetHandle"]
            if rel_layout:
                layout_positions[f"rel_{association_name}"] = rel_layout

        elif rel_type == "ClassInheritance":
            generalization = Generalization(general=target_class, specific=source_class)
            domain_model.generalizations.add(generalization)
            gen_layout: dict[str, Any] = {}
            if edge_data.get("points"):
                gen_layout["path"] = edge_data["points"]
            if gen_layout:
                layout_positions[f"gen_{source_class.name}_{target_class.name}"] = gen_layout

    return association_class_candidates, association_by_id


def _process_association_classes(
    association_class_candidates: dict[str, set[str]],
    association_by_id: dict[str, BinaryAssociation],
    nodes: list[dict],
    domain_model: DomainModel,
    all_warnings: list[str],
) -> None:
    """Promote regular classes to association classes where linked."""
    nodes_by_id = {n.get("id"): n for n in nodes if n.get("id")}
    for class_id, association_ids in association_class_candidates.items():
        node = nodes_by_id.get(class_id)
        if not node:
            continue
        class_name = node_data(node).get("name", "")
        class_obj = domain_model.get_class_by_name(class_name)
        if not class_obj:
            continue

        if len(association_ids) > 1:
            msg = f"Class '{class_name}' is linked to multiple associations. Only using the first one."
            logger.warning(msg)
            all_warnings.append(msg)

        association_id = next(iter(association_ids))
        association = association_by_id.get(association_id)
        if not association:
            continue

        attributes = set(class_obj.attributes)
        methods = set(class_obj.methods)

        association_class = AssociationClass(
            name=class_name,
            attributes=attributes,
            association=association,
        )
        if methods:
            association_class.methods = methods

        domain_model.types.discard(class_obj)
        domain_model.types.add(association_class)


def _ocl_box_to_full_text(
    ocl_row: dict[str, Any],
    owner_class: Optional[Class],
    method_id_to_method: dict[str, Method],
    warnings: list[str],
) -> Optional[str]:
    """Coerce an OCL constraint row to its canonical full-text form.

    Per the v4 spec, OCL constraints collapse onto the owner class's
    ``data.oclConstraints`` array; each row carries an ``expression``
    field with the canonical full text. Legacy body-only fields are
    accepted for backward compatibility with hand-authored fixtures.
    """
    raw = ocl_row.get("expression") or ocl_row.get("constraint")
    if not raw:
        return None

    if raw.lstrip().lower().startswith("context"):
        return raw

    legacy_kind = ocl_row.get("kind")
    if not legacy_kind:
        warnings.append(
            f"Warning: OCL constraint {ocl_row.get('id')!r} has no recognisable header "
            f"and no legacy 'kind' field; skipping."
        )
        return None
    if legacy_kind not in ("invariant", "precondition", "postcondition"):
        warnings.append(
            f"Warning: OCL constraint {ocl_row.get('id')!r} has unknown kind {legacy_kind!r}; skipping."
        )
        return None
    if owner_class is None:
        warnings.append(
            f"Warning: legacy body-only OCL constraint {ocl_row.get('id')!r} has no owner class; skipping."
        )
        return None

    method = None
    if legacy_kind in ("precondition", "postcondition"):
        target_method_id = ocl_row.get("targetMethodId")
        if not target_method_id:
            warnings.append(
                f"Warning: legacy {legacy_kind} constraint {ocl_row.get('id')!r} has no targetMethodId; skipping."
            )
            return None
        method = method_id_to_method.get(target_method_id)
        if method is None:
            warnings.append(
                f"Warning: legacy {legacy_kind} constraint {ocl_row.get('id')!r} targets missing method "
                f"{target_method_id}; skipping."
            )
            return None

    try:
        return legacy_body_only_to_text(
            body=raw,
            kind=legacy_kind,
            name=ocl_row.get("constraintName") or ocl_row.get("name"),
            context_class=owner_class,
            method=method,
        )
    except ValueError as e:
        warnings.append(f"Warning: legacy {legacy_kind} constraint {ocl_row.get('id')!r}: {e}")
        return None


def _process_constraints(
    nodes: list[dict],
    domain_model: DomainModel,
    all_warnings: list[str],
    class_id_to_class: dict[str, Class],
    method_id_to_method: dict[str, Method],
) -> None:
    """Walk every class node's ``data.oclConstraints`` and any free-standing
    ``ClassOCLConstraint`` nodes (rare fallback per the v4 spec)."""
    method_by_qualified_name: dict[tuple[str, str], Method] = {}
    duplicates: set[tuple[str, str]] = set()

    def _walk_inherited_methods(start: Class):
        seen: set[Class] = set()
        frontier: list[Class] = [start]
        while frontier:
            next_frontier: list[Class] = []
            for cls in frontier:
                if cls in seen:
                    continue
                seen.add(cls)
                for m in cls.methods:
                    yield m
                next_frontier.extend(cls.parents())
            frontier = next_frontier

    for cls in domain_model.types:
        if not isinstance(cls, Class):
            continue
        for m in _walk_inherited_methods(cls):
            key = (cls.name, m.name)
            if key in method_by_qualified_name:
                if method_by_qualified_name[key] is not m:
                    duplicates.add(key)
                continue
            method_by_qualified_name[key] = m
    for cls_name, m_name in duplicates:
        all_warnings.append(
            f"Warning: class {cls_name!r} has multiple methods named "
            f"{m_name!r} (across inheritance); pre/post lookup may be "
            f"ambiguous. Closest definition wins."
        )

    extra_invariants: list = []
    counter = 0

    # Collected OCL rows: (row_dict, owner_class_or_None).
    ocl_rows: list[tuple[dict, Optional[Class]]] = []
    for node in nodes:
        if node.get("type") != "class":
            continue
        kind = _class_v4_kind(node)
        data = node_data(node)
        node_id = node.get("id")
        owner_class = class_id_to_class.get(node_id)
        if kind == "ClassOCLConstraint":
            # Free-standing fallback: synthesise a row dict from the node's
            # data so the same parsing path applies.
            row = dict(data)
            row.setdefault("id", node_id)
            ocl_rows.append((row, owner_class))
            continue
        # Class / Abstract / Interface / Enumeration: walk their oclConstraints.
        for row in data.get("oclConstraints") or []:
            ocl_rows.append((row, owner_class))

    for row, owner_class in ocl_rows:
        text = _ocl_box_to_full_text(row, owner_class, method_id_to_method, all_warnings)
        if not text:
            continue
        description = row.get("description")
        counter += 1
        try:
            routing, warnings = process_ocl_constraints(
                text, domain_model, counter, default_description=description,
            )
        except (BOCLSyntaxError, ValueError) as e:
            all_warnings.append(f"Warning: Error processing OCL element {row.get('id')!r}: {e}")
            continue
        all_warnings.extend(warnings)

        for kind, constraint, class_name, method_name in routing:
            try:
                if kind == "invariant":
                    extra_invariants.append(constraint)
                else:
                    method = method_by_qualified_name.get((class_name, method_name)) if method_name else None
                    if method is None:
                        all_warnings.append(
                            f"Warning: {kind} '{constraint.name}' targets unknown method "
                            f"{class_name}::{method_name}; skipping."
                        )
                        continue
                    if kind == "precondition":
                        method.add_pre(constraint)
                    else:
                        method.add_post(constraint)
            except ValueError as e:
                all_warnings.append(f"Warning: Could not attach {kind} '{constraint.name}': {e}")

    by_name: dict[str, Any] = {c.name: c for c in domain_model.constraints}
    for c in extra_invariants:
        if c.name in by_name and by_name[c.name] is not c:
            all_warnings.append(
                f"Warning: duplicate constraint name {c.name!r} across OCL boxes; "
                f"keeping the first occurrence."
            )
            continue
        by_name[c.name] = c
    domain_model.ocl_warnings = all_warnings
    domain_model.constraints = set(by_name.values())


def process_class_diagram(json_data: dict[str, Any]) -> DomainModel:
    """Process a Class Diagram in the v4 wire shape.

    Accepts only the v4 ``{nodes, edges}`` shape. v3-shape input is not
    supported on the backend; the editor migrator at
    ``packages/library/lib/utils/versionConverter.ts`` is the only place
    that lifts v3 to v4 in the system. See
    ``docs/source/migrations/uml-v4-shape.md``.
    """
    title = json_data.get('title', '')
    if ' ' in title:
        title = title.replace(' ', '_')

    domain_model = DomainModel(title)
    model_payload = json_data.get('model') or {}
    nodes = model_payload.get('nodes') or []
    edges = model_payload.get('edges') or []
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []

    layout_positions: dict[str, Any] = {}
    comment_elements: dict[str, str] = {}
    comment_links: dict[str, list[str]] = {}
    all_warnings: list[str] = []
    method_diagram_refs: dict[tuple[str, str], dict[str, str]] = {}

    _process_enumerations(nodes, domain_model, layout_positions, comment_elements)

    type_lookup = _build_type_lookup(domain_model)

    class_id_to_class, method_id_to_method = _process_classes(
        nodes, domain_model, type_lookup, layout_positions, method_diagram_refs,
    )

    type_lookup = _build_type_lookup(domain_model)

    association_class_candidates, association_by_id = _process_relationships(
        nodes, edges, domain_model, layout_positions,
        comment_elements, comment_links, all_warnings,
    )

    _process_association_classes(
        association_class_candidates, association_by_id,
        nodes, domain_model, all_warnings,
    )

    _process_constraints(
        nodes, domain_model, all_warnings,
        class_id_to_class, method_id_to_method,
    )

    # Apply collected comments.
    nodes_by_id = {n.get("id"): n for n in nodes if n.get("id")}
    for comment_id, comment_text in comment_elements.items():
        if comment_id in comment_links:
            for linked_node_id in comment_links[comment_id]:
                linked_node = nodes_by_id.get(linked_node_id)
                if not linked_node:
                    continue
                element_name = node_data(linked_node).get("name", "").strip()
                type_obj = type_lookup.get(element_name)
                if isinstance(type_obj, Class):
                    if not type_obj.metadata:
                        type_obj.metadata = Metadata(description=comment_text)
                    else:
                        if type_obj.metadata.description:
                            type_obj.metadata.description += f"\n{comment_text}"
                        else:
                            type_obj.metadata.description = comment_text
        else:
            if not domain_model.metadata:
                domain_model.metadata = Metadata(description=comment_text)
            else:
                if domain_model.metadata.description:
                    domain_model.metadata.description += f"\n{comment_text}"
                else:
                    domain_model.metadata.description = comment_text

    domain_model.association_by_id = association_by_id
    domain_model.method_diagram_refs = method_diagram_refs
    domain_model._layout_positions = layout_positions

    return domain_model
