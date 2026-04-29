"""
Class diagram processing for converting JSON to BUML format.
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
    parse_attribute, parse_method, parse_multiplicity, process_ocl_constraints
)


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
    """Build a name-to-type lookup dict for O(1) type resolution.

    Only includes Class and Enumeration types (the user-defined types that
    can appear as attribute types, parameter types, or return types).
    """
    return {
        t.name: t
        for t in domain_model.types
        if isinstance(t, (Enumeration, Class))
    }


def _resolve_type(type_name: str, type_lookup: dict[str, Union[Class, Enumeration]]) -> Type:
    """Resolve a type name to a metamodel type object.

    Returns the Class/Enumeration from *type_lookup* if found, otherwise
    returns a new ``PrimitiveDataType``.  Raises ``ConversionError`` with
    a descriptive message if the name is neither a known user type nor a
    valid primitive.
    """
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


def _process_enumerations(
    elements: dict[str, Any],
    domain_model: DomainModel,
    layout_positions: dict[str, Any],
    comment_elements: dict[str, str],
) -> None:
    """Process enumeration elements and collect comments from the element map.

    Iterates over *elements*, creates ``Enumeration`` instances for each
    enumeration element, adds them to *domain_model*, and records their
    layout bounds in *layout_positions*.  Comment elements encountered
    during iteration are stored in *comment_elements* for later processing.
    """
    for element_id, element in elements.items():
        # Collect comments
        if element.get("type") == "Comments":
            comment_text = element.get("name", "").strip()
            comment_elements[element_id] = comment_text
            continue

        if element.get("type") == "Enumeration":
            element_name = element.get("name", "").strip()
            if not element_name or any(char.isspace() for char in element_name):
                raise ConversionError(
                    f"Invalid enumeration name: '{element_name}'. Names cannot contain whitespace or be empty."
                )

            literals = []
            seen_literal_names = set()
            for literal_id in element.get("attributes", []):
                literal = elements.get(literal_id)
                if literal:
                    literal_name = literal.get("name", "").strip()
                    if not literal_name:
                        raise ConversionError(
                            f"Empty enumeration literal name in '{element_name}'."
                        )
                    if literal_name in seen_literal_names:
                        raise ConversionError(
                            f"Duplicate enumeration literal '{literal_name}' in '{element_name}'."
                        )
                    seen_literal_names.add(literal_name)
                    literal_obj = EnumerationLiteral(name=literal_name)
                    literals.append(literal_obj)
            # Store the ordered list for later use in buml_to_json conversion,
            # but pass as set to Enumeration (which requires set internally).
            enum = Enumeration(name=element_name, literals=set(literals))
            # Preserve insertion order from the JSON for round-trip fidelity.
            enum._ordered_literals = list(literals)
            # Use add_type() which triggers validation through the setter
            try:
                domain_model.add_type(enum)
            except ValueError as e:
                raise ConversionError(str(e))

            # Capture layout bounds for round-trip fidelity
            if "bounds" in element:
                layout_positions[element_name] = element["bounds"]


def _process_classes(
    elements: dict[str, Any],
    domain_model: DomainModel,
    type_lookup: dict[str, Union[Class, Enumeration]],
    layout_positions: dict[str, Any],
    method_diagram_refs: dict[tuple[str, str], dict[str, str]],
) -> None:
    """Create classes with their attributes and methods.

    First creates all class shells (without attributes/methods) so that
    cross-references between classes resolve correctly, then populates
    each class with its attributes and methods in a second pass.
    """
    # First create all class structures without attributes or methods
    for element_id, element in elements.items():
        if element.get("type") in ["Class", "AbstractClass"]:
            class_name = element.get("name", "").strip()
            if not class_name or any(char.isspace() for char in class_name):
                raise ConversionError(
                    f"Invalid class name: '{class_name}'. Names cannot contain whitespace or be empty."
                )

            is_abstract = element.get("type") == "AbstractClass"
              # Handle metadata with description and URI
            metadata = None
            description = element.get("description")
            uri = element.get("uri")
            icon = element.get("icon")

            if description or uri or icon:
                metadata = Metadata(description=description, uri=uri, icon=icon)
            try:
                cls = Class(name=class_name, is_abstract=is_abstract, metadata=metadata)
                # Use add_type() which triggers validation through the setter
                domain_model.add_type(cls)
            except ValueError as e:
                raise ConversionError(str(e))

            # Capture layout bounds for round-trip fidelity
            if "bounds" in element:
                layout_positions[class_name] = element["bounds"]

    # Rebuild the type lookup now that all class shells have been added,
    # so that cross-class references in attributes/methods resolve correctly.
    type_lookup = _build_type_lookup(domain_model)

    # Now add attributes and methods to classes
    for element_id, element in elements.items():
        if element.get("type") in ["Class", "AbstractClass"]:
            class_name = element.get("name", "").strip()
            cls = domain_model.get_class_by_name(class_name)

            if not cls:
                continue  # Skip if class wasn't created successfully in first pass

            # Add attributes
            attribute_names = set()
            for attr_id in element.get("attributes", []):
                attr = elements.get(attr_id)
                if attr:
                    # Check for new format (separate visibility and attributeType properties)
                    if "visibility" in attr and "attributeType" in attr:
                        # New format - use separate properties
                        visibility = attr.get("visibility", "public")
                        name = attr.get("name", "").strip()
                        attr_type = attr.get("attributeType", "str")
                        is_optional = attr.get("isOptional", False)
                        is_id = attr.get("isId", False)
                        is_external_id = attr.get("isExternalId", False)
                        is_derived = attr.get("isDerived", False)
                        default_value = attr.get("defaultValue", None)
                    else:
                        # Legacy format - parse from name string
                        visibility, name, attr_type = parse_attribute(attr.get("name", ""), domain_model, type_lookup=type_lookup)
                        is_optional = False
                        is_id = False
                        is_external_id = False
                        is_derived = False
                        default_value = None

                    if not name:  # Skip if no name was returned
                        continue
                    if name in attribute_names:
                        raise ConversionError(f"Duplicate attribute name '{name}' found in class '{class_name}'")
                    attribute_names.add(name)

                    # Resolve the attribute type via O(1) lookup
                    type_obj = _resolve_type(attr_type, type_lookup)
                    property_ = Property(name=name, type=type_obj, visibility=visibility, is_optional=is_optional, is_id=is_id, is_external_id=is_external_id, is_derived=is_derived, default_value=default_value)
                    cls.add_attribute(property_)

            # Add methods
            for method_id in element.get("methods", []):
                method = elements.get(method_id)
                if method:
                    visibility, name, parameters, return_type = parse_method(method.get("name", ""), domain_model, type_lookup=type_lookup)

                    # Get the code attribute for the method
                    method_code = method.get("code", "")

                    method_name_is_malformed = (
                        isinstance(name, str)
                        and name.count("(") != name.count(")")
                    )
                    if method_name_is_malformed or (not parameters and method_code):
                        parsed_from_code = parse_method_signature_from_code(method_code, domain_model, type_lookup=type_lookup)
                        if parsed_from_code:
                            parsed_name, parsed_parameters, parsed_return_type = parsed_from_code
                            if method_name_is_malformed and parsed_name:
                                name = parsed_name
                            if not parameters and parsed_parameters:
                                parameters = parsed_parameters
                            if not return_type and parsed_return_type:
                                return_type = parsed_return_type

                    # Get implementation type and diagram references
                    impl_type_str = method.get("implementationType", "none")
                    if isinstance(impl_type_str, str):
                        impl_type_str = impl_type_str.strip().lower()
                    state_machine_id = method.get("stateMachineId", "")
                    quantum_circuit_id = method.get("quantumCircuitId", "")

                    # Map string to MethodImplementationType enum
                    impl_type_map = {
                        "none": MethodImplementationType.NONE,
                        "code": MethodImplementationType.CODE,
                        "bal": MethodImplementationType.BAL,
                        "state_machine": MethodImplementationType.STATE_MACHINE,
                        "quantum_circuit": MethodImplementationType.QUANTUM_CIRCUIT,
                    }
                    implementation_type = impl_type_map.get(impl_type_str, MethodImplementationType.NONE)

                    # Auto-detect implementation type if not set but code exists
                    if implementation_type == MethodImplementationType.NONE and method_code:
                        implementation_type = MethodImplementationType.CODE

                    # Create method parameters
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

                        # Resolve parameter type via O(1) lookup
                        param_type_obj = _resolve_type(param_type_name, type_lookup)

                        param_obj = Parameter(
                            name=param_name,
                            type=param_type_obj
                        )
                        if 'default' in param:
                            param_obj.default_value = param['default']
                        method_params.append(param_obj)

                    # Create method with parameters, return type, code, and implementation type
                    method_obj = Method(
                        name=name,
                        visibility=visibility,
                        parameters=method_params,
                        code=method_code,
                        implementation_type=implementation_type
                    )

                    # Store diagram references in a separate mapping for later resolution.
                    # These will be used by project-level processing to link to actual diagrams.
                    if state_machine_id or quantum_circuit_id:
                        method_diagram_refs[(class_name, name)] = {
                            "stateMachineId": state_machine_id or "",
                            "quantumCircuitId": quantum_circuit_id or "",
                        }

                    # Handle return type via O(1) lookup
                    if return_type:
                        method_obj.type = _resolve_type(return_type, type_lookup)

                    cls.add_method(method_obj)


def _process_relationships(
    relationships: dict[str, Any],
    elements: dict[str, Any],
    domain_model: DomainModel,
    layout_positions: dict[str, Any],
    comment_elements: dict[str, str],
    comment_links: dict[str, list[str]],
    all_warnings: list[str],
) -> tuple[dict[str, set[str]], dict[str, BinaryAssociation]]:
    """Process associations, generalizations, and link relationships.

    Returns a tuple of (*association_class_candidates*, *association_by_id*)
    for downstream association-class processing.
    """
    # Store association classes candidates and their links for third pass processing
    association_class_candidates: dict[str, set[str]] = {}
    association_by_id: dict[str, BinaryAssociation] = {}

    for rel_id, relationship in relationships.items():
        rel_type = relationship.get("type")
        source = relationship.get("source")
        target = relationship.get("target")

        if not rel_type or not source or not target:
            logger.warning("Skipping relationship %s due to missing data.", rel_id)
            all_warnings.append(f"Skipped relationship '{rel_id}': missing type, source, or target.")
            continue

        # Skip OCL links -- these are visual-only relationships that connect a
        # ClassOCLConstraint element to the class it constrains.  The constraint
        # data itself is stored in the ClassOCLConstraint element and processed
        # separately below; the context class is derived from the OCL expression.
        if rel_type == "ClassOCLLink":
            continue

        # Handle Link (comment links)
        if rel_type == "Link":
            source_element_id = source.get("element")
            target_element_id = target.get("element")

            # Determine which is the comment and which is the target
            comment_id = None
            target_id = None

            if source_element_id in comment_elements:
                comment_id = source_element_id
                target_id = target_element_id
            elif target_element_id in comment_elements:
                comment_id = target_element_id
                target_id = source_element_id

            if comment_id and target_id:
                if comment_id not in comment_links:
                    comment_links[comment_id] = []
                comment_links[comment_id].append(target_id)

            continue

        # Handle ClassLinkRel (association class links) later
        if rel_type == "ClassLinkRel":
            source_element_id = source.get("element")
            target_element_id = target.get("element")

            # Check if source is a class and target is a relationship
            if source_element_id in elements and target_element_id in relationships:
                # Source is a class, target is an association
                if source_element_id not in association_class_candidates:
                    association_class_candidates[source_element_id] = set()
                association_class_candidates[source_element_id].add(target_element_id)

            # Check if target is a class and source is a relationship
            elif target_element_id in elements and source_element_id in relationships:
                # Target is a class, source is an association
                if target_element_id not in association_class_candidates:
                    association_class_candidates[target_element_id] = set()
                association_class_candidates[target_element_id].add(source_element_id)

            continue

        # Retrieve source and target elements
        source_element = elements.get(source.get("element"))
        target_element = elements.get(target.get("element"))

        if not source_element or not target_element:
            logger.warning("Skipping relationship %s due to missing elements.", rel_id)
            all_warnings.append(f"Skipped relationship '{rel_id}': source or target element not found.")
            continue

        source_class = domain_model.get_class_by_name(source_element.get("name", ""))
        target_class = domain_model.get_class_by_name(target_element.get("name", ""))

        if not source_class or not target_class:
            logger.warning("Skipping relationship %s: classes not found in domain model.", rel_id)
            all_warnings.append(f"Skipped relationship '{rel_id}': source or target class not in domain model.")
            continue

        # Handle each type of relationship
        if rel_type == "ClassBidirectional" or rel_type == "ClassUnidirectional" or rel_type == "ClassComposition" or rel_type == "ClassAggregation" :
            is_composite = rel_type == "ClassComposition"
            source_navigable = rel_type != "ClassUnidirectional"
            target_navigable = True

            source_multiplicity = parse_multiplicity(source.get("multiplicity", "1"))
            target_multiplicity = parse_multiplicity(target.get("multiplicity", "1"))

            source_role = source.get("role")
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
                is_navigable=source_navigable
            )

            target_role = target.get("role")
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
                is_composite=is_composite
            )

            association_name = relationship.get("name") or f"{source_class.name}_{target_class.name}"

            # Check if association name already exists and add increment if needed
            existing_assoc_names = {assoc.name for assoc in domain_model.associations}
            if association_name in existing_assoc_names:
                counter = 1
                while f"{association_name}_{counter}" in existing_assoc_names:
                    counter += 1
                association_name = f"{association_name}_{counter}"

            association = BinaryAssociation(
                name=association_name,
                ends={source_property, target_property}
            )
            domain_model.associations.add(association)

            # Store the association for association class processing
            association_by_id[rel_id] = association

            # Capture layout bounds for the relationship and its endpoints
            rel_layout: dict[str, Any] = {}
            if "bounds" in relationship:
                rel_layout["bounds"] = relationship["bounds"]
            if "path" in relationship:
                rel_layout["path"] = relationship["path"]
            if "isManuallyLayouted" in relationship:
                rel_layout["isManuallyLayouted"] = relationship["isManuallyLayouted"]
            # Capture source/target endpoint bounds and directions
            if source.get("bounds"):
                rel_layout["source_bounds"] = source["bounds"]
            if source.get("direction"):
                rel_layout["source_direction"] = source["direction"]
            if target.get("bounds"):
                rel_layout["target_bounds"] = target["bounds"]
            if target.get("direction"):
                rel_layout["target_direction"] = target["direction"]
            if rel_layout:
                layout_positions[f"rel_{association_name}"] = rel_layout

        elif rel_type == "ClassInheritance":
            generalization = Generalization(general=target_class, specific=source_class)
            domain_model.generalizations.add(generalization)

            # Capture layout bounds for the generalization
            gen_layout: dict[str, Any] = {}
            if "bounds" in relationship:
                gen_layout["bounds"] = relationship["bounds"]
            if "path" in relationship:
                gen_layout["path"] = relationship["path"]
            if source.get("bounds"):
                gen_layout["source_bounds"] = source["bounds"]
            if target.get("bounds"):
                gen_layout["target_bounds"] = target["bounds"]
            if gen_layout:
                layout_positions[f"gen_{source_class.name}_{target_class.name}"] = gen_layout

    return association_class_candidates, association_by_id


def _process_association_classes(
    association_class_candidates: dict[str, set[str]],
    association_by_id: dict[str, BinaryAssociation],
    elements: dict[str, Any],
    domain_model: DomainModel,
    all_warnings: list[str],
) -> None:
    """Promote regular classes to association classes where linked.

    For each class that was linked to an association via a ``ClassLinkRel``,
    create an ``AssociationClass`` that wraps the original class's attributes
    and methods together with the association, then replace the plain class
    in *domain_model*.
    """
    for class_id, association_ids in association_class_candidates.items():
        class_element = elements.get(class_id)
        if not class_element:
            continue

        class_name = class_element.get("name", "")
        class_obj = domain_model.get_class_by_name(class_name)

        if not class_obj:
            continue

        # An association class should only be linked to one association
        if len(association_ids) > 1:
            msg = f"Class '{class_name}' is linked to multiple associations. Only using the first one."
            logger.warning(msg)
            all_warnings.append(msg)

        # Get the first association
        association_id = next(iter(association_ids))
        association = association_by_id.get(association_id)

        if not association:
            continue

        # Get attributes and methods from the original class.
        # Copy the sets — the Class.attributes setter re-parents each member by
        # discarding it from its previous owner's attribute set. If we passed
        # ``class_obj.attributes`` directly that owner *is* class_obj, so the
        # discard would mutate the same set being iterated and raise
        # "set changed size during iteration".
        attributes = set(class_obj.attributes)
        methods = set(class_obj.methods)

        # Create the association class with attributes and methods
        association_class = AssociationClass(
            name=class_name,
            attributes=attributes,
            association=association
        )

        # Add methods to the association class if they exist
        if methods:
            association_class.methods = methods

        # Update the domain model - remove the regular class and add the association class
        domain_model.types.discard(class_obj)
        domain_model.types.add(association_class)


def _process_constraints(
    elements: dict[str, Any],
    domain_model: DomainModel,
    all_warnings: list[str],
) -> None:
    """Process OCL constraint elements and attach them to the domain model."""
    all_constraints = set()
    constraint_counter = 0
    for element_id, element in elements.items():
        if element.get("type") in ["ClassOCLConstraint"]:
            ocl = element.get("constraint")
            if ocl:
                try:
                    new_constraints, warnings = process_ocl_constraints(ocl, domain_model, constraint_counter)
                    all_constraints.update(new_constraints)
                    all_warnings.extend(warnings)
                    constraint_counter += 1
                except Exception as e:
                    error_msg = f"Error processing OCL constraint for element {element_id}: {e}"
                    all_warnings.append(error_msg)
                    continue    # Attach warnings to domain model for later use
    domain_model.ocl_warnings = all_warnings
    domain_model.constraints = all_constraints


def process_class_diagram(json_data: dict[str, Any]) -> DomainModel:
    """Process Class Diagram specific elements."""
    title = json_data.get('title', '')
    if ' ' in title:
        title = title.replace(' ', '_')

    domain_model = DomainModel(title)
    # Get elements and OCL constraints from the JSON data
    elements = (json_data.get('model') or {}).get('elements', {})
    relationships = (json_data.get('model') or {}).get('relationships', {})

    # Collect layout positions from the original JSON for round-trip fidelity.
    # Keyed by element name (for classes/enums) or composite key (for relationships).
    layout_positions: dict[str, Any] = {}

    # Store comments for later processing
    comment_elements: dict[str, str] = {}  # {comment_id: comment_text}
    comment_links: dict[str, list[str]] = {}  # {comment_id: [linked_element_ids]}
    all_warnings: list[str] = []  # Collect non-fatal warnings for the caller
    # Track diagram references for methods (state machine / quantum circuit IDs).
    # Keyed by (class_name, method_name) to avoid collisions across classes.
    method_diagram_refs: dict[tuple[str, str], dict[str, str]] = {}

    # FIRST PASS: Process all type declarations (enumerations and classes)
    _process_enumerations(elements, domain_model, layout_positions, comment_elements)

    # Build a name->type lookup dict for O(1) type resolution in the second pass.
    # Must happen after enumerations are registered but before classes need
    # cross-referencing (the lookup is rebuilt after class shells are added).
    type_lookup = _build_type_lookup(domain_model)

    _process_classes(elements, domain_model, type_lookup, layout_positions, method_diagram_refs)

    # Rebuild lookup after classes have been added so that subsequent helpers
    # (e.g., comment processing) can resolve class names.
    type_lookup = _build_type_lookup(domain_model)

    # Process relationships (Associations, Generalizations, and Compositions)
    association_class_candidates, association_by_id = _process_relationships(
        relationships, elements, domain_model, layout_positions,
        comment_elements, comment_links, all_warnings,
    )

    # THIRD PASS: Process association classes
    _process_association_classes(
        association_class_candidates, association_by_id,
        elements, domain_model, all_warnings,
    )

    # Process OCL constraints
    _process_constraints(elements, domain_model, all_warnings)

    # Process comments and apply them to class or domain model metadata
    for comment_id, comment_text in comment_elements.items():
        if comment_id in comment_links:
            # Comment is linked to specific elements
            for linked_element_id in comment_links[comment_id]:
                linked_element = elements.get(linked_element_id)
                if linked_element:
                    element_name = linked_element.get("name", "").strip()
                    # Find the class in the domain model via O(1) lookup
                    type_obj = type_lookup.get(element_name)
                    if isinstance(type_obj, Class):
                        # Add comment to class metadata
                        if not type_obj.metadata:
                            type_obj.metadata = Metadata(description=comment_text)
                        else:
                            # Append to existing description
                            if type_obj.metadata.description:
                                type_obj.metadata.description += f"\n{comment_text}"
                            else:
                                type_obj.metadata.description = comment_text
        else:
            # Comment is not linked, add to domain model metadata
            if not domain_model.metadata:
                domain_model.metadata = Metadata(description=comment_text)
            else:
                # Append to existing description
                if domain_model.metadata.description:
                    domain_model.metadata.description += f"\n{comment_text}"
                else:
                    domain_model.metadata.description = comment_text

    # Store the association_by_id mapping for object diagram processing
    domain_model.association_by_id = association_by_id

    # Store method diagram references for buml_to_json round-trip fidelity.
    # Keyed by (class_name, method_name) -> {"stateMachineId": ..., "quantumCircuitId": ...}
    domain_model.method_diagram_refs = method_diagram_refs

    # Store layout positions for buml_to_json round-trip fidelity.
    # Keyed by element name (classes/enums) or composite key (relationships).
    domain_model._layout_positions = layout_positions

    return domain_model
