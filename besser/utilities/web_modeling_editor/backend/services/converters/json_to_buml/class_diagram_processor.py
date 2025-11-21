"""
Class diagram processing for converting JSON to BUML format.
"""

import json
from fastapi import HTTPException

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Enumeration, Property, Method, BinaryAssociation,
    Generalization, PrimitiveDataType, EnumerationLiteral, AssociationClass, Metadata
)
from besser.utilities.web_modeling_editor.backend.services.converters.parsers import (
    parse_attribute, parse_method, parse_multiplicity, process_ocl_constraints
)


def process_class_diagram(json_data):
    """Process Class Diagram specific elements."""
    title = json_data.get('title', '')
    if ' ' in title:
        title = title.replace(' ', '_')

    domain_model = DomainModel(title)
    # Get elements and OCL constraints from the JSON data
    elements = json_data.get('model', {}).get('elements', {})
    relationships = json_data.get('model', {}).get('relationships', {})
    
    # Store comments for later processing
    comment_elements = {}  # {comment_id: comment_text}
    comment_links = {}  # {comment_id: [linked_element_ids]}

    # FIRST PASS: Process all type declarations (enumerations and classes)
    # 1. First process enumerations
    for element_id, element in elements.items():
        # Collect comments
        if element.get("type") == "Comments":
            comment_text = element.get("name", "").strip()
            comment_elements[element_id] = comment_text
            continue
            
        if element.get("type") == "Enumeration":
            element_name = element.get("name", "").strip()
            if not element_name or any(char.isspace() for char in element_name):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid enumeration name: '{element_name}'. Names cannot contain whitespace or be empty."
                )
            
            literals = set()
            for literal_id in element.get("attributes", []):
                literal = elements.get(literal_id)
                if literal:
                    literal_obj = EnumerationLiteral(name=literal.get("name", ""))
                    literals.add(literal_obj)
            enum = Enumeration(name=element_name, literals=literals)
            # Use add_type() which triggers validation through the setter
            try:
                domain_model.add_type(enum)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
    
    # 2. Then create all class structures without attributes or methods
    for element_id, element in elements.items():
        if element.get("type") in ["Class", "AbstractClass"]:
            class_name = element.get("name", "").strip()
            if not class_name or any(char.isspace() for char in class_name):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid class name: '{class_name}'. Names cannot contain whitespace or be empty."
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
                raise HTTPException(status_code=400, detail=str(e))

    # SECOND PASS: Now add attributes and methods to classes
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
                    visibility, name, attr_type = parse_attribute(attr.get("name", ""), domain_model)
                    if name is None:  # Skip if no name was returned
                        continue
                    if name in attribute_names:
                        raise HTTPException(status_code=400, detail=f"Duplicate attribute name '{name}' found in class '{class_name}'")
                    attribute_names.add(name)
                    
                    # Find the type in the domain model
                    type_obj = None
                    for t in domain_model.types:
                        if isinstance(t, (Enumeration, Class)) and t.name == attr_type:
                            type_obj = t
                            break
                    
                    if type_obj:
                        property_ = Property(name=name, type=type_obj, visibility=visibility)
                    else:
                        property_ = Property(name=name, type=PrimitiveDataType(attr_type), visibility=visibility)
                    cls.add_attribute(property_)

            # Add methods
            for method_id in element.get("methods", []):
                method = elements.get(method_id)
                if method:
                    visibility, name, parameters, return_type = parse_method(method.get("name", ""), domain_model)

                    # Create method parameters
                    method_params = []
                    for param in parameters:
                        param_type_obj = None
                        param_type_name = param['type']
                        
                        # Try to find parameter type in domain model
                        for t in domain_model.types:
                            if isinstance(t, (Enumeration, Class)) and t.name == param_type_name:
                                param_type_obj = t
                                break
                                
                        if not param_type_obj:
                            param_type_obj = PrimitiveDataType(param_type_name)
                            
                        param_obj = Property(
                            name=param['name'],
                            type=param_type_obj,
                            visibility='public'
                        )
                        if 'default' in param:
                            param_obj.default_value = param['default']
                        method_params.append(param_obj)

                    # Create method with parameters and return type
                    method_obj = Method(
                        name=name,
                        visibility=visibility,
                        parameters=method_params
                    )
                    
                    # Handle return type
                    if return_type:
                        return_type_obj = None
                        # Find return type in domain model
                        for t in domain_model.types:
                            if isinstance(t, (Enumeration, Class)) and t.name == return_type:
                                return_type_obj = t
                                break
                                
                        if return_type_obj:
                            method_obj.type = return_type_obj
                        else:
                            method_obj.type = PrimitiveDataType(return_type)
                    
                    cls.add_method(method_obj)

    # Processing relationships (Associations, Generalizations, and Compositions)
    # Store association classes candidates and their links for third pass processing
    association_class_candidates = {}  # {class_id: {association_id}}
    association_by_id = {}  # {association_id: association_object}

    for rel_id, relationship in relationships.items():
        rel_type = relationship.get("type")
        source = relationship.get("source")
        target = relationship.get("target")

        if not rel_type or not source or not target:
            print(f"Skipping relationship {rel_id} due to missing data.")
            continue

        # Skip OCL links
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
            print(f"Skipping relationship {rel_id} due to missing elements.")
            continue

        source_class = domain_model.get_class_by_name(source_element.get("name", ""))
        target_class = domain_model.get_class_by_name(target_element.get("name", ""))

        if not source_class or not target_class:
            print(f"Skipping relationship {rel_id} because classes are missing in the domain model.")
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
            if association_name in [assoc.name for assoc in domain_model.associations]:
                counter = 1
                while f"{association_name}_{counter}" in [assoc.name for assoc in domain_model.associations]:
                    counter += 1
                association_name = f"{association_name}_{counter}"

            association = BinaryAssociation(
                name=association_name,
                ends={source_property, target_property}
            )
            domain_model.associations.add(association)

            # Store the association for association class processing
            association_by_id[rel_id] = association

        elif rel_type == "ClassInheritance":
            generalization = Generalization(general=target_class, specific=source_class)
            domain_model.generalizations.add(generalization)

    # THIRD PASS: Process association classes
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
            print(f"Warning: Class '{class_name}' is linked to multiple associations. Only using the first one.")

        # Get the first association
        association_id = next(iter(association_ids))
        association = association_by_id.get(association_id)

        if not association:
            continue

        # Get attributes and methods from the original class
        attributes = class_obj.attributes
        methods = class_obj.methods

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

    # Process OCL constraints
    all_constraints = set()
    all_warnings = []
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

    # Process comments and apply them to class or domain model metadata
    for comment_id, comment_text in comment_elements.items():
        if comment_id in comment_links:
            # Comment is linked to specific elements
            for linked_element_id in comment_links[comment_id]:
                linked_element = elements.get(linked_element_id)
                if linked_element:
                    element_name = linked_element.get("name", "").strip()
                    # Find the class in the domain model
                    for type_obj in domain_model.types:
                        if isinstance(type_obj, Class) and type_obj.name == element_name:
                            # Add comment to class metadata
                            if not type_obj.metadata:
                                type_obj.metadata = Metadata(description=comment_text)
                            else:
                                # Append to existing description
                                if type_obj.metadata.description:
                                    type_obj.metadata.description += f"\n{comment_text}"
                                else:
                                    type_obj.metadata.description = comment_text
                            break
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

    return domain_model
