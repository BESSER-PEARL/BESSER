import re
import json

from besser.BUML.metamodel.structural import DomainModel, Class, Enumeration, Property, Method, BinaryAssociation, \
    Generalization, PrimitiveDataType, EnumerationLiteral, Multiplicity, UNLIMITED_MAX_MULTIPLICITY, Constraint, AnyType, \
    AssociationClass, Metadata
from besser.BUML.metamodel.object import ObjectModel, Object, AttributeLink, DataValue, Link, LinkEnd
from besser.utilities.web_modeling_editor.backend.constants.constants import VISIBILITY_MAP, VALID_PRIMITIVE_TYPES
from fastapi import HTTPException


def parse_attribute(attribute_name, domain_model=None):
    """Parse an attribute string to extract visibility, name, and type, removing any colons."""
    # Split the string by colon first to separate name and type
    name_type_parts = attribute_name.split(":")

    if len(name_type_parts) > 1:
        name_part = name_type_parts[0].strip()
        type_part = name_type_parts[1].strip()

        # Check for visibility symbol at start of name
        if name_part[0] in VISIBILITY_MAP:
            visibility = VISIBILITY_MAP[name_part[0]]
            name = name_part[1:].strip()
        else:
            # Existing split logic for space-separated visibility
            name_parts = name_part.split()
            if len(name_parts) > 1:
                visibility_symbol = name_parts[0] if name_parts[0] in VISIBILITY_MAP else "+"
                visibility = VISIBILITY_MAP.get(visibility_symbol, "public")
                name = name_parts[1]
            else:
                visibility = "public"
                name = name_parts[0]

        # Handle the type
        if domain_model and any(isinstance(t, (Enumeration, Class)) and t.name == type_part for t in domain_model.types):
            attr_type = type_part
        else:
            attr_type = VALID_PRIMITIVE_TYPES.get(type_part.lower(), None)
            if attr_type is None:
                raise ValueError(f"Invalid type: {type_part}")
    else:
        # Handle case without type specification
        parts = attribute_name.split()

        if len(parts) == 1:
            part = parts[0].strip()
            if part and part[0] in VISIBILITY_MAP:
                visibility = VISIBILITY_MAP[part[0]]
                name = part[1:].strip()
                attr_type = "str"
            else:
                visibility = "public"
                name = part
                attr_type = "str"
        else:
            visibility_symbol = parts[0] if parts[0] in VISIBILITY_MAP else "+"
            visibility = VISIBILITY_MAP.get(visibility_symbol, "public")
            name = parts[1]
            attr_type = "str"
    if not name:  # Skip if name is empty
        return None, None, None
    return visibility, name, attr_type

def parse_method(method_str, domain_model=None):
    """
    Parse a method string to extract visibility, name, parameters, and return type.
    Examples:
    "+ notify(sms: str = 'message')" -> ("public", "notify", [{"name": "sms", "type": "str", "default": "message"}], None)
    "- findBook(title: str): Book" -> ("private", "findBook", [{"name": "title", "type": "str"}], "Book")
    "validate()" -> ("public", "validate", [], None)
    """

    # Default values
    visibility = "public"
    parameters = []
    return_type = None

    # Check if this is actually a method (contains parentheses)
    if '(' not in method_str:
        return visibility, method_str, parameters, return_type

    # Extract visibility if present
    method_str = method_str.strip()
    if method_str.startswith(tuple(VISIBILITY_MAP.keys())):
        visibility = VISIBILITY_MAP.get(method_str[0], "public")
        method_str = method_str[2:].strip()

    # Parse method using regex
    pattern = r"([^(]+)\((.*?)\)(?:\s*:\s*(.+))?"
    match = re.match(pattern, method_str)

    if not match:
        return visibility, method_str.replace("()", ""), parameters, return_type

    method_name, params_str, return_type = match.groups()
    method_name = method_name.strip()

    # Parse parameters if present
    if params_str:
        # Handle nested parentheses in default values
        param_list = []
        current_param = []
        paren_count = 0

        for char in params_str + ',':
            if char == '(' and paren_count >= 0:
                paren_count += 1
                current_param.append(char)
            elif char == ')' and paren_count > 0:
                paren_count -= 1
                current_param.append(char)
            elif char == ',' and paren_count == 0:
                param_list.append(''.join(current_param).strip())
                current_param = []
            else:
                current_param.append(char)

        for param in param_list:
            if not param:
                continue

            param_dict = {'name': param, 'type': 'any'}

            # Handle parameter with default value
            if '=' in param:
                param_parts = param.split('=', 1)
                param_name_type = param_parts[0].strip()
                default_value = param_parts[1].strip().strip('"\'')

                if ':' in param_name_type:
                    param_name, param_type = [p.strip() for p in param_name_type.split(':')]
                    param_dict.update({
                        'name': param_name,
                        'type': VALID_PRIMITIVE_TYPES.get(param_type.lower(), param_type),
                        'default': default_value
                    })
                else:
                    param_dict.update({
                        'name': param_name_type,
                        'default': default_value
                    })

            # Handle parameter with type annotation
            elif ':' in param:
                param_name, param_type = [p.strip() for p in param.split(':')]

                # Handle the type
                if domain_model and any(isinstance(t, (Enumeration, Class)) and t.name == param_type for t in domain_model.types):
                    type_param = param_type
                else:
                    type_param = VALID_PRIMITIVE_TYPES.get(param_type.lower(), None)
                    if type_param is None:
                        raise ValueError(f"Invalid type '{param_type}' for the parameter '{param_name}'")

                param_dict.update({
                    'name': param_name,
                    'type': type_param
                })
            else:
                param_dict['name'] = param.strip()

            parameters.append(param_dict)

    # Clean up return type if present
    if return_type:
        return_type = return_type.strip()
        # Keep the original return type if it's not a primitive type
        if domain_model and any(isinstance(t, (Enumeration, Class)) and t.name == return_type for t in domain_model.types):
            type_return = return_type
        else:
            type_return = VALID_PRIMITIVE_TYPES.get(return_type.lower(), None)
            if type_return is None:
                raise ValueError(f"Invalid return type '{return_type}' for the method '{method_name}'")

    return visibility, method_name, parameters, return_type

def parse_multiplicity(multiplicity_str):
    """Parse a multiplicity string and return a Multiplicity object with defaults."""
    if not multiplicity_str:
        return Multiplicity(min_multiplicity=1, max_multiplicity=1)

    # Handle single "*" case
    if multiplicity_str == "*":
        return Multiplicity(min_multiplicity=0, max_multiplicity=UNLIMITED_MAX_MULTIPLICITY)

    parts = multiplicity_str.split("..")
    try:
        min_multiplicity = int(parts[0]) if parts[0] and parts[0] != "*" else 0
        max_multiplicity = (
            UNLIMITED_MAX_MULTIPLICITY if len(parts) > 1 and (not parts[1] or parts[1] == "*")
            else int(parts[1]) if len(parts) > 1
            else min_multiplicity
        )
    except ValueError:
        # If parsing fails, return default multiplicity of 1..1
        return Multiplicity(min_multiplicity=1, max_multiplicity=1)

    return Multiplicity(min_multiplicity=min_multiplicity, max_multiplicity=max_multiplicity)

def process_ocl_constraints(ocl_text: str, domain_model: DomainModel, counter: int) -> tuple[list, list]:
    """Process OCL constraints and convert them to BUML Constraint objects."""
    if not ocl_text:
        return [], []

    constraints = []
    warnings = []
    lines = re.split(r'[,]', ocl_text)
    constraint_count = 1

    domain_classes = {cls.name.lower(): cls for cls in domain_model.types}

    for line in lines:

        line = line.strip().replace('\n', '')
        if not line or not line.lower().startswith('context'):
            continue

        # Extract context class name
        parts = line.split()
        if len(parts) < 4:  # Minimum: "context ClassName inv name:"
            continue

        context_class_name = parts[1]
        context_class = domain_classes.get(context_class_name.lower())

        if not context_class:
            warning_msg = f"Warning: Context class {context_class_name} not found"
            warnings.append(warning_msg)
            continue

        constraint_name = f"constraint_{context_class_name}_{counter}_{constraint_count}"
        constraint_count += 1

        constraints.append(
            Constraint(
                name=constraint_name,
                context=context_class,
                expression=line,
                language="OCL"
            )
        )

    return constraints, warnings

def process_class_diagram(json_data):
    """Process Class Diagram specific elements."""
    title = json_data.get('diagramTitle', '')
    if ' ' in title:
        title = title.replace(' ', '_')

    domain_model = DomainModel(title)
    # Get elements and OCL constraints from the JSON data
    elements = json_data.get('elements', {}).get('elements', {})
    relationships = json_data.get('elements', {}).get('relationships', {})

    # FIRST PASS: Process all type declarations (enumerations and classes)
    # 1. First process enumerations
    for element_id, element in elements.items():
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
            domain_model.types.add(enum)
    
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
            
            if description or uri:
                metadata = Metadata(description=description, uri=uri)
            try:
                cls = Class(name=class_name, is_abstract=is_abstract, metadata=metadata)
                domain_model.types.add(cls)
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
                    cls.attributes.add(property_)

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
                    
                    cls.methods.add(method_obj)

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
    
    # Store the association_by_id mapping for object diagram processing
    domain_model.association_by_id = association_by_id

    return domain_model

def process_object_diagram(json_data, domain_model):
    """Process Object Diagram specific elements and return an ObjectModel."""
    from besser.BUML.metamodel.object.builder import ObjectBuilder
    
    title = json_data.get('diagramTitle', 'Generated_Object_Model')
    if ' ' in title:
        title = title.replace(' ', '_')
    
    object_model = ObjectModel(title)
    
    # Get elements and relationships from the JSON data
    elements = json_data.get('elements', {}).get('elements', {})
    relationships = json_data.get('elements', {}).get('relationships', {})
    
    # Track objects by their ID for link creation
    objects_by_id = {}
    
    # First pass: Create objects using fluent API
    for element_id, element in elements.items():
        if element.get("type") == "ObjectName":
            object_name = element.get("name", "")
            class_id = element.get("classId")
            
            # Find the corresponding class in the domain model using classId
            class_obj = None
            if class_id:
                # Get the reference data to find the class name by ID
                reference_data = json_data.get('elements', {}).get('referenceDiagramData', {})
                if not reference_data:
                    reference_data = json_data.get('referenceDiagramData', {})
                
                if reference_data:
                    reference_elements = reference_data.get('elements', {})
                    class_element = reference_elements.get(class_id)
                    if class_element:
                        class_name = class_element.get("name", "")
                        class_obj = domain_model.get_class_by_name(class_name)
                        if not class_obj:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Class '{class_name}' with ID '{class_id}' not found in domain model. Please ensure the class diagram contains all required classes."
                            )
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Could not find class element with ID '{class_id}' in reference diagram data. Please verify the object diagram references are correct."
                        )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="No reference diagram data found. Please ensure the class diagram is properly linked to the object diagram."
                    )

            # Fall back to searching by object name if class ID lookup fails
            if not class_obj:
                class_obj = domain_model.get_class_by_name(object_name)
               
            
            if not class_obj:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not find class for object '{object_name}' with class ID '{class_id}'. Please ensure all objects have corresponding classes in the class diagram."
                )
            
            # Create object using fluent API
            builder = ObjectBuilder(class_obj).name(object_name)
            
            # Process object attributes (slots) and add them to builder
            attributes_dict = {}
            for attr_id in element.get("attributes", []):
                attr_element = elements.get(attr_id)
                if attr_element and attr_element.get("type") == "ObjectAttribute":
                    attr_string = attr_element.get("name", "")
                    
                    # Parse the attribute string to extract name, type, and value
                    # Format: "+ name: type = value"
                    value = None
                    if " = " in attr_string:
                        attr_part, value_part = attr_string.split(" = ", 1)
                        value = value_part.strip()
                        attr_string = attr_part.strip()
                    
                    # Parse the attribute definition
                    try:
                        visibility, attr_name, attr_type = parse_attribute(attr_string, domain_model)
                        if attr_name and value is not None:
                            # Find the corresponding property in the class or its parents
                            property_obj = None
                            
                            # First check the class itself
                            for prop in class_obj.attributes:
                                if prop.name == attr_name:
                                    property_obj = prop
                                    break
                            
                            # If not found, check parent classes (for inheritance)
                            if not property_obj:
                                for gen in domain_model.generalizations:
                                    if gen.specific == class_obj:
                                        for prop in gen.general.attributes:
                                            if prop.name == attr_name:
                                                property_obj = prop
                                                break
                                        if property_obj:
                                            break
                            
                            if property_obj:
                                # Convert value to appropriate type
                                converted_value = value
                                if hasattr(property_obj.type, 'name'):
                                    type_name = property_obj.type.name if hasattr(property_obj.type, 'name') else str(property_obj.type)
                                    if type_name in ['int', 'IntegerType']:
                                        try:
                                            converted_value = int(value)
                                        except ValueError:
                                            converted_value = value
                                    elif type_name in ['float', 'FloatType']:
                                        try:
                                            converted_value = float(value)
                                        except ValueError:
                                            converted_value = value
                                    elif type_name in ['bool', 'BooleanType']:
                                        converted_value = value.lower() in ['true', '1', 'yes']
                                
                                attributes_dict[attr_name] = converted_value
                                
                    except Exception as e:
                        print(f"Warning: Could not process attribute '{attr_string}' for object '{object_name}': {e}")
                        continue
            
            # Add attributes to builder if any were found
            if attributes_dict:
                builder = builder.attributes(**attributes_dict)
            
            # Build the object
            obj = builder.build()
            # print(f"Created object '{object_name}' of class '{class_obj.name}'")
            
            # Add the object to the model and track it
            object_model.add_object(obj)
            objects_by_id[element_id] = obj
    
    # Second pass: Create links between objects
    for rel_id, relationship in relationships.items():
        if relationship.get("type") == "ObjectLink":
            source_id = relationship.get("source", {}).get("element")
            target_id = relationship.get("target", {}).get("element")
            link_name = relationship.get("name", "")
            association_id = relationship.get("associationId")
            
            source_obj = objects_by_id.get(source_id)
            target_obj = objects_by_id.get(target_id)
            
            if not source_obj or not target_obj:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not find objects for link '{link_name}'. Please ensure all objects in the link exist in the diagram."
                )
          # Find the corresponding association in the domain model
            association_obj = None
            if association_id:
                # First try to find the association directly by ID from the domain model
                if hasattr(domain_model, 'association_by_id') and domain_model.association_by_id:
                    association_obj = domain_model.association_by_id.get(association_id)
                
                # If not found by direct ID lookup, try the reference diagram approach
                if not association_obj:
                    # Look for the association by ID in the reference diagram data
                    reference_data = json_data.get('elements', {}).get('referenceDiagramData', {})
                    if not reference_data:
                        reference_data = json_data.get('referenceDiagramData', {})
                    if reference_data:
                        reference_relationships = reference_data.get('relationships', {})
                        assoc_element = reference_relationships.get(association_id)
                        if assoc_element:
                            # Found association in reference data
                            assoc_name = assoc_element.get("name", "")
                            # Only try to find by name if the association name is not empty
                            if assoc_name:
                                # Try to find the association by name
                                for assoc in domain_model.associations:
                                    if assoc.name == assoc_name:
                                        association_obj = assoc
                                        break
            
            # If still not found, try to find association by matching the connected classes
            if not association_obj:
                for assoc in domain_model.associations:
                    # Check if this association connects the right classes
                    end_types = [end.type for end in assoc.ends]
                    end_type_names = [end_type.name for end_type in end_types]
                    if (source_obj.classifier in end_types and target_obj.classifier in end_types):
                        association_obj = assoc
                        break

            if association_obj:
                # Create link using association ends
                link_ends = []
                for end in association_obj.ends:
                    # Check if end type matches source object (considering inheritance)
                    source_matches = False
                    if end.type == source_obj.classifier:
                        source_matches = True
                    else:
                        # Check if source object's class inherits from end type
                        for gen in domain_model.generalizations:
                            if gen.specific == source_obj.classifier and gen.general == end.type:
                                source_matches = True
                                break
                    
                    # Check if end type matches target object (considering inheritance)
                    target_matches = False
                    if end.type == target_obj.classifier:
                        target_matches = True
                    else:
                        # Check if target object's class inherits from end type
                        for gen in domain_model.generalizations:
                            if gen.specific == target_obj.classifier and gen.general == end.type:
                                target_matches = True
                                break
                    
                    if source_matches:
                        link_end = LinkEnd(name=f"{end.name}_end", association_end=end, object=source_obj)
                        link_ends.append(link_end)
                        # print(f"  -> Created link end for source: {end.name}_end")
                    elif target_matches:
                        link_end = LinkEnd(name=f"{end.name}_end", association_end=end, object=target_obj)
                        link_ends.append(link_end)

                if len(link_ends) == 2:
                    # Use link_name if provided, otherwise generate a default name
                    link_display_name = link_name if link_name else f"{source_obj.name}_{target_obj.name}_link"
                    link = Link(name=link_display_name, association=association_obj, connections=link_ends)
                    # Links are automatically added to objects via the Link constructor
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error: Expected 2 link ends but got {len(link_ends)} for link '{link_name}'. There may be an issue with the association structure."
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not find association for link '{link_name}'. Please ensure all links correspond to valid associations in the class diagram."
                )
    return object_model

def process_state_machine(json_data):
    """Process State Machine Diagram specific elements and return Python code as string."""
    code_lines = []
    code_lines.append("import datetime")
    code_lines.append("from besser.BUML.metamodel.state_machine.state_machine import StateMachine, Session, Body, Event\n")
    sm_name = json_data.get("name", "Generated_State_Machine")
    code_lines.append(f"sm = StateMachine(name='{sm_name}')\n")

    elements = json_data.get("elements", {})
    relationships = json_data.get("relationships", {})

    # Track states by ID for later reference
    states_by_id = {}
    body_names = set()
    event_names = set()

    # Collect all body and event names first
    for element in elements.values():
        if element.get("type") == "StateBody":
            body_names.add(element.get("name"))
        elif element.get("type") == "StateFallbackBody":
            body_names.add(element.get("name"))

    # Collect event names from transitions
    for rel in relationships.values():
        if rel.get("type") == "StateTransition" and rel.get("name"):
            event_names.add(rel.get("name"))

    # Write function definitions first
    for element in elements.values():
        if element.get("type") == "StateCodeBlock":
            name = element.get("name", "")
            code_content = element.get("code", {})
            
            # If name is empty, try to extract function name from code content
            if not name:
                # Look for "def function_name(" pattern in the code
                function_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_content)
                if function_match:
                    name = function_match.group(1)
                    
            # Clean up the code content by removing extra newlines
            cleaned_code = "\n".join(line for line in code_content.splitlines() if line.strip())
            # Write the function definition with its code content
            code_lines.append(cleaned_code)  # Write the actual function code
            code_lines.append("")  # Add single blank line after function

            if name in body_names:
                code_lines.append(f"{name} = Body(name='{name}', callable={name})")
            if name in event_names:
                code_lines.append(f"{name} = Event(name='{name}', callable={name})")
            code_lines.append("")  # Add blank line after Body/Event creation

    # Create states
    for element_id, element in elements.items():
        if element.get("type") == "State":
            is_initial = False
            for rel in relationships.values():
                if (rel.get("type") == "StateTransition" and
                    rel.get("target", {}).get("element") == element_id and
                    elements.get(rel.get("source", {}).get("element", ""), {}).get("type") == "StateInitialNode"):
                    is_initial = True
                    break

            state_name = element.get("name", "")
            code_lines.append(f"{state_name}_state = sm.new_state(name='{state_name}', initial={str(is_initial)})")
            states_by_id[element_id] = state_name
    code_lines.append("")

    # Assign bodies to states
    for element_id, element in elements.items():
        if element.get("type") == "State":
            state_name = element.get("name", "")
            for body_id in element.get("bodies", []):
                body_element = elements.get(body_id)
                if body_element:
                    body_name = body_element.get("name")
                    if body_name in body_names:
                        code_lines.append(f"{state_name}_state.set_body(body={body_name})")

            for fallback_id in element.get("fallbackBodies", []):
                fallback_element = elements.get(fallback_id)
                if fallback_element:
                    fallback_name = fallback_element.get("name")
                    if fallback_name in body_names:
                        code_lines.append(f"{state_name}_state.set_fallback_body({fallback_name})")
    code_lines.append("")

    # Write transitions
    for relationship in relationships.values():
        if relationship.get("type") == "StateTransition":
            source_id = relationship.get("source", {}).get("element")
            target_id = relationship.get("target", {}).get("element")

            if elements.get(source_id, {}).get("type") == "StateInitialNode":
                continue

            source_name = states_by_id.get(source_id)
            target_name = states_by_id.get(target_id)

            if source_name and target_name:
                event_name = relationship.get("name", "")
                params = relationship.get("params")

                if event_name:
                    event_params = f"event_params={{ {params} }}" if params else "event_params={}"
                    code_lines.append(f"{source_name}_state.when_event_go_to(")
                    code_lines.append(f"    event={event_name},")
                    code_lines.append(f"    dest={target_name}_state,")
                    code_lines.append(f"    {event_params}")
                    code_lines.append(")")

    return "\n".join(code_lines)

import unicodedata

def sanitize_text(text):
    if not isinstance(text, str):
        return text
    # Normalize and strip accents or special symbols
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    #text = text.replace("'", "\\'")
    text = text.replace("'", " ")
    # Escape single quotes for code safety
    return text

def process_agent_diagram(json_data):
    """Process Agent Diagram specific elements and return an Agent model."""
    from besser.BUML.metamodel.state_machine.state_machine import Body, Condition, Event, ConfigProperty
    from besser.BUML.metamodel.state_machine.agent import Agent, Intent, Auto, IntentMatcher, ReceiveTextEvent
    import operator
    import json as json_lib

    # Create the agent model
    title = json_data.get('diagramTitle', 'Generated_Agent')
    if ' ' in title:
        title = title.replace(' ', '_')

    agent = Agent(title)
    
    # Add default configuration properties
    agent.add_property(ConfigProperty('websocket_platform', 'websocket.host', 'localhost'))
    agent.add_property(ConfigProperty('websocket_platform', 'websocket.port', 8765))
    agent.add_property(ConfigProperty('websocket_platform', 'streamlit.host', 'localhost'))
    agent.add_property(ConfigProperty('websocket_platform', 'streamlit.port', 5000))
    agent.add_property(ConfigProperty('nlp', 'nlp.language', 'en'))
    agent.add_property(ConfigProperty('nlp', 'nlp.region', 'US'))
    agent.add_property(ConfigProperty('nlp', 'nlp.timezone', 'Europe/Madrid'))
    agent.add_property(ConfigProperty('nlp', 'nlp.pre_processing', True))
    agent.add_property(ConfigProperty('nlp', 'nlp.intent_threshold', 0.4))
    agent.add_property(ConfigProperty('nlp', 'nlp.openai.api_key', 'YOUR-API-KEY'))
    agent.add_property(ConfigProperty('nlp', 'nlp.hf.api_key', 'YOUR-API-KEY'))
    agent.add_property(ConfigProperty('nlp', 'nlp.replicate.api_key', 'YOUR-API-KEY'))

    # Get elements and relationships from the JSON data
    elements = json_data.get('elements', {}).get('elements', {})
    relationships = json_data.get('elements', {}).get('relationships', {})
    
    # Track states and bodies for later reference
    states_by_id = {}
    bodies_by_id = {}
    fallback_bodies_by_id = {}
    intents_by_id = {}
    
    # First pass: Process intents
    intent_count = 0
    for element_id, element in elements.items():
        if element.get("type") == "AgentIntent":
            intent_name = element.get("name")
            training_sentences = []
            
            # Collect training sentences
            for body_id in element.get("bodies", []):
                body_element = elements.get(body_id)
                if body_element:
                    training_sentence = sanitize_text(body_element.get("name", ""))
                    if training_sentence:
                        training_sentences.append(training_sentence)
            
            # Create intent and add to agent
            intent = Intent(intent_name, training_sentences)
            agent.add_intent(intent)
            intents_by_id[element_id] = intent
            intent_count += 1
    
    # First identify the initial state
    initial_state_id = None
    for element_id, element in elements.items():
        if element.get("type") == "AgentState":
            # Check if this is an initial state
            for rel in relationships.values():
                if ((rel.get("type") == "AgentStateTransition" or rel.get("type") == "AgentStateTransitionInit") and
                    rel.get("target", {}).get("element") == element_id and
                    elements.get(rel.get("source", {}).get("element", ""), {}).get("type") == "StateInitialNode"):
                    initial_state_id = element_id
                    break
            if initial_state_id:
                break
    
    # Process the initial state first if found
    if initial_state_id:
        element = elements.get(initial_state_id)
        state_name = element.get("name", "")
        
        agent_state = agent.new_state(name=state_name, initial=True)
        states_by_id[initial_state_id] = agent_state
        
        # Process state bodies
        body_count = 0
        body_messages = []
        for body_id in element.get("bodies", []):
            body_element = elements.get(body_id)
            if body_element:
                body_name = f"{state_name}_body"
                body_type = body_element.get("replyType")
                body_content = body_element.get("name", "")
                
                # Collect messages for this body
                if body_type == "text":
                    body_messages.append(sanitize_text(body_content))
                elif body_type == "llm":
                    # For LLM replies, we need to use llm.predict(session.event.message)
                    body_messages.append(f"LLM:{sanitize_text(body_content)}")
                elif body_type == "code":
                    # For code, store as a special code message
                    body_messages.append(f"CODE:{sanitize_text(body_content)}")
                
                body_count += 1
        
        # Create a single body function that combines all messages
        if body_messages:
            # Check if any of the messages are LLM messages
            has_llm = any(message.startswith("LLM:") for message in body_messages)
            
            # If we have an LLM message, create a function that uses llm.predict
            if has_llm:
                f_name = f"{state_name}_body"
                def create_llm_body_function(name):
                    def body_function(session):
                        session.reply(llm.predict(session.event.message))
                    return body_function
                
                body = Body(f_name, create_llm_body_function(f_name))
            else:
                # Otherwise, create a regular function with the messages
                def create_body_function(messages):
                    def body_function(session):
                        for message in messages:
                            if message.startswith("CODE:"):
                                # This is code to be executed
                                try:
                                    # Just store the code for later execution
                                    code_content = message[5:]
                                    exec(code_content)
                                except Exception as e:
                                    print(f"Error executing code: {str(e)}")
                            else:
                                session.reply(message)
                    return body_function
                
                body = Body(f"{state_name}_body", create_body_function(body_messages))
            
            # Store the messages directly in the Body object for easier extraction
            body.messages = body_messages
            agent_state.set_body(body)
        
        # Process fallback bodies
        fallback_count = 0
        fallback_messages = []
        for fallback_id in element.get("fallbackBodies", []):
            fallback_element = elements.get(fallback_id)
            if fallback_element:
                fallback_name = f"{state_name}_fallback_body"
                fallback_type = fallback_element.get("replyType")
                fallback_content = fallback_element.get("name", "")
                
                # Collect messages for this fallback body
                if fallback_type == "text":
                    fallback_messages.append(sanitize_text(fallback_content))
                elif fallback_type == "llm":
                    # For LLM replies, store as a special LLM message
                    fallback_messages.append(f"LLM:{sanitize_text(fallback_content)}")
                elif fallback_type == "code":
                    # For code, store as a special code message
                    fallback_messages.append(f"CODE:{sanitize_text(fallback_content)}")
                
                fallback_count += 1
        
        # Create a single fallback body function that combines all messages
        if fallback_messages:
            # Check if any of the messages are LLM messages
            has_llm = any(message.startswith("LLM:") for message in fallback_messages)
            
            # If we have an LLM message, create a function that uses llm.predict
            if has_llm:
                f_name = f"{state_name}_fallback_body"
                def create_llm_fallback_function(name):
                    def fallback_function(session):
                        session.reply(llm.predict(session.event.message))
                    return fallback_function
                
                fallback_body = Body(f_name, create_llm_fallback_function(f_name))
            else:
                # Otherwise, create a regular function with the messages
                def create_fallback_function(messages):
                    def fallback_function(session):
                        for message in messages:
                            if message.startswith("CODE:"):
                                # This is code to be executed
                                try:
                                    code_content = message[5:]
                                    exec(code_content)
                                except Exception as e:
                                    print(f"Error executing code: {str(e)}")
                            else:
                                session.reply(message)
                    return fallback_function
                
                fallback_body = Body(f"{state_name}_fallback_body", create_fallback_function(fallback_messages))
            
            # Store the messages directly in the Body object for easier extraction
            fallback_body.messages = fallback_messages
            agent_state.set_fallback_body(fallback_body)
    
    # Now process the rest of the states
    for element_id, element in elements.items():
        if element.get("type") == "AgentState" and element_id != initial_state_id:
            # Create state and add to agent
            state_name = element.get("name", "")
            
            agent_state = agent.new_state(name=state_name, initial=False)
            states_by_id[element_id] = agent_state
            
            # Process state bodies
            body_count = 0
            body_messages = []
            for body_id in element.get("bodies", []):
                body_element = elements.get(body_id)
                if body_element:
                    body_name = f"{state_name}_body"
                    body_type = body_element.get("replyType")
                    body_content = body_element.get("name", "")
                    
                    # Collect messages for this body
                    if body_type == "text":
                        body_messages.append(sanitize_text(body_content))
                    elif body_type == "llm":
                        # For LLM replies, we need to use llm.predict(session.event.message)
                        body_messages.append(f"LLM:{sanitize_text(body_content)}")
                    elif body_type == "code":
                        # For code, store as a special code message
                        body_messages.append(f"CODE:{sanitize_text(body_content)}")
                    
                    body_count += 1
            
            # Create a single body function that combines all messages
            if body_messages:
                # Check if any of the messages are LLM messages
                has_llm = any(message.startswith("LLM:") for message in body_messages)
                
                # If we have an LLM message, create a function that uses llm.predict
                if has_llm:
                    f_name = f"{state_name}_body"
                    def create_llm_body_function(name):
                        def body_function(session):
                            session.reply(llm.predict(session.event.message))
                        return body_function
                    
                    body = Body(f_name, create_llm_body_function(f_name))
                else:
                    # Otherwise, create a regular function with the messages
                    def create_body_function(messages):
                        def body_function(session):
                            for message in messages:
                                if message.startswith("CODE:"):
                                    # This is code to be executed
                                    try:
                                        # Just store the code for later execution
                                        code_content = message[5:]
                                        exec(code_content)
                                    except Exception as e:
                                        print(f"Error executing code: {str(e)}")
                                else:
                                    session.reply(message)
                        return body_function
                    
                    body = Body(f"{state_name}_body", create_body_function(body_messages))
                
                # Store the messages directly in the Body object for easier extraction
                body.messages = body_messages
                agent_state.set_body(body)
            
            # Process fallback bodies
            fallback_count = 0
            fallback_messages = []
            for fallback_id in element.get("fallbackBodies", []):
                fallback_element = elements.get(fallback_id)
                if fallback_element:
                    fallback_name = f"{state_name}_fallback_body"
                    fallback_type = fallback_element.get("replyType")
                    fallback_content = fallback_element.get("name", "")
                    
                    # Collect messages for this fallback body
                    if fallback_type == "text":
                        fallback_messages.append(sanitize_text(fallback_content))
                    elif fallback_type == "llm":
                        # For LLM replies, store as a special LLM message
                        fallback_messages.append(f"LLM:{sanitize_text(fallback_content)}")
                    elif fallback_type == "code":
                        # For code, store as a special code message
                        fallback_messages.append(f"CODE:{sanitize_text(fallback_content)}")
                    
                    fallback_count += 1
            
            # Create a single fallback body function that combines all messages
            if fallback_messages:
                # Check if any of the messages are LLM messages
                has_llm = any(message.startswith("LLM:") for message in fallback_messages)
                
                # If we have an LLM message, create a function that uses llm.predict
                if has_llm:
                    f_name = f"{state_name}_fallback_body"
                    def create_llm_fallback_function(name):
                        def fallback_function(session):
                            session.reply(llm.predict(session.event.message))
                        return fallback_function
                    
                    fallback_body = Body(f_name, create_llm_fallback_function(f_name))
                else:
                    # Otherwise, create a regular function with the messages
                    def create_fallback_function(messages):
                        def fallback_function(session):
                            for message in messages:
                                if message.startswith("CODE:"):
                                    # This is code to be executed
                                    try:
                                        code_content = message[5:]
                                        exec(code_content)
                                    except Exception as e:
                                        print(f"Error executing code: {str(e)}")
                                else:
                                    session.reply(message)
                        return fallback_function
                    
                    fallback_body = Body(f"{state_name}_fallback_body", create_fallback_function(fallback_messages))
                
                # Store the messages directly in the Body object for easier extraction
                fallback_body.messages = fallback_messages
                agent_state.set_fallback_body(fallback_body)
    
    # Third pass: Process transitions
    transition_count = 0
    for relationship in relationships.values():
        if relationship.get("type") in ["AgentStateTransition", "AgentStateTransitionInit"]:
            source_id = relationship.get("source", {}).get("element")
            target_id = relationship.get("target", {}).get("element")
            
            # Skip initial node transitions (already handled when creating states)
            if elements.get(source_id, {}).get("type") == "StateInitialNode":
                continue
            
            source_state = states_by_id.get(source_id)
            target_state = states_by_id.get(target_id)
            
            if source_state and target_state:
                condition_name = relationship.get("condition", "")
                condition_value = relationship.get("conditionValue", "")
                
                # Create appropriate transition based on condition
                if condition_name == "when_intent_matched":
                    # Find the intent by name
                    intent_to_match = None
                    for intent in agent.intents:
                        if intent.name == condition_value:
                            intent_to_match = intent
                            break
                    
                    if intent_to_match:
                        source_state.when_intent_matched(intent_to_match).go_to(target_state)
                        transition_count += 1
                
                elif condition_name == "when_no_intent_matched":
                    source_state.when_no_intent_matched().go_to(target_state)
                    transition_count += 1
                
                elif condition_name == "when_variable_operation_matched":
                    # Check if condition_value is a dictionary
                    if isinstance(condition_value, dict):
                        variable_name = condition_value.get("variable")
                        operator_value = condition_value.get("operator")
                        target_value = condition_value.get("targetValue")
                        
                        # Map string operators to actual operator functions
                        operator_map = {
                            "<": operator.lt,
                            "<=": operator.le,
                            "==": operator.eq,
                            ">=": operator.ge,
                            ">": operator.gt,
                            "!=": operator.ne
                        }
                        
                        op_func = operator_map.get(operator_value)
                        if op_func:
                            source_state.when_variable_matches_operation(
                                var_name=variable_name,
                                operation=op_func,
                                target=target_value
                            ).go_to(target_state)
                            transition_count += 1
                    else:
                        # If condition_value is not a dictionary, add a simple transition
                        source_state.when_no_intent_matched().go_to(target_state)
                        transition_count += 1
                
                elif condition_name == "when_file_received":
                    mime_types = {
                        "PDF": "application/pdf",
                        "TXT": "text/plain",
                        "JSON": "application/json"
                    }
                    file_type = mime_types.get(condition_value)
                    if file_type:
                        source_state.when_file_received(file_type).go_to(target_state)
                        transition_count += 1
                
                elif condition_name == "auto":
                    source_state.go_to(target_state)
                    transition_count += 1
                
                else:
                    # Default to no_intent_matched if no condition specified
                    source_state.when_no_intent_matched().go_to(target_state)
                    transition_count += 1
    
    return agent