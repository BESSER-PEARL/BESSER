"""
Domain model conversion from BUML to JSON format.
"""

import uuid
import ast
from besser.BUML.metamodel.structural import (
    Class, Property, Method, DomainModel, PrimitiveDataType, Enumeration,
    EnumerationLiteral, BinaryAssociation, Generalization, Multiplicity,
    UNLIMITED_MAX_MULTIPLICITY, Constraint, AssociationClass, Metadata
)
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    VISIBILITY_MAP, RELATIONSHIP_TYPES
)
from besser.utilities.web_modeling_editor.backend.services.utils import (
    calculate_center_point, determine_connection_direction, calculate_connection_points,
    calculate_path_points, calculate_relationship_bounds
)


def parse_buml_content(content: str) -> DomainModel:
    """Parse B-UML content from a Python file and return a DomainModel and OCL constraints."""
    try:
        # If caller already passed a DomainModel instance, return it directly.
        if isinstance(content, DomainModel):
            return content

        # Create a safe environment for eval without any generators
        safe_globals = {
            "Class": Class,
            "Property": Property,
            "Method": Method,
            "PrimitiveDataType": PrimitiveDataType,
            "BinaryAssociation": BinaryAssociation,
            "Constraint": Constraint,
            "Multiplicity": Multiplicity,
            "UNLIMITED_MAX_MULTIPLICITY": UNLIMITED_MAX_MULTIPLICITY,
            "Generalization": Generalization,
            "Enumeration": Enumeration,
            "EnumerationLiteral": EnumerationLiteral,
            "set": set,
            "StringType": PrimitiveDataType("str"),
            "IntegerType": PrimitiveDataType("int"),
            "DateType": PrimitiveDataType("date"),
        }

        # Ensure we have a string before preprocessing
        if not isinstance(content, str):
            raise TypeError(f"Expected B-UML content as str or DomainModel, got {type(content)!r}")

        # Pre-process the content to remove generator-related lines
        cleaned_lines = []
        for line in content.splitlines():
            if not any(gen in line for gen in ["Generator(", ".generate("]):
                cleaned_lines.append(line)
        cleaned_content = "\n".join(cleaned_lines)

        # Execute the cleaned B-UML content
        local_vars = {}
        exec(cleaned_content, safe_globals, local_vars)

        domain_name = "Imported_Domain_Model"
        for var_name, var_value in local_vars.items():
            if isinstance(var_value, DomainModel):
                domain_name = var_value.name

        domain_model = DomainModel(domain_name)
        # First pass: Add all classes and enumerations
        classes = {}
        for var_name, var_value in local_vars.items():
            if isinstance(var_value, (Class, Enumeration)):
                domain_model.types.add(var_value)
                classes[var_name] = var_value
            elif isinstance(var_value, Constraint):
                domain_model.constraints.add(var_value)

        # Second pass: Add associations and generalizations
        for var_name, var_value in local_vars.items():
            if isinstance(var_value, BinaryAssociation):
                domain_model.associations.add(var_value)
            elif isinstance(var_value, Generalization):
                domain_model.generalizations.add(var_value)

        return domain_model

    except Exception as e:
        print(f"Error parsing B-UML content: {e}")
        raise ValueError(f"Failed to parse B-UML content: {str(e)}")


def class_buml_to_json(domain_model):
    """Convert a B-UML DomainModel object to JSON format matching the frontend structure."""
    elements = {}
    relationships = {}
    # Default diagram size
    default_size = {
        "width": 1200,
        "height": 800,
    }

    # Grid layout configuration
    grid_size = {
        "x_spacing": 300,
        "y_spacing": 200,
        "max_columns": 3,
    }

    # Track position
    current_column = 0
    current_row = 0
    
    # Track comments to create
    comments_to_create = []  # [(comment_text, linked_class_id)]

    def get_position():
        nonlocal current_column, current_row
        x = -600 + (current_column * grid_size["x_spacing"])
        y = -300 + (current_row * grid_size["y_spacing"])

        # Move to next position
        current_column += 1
        if current_column >= grid_size["max_columns"]:
            current_column = 0
            current_row += 1

        return x, y

    # First pass: Create all class and enumeration elements
    class_id_map = {}  # Store mapping between Class objects and their IDs

    for type_obj in domain_model.types | domain_model.constraints:
        if isinstance(type_obj, (Class, Enumeration, Constraint)):
            # Generate UUID for the element
            element_id = str(uuid.uuid4())
            class_id_map[type_obj] = element_id

            # Get position for this element
            x, y = get_position()

            # Initialize lists for attributes and methods IDs
            attribute_ids = []
            method_ids = []

            # Process attributes/literals
            y_offset = y + 40  # Starting position for attributes
            if isinstance(type_obj, Class):
                for attr in type_obj.attributes:
                    attr_id = str(uuid.uuid4())
                    visibility_symbol = next(
                        k for k, v in VISIBILITY_MAP.items() if v == attr.visibility
                    )
                    attr_type = (
                        attr.type.name if hasattr(attr.type, "name") else str(attr.type)
                    )

                    elements[attr_id] = {
                        "id": attr_id,
                        "name": f"{visibility_symbol} {attr.name}: {attr_type}",
                        "type": "ClassAttribute",
                        "owner": element_id,
                        "bounds": {
                            "x": x + 0.5,
                            "y": y_offset,
                            "width": 159,
                            "height": 30,
                        },
                    }
                    attribute_ids.append(attr_id)
                    y_offset += 30

                # Process methods
                for method in type_obj.methods:
                    method_id = str(uuid.uuid4())
                    visibility_symbol = next(
                        k for k, v in VISIBILITY_MAP.items() if v == method.visibility
                    )

                    # Build method signature with parameters and return type
                    param_str = []
                    for param in method.parameters:
                        param_type = (
                            param.type.name
                            if hasattr(param.type, "name")
                            else str(param.type)
                        )
                        param_signature = f"{param.name}: {param_type}"
                        if (
                            hasattr(param, "default_value")
                            and param.default_value is not None
                        ):
                            param_signature += f" = {param.default_value}"
                        param_str.append(param_signature)

                    # Build complete method signature
                    method_signature = (
                        f"{visibility_symbol} {method.name}({', '.join(param_str)})"
                    )
                    if hasattr(method, "type") and method.type:
                        return_type = (
                            method.type.name
                            if hasattr(method.type, "name")
                            else str(method.type)
                        )
                        method_signature += f": {return_type}"

                    elements[method_id] = {
                        "id": method_id,
                        "name": method_signature,
                        "type": "ClassMethod",
                        "owner": element_id,
                        "bounds": {
                            "x": x + 0.5,
                            "y": y_offset,
                            "width": 159,
                            "height": 30,
                        },
                    }
                    method_ids.append(method_id)
                    y_offset += 30

            elif isinstance(type_obj, Enumeration):
                # Handle enumeration literals
                for literal in type_obj.literals:
                    literal_id = str(uuid.uuid4())
                    elements[literal_id] = {
                        "id": literal_id,
                        "name": literal.name,
                        "type": "ClassAttribute",  # We use ClassAttribute type for literals
                        "owner": element_id,
                        "bounds": {
                            "x": x + 0.5,
                            "y": y_offset,
                            "width": 159,
                            "height": 30,
                        },
                    }
                    attribute_ids.append(literal_id)
                    y_offset += 30

            # Create the element
            element_data = {
                "id": element_id,
                "name": type_obj.name,
                "type": (
                    "Enumeration"
                    if isinstance(type_obj, Enumeration)
                    else (
                        "ClassOCLConstraint"
                        if isinstance(type_obj, Constraint)
                        else "AbstractClass" if type_obj.is_abstract else "Class"
                    )
                ),
                "owner": None,
                "bounds": {
                    "x": x,
                    "y": y,
                    "width": 160,
                    "height": max(100, 30 * (len(attribute_ids) + len(method_ids) + 1)),
                },
                **(
                    {
                        "attributes": attribute_ids,
                        "methods": method_ids,
                        "stereotype": (
                            "enumeration" if isinstance(type_obj, Enumeration) else None
                        ),
                    }
                    if not isinstance(type_obj, Constraint)
                    else {"constraint": type_obj.expression}
                ),
            }
            
            # Add metadata fields for classes if they exist
            if isinstance(type_obj, Class) and hasattr(type_obj, 'metadata') and type_obj.metadata:
                if type_obj.metadata.description:
                    element_data["description"] = type_obj.metadata.description
                    # Also create a comment element linked to this class
                    comments_to_create.append((type_obj.metadata.description, element_id))
                if type_obj.metadata.uri:
                    element_data["uri"] = type_obj.metadata.uri
                if type_obj.metadata.icon:
                    element_data["icon"] = type_obj.metadata.icon

            elements[element_id] = element_data

    # Second pass: Create relationships
    for association in domain_model.associations:
        try:
            rel_id = str(uuid.uuid4())
            name = association.name if association.name else ""
            ends = list(association.ends)
            if len(ends) == 2:
                source_prop, target_prop = ends

                # Check navigability and composition, swap if needed
                if source_prop.is_composite and not target_prop.is_composite:
                    source_prop, target_prop = target_prop, source_prop
                elif not source_prop.is_composite and not target_prop.is_composite:
                    if not source_prop.is_navigable and target_prop.is_navigable:
                        pass
                    elif source_prop.is_navigable and not target_prop.is_navigable:
                        source_prop, target_prop = target_prop, source_prop
                    elif not source_prop.is_navigable and not target_prop.is_navigable:
                        print(f"Warning: Both ends of association {name} are not navigable. Skipping this association.")
                        continue

                source_class = source_prop.type
                target_class = target_prop.type

                if source_class in class_id_map and target_class in class_id_map:
                    # Get source and target elements
                    source_element = elements[class_id_map[source_class]]
                    target_element = elements[class_id_map[target_class]]

                    # Calculate connection directions and points
                    source_dir, target_dir = determine_connection_direction(
                        source_element["bounds"], target_element["bounds"]
                    )

                    source_point = calculate_connection_points(
                        source_element["bounds"], source_dir
                    )
                    target_point = calculate_connection_points(
                        target_element["bounds"], target_dir
                    )

                    # Calculate path points
                    path_points = calculate_path_points(
                        source_point, target_point, source_dir, target_dir
                    )

                    # Calculate bounds
                    rel_bounds = calculate_relationship_bounds(path_points)

                    # Determine relationship type
                    rel_type = (
                        RELATIONSHIP_TYPES["composition"]
                        if target_prop.is_composite
                        else (
                            RELATIONSHIP_TYPES["bidirectional"]
                            if source_prop.is_navigable and target_prop.is_navigable
                            else RELATIONSHIP_TYPES["unidirectional"]
                        )
                    )

                    relationships[rel_id] = {
                        "id": rel_id,
                        "name": name,
                        "type": rel_type,
                        "source": {
                            "element": class_id_map[source_class],
                            "multiplicity": f"{source_prop.multiplicity.min}..{'*' if source_prop.multiplicity.max == 9999 else source_prop.multiplicity.max}",
                            "role": source_prop.name,
                            "direction": source_dir,
                            "bounds": {
                                "x": source_point["x"],
                                "y": source_point["y"],
                                "width": 0,
                                "height": 0,
                            },
                        },
                        "target": {
                            "element": class_id_map[target_class],
                            "multiplicity": f"{target_prop.multiplicity.min}..{'*' if target_prop.multiplicity.max == 9999 else target_prop.multiplicity.max}",
                            "role": target_prop.name,
                            "direction": target_dir,
                            "bounds": {
                                "x": target_point["x"],
                                "y": target_point["y"],
                                "width": 0,
                                "height": 0,
                            },
                        },
                        "bounds": rel_bounds,
                        "path": path_points,
                        "isManuallyLayouted": False,
                    }
        except Exception as e:
            print(f"Error creating relationship: {e}")
            continue

    # Handle generalizations
    for generalization in domain_model.generalizations:
        rel_id = str(uuid.uuid4())
        if (
            generalization.general in class_id_map
            and generalization.specific in class_id_map
        ):
            relationships[rel_id] = {
                "id": rel_id,
                "type": "ClassInheritance",
                "source": {
                    "element": class_id_map[generalization.specific],
                    "bounds": {"x": 0, "y": 0, "width": 0, "height": 0},
                },
                "target": {
                    "element": class_id_map[generalization.general],
                    "bounds": {"x": 0, "y": 0, "width": 0, "height": 0},
                },
                "path": [
                    {"x": 0, "y": 0},
                    {"x": 50, "y": 0},
                    {"x": 50, "y": 50},
                    {"x": 100, "y": 50},
                ],
            }

    # Handle association classes
    for type_obj in domain_model.types:
        if isinstance(type_obj, AssociationClass) and type_obj in class_id_map:
            # Track associations by name for easier lookup
            association_by_name = {}
            for rel_id, rel in relationships.items():
                if rel.get("type") in [
                    "ClassBidirectional",
                    "ClassUnidirectional",
                    "ClassComposition",
                ]:
                    association_by_name[rel.get("name", "")] = rel_id

            # Find the association relationship ID by name
            association_rel_id = association_by_name.get(type_obj.association.name)
            if association_rel_id:
                # Create a ClassLinkRel relationship
                rel_id = str(uuid.uuid4())

                relationships[rel_id] = {
                    "id": rel_id,
                    "name": "",
                    "type": "ClassLinkRel",
                    "owner": None,
                    "source": {"element": association_rel_id, "direction": "Center"},
                    "target": {"element": class_id_map[type_obj], "direction": "Up"},
                    "bounds": {"x": 0, "y": 0, "width": 0, "height": 0},
                    "path": [{"x": 0, "y": 0}, {"x": 0, "y": 0}],
                    "isManuallyLayouted": False,
                }

    # Handle OCL constraint links
    for type_obj in domain_model.constraints:
        if isinstance(type_obj, Constraint) and type_obj.context in class_id_map:
            rel_id = str(uuid.uuid4())
            relationships[rel_id] = {
                "id": rel_id,
                "name": "",
                "type": "ClassOCLLink",
                "owner": None,
                "source": {
                    "direction": "Left",
                    "element": class_id_map[type_obj],
                    "multiplicity": "",
                    "role": "",
                },
                "target": {
                    "direction": "Right",
                    "element": class_id_map[type_obj.context],
                    "multiplicity": "",
                    "role": "",
                },
                "bounds": {"x": 0, "y": 0, "width": 0, "height": 0},
                "path": [{"x": 0, "y": 0}, {"x": 0, "y": 0}],
                "isManuallyLayouted": False,
            }
    
    # Create comment elements from metadata descriptions
    for comment_text, linked_class_id in comments_to_create:
        comment_id = str(uuid.uuid4())
        x, y = get_position()
        
        # Create comment element
        elements[comment_id] = {
            "id": comment_id,
            "name": comment_text,
            "type": "Comments",
            "owner": None,
            "bounds": {
                "x": x,
                "y": y,
                "width": 160,
                "height": 100,
            },
        }
        
        # Create Link relationship from comment to class
        rel_id = str(uuid.uuid4())
        relationships[rel_id] = {
            "id": rel_id,
            "name": "",
            "type": "Link",
            "owner": None,
            "source": {
                "direction": "Right",
                "element": comment_id,
                "multiplicity": "",
                "role": "",
            },
            "target": {
                "direction": "Left",
                "element": linked_class_id,
                "multiplicity": "",
                "role": "",
            },
            "bounds": {"x": 0, "y": 0, "width": 0, "height": 0},
            "path": [{"x": 0, "y": 0}, {"x": 0, "y": 0}],
            "isManuallyLayouted": False,
        }
    
    # Handle domain model level comments (unlinked comments)
    if hasattr(domain_model, 'metadata') and domain_model.metadata and domain_model.metadata.description:
        comment_id = str(uuid.uuid4())
        x, y = get_position()
        
        elements[comment_id] = {
            "id": comment_id,
            "name": domain_model.metadata.description,
            "type": "Comments",
            "owner": None,
            "bounds": {
                "x": x,
                "y": y,
                "width": 160,
                "height": 100,
            },
        }

    # Create the final structure
    result = {
        "version": "3.0.0",
        "type": "ClassDiagram",
        "size": default_size,
        "interactive": {"elements": {}, "relationships": {}},
        "elements": elements,
        "relationships": relationships,
        "assessments": {},
    }

    return result
