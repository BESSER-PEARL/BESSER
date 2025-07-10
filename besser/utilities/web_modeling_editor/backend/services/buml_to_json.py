import uuid
import ast
from datetime import datetime, UTC
import re

from besser.BUML.metamodel.structural import (
    Class,
    Property,
    Method,
    DomainModel,
    PrimitiveDataType,
    Enumeration,
    EnumerationLiteral,
    BinaryAssociation,
    Generalization,
    Multiplicity,
    UNLIMITED_MAX_MULTIPLICITY,
    Constraint,
    AssociationClass,
    Metadata,
)
from besser.BUML.metamodel.project import Project
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    VISIBILITY_MAP,
    RELATIONSHIP_TYPES,
)
from besser.utilities.web_modeling_editor.backend.services.layout_calculator import (
    calculate_center_point,
    determine_connection_direction,
    calculate_connection_points,
    calculate_path_points,
    calculate_relationship_bounds,
)
import inspect
from besser.BUML.metamodel.state_machine import StateMachine, Session, Body, Condition
import traceback


def parse_buml_content(content: str) -> DomainModel:
    """Parse B-UML content from a Python file and return a DomainModel and OCL constraints."""
    try:
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
                # print(f"Found association: {var_name} = {var_value}")
                # print(f"Association ends: {var_value.ends}")
                domain_model.associations.add(var_value)
            elif isinstance(var_value, Generalization):
                domain_model.generalizations.add(var_value)

        return domain_model

    except Exception as e:
        print(f"Error parsing B-UML content: {e}")
        raise ValueError(f"Failed to parse B-UML content: {str(e)}")


def domain_model_to_json(domain_model):
    """Convert a B-UML DomainModel object to JSON format matching the frontend structure."""
    elements = {}
    relationships = {}
    # Default diagram size
    default_size = {
        "width": 1200,
        "height": 800,  # Increased height for better visibility
    }

    # Grid layout configuration
    grid_size = {
        "x_spacing": 300,  # Space between elements horizontally
        "y_spacing": 200,  # Space between elements vertically
        "max_columns": 3,  # Maximum elements per row
    }

    # Track position
    current_column = 0
    current_row = 0

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
                    y_offset += 30            # Create the element
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
                if type_obj.metadata.uri:
                    element_data["uri"] = type_obj.metadata.uri
            
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
                    # If source is composite, swap them to make target composite
                    source_prop, target_prop = target_prop, source_prop
                elif not source_prop.is_composite and not target_prop.is_composite:
                    # Check navigability only if neither end is composite
                    if not source_prop.is_navigable and target_prop.is_navigable:
                        # If source is not navigable but target is, keep current order
                        pass
                    elif source_prop.is_navigable and not target_prop.is_navigable:
                        # If target is not navigable but source is, swap them
                        source_prop, target_prop = target_prop, source_prop
                    elif not source_prop.is_navigable and not target_prop.is_navigable:
                        # If both are not navigable, raise error but continue
                        print(
                            f"Warning: Both ends of association {name} are not navigable. Skipping this association."
                        )
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

                # Create a relationship from association path center to class
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


def state_machine_to_json(content: str):
    """Convert a state machine Python file content to JSON format matching the frontend structure."""

    elements = {}
    relationships = {}

    # Default diagram size
    default_size = {"width": 1980, "height": 640}

    # Track positions for layout
    states_x = -550
    states_y = -300
    code_blocks_x = -970
    code_blocks_y = 80

    # Parse the Python code
    tree = ast.parse(content)
    # Track states and functions
    states = {}  # name -> state_id mapping
    functions = {}  # name -> function_node mapping
    state_machine_name = "Generated_State_Machine"

    # First pass: collect all functions and state machine name
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = {
                "node": node,
                "source": ast.get_source_segment(content, node),
            }
        elif isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call):
                if (
                    isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "StateMachine"
                ):
                    for kw in node.value.keywords:
                        if kw.arg == "name":
                            state_machine_name = ast.literal_eval(kw.value)

    # Create initial node
    initial_node_id = str(uuid.uuid4())
    elements[initial_node_id] = {
        "id": initial_node_id,
        "name": "",
        "type": "StateInitialNode",
        "owner": None,
        "bounds": {"x": states_x - 300, "y": states_y + 20, "width": 45, "height": 45},
    }

    # Second pass: collect states and their configurations
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call):
                if (
                    isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "new_state"
                ):
                    state_id = str(uuid.uuid4())
                    state_name = None
                    is_initial = False

                    for kw in node.value.keywords:
                        if kw.arg == "name":
                            state_name = ast.literal_eval(kw.value)
                        elif kw.arg == "initial":
                            is_initial = ast.literal_eval(kw.value)

                    if state_name:
                        states[node.targets[0].id] = {
                            "id": state_id,
                            "name": state_name,
                            "is_initial": is_initial,
                            "bodies": [],
                            "fallback_bodies": [],
                        }

                        elements[state_id] = {
                            "id": state_id,
                            "name": state_name,
                            "type": "State",
                            "owner": None,
                            "bounds": {
                                "x": states_x,
                                "y": states_y,
                                "width": 160,
                                "height": 100,
                            },
                            "bodies": [],
                            "fallbackBodies": [],
                        }

                        if states_x < 200:
                            states_x += 490
                        else:
                            states_x = -280
                            states_y += 220

    # After creating all states, add initial node transition
    for state_info in states.values():
        if state_info["is_initial"]:
            initial_rel_id = str(uuid.uuid4())
            relationships[initial_rel_id] = {
                "id": initial_rel_id,
                "name": "",
                "type": "StateTransition",
                "owner": None,
                "source": {
                    "direction": "Right",
                    "element": initial_node_id,
                    "bounds": {
                        "x": elements[initial_node_id]["bounds"]["x"] + 45,
                        "y": elements[initial_node_id]["bounds"]["y"] + 22.5,
                        "width": 0,
                        "height": 0,
                    },
                },
                "target": {
                    "direction": "Left",
                    "element": state_info["id"],
                    "bounds": {
                        "x": elements[state_info["id"]]["bounds"]["x"],
                        "y": elements[state_info["id"]]["bounds"]["y"] + 35,
                        "width": 0,
                        "height": 0,
                    },
                },
                "bounds": {
                    "x": elements[initial_node_id]["bounds"]["x"] + 45,
                    "y": elements[initial_node_id]["bounds"]["y"] + 22.5,
                    "width": elements[state_info["id"]]["bounds"]["x"]
                    - (elements[initial_node_id]["bounds"]["x"] + 45),
                    "height": 1,
                },
                "path": [
                    {
                        "x": elements[initial_node_id]["bounds"]["x"] + 45,
                        "y": elements[initial_node_id]["bounds"]["y"] + 22.5,
                    },
                    {
                        "x": elements[state_info["id"]]["bounds"]["x"],
                        "y": elements[initial_node_id]["bounds"]["y"] + 22.5,
                    },
                ],
                "isManuallyLayouted": False,
            }
            break  # Only one initial state should exist

    # Track created code blocks to avoid duplication
    created_code_blocks = {}

    # When processing functions
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            if function_name not in created_code_blocks:
                code_block_id = str(uuid.uuid4())
                function_source = ast.get_source_segment(content, node)

                # Clean up the source code
                cleaned_source = "\n".join(
                    line.rstrip()
                    for line in function_source.splitlines()
                    if line.strip()  # Only include non-empty lines
                )

                elements[code_block_id] = {
                    "id": code_block_id,
                    "name": function_name,
                    "type": "StateCodeBlock",
                    "owner": None,
                    "bounds": {
                        "x": code_blocks_x,
                        "y": code_blocks_y,
                        "width": 580,
                        "height": 200,
                    },
                    "code": cleaned_source,
                    "language": "python",
                }
                created_code_blocks[function_name] = {
                    "id": code_block_id,
                    "source": cleaned_source,
                }
                code_blocks_x += 610

    # Third pass: process state bodies and transitions
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Attribute):
                    # Handle set_body
                    if node.value.func.attr == "set_body":
                        state_var = node.value.func.value.id
                        body_func = node.value.keywords[0].value.id
                        if state_var in states and body_func in created_code_blocks:
                            state = states[state_var]
                            body_id = str(uuid.uuid4())
                            elements[body_id] = {
                                "id": body_id,
                                "name": body_func,
                                "type": "StateBody",
                                "owner": state["id"],
                                "bounds": {
                                    "x": elements[state["id"]]["bounds"]["x"] + 0.5,
                                    "y": elements[state["id"]]["bounds"]["y"] + 40.5,
                                    "width": 159,
                                    "height": 30,
                                },
                            }
                            elements[state["id"]]["bodies"].append(body_id)

                    # Handle when_event_go_to
                    elif node.value.func.attr == "when_event_go_to":
                        source_state = node.value.func.value.id
                        rel_id = str(uuid.uuid4())

                        event_name = None
                        target_state = None
                        event_params = None

                        for kw in node.value.keywords:
                            if kw.arg == "event":
                                event_name = kw.value.id
                            elif kw.arg == "dest":
                                target_state = kw.value.id
                            elif kw.arg == "event_params":
                                event_params = ast.literal_eval(kw.value)

                        if source_state in states and target_state in states:
                            source_element = elements[states[source_state]["id"]]
                            target_element = elements[states[target_state]["id"]]

                            source_dir, target_dir = determine_connection_direction(
                                source_element["bounds"], target_element["bounds"]
                            )

                            source_point = calculate_connection_points(
                                source_element["bounds"], source_dir
                            )
                            target_point = calculate_connection_points(
                                target_element["bounds"], target_dir
                            )

                            path_points = calculate_path_points(
                                source_point, target_point, source_dir, target_dir
                            )
                            rel_bounds = calculate_relationship_bounds(path_points)

                            relationships[rel_id] = {
                                "id": rel_id,
                                "name": event_name,
                                "type": "StateTransition",
                                "owner": None,
                                "bounds": rel_bounds,
                                "path": path_points,
                                "source": {
                                    "direction": source_dir,
                                    "element": states[source_state]["id"],
                                    "bounds": {
                                        "x": source_point["x"],
                                        "y": source_point["y"],
                                        "width": 0,
                                        "height": 0,
                                    },
                                },
                                "target": {
                                    "direction": target_dir,
                                    "element": states[target_state]["id"],
                                    "bounds": {
                                        "x": target_point["x"],
                                        "y": target_point["y"],
                                        "width": 0,
                                        "height": 0,
                                    },
                                },
                                "isManuallyLayouted": False,
                            }

                            if event_params:
                                relationships[rel_id]["params"] = str(event_params)

                    # Add handling for fallback bodies
                    elif node.value.func.attr == "set_fallback_body":
                        state_var = node.value.func.value.id
                        fallback_func = node.value.args[0].id
                        if state_var in states and fallback_func in functions:
                            state = states[state_var]
                            fallback_id = str(uuid.uuid4())
                            elements[fallback_id] = {
                                "id": fallback_id,
                                "name": fallback_func,
                                "type": "StateFallbackBody",
                                "owner": state["id"],
                                "bounds": {
                                    "x": elements[state["id"]]["bounds"]["x"] + 0.5,
                                    "y": elements[state["id"]]["bounds"]["y"] + 70.5,
                                    "width": 159,
                                    "height": 30,
                                },
                            }
                            elements[state["id"]]["fallbackBodies"].append(fallback_id)

    return {
        "version": "3.0.0",
        "type": "StateMachineDiagram",
        "size": default_size,
        "interactive": {"elements": {}, "relationships": {}},
        "elements": elements,
        "relationships": relationships,
        "assessments": {},
    }



def agent_buml_to_json(content: str):
    """Convert an agent Python file content to JSON format matching the frontend structure."""

    elements = {}
    relationships = {}

    # Default diagram size
    default_size = {"width": 1980, "height": 640}

    # Track positions for layout
    states_x = -550
    states_y = -300

    # Parse the Python code
    tree = ast.parse(content)
    # Track states and functions
    states = {}  # name -> state_id mapping
    functions = {}  # name -> function_node mapping
    intents = {}  # name -> intent_id mapping
    state_machine_name = "Generated_State_Machine"
    try:

        # First pass: collect all intents
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Call):
                    if (
                        isinstance(node.value.func, ast.Attribute)
                        and node.value.func.attr == "new_intent"
                    ):
                        intent_id = str(uuid.uuid4())
                        intent_name = None
                        sentences = []

                        args = node.value.args
                        intent_name = ast.literal_eval(args[0])

                        if len(args) >= 2 and isinstance(args[1], ast.List):
                            for elt in args[1].elts:
                                if isinstance(elt, ast.Constant) and isinstance(
                                    elt.value, str
                                ):
                                    sentence_id = str(uuid.uuid4())
                                    elements[sentence_id] = {
                                        "id": sentence_id,
                                        "name": elt.value,
                                        "type": "AgentIntentBody",
                                        "owner": intent_id,
                                        "bounds": {
                                            "x": states_x,
                                            "y": states_y,
                                            "width": 160,
                                            "height": 30,
                                        },
                                    }
                                    sentences.append(sentence_id)

                        if intent_name:
                            intents[intent_name] = {
                                "id": intent_id,
                                "name": intent_name,
                            }

                            elements[intent_id] = {
                                "id": intent_id,
                                "name": intent_name,
                                "type": "AgentIntent",
                                "owner": None,
                                "bounds": {
                                    "x": states_x,
                                    "y": states_y,
                                    "width": 160,
                                    "height": 100,
                                },
                                "bodies": sentences,
                            }

                            if states_x < 200:
                                states_x += 300
                            else:
                                states_x = -280
                                states_y += 220
        # Second pass: collect all functions
        states_x = -280
        states_y += 220
        print("DEBUG: Collecting functions...")
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                function_source = ast.get_source_segment(content, node)
                functions[function_name] = {
                    "node": node,
                    "source": function_source,
                }
        
        # Create initial node
        initial_node_id = str(uuid.uuid4())
        elements[initial_node_id] = {
            "id": initial_node_id,
            "name": "",
            "type": "StateInitialNode",
            "owner": None,
            "bounds": {
                "x": states_x - 300,
                "y": states_y + 20,
                "width": 45,
                "height": 45,
            },
        }
        
        # Store the initial node ID for later use with transitions

        # Second pass: collect states and their configurations
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "new_state":
                    state_id = str(uuid.uuid4())
                    state_name = var_name  # Default to variable name
                    is_initial = False
                    
                    # Try to extract state name and initial flag from keywords
                    for kw in node.value.keywords:
                        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                            state_name = kw.value.value
                        elif kw.arg == "initial" and isinstance(kw.value, ast.Constant):
                            is_initial = kw.value.value
                    
                    # Create the state object
                    state_obj = {
                        "id": state_id,
                        "name": state_name,
                        "is_initial": is_initial,
                        "bodies": [],
                        "fallback_bodies": [],
                    }
                    
                    # Store by variable name for transitions
                    states[var_name] = state_obj
                    
                    # Also store by state name for body lookup
                    if state_name != var_name:
                        states[state_name] = state_obj

                    # Create element for visualization
                    elements[state_id] = {
                        "id": state_id,
                        "name": state_name,
                        "type": "AgentState",
                        "owner": None,
                        "bounds": {
                            "x": states_x,
                            "y": states_y,
                            "width": 160,
                            "height": 100,
                        },
                        "bodies": [],
                        "fallbackBodies": [],
                    }
                    
                    # Update position for next element
                    if states_x < 200:
                        states_x += 490
                    else:
                        states_x = -280
                        states_y += 220
        
        # Find initial state and create initial transition
        initial_state = None
        for state_key, state_info in states.items():
            if state_info["is_initial"]:
                initial_state = state_info
                break
        
        # If no initial state is marked, use the first state as fallback
        if not initial_state and states:
            # Get the first state
            first_state_key = next(iter(states))
            initial_state = states[first_state_key]
            
        if initial_state:
            initial_rel_id = str(uuid.uuid4())
            relationships[initial_rel_id] = {
                "id": initial_rel_id,
                "name": "",
                "type": "AgentStateTransitionInit",
                "owner": None,
                "source": {
                    "direction": "Right",
                    "element": initial_node_id,
                    "bounds": {
                        "x": elements[initial_node_id]["bounds"]["x"] + 45,
                        "y": elements[initial_node_id]["bounds"]["y"] + 22.5,
                        "width": 0,
                        "height": 0,
                    },
                },
                "target": {
                    "direction": "Left",
                    "element": initial_state["id"],
                    "bounds": {
                        "x": elements[initial_state["id"]]["bounds"]["x"],
                        "y": elements[initial_state["id"]]["bounds"]["y"] + 35,
                        "width": 0,
                        "height": 0,
                    },
                },
                "bounds": {
                    "x": elements[initial_node_id]["bounds"]["x"] + 45,
                    "y": elements[initial_node_id]["bounds"]["y"] + 22.5,
                    "width": elements[initial_state["id"]]["bounds"]["x"]
                    - (elements[initial_node_id]["bounds"]["x"] + 45),
                    "height": 1,
                },
                "path": [
                    {
                        "x": elements[initial_node_id]["bounds"]["x"] + 45,
                        "y": elements[initial_node_id]["bounds"]["y"] + 22.5,
                    },
                    {
                        "x": elements[initial_state["id"]]["bounds"]["x"],
                        "y": elements[initial_node_id]["bounds"]["y"] + 22.5,
                    },
                ],
                "isManuallyLayouted": False,
            }
                                
        # Third pass: process state bodies and transitions
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
            ):
                if (
                    isinstance(node.value.func.value, ast.Call)
                    and isinstance(node.value.func.value.func, ast.Attribute)
                    and node.value.func.value.func.attr
                    in [
                        "when_event_go_to",
                        "when_intent_matched",
                        "when_no_intent_matched",
                        "when_variable_matches_operation",
                        "when_file_received"
                    ]
                ):
                    source_state = node.value.func.value.func.value.id
                    rel_id = str(uuid.uuid4())

                    condition_name = node.value.func.value.func.attr
                    condition_value = ""
                    if condition_name == "when_intent_matched":
                        condition_value = node.value.func.value.args[0].id
                    elif condition_name == "when_file_received":
                        condition_value = node.value.func.value.args[0].value
                    elif condition_name == "when_variable_matches_operation":
                        condition_name = "when_variable_operation_matched"
                        condition_value = {}
                        for kw in node.value.func.value.keywords:
                            if kw.arg == "operation":
                                operator = kw.value.attr
                                operator_map = {
                                    "eq": "==",
                                    "lt": "<",
                                    "le": "<=",
                                    "ge": ">=",
                                    "gt": ">",
                                    "ne": "!=",
                                }
                                condition_value["operator"] = operator_map.get(operator, operator)
                            elif kw.arg == "var_name":
                                condition_value["variable"] = kw.value.value
                            elif kw.arg == "target":
                                condition_value["targetValue"] = kw.value.value
                    event_name = None
                    target_state = node.value.args[0].id
                    event_params = None

                    if source_state in states and target_state in states:
                        source_element = elements[states[source_state]["id"]]
                        target_element = elements[states[target_state]["id"]]

                        source_dir, target_dir = determine_connection_direction(
                            source_element["bounds"], target_element["bounds"]
                        )

                        source_point = calculate_connection_points(
                            source_element["bounds"], source_dir
                        )
                        target_point = calculate_connection_points(
                            target_element["bounds"], target_dir
                        )

                        path_points = calculate_path_points(
                            source_point, target_point, source_dir, target_dir
                        )
                        rel_bounds = calculate_relationship_bounds(path_points)

                        relationships[rel_id] = {
                            "id": rel_id,
                            "name": event_name,
                            "type": "AgentStateTransition",
                            "owner": None,
                            "bounds": rel_bounds,
                            "path": path_points,
                            "source": {
                                "direction": source_dir,
                                "element": states[source_state]["id"],
                                "bounds": {
                                    "x": source_point["x"],
                                    "y": source_point["y"],
                                    "width": 0,
                                    "height": 0,
                                },
                            },
                            "target": {
                                "direction": target_dir,
                                "element": states[target_state]["id"],
                                "bounds": {
                                    "x": target_point["x"],
                                    "y": target_point["y"],
                                    "width": 0,
                                    "height": 0,
                                },
                            },
                            "isManuallyLayouted": False,
                            "condition": condition_name,
                            "conditionValue": condition_value,
                        }

                        if event_params:
                            relationships[rel_id]["params"] = str(event_params)
            
                elif node.value.func.attr == "go_to":
                    source_state = node.value.func.value.id
                    rel_id = str(uuid.uuid4())

                    condition_name = "auto"
                    condition_value = ""
                    target_state = node.value.args[0].id
                    
                    if source_state in states and target_state in states:
                        source_element = elements[states[source_state]["id"]]
                        target_element = elements[states[target_state]["id"]]

                        source_dir, target_dir = determine_connection_direction(
                            source_element["bounds"], target_element["bounds"]
                        )

                        source_point = calculate_connection_points(
                            source_element["bounds"], source_dir
                        )
                        target_point = calculate_connection_points(
                            target_element["bounds"], target_dir
                        )

                        path_points = calculate_path_points(
                            source_point, target_point, source_dir, target_dir
                        )
                        rel_bounds = calculate_relationship_bounds(path_points)

                        relationships[rel_id] = {
                            "id": rel_id,
                            "name": event_name,
                            "type": "AgentStateTransition",
                            "owner": None,
                            "bounds": rel_bounds,
                            "path": path_points,
                            "source": {
                                "direction": source_dir,
                                "element": states[source_state]["id"],
                                "bounds": {
                                    "x": source_point["x"],
                                    "y": source_point["y"],
                                    "width": 0,
                                    "height": 0,
                                },
                            },
                            "target": {
                                "direction": target_dir,
                                "element": states[target_state]["id"],
                                "bounds": {
                                    "x": target_point["x"],
                                    "y": target_point["y"],
                                    "width": 0,
                                    "height": 0,
                                },
                            },
                            "isManuallyLayouted": False,
                            "condition": condition_name,
                            "conditionValue": condition_value,
                        }

                        if event_params:
                            relationships[rel_id]["params"] = str(event_params)
                
                # Handle set_body
                elif node.value.func.attr == "set_body":
                    try:
                        # Extract function name from Body('function_name', function_name) pattern
                        body_args = node.value.args[0].args
                        function_name = None
                        if len(body_args) >= 2:
                            if isinstance(body_args[1], ast.Name):
                                function_name = body_args[1].id
                            elif isinstance(body_args[0], ast.Constant) and isinstance(body_args[0].value, str):
                                function_name = body_args[0].value
                                
                        if not function_name:
                            continue

                        state_name = node.value.func.value.id
                        if state_name not in states:
                            continue
                            
                        state = states[state_name]
                        
                        if function_name in functions:
                            result = analyze_function_node(functions[function_name]["node"], functions[function_name]["source"])
                            if result["replyType"] == "text":
                                for reply in result["replies"]:
                                    body_id = str(uuid.uuid4())
                                    elements[body_id] = {
                                        "id": body_id,
                                        "name": reply,
                                        "type": "AgentStateBody",
                                        "owner": state["id"],
                                        "bounds": {
                                            "x": elements[state["id"]]["bounds"]["x"],
                                            "y": elements[state["id"]]["bounds"]["y"],
                                            "width": 159,
                                            "height": 30,
                                        },
                                        "replyType": "text"
                                    }
                                    elements[state["id"]]["bodies"].append(body_id)
                            elif result["replyType"] == "llm":
                                body_id = str(uuid.uuid4())
                                elements[body_id] = {
                                    "id": body_id,
                                    "name": "AI response 🪄",
                                    "type": "AgentStateBody",
                                    "owner": state["id"],
                                    "bounds": {
                                        "x": elements[state["id"]]["bounds"]["x"],
                                        "y": elements[state["id"]]["bounds"]["y"],
                                        "width": 159,
                                        "height": 30,
                                    },
                                    "replyType": "llm"
                                }
                                elements[state["id"]]["bodies"].append(body_id)
                            elif result["replyType"] == "code":
                                body_id = str(uuid.uuid4())
                                elements[body_id] = {
                                    "id": body_id,
                                    "name": result["code"],
                                    "type": "AgentStateBody",
                                    "owner": state["id"],
                                    "bounds": {
                                        "x": elements[state["id"]]["bounds"]["x"],
                                        "y": elements[state["id"]]["bounds"]["y"],
                                        "width": 159,
                                        "height": 30,
                                    },
                                    "replyType": "code"
                                }
                                elements[state["id"]]["bodies"].append(body_id)
                        else:
                            # Fallback if function not found
                            body_id = str(uuid.uuid4())
                            elements[body_id] = {
                                "id": body_id,
                                "name": function_name,
                                "type": "AgentStateBody",
                                "owner": state["id"],
                                "bounds": {
                                    "x": elements[state["id"]]["bounds"]["x"],
                                    "y": elements[state["id"]]["bounds"]["y"],
                                    "width": 159,
                                    "height": 30,
                                },
                            }
                            elements[state["id"]]["bodies"].append(body_id)
                    except Exception as e:
                        continue

                # Add handling for fallback bodies
                elif node.value.func.attr == "set_fallback_body":
                    try:
                        # Extract function name from Body('function_name', function_name) pattern
                        body_args = node.value.args[0].args
                        function_name = None
                        if len(body_args) >= 2:
                            if isinstance(body_args[1], ast.Name):
                                function_name = body_args[1].id
                            elif isinstance(body_args[0], ast.Constant) and isinstance(body_args[0].value, str):
                                function_name = body_args[0].value
                                
                        if not function_name:
                            continue

                        state_name = node.value.func.value.id
                        if state_name not in states:
                            continue
                            
                        state = states[state_name]
                        
                        if function_name in functions:
                            result = analyze_function_node(functions[function_name]["node"], functions[function_name]["source"])
                            if result["replyType"] == "text":
                                for reply in result["replies"]:
                                    body_id = str(uuid.uuid4())
                                    elements[body_id] = {
                                        "id": body_id,
                                        "name": reply,
                                        "type": "AgentStateFallbackBody",
                                        "owner": state["id"],
                                        "bounds": {
                                            "x": elements[state["id"]]["bounds"]["x"],
                                            "y": elements[state["id"]]["bounds"]["y"],
                                            "width": 159,
                                            "height": 30,
                                        },
                                        "replyType": "text"
                                    }
                                    elements[state["id"]]["fallbackBodies"].append(body_id)
                            elif result["replyType"] == "llm":
                                body_id = str(uuid.uuid4())
                                elements[body_id] = {
                                    "id": body_id,
                                    "name": "AI response 🪄",
                                    "type": "AgentStateFallbackBody",
                                    "owner": state["id"],
                                    "bounds": {
                                        "x": elements[state["id"]]["bounds"]["x"],
                                        "y": elements[state["id"]]["bounds"]["y"],
                                        "width": 159,
                                        "height": 30,
                                    },
                                    "replyType": "llm"
                                }
                                elements[state["id"]]["fallbackBodies"].append(body_id)
                            elif result["replyType"] == "code":
                                body_id = str(uuid.uuid4())
                                elements[body_id] = {
                                    "id": body_id,
                                    "name": result["code"],
                                    "type": "AgentStateFallbackBody",
                                    "owner": state["id"],
                                    "bounds": {
                                        "x": elements[state["id"]]["bounds"]["x"],
                                        "y": elements[state["id"]]["bounds"]["y"],
                                        "width": 159,
                                        "height": 30,
                                    },
                                    "replyType": "code"
                                }
                                elements[state["id"]]["fallbackBodies"].append(body_id)
                        else:
                            # Fallback if function not found
                            body_id = str(uuid.uuid4())
                            elements[body_id] = {
                                "id": body_id,
                                "name": function_name,
                                "type": "AgentStateFallbackBody",
                                "owner": state["id"],
                                "bounds": {
                                    "x": elements[state["id"]]["bounds"]["x"],
                                    "y": elements[state["id"]]["bounds"]["y"],
                                    "width": 159,
                                    "height": 30,
                                },
                            }
                            elements[state["id"]]["fallbackBodies"].append(body_id)
                    except Exception as e:
                        continue

        # Find initial state and create initial transition
        initial_state = None
        for state_key, state_info in states.items():
            if state_info["is_initial"]:
                initial_state = state_info
                break
        
        # If no initial state is marked, use the first state as fallback
        if not initial_state and states:
            # Get the first state
            first_state_key = next(iter(states))
            initial_state = states[first_state_key]
            
        if initial_state:
            initial_rel_id = str(uuid.uuid4())
            relationships[initial_rel_id] = {
                "id": initial_rel_id,
                "name": "",
                "type": "AgentStateTransitionInit",
                "owner": None,
                "source": {
                    "direction": "Right",
                    "element": initial_node_id,
                    "bounds": {
                        "x": elements[initial_node_id]["bounds"]["x"] + 45,
                        "y": elements[initial_node_id]["bounds"]["y"] + 22.5,
                        "width": 0,
                        "height": 0,
                    },
                },
                "target": {
                    "direction": "Left",
                    "element": initial_state["id"],
                    "bounds": {
                        "x": elements[initial_state["id"]]["bounds"]["x"],
                        "y": elements[initial_state["id"]]["bounds"]["y"] + 35,
                        "width": 0,
                        "height": 0,
                    },
                },
                "bounds": {
                    "x": elements[initial_node_id]["bounds"]["x"] + 45,
                    "y": elements[initial_node_id]["bounds"]["y"] + 22.5,
                    "width": elements[initial_state["id"]]["bounds"]["x"]
                    - (elements[initial_node_id]["bounds"]["x"] + 45),
                    "height": 1,
                },
                "path": [
                    {
                        "x": elements[initial_node_id]["bounds"]["x"] + 45,
                        "y": elements[initial_node_id]["bounds"]["y"] + 22.5,
                    },
                    {
                        "x": elements[initial_state["id"]]["bounds"]["x"],
                        "y": elements[initial_node_id]["bounds"]["y"] + 22.5,
                    },
                ],
                "isManuallyLayouted": False,
            }
                                
        return {
            "version": "3.0.0",
            "type": "AgentDiagram",
            "size": default_size,
            "interactive": {"elements": {}, "relationships": {}},
            "elements": elements,
            "relationships": relationships,
            "assessments": {},
        }

    except Exception as e:
        # Return an empty diagram on error
        return {
            "version": "3.0.0",
            "type": "AgentDiagram",
            "size": default_size,
            "interactive": {"elements": {}, "relationships": {}},
            "elements": elements,
            "relationships": relationships,
            "assessments": {},
        }

def analyze_function_node(node: ast.FunctionDef, source_code: str) -> dict:
    body = node.body

    # Case 1: Only session.reply("constant")
    replies = []
    if all(
        isinstance(stmt, ast.Expr) and
        isinstance(stmt.value, ast.Call) and
        isinstance(stmt.value.func, ast.Attribute) and
        stmt.value.func.attr == 'reply' and
        isinstance(stmt.value.func.value, ast.Name) and
        stmt.value.func.value.id == 'session' and
        len(stmt.value.args) == 1 and
        isinstance(stmt.value.args[0], ast.Constant) and
        isinstance(stmt.value.args[0].value, str)
        for stmt in body
    ):
        for stmt in body:
            replies.append(stmt.value.args[0].value)
        return {
            "replyType": "text",
            "replies": replies
        }

    # Case 2: One line session.reply(llm.predict(session.event.message))
    if len(body) == 1:
        stmt = body[0]
        if (
            isinstance(stmt, ast.Expr) and
            isinstance(stmt.value, ast.Call) and
            isinstance(stmt.value.func, ast.Attribute) and
            stmt.value.func.attr == 'reply' and
            isinstance(stmt.value.func.value, ast.Name) and
            stmt.value.func.value.id == 'session' and
            len(stmt.value.args) == 1
        ):
            arg = stmt.value.args[0]
            if (
                isinstance(arg, ast.Call) and
                isinstance(arg.func, ast.Attribute) and
                arg.func.attr == 'predict' and
                isinstance(arg.func.value, ast.Name) and
                arg.func.value.id == 'llm' and
                len(arg.args) == 1
            ):
                msg_arg = arg.args[0]
                if (
                    isinstance(msg_arg, ast.Attribute) and
                    msg_arg.attr == 'message' and
                    isinstance(msg_arg.value, ast.Attribute) and
                    msg_arg.value.attr == 'event' and
                    isinstance(msg_arg.value.value, ast.Name) and
                    msg_arg.value.value.id == 'session'
                ):
                    return {
                        "replyType": "llm"
                    }

    # Case 3: Default fallback
    return {
        "replyType": "code",
        "code": source_code
    }

def object_buml_to_json(content: str):
    """Convert an object model Python file content to JSON format matching the frontend structure."""
    import ast
    import uuid
    from besser.BUML.metamodel.object import ObjectModel, Object, Link
    
    elements = {}
    relationships = {}
    
    # Default diagram size
    default_size = {"width": 960, "height": 670}
    
    # Grid layout configuration for positioning objects
    grid_size = {
        "x_spacing": 250,
        "y_spacing": 180,
        "max_columns": 4,
    }
    
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
        # First, parse the domain model (class diagram) using parse_buml_content
        domain_model = parse_buml_content(content)
        reference_diagram_json = domain_model_to_json(domain_model)
        
        # Create mapping from class names to class IDs in reference diagram
        class_name_to_id = {}
        class_id_to_attributes = {}
        
        for elem_id, elem in reference_diagram_json["elements"].items():
            if elem["type"] == "Class":
                class_name_to_id[elem["name"]] = elem_id
                # Store class attributes for object attribute mapping
                class_attributes = {}
                for attr_id in elem.get("attributes", []):
                    if attr_id in reference_diagram_json["elements"]:
                        attr_elem = reference_diagram_json["elements"][attr_id]
                        # Extract attribute name (remove visibility and type info)
                        attr_name = attr_elem["name"].split(":")[0].strip().lstrip("+-#~")
                        class_attributes[attr_name] = attr_id
                class_id_to_attributes[elem_id] = class_attributes
        
        # Parse the Python code to extract object instances
        tree = ast.parse(content)
        
        # Track objects and their information
        objects_by_name = {}
        object_class_mapping = {}
        
        # Extract object instantiations using fluent API
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        
                        # Check if this is an object instantiation using fluent API
                        if isinstance(node.value, ast.Call):
                            # Look for pattern: ClassName("ObjectName").attributes(...).build()
                            call_chain = []
                            current = node.value
                            
                            # Walk back through the method chain
                            while isinstance(current, ast.Call):
                                if isinstance(current.func, ast.Attribute):
                                    call_chain.append(current.func.attr)
                                    current = current.func.value
                                elif isinstance(current.func, ast.Name):
                                    call_chain.append(current.func.id)
                                    break
                                else:
                                    break
                            
                            # Reverse to get the correct order
                            call_chain.reverse()
                            
                            # Check if this matches the fluent pattern
                            if len(call_chain) >= 3 and call_chain[-1] == "build":
                                class_name = call_chain[0]
                                if class_name in class_name_to_id:
                                    # Extract object name and attributes
                                    object_name = var_name
                                    object_instance_name = None
                                    attributes = {}
                                    
                                    # Get the initial call arguments for object name
                                    init_call = node.value
                                    while isinstance(init_call, ast.Call) and isinstance(init_call.func, ast.Attribute):
                                        init_call = init_call.func.value
                                    
                                    if isinstance(init_call, ast.Call) and len(init_call.args) > 0:
                                        if isinstance(init_call.args[0], ast.Constant):
                                            object_instance_name = init_call.args[0].value
                                    
                                    # Find attributes call in the chain
                                    attr_call = node.value
                                    while isinstance(attr_call, ast.Call):
                                        if isinstance(attr_call.func, ast.Attribute) and attr_call.func.attr == "attributes":
                                            # Extract attributes from keyword arguments
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
                                        "attributes": attributes
                                    }
                                    object_class_mapping[object_name] = class_name
        
        # Create object elements in JSON format
        for obj_name, obj_info in objects_by_name.items():
            x, y = get_position()
            object_id = str(uuid.uuid4())
            
            # Create object attribute elements
            object_attribute_ids = []
            class_id = obj_info["class_id"]
            class_attributes = class_id_to_attributes.get(class_id, {})
            
            attr_y_offset = 30
            for attr_name, attr_value in obj_info["attributes"].items():
                attr_id = str(uuid.uuid4())
                object_attribute_ids.append(attr_id)
                
                # Format the value appropriately
                if isinstance(attr_value, str):
                    formatted_value = attr_value
                else:
                    formatted_value = str(attr_value)
                
                # Get the corresponding class attribute ID if available
                class_attr_id = class_attributes.get(attr_name, None)
                
                elements[attr_id] = {
                    "id": attr_id,
                    "name": f"+ {attr_name}: str = {formatted_value}",
                    "type": "ObjectAttribute",
                    "owner": object_id,
                    "bounds": {
                        "x": x + 0.5,
                        "y": y + attr_y_offset - 0.5,
                        "width": 199,
                        "height": 30
                    },
                    "attributeId": class_attr_id
                }
                attr_y_offset += 30
            
            # Calculate object height based on attributes
            object_height = max(70, 40 + len(object_attribute_ids) * 30)
            
            elements[object_id] = {
                "id": object_id,
                "name": obj_info["instance_name"],
                "type": "ObjectName",
                "owner": None,
                "bounds": {
                    "x": x,
                    "y": y,
                    "width": 200,
                    "height": object_height
                },
                "attributes": object_attribute_ids,
                "methods": [],
                "classId": class_id
            }
        
        # Look for object links/relationships
        association_id_map = {}
        for rel_id, rel in reference_diagram_json["relationships"].items():
            if rel["type"] in ["ClassBidirectional", "ClassUnidirectional"]:
                source_role = rel.get("source", {}).get("role", "")
                target_role = rel.get("target", {}).get("role", "")
                if source_role or target_role:
                    association_id_map[f"{source_role}-{target_role}"] = rel_id
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Look for assignments like: obj1.relationship = obj2
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        obj_name = target.value.id if isinstance(target.value, ast.Name) else None
                        relationship_name = target.attr
                        
                        if obj_name in objects_by_name:
                            # Get the target object
                            if isinstance(node.value, ast.Name) and node.value.id in objects_by_name:
                                target_obj = node.value.id
                                
                                # Create a link relationship
                                rel_id = str(uuid.uuid4())
                                
                                # Find the object IDs
                                source_id = None
                                target_id = None
                                
                                for elem_id, elem in elements.items():
                                    if elem["type"] == "ObjectName" and elem["name"] == objects_by_name[obj_name]["instance_name"]:
                                        source_id = elem_id
                                    if elem["type"] == "ObjectName" and elem["name"] == objects_by_name[target_obj]["instance_name"]:
                                        target_id = elem_id
                                
                                if source_id and target_id:
                                    # Find corresponding association ID
                                    assoc_id = None
                                    for key, aid in association_id_map.items():
                                        if relationship_name in key:
                                            assoc_id = aid
                                            break
                                    
                                    relationships[rel_id] = {
                                        "id": rel_id,
                                        "name": f"{relationship_name}",
                                        "type": "ObjectLink",
                                        "owner": None,
                                        "bounds": {
                                            "x": -260,
                                            "y": -315,
                                            "width": 300,
                                            "height": 80
                                        },
                                        "path": [
                                            {"x": 0, "y": 80},
                                            {"x": 40, "y": 80},
                                            {"x": 40, "y": 0},
                                            {"x": 300, "y": 0},
                                            {"x": 300, "y": 65}
                                        ],
                                        "source": {
                                            "direction": "Right",
                                            "element": source_id
                                        },
                                        "target": {
                                            "direction": "Topleft",
                                            "element": target_id
                                        },
                                        "isManuallyLayouted": False,
                                        "associationId": assoc_id
                                    }

        return {
            "version": "3.0.0",
            "type": "ObjectDiagram",
            "size": default_size,
            "interactive": {"elements": {}, "relationships": {}},
            "elements": elements,
            "relationships": relationships,
            "assessments": {},
            "referenceDiagramData": reference_diagram_json
        }
  
    except Exception as e:
        print(f"Error parsing object BUML content: {str(e)}")
        return {
            "version": "3.0.0",
            "type": "ObjectDiagram", 
            "size": default_size,
            "interactive": {"elements": {}, "relationships": {}},
            "elements": {},
            "relationships": {},
            "assessments": {},
        }


def empty_model(diagram_type: str) -> dict:
    return {
        "version": "3.0.0",
        "type": diagram_type,
        "size": {"width": 1400, "height": 740},
        "elements": {},
        "relationships": {},
        "interactive": {"elements": {}, "relationships": {}},
        "assessments": {}
    }

def project_to_json(content: str) -> dict:
    """Convert a BUML project content to JSON format matching the frontend structure."""

    def extract_section(name: str, next_headers: list) -> str:
        pattern = rf"# {name.upper()} MODEL #(.*?)# ({'|'.join(next_headers)})"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()

        pattern = rf"# {name.upper()} MODEL #(.*?)(# PROJECT DEFINITION|$)"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""

    # Detect models included in the project
    model_match = re.search(r"models\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if not model_match:
        raise ValueError("No models defined in 'models=[...]'")

    model_names = re.findall(r'\b(\w+)\b', model_match.group(1))

    # Extract project metadata
    project_name = re.search(r'Project\s*\(\s*name\s*=\s*"(.*?)"', content)
    project_description = re.search(r'Metadata\s*\(\s*description\s*=\s*"(.*?)"', content)
    project_owner = re.search(r'owner\s*=\s*"(.*?)"', content)

    project_name = project_name.group(1) if project_name else "Unnamed Project"
    project_description = project_description.group(1) if project_description else "No description"
    project_owner = project_owner.group(1) if project_owner else "Unknown"

    section_extractors = {
        'domain_model': ('STRUCTURAL', ['OBJECT', 'AGENT', 'STATE MACHINE']),
        'object_model': ('OBJECT', ['AGENT', 'STATE MACHINE']),
        'agent': ('AGENT', ['STATE MACHINE']),
        'sm': ('STATE MACHINE', []),
    }

    diagram_jsons = {}

    # Extraemos primero el domain_model si existe (porque lo vamos a usar también para object_model)
    domain_code = ""
    if "domain_model" in model_names:
        domain_code = extract_section(*section_extractors["domain_model"])

    for model_name in model_names:
        if model_name not in section_extractors:
            continue

        header, next_headers = section_extractors[model_name]
        section_code = extract_section(header, next_headers)

        diagram_id = str(uuid.uuid4())
        last_update = datetime.now(UTC).isoformat()

        if model_name == "domain_model":
            parsed_domain_model = parse_buml_content(section_code)
            model = domain_model_to_json(parsed_domain_model)
            diagram_jsons["ClassDiagram"] = {
                "id": diagram_id,
                "title": "Class Diagram",
                "model": model,
                "lastUpdate": last_update
            }

        elif model_name == "object_model":
            combined_code = domain_code + "\n" + section_code
            model = object_buml_to_json(combined_code)
            diagram_jsons["ObjectDiagram"] = {
                "id": diagram_id,
                "title": "Object Diagram",
                "model": model,
                "lastUpdate": last_update
            }

        elif model_name == "agent":
            model = agent_buml_to_json(section_code)
            diagram_jsons["AgentDiagram"] = {
                "id": diagram_id,
                "title": "Agent Diagram",
                "model": model,
                "lastUpdate": last_update
            }

        elif model_name == "sm":
            model = state_machine_to_json(section_code)
            diagram_jsons["StateMachineDiagram"] = {
                "id": diagram_id,
                "title": "State Machine Diagram",
                "model": model,
                "lastUpdate": last_update
            }

    project_id = str(uuid.uuid4())
    created_at = datetime.now(UTC).isoformat()

    diagram_defaults = {
        "ClassDiagram": "ClassDiagram",
        "ObjectDiagram": "ObjectDiagram",
        "AgentDiagram": "AgentDiagram",
        "StateMachineDiagram": "StateMachineDiagram"
    }

    for diagram_type, model_type in diagram_defaults.items():
        if diagram_type not in diagram_jsons:
            diagram_jsons[diagram_type] = {
                "id": str(uuid.uuid4()),
                "title": diagram_type.replace("Diagram", " Diagram"),
                "model": empty_model(model_type),
                "lastUpdate": datetime.now(UTC).isoformat()
            }

    return {
        "id": project_id,
        "type": "Project",
        "name": project_name,
        "description": project_description,
        "owner": project_owner,
        "createdAt": created_at,
        "currentDiagramType": "ClassDiagram",
        "diagrams": diagram_jsons,
        "settings": {
            "defaultDiagramType": "ClassDiagram",
            "autoSave": True,
            "collaborationEnabled": False
        }
    }
