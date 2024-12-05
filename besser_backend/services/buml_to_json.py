import uuid
from besser.BUML.metamodel.structural import (
    Class, Property, Method, DomainModel, PrimitiveDataType,
    Enumeration, EnumerationLiteral, BinaryAssociation, Generalization, 
    Multiplicity, UNLIMITED_MAX_MULTIPLICITY
)
from besser_backend.utils.constants import VISIBILITY_MAP, RELATIONSHIP_TYPES
from besser_backend.services.layout_calculator import calculate_center_point, determine_connection_direction, calculate_connection_points, calculate_path_points, calculate_relationship_bounds
import inspect
from besser.BUML.metamodel.state_machine import (
    StateMachine, Session, Body, Event
)


def parse_buml_content(content: str) -> DomainModel:
    """Parse BUML content from a Python file and return a DomainModel."""
    try:
        # Create a safe environment for eval
        safe_globals = {
            'Class': Class,
            'Property': Property,
            'Method': Method,
            'PrimitiveDataType': PrimitiveDataType,
            'BinaryAssociation': BinaryAssociation,
            'Multiplicity': Multiplicity,
            'UNLIMITED_MAX_MULTIPLICITY': UNLIMITED_MAX_MULTIPLICITY,
            'Generalization': Generalization,
            'Enumeration': Enumeration,
            'EnumerationLiteral': EnumerationLiteral,
            'set': set,
            'StringType': PrimitiveDataType("str"),
            'IntegerType': PrimitiveDataType("int"),
            'DateType': PrimitiveDataType("date"),
            # Add mock generators that do nothing
            'PythonGenerator': lambda model: type('MockGenerator', (), {'generate': lambda: None}),
            'DjangoGenerator': lambda model: type('MockGenerator', (), {'generate': lambda: None}),
            'SQLAlchemyGenerator': lambda model: type('MockGenerator', (), {'generate': lambda: None}),
            'SQLGenerator': lambda model: type('MockGenerator', (), {'generate': lambda: None}),
            'RESTAPIGenerator': lambda model: type('MockGenerator', (), {'generate': lambda: None}),
            'BackendGenerator': lambda model, **kwargs: type('MockGenerator', (), {'generate': lambda: None}),
            'RDFGenerator': lambda model: type('MockGenerator', (), {'generate': lambda: None})
        }
        
        # Create a new domain model
        domain_model = DomainModel("Generated Model")
        
        # Execute the BUML content in a safe environment
        local_vars = {}
        exec(content, safe_globals, local_vars)
        
        #print("Local variables after execution:", local_vars.keys())
        
        # First pass: Add all classes and enumerations
        classes = {}
        for var_name, var_value in local_vars.items():
            if isinstance(var_value, (Class, Enumeration)):
                #print(f"Found type: {var_name} = {var_value}")
                domain_model.types.add(var_value)
                classes[var_name] = var_value
        
        # Second pass: Add associations and generalizations
        for var_name, var_value in local_vars.items():
            if isinstance(var_value, BinaryAssociation):
                #print(f"Found association: {var_name} = {var_value}")
                #print(f"Association ends: {var_value.ends}")
                domain_model.associations.add(var_value)
            elif isinstance(var_value, Generalization):
                #print(f"Found generalization: {var_name} = {var_value}")
                domain_model.generalizations.add(var_value)
        
        #print(f"Domain model types: {domain_model.types}")
        #print(f"Domain model associations: {domain_model.associations}")
        
        return domain_model
            
    except Exception as e:
        print(f"Error parsing BUML content: {e}")
        raise ValueError(f"Failed to parse BUML content: {str(e)}")
    
def domain_model_to_json(domain_model):
    """Convert a BUML DomainModel object to JSON format matching the frontend structure."""
    elements = {}
    relationships = {}
    
    # Default diagram size
    default_size = {
        "width": 1200,
        "height": 800  # Increased height for better visibility
    }
    
    # Grid layout configuration
    grid_size = {
        "x_spacing": 300,  # Space between elements horizontally
        "y_spacing": 200,  # Space between elements vertically
        "max_columns": 3   # Maximum elements per row
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
    
    for type_obj in domain_model.types:
        if isinstance(type_obj, (Class, Enumeration)):
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
                    visibility_symbol = next(k for k, v in VISIBILITY_MAP.items() if v == attr.visibility)
                    attr_type = attr.type.name if hasattr(attr.type, 'name') else str(attr.type)
                    
                    elements[attr_id] = {
                        "id": attr_id,
                        "name": f"{visibility_symbol} {attr.name}: {attr_type}",
                        "type": "ClassAttribute",
                        "owner": element_id,
                        "bounds": {
                            "x": x + 0.5,
                            "y": y_offset,
                            "width": 159,
                            "height": 30
                        }
                    }
                    attribute_ids.append(attr_id)
                    y_offset += 30

                # Process methods
                for method in type_obj.methods:
                    method_id = str(uuid.uuid4())
                    visibility_symbol = next(k for k, v in VISIBILITY_MAP.items() if v == method.visibility)
                    
                    # Build method signature with parameters and return type
                    param_str = []
                    for param in method.parameters:
                        param_type = param.type.name if hasattr(param.type, 'name') else str(param.type)
                        param_signature = f"{param.name}: {param_type}"
                        if hasattr(param, 'default_value') and param.default_value is not None:
                            param_signature += f" = {param.default_value}"
                        param_str.append(param_signature)
                    
                    # Build complete method signature
                    method_signature = f"{visibility_symbol} {method.name}({', '.join(param_str)})"
                    if hasattr(method, 'type') and method.type:
                        return_type = method.type.name if hasattr(method.type, 'name') else str(method.type)
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
                            "height": 30
                        }
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
                            "height": 30
                        }
                    }
                    attribute_ids.append(literal_id)
                    y_offset += 30

            # Create the element
            elements[element_id] = {
                "id": element_id,
                "name": type_obj.name,
                "type": "Enumeration" if isinstance(type_obj, Enumeration) else 
                       "AbstractClass" if type_obj.is_abstract else "Class",
                "owner": None,
                "bounds": {
                    "x": x,
                    "y": y,
                    "width": 160,
                    "height": max(100, 30 * (len(attribute_ids) + len(method_ids) + 1))
                },
                "attributes": attribute_ids,
                "methods": method_ids,
                "stereotype": "enumeration" if isinstance(type_obj, Enumeration) else None
            }

    # Second pass: Create relationships
    for association in domain_model.associations:
        rel_id = str(uuid.uuid4())
        ends = list(association.ends)
        if len(ends) == 2:
            source_prop, target_prop = ends
            source_class = source_prop.type
            target_class = target_prop.type
            
            if source_class in class_id_map and target_class in class_id_map:
                # Get source and target elements
                source_element = elements[class_id_map[source_class]]
                target_element = elements[class_id_map[target_class]]
                
                # Calculate connection directions and points
                source_dir, target_dir = determine_connection_direction(
                    source_element['bounds'], 
                    target_element['bounds']
                )
                
                source_point = calculate_connection_points(source_element['bounds'], source_dir)
                target_point = calculate_connection_points(target_element['bounds'], target_dir)
                
                # Calculate path points
                path_points = calculate_path_points(source_point, target_point, source_dir, target_dir)
                
                # Calculate bounds
                rel_bounds = calculate_relationship_bounds(path_points)
                
                # Determine relationship type
                rel_type = RELATIONSHIP_TYPES["composition"] if source_prop.is_composite else (
                    RELATIONSHIP_TYPES["bidirectional"] if source_prop.is_navigable and target_prop.is_navigable
                    else RELATIONSHIP_TYPES["unidirectional"]
                )
                
                relationships[rel_id] = {
                    "id": rel_id,
                    "type": rel_type,
                    "source": {
                        "element": class_id_map[source_class],
                        "multiplicity": f"{source_prop.multiplicity.min}..{'*' if source_prop.multiplicity.max == 9999 else source_prop.multiplicity.max}",
                        "role": source_prop.name,
                        "direction": source_dir,
                        "bounds": {
                            "x": source_point['x'],
                            "y": source_point['y'],
                            "width": 0,
                            "height": 0
                        }
                    },
                    "target": {
                        "element": class_id_map[target_class],
                        "multiplicity": f"{target_prop.multiplicity.min}..{'*' if target_prop.multiplicity.max == 9999 else target_prop.multiplicity.max}",
                        "role": target_prop.name,
                        "direction": target_dir,
                        "bounds": {
                            "x": target_point['x'],
                            "y": target_point['y'],
                            "width": 0,
                            "height": 0
                        }
                    },
                    "bounds": rel_bounds,
                    "path": path_points,
                    "isManuallyLayouted": False
                }

    # Handle generalizations
    for generalization in domain_model.generalizations:
        rel_id = str(uuid.uuid4())
        if generalization.general in class_id_map and generalization.specific in class_id_map:
            relationships[rel_id] = {
                "id": rel_id,
                "type": "ClassInheritance",
                "source": {
                    "element": class_id_map[generalization.specific],
                    "bounds": {
                        "x": 0,
                        "y": 0,
                        "width": 0,
                        "height": 0
                    }
                },
                "target": {
                    "element": class_id_map[generalization.general],
                    "bounds": {
                        "x": 0,
                        "y": 0,
                        "width": 0,
                        "height": 0
                    }
                },
                "path": [
                    {"x": 0, "y": 0},
                    {"x": 50, "y": 0},
                    {"x": 50, "y": 50},
                    {"x": 100, "y": 50}
                ]
            }

    # Create the final structure
    result = {
        "version": "3.0.0",
        "type": "ClassDiagram",
        "size": default_size,
        "interactive": {
            "elements": {},
            "relationships": {}
        },
        "elements": elements,
        "relationships": relationships,
        "assessments": {}
    }

    return result

def state_machine_to_json(content: str):
    """Convert a state machine Python file content to JSON format matching the frontend structure."""
    import ast
    
    elements = {}
    relationships = {}
    
    # Default diagram size
    default_size = {
        "width": 1980,
        "height": 640
    }
    
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
    state_machine_name = "Generated State Machine"
    
    # First pass: collect all functions and state machine name
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = {
                'node': node,
                'source': ast.get_source_segment(content, node)
            }
        elif isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name) and node.value.func.id == 'StateMachine':
                    for kw in node.value.keywords:
                        if kw.arg == 'name':
                            state_machine_name = ast.literal_eval(kw.value)
    
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
            "height": 45
        }
    }
    
    # Second pass: collect states and their configurations
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'new_state':
                    state_id = str(uuid.uuid4())
                    state_name = None
                    is_initial = False
                    
                    for kw in node.value.keywords:
                        if kw.arg == 'name':
                            state_name = ast.literal_eval(kw.value)
                        elif kw.arg == 'initial':
                            is_initial = ast.literal_eval(kw.value)
                    
                    if state_name:
                        states[node.targets[0].id] = {
                            'id': state_id,
                            'name': state_name,
                            'is_initial': is_initial,
                            'bodies': [],
                            'fallback_bodies': []
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
                                "height": 100
                            },
                            "bodies": [],
                            "fallbackBodies": []
                        }
                        
                        if states_x < 200:
                            states_x += 490
                        else:
                            states_x = -280
                            states_y += 220
    
    # After creating all states, add initial node transition
    for state_info in states.values():
        if state_info['is_initial']:
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
                        "x": elements[initial_node_id]['bounds']['x'] + 45,
                        "y": elements[initial_node_id]['bounds']['y'] + 22.5,
                        "width": 0,
                        "height": 0
                    }
                },
                "target": {
                    "direction": "Left",
                    "element": state_info['id'],
                    "bounds": {
                        "x": elements[state_info['id']]['bounds']['x'],
                        "y": elements[state_info['id']]['bounds']['y'] + 35,
                        "width": 0,
                        "height": 0
                    }
                },
                "bounds": {
                    "x": elements[initial_node_id]['bounds']['x'] + 45,
                    "y": elements[initial_node_id]['bounds']['y'] + 22.5,
                    "width": elements[state_info['id']]['bounds']['x'] - (elements[initial_node_id]['bounds']['x'] + 45),
                    "height": 1
                },
                "path": [
                    {"x": elements[initial_node_id]['bounds']['x'] + 45, "y": elements[initial_node_id]['bounds']['y'] + 22.5},
                    {"x": elements[state_info['id']]['bounds']['x'], "y": elements[initial_node_id]['bounds']['y'] + 22.5}
                ],
                "isManuallyLayouted": False
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
                )
                
                elements[code_block_id] = {
                    "id": code_block_id,
                    "name": "",
                    "type": "StateCodeBlock",
                    "owner": None,
                    "bounds": {
                        "x": code_blocks_x,
                        "y": code_blocks_y,
                        "width": 580,
                        "height": 200
                    },
                    "text": cleaned_source,
                    "language": "python",
                    "code": {
                        "content": cleaned_source,
                        "language": "python",
                        "version": "1.0"
                    }
                }
                created_code_blocks[function_name] = code_block_id
                code_blocks_x += 610
    
    # Third pass: process state bodies and transitions
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Attribute):
                    # Handle set_body
                    if node.value.func.attr == 'set_body':
                        state_var = node.value.func.value.id
                        body_func = node.value.keywords[0].value.id
                        if state_var in states and body_func in functions:
                            state = states[state_var]
                            body_id = str(uuid.uuid4())
                            elements[body_id] = {
                                "id": body_id,
                                "name": body_func,
                                "type": "StateBody",
                                "owner": state['id'],
                                "bounds": {
                                    "x": elements[state['id']]['bounds']['x'] + 0.5,
                                    "y": elements[state['id']]['bounds']['y'] + 40.5,
                                    "width": 159,
                                    "height": 30
                                }
                            }
                            elements[state['id']]["bodies"].append(body_id)
                            
                            # Create code block
                            code_block_id = str(uuid.uuid4())
                            function_source = functions[body_func]['source']
                            
                            # Clean up the source code and preserve all content
                            cleaned_source = "\n".join(
                                line.rstrip() 
                                for line in function_source.splitlines()
                            )
                            
                            elements[code_block_id] = {
                                "id": code_block_id,
                                "name": "",
                                "type": "StateCodeBlock",
                                "owner": None,
                                "bounds": {
                                    "x": code_blocks_x,
                                    "y": code_blocks_y,
                                    "width": 580,
                                    "height": 200
                                },
                                "text": cleaned_source,
                                "language": "python",
                                "code": {
                                    "content": cleaned_source,
                                    "language": "python",
                                    "version": "1.0"
                                }
                            }
                            code_blocks_x += 610
                    
                    # Handle when_event_go_to
                    elif node.value.func.attr == 'when_event_go_to':
                        source_state = node.value.func.value.id
                        rel_id = str(uuid.uuid4())
                        
                        event_name = None
                        target_state = None
                        event_params = None
                        
                        for kw in node.value.keywords:
                            if kw.arg == 'event':
                                event_name = kw.value.id
                            elif kw.arg == 'dest':
                                target_state = kw.value.id
                            elif kw.arg == 'event_params':
                                event_params = ast.literal_eval(kw.value)
                        
                        if source_state in states and target_state in states:
                            source_element = elements[states[source_state]['id']]
                            target_element = elements[states[target_state]['id']]
                            
                            source_dir, target_dir = determine_connection_direction(
                                source_element['bounds'],
                                target_element['bounds']
                            )
                            
                            source_point = calculate_connection_points(source_element['bounds'], source_dir)
                            target_point = calculate_connection_points(target_element['bounds'], target_dir)
                            
                            path_points = calculate_path_points(source_point, target_point, source_dir, target_dir)
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
                                    "element": states[source_state]['id'],
                                    "bounds": {
                                        "x": source_point['x'],
                                        "y": source_point['y'],
                                        "width": 0,
                                        "height": 0
                                    }
                                },
                                "target": {
                                    "direction": target_dir,
                                    "element": states[target_state]['id'],
                                    "bounds": {
                                        "x": target_point['x'],
                                        "y": target_point['y'],
                                        "width": 0,
                                        "height": 0
                                    }
                                },
                                "isManuallyLayouted": False
                            }
                            
                            if event_params:
                                relationships[rel_id]["params"] = str(event_params)
                    
                    # Add handling for fallback bodies
                    elif node.value.func.attr == 'set_fallback_body':
                        state_var = node.value.func.value.id
                        fallback_func = node.value.args[0].id
                        if state_var in states and fallback_func in functions:
                            state = states[state_var]
                            fallback_id = str(uuid.uuid4())
                            elements[fallback_id] = {
                                "id": fallback_id,
                                "name": fallback_func,
                                "type": "StateFallbackBody",
                                "owner": state['id'],
                                "bounds": {
                                    "x": elements[state['id']]['bounds']['x'] + 0.5,
                                    "y": elements[state['id']]['bounds']['y'] + 70.5,
                                    "width": 159,
                                    "height": 30
                                }
                            }
                            elements[state['id']]["fallbackBodies"].append(fallback_id)
    
    return {
        "version": "3.0.0",
        "type": "StateMachineDiagram",
        "size": default_size,
        "interactive": {
            "elements": {},
            "relationships": {}
        },
        "elements": elements,
        "relationships": relationships,
        "assessments": {}
    }