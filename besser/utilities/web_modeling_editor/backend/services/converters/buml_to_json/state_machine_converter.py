"""
State machine conversion from BUML to JSON format.

Provides two entry points:
- ``state_machine_object_to_json(sm)`` - converts a StateMachine metamodel object directly
- ``state_machine_to_json(content)`` - legacy: AST-parses a Python code string (for file import)
"""

import uuid
import re
import ast
from besser.BUML.metamodel.state_machine.state_machine import (
    StateMachine, CustomCodeAction,
)
from besser.utilities.web_modeling_editor.backend.services.utils import (
    determine_connection_direction, calculate_connection_points,
    calculate_path_points, calculate_relationship_bounds
)


def _sanitize_identifier(name: str) -> str:
    """Sanitize a string to be a valid Python identifier."""
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    sanitized = re.sub(r'^[^a-zA-Z_]+', '', sanitized)
    return sanitized or 'unnamed'


def state_machine_object_to_json(sm: StateMachine) -> dict:
    """Convert a StateMachine metamodel instance to JSON format matching the frontend structure.

    This is the preferred converter that works directly from the metamodel object,
    avoiding the need to generate and re-parse Python code strings.

    Args:
        sm: A StateMachine metamodel instance.

    Returns:
        Dictionary in the frontend JSON format for state machine diagrams.
    """
    elements = {}
    relationships = {}

    default_size = {"width": 1980, "height": 640}

    # Track positions for layout
    states_x = -550
    states_y = -300
    code_blocks_x = -970
    code_blocks_y = 80

    # Create initial node
    initial_node_id = str(uuid.uuid4())
    elements[initial_node_id] = {
        "id": initial_node_id,
        "name": "",
        "type": "StateInitialNode",
        "owner": None,
        "bounds": {"x": states_x - 300, "y": states_y + 20, "width": 45, "height": 45},
    }

    # Collect all unique Body, Event, and Condition objects for code blocks
    all_bodies = {}  # name -> Body
    all_events = {}  # name -> Event
    all_conditions = {}  # name -> Condition
    state_id_map = {}  # State object -> element_id
    state_var_map = {}  # State object -> variable name (sanitized)

    for state in sm.states:
        if state.body and state.body.name not in all_bodies:
            all_bodies[state.body.name] = state.body
        if state.fallback_body and state.fallback_body.name not in all_bodies:
            all_bodies[state.fallback_body.name] = state.fallback_body
        for transition in state.transitions:
            if transition.event and transition.event.name not in all_events:
                all_events[transition.event.name] = transition.event
            for condition in transition.conditions:
                if condition.name not in all_conditions:
                    all_conditions[condition.name] = condition

    # Create code block elements for functions
    created_code_blocks = {}  # function name -> {"id": ..., "source": ...}

    # Process bodies
    for name, body in all_bodies.items():
        if name in created_code_blocks:
            continue
        source_code = ""
        for action in body.actions:
            if isinstance(action, CustomCodeAction):
                source_code = action.code
                break
        if not source_code and body.code:
            source_code = body.code

        if source_code:
            code_block_id = str(uuid.uuid4())
            cleaned_source = "\n".join(
                line.rstrip() for line in source_code.splitlines() if line.strip()
            )
            elements[code_block_id] = {
                "id": code_block_id,
                "name": name,
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
            created_code_blocks[name] = {"id": code_block_id, "source": cleaned_source}
            code_blocks_x += 610

    # Process events
    for name, event in all_events.items():
        if name in created_code_blocks:
            continue
        source_code = getattr(event, '_source_code', "")
        if source_code:
            code_block_id = str(uuid.uuid4())
            cleaned_source = "\n".join(
                line.rstrip() for line in source_code.splitlines() if line.strip()
            )
            elements[code_block_id] = {
                "id": code_block_id,
                "name": name,
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
            created_code_blocks[name] = {"id": code_block_id, "source": cleaned_source}
            code_blocks_x += 610

    # Process conditions (guard expressions backed by code blocks).
    # Only create code blocks for function-backed conditions (code starts with "def ").
    # Inline expression guards (e.g. "x > 10") are stored directly in the
    # transition's "guard" field without a separate code block.
    for name, condition in all_conditions.items():
        if name in created_code_blocks:
            continue
        source_code = getattr(condition, 'code', "") or ""
        if source_code and source_code.lstrip().startswith("def "):
            code_block_id = str(uuid.uuid4())
            cleaned_source = "\n".join(
                line.rstrip() for line in source_code.splitlines() if line.strip()
            )
            elements[code_block_id] = {
                "id": code_block_id,
                "name": name,
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
            created_code_blocks[name] = {"id": code_block_id, "source": cleaned_source}
            code_blocks_x += 610

    # Create state elements
    for state in sm.states:
        state_id = str(uuid.uuid4())
        state_id_map[state] = state_id
        state_var_map[state] = _sanitize_identifier(state.name)

        elements[state_id] = {
            "id": state_id,
            "name": state.name,
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

    # Add initial node transition
    for state in sm.states:
        if state.initial:
            state_id = state_id_map[state]
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
                    "element": state_id,
                    "bounds": {
                        "x": elements[state_id]["bounds"]["x"],
                        "y": elements[state_id]["bounds"]["y"] + 35,
                        "width": 0,
                        "height": 0,
                    },
                },
                "bounds": {
                    "x": elements[initial_node_id]["bounds"]["x"] + 45,
                    "y": elements[initial_node_id]["bounds"]["y"] + 22.5,
                    "width": elements[state_id]["bounds"]["x"]
                    - (elements[initial_node_id]["bounds"]["x"] + 45),
                    "height": 1,
                },
                "path": [
                    {
                        "x": elements[initial_node_id]["bounds"]["x"] + 45,
                        "y": elements[initial_node_id]["bounds"]["y"] + 22.5,
                    },
                    {
                        "x": elements[state_id]["bounds"]["x"],
                        "y": elements[initial_node_id]["bounds"]["y"] + 22.5,
                    },
                ],
                "isManuallyLayouted": False,
            }
            break  # Only one initial state

    # Process state bodies and fallback bodies
    for state in sm.states:
        state_id = state_id_map[state]

        if state.body and state.body.name in created_code_blocks:
            body_id = str(uuid.uuid4())
            elements[body_id] = {
                "id": body_id,
                "name": state.body.name,
                "type": "StateBody",
                "owner": state_id,
                "bounds": {
                    "x": elements[state_id]["bounds"]["x"] + 0.5,
                    "y": elements[state_id]["bounds"]["y"] + 40.5,
                    "width": 159,
                    "height": 30,
                },
            }
            elements[state_id]["bodies"].append(body_id)

        if state.fallback_body and state.fallback_body.name in created_code_blocks:
            fallback_id = str(uuid.uuid4())
            elements[fallback_id] = {
                "id": fallback_id,
                "name": state.fallback_body.name,
                "type": "StateFallbackBody",
                "owner": state_id,
                "bounds": {
                    "x": elements[state_id]["bounds"]["x"] + 0.5,
                    "y": elements[state_id]["bounds"]["y"] + 70.5,
                    "width": 159,
                    "height": 30,
                },
            }
            elements[state_id]["fallbackBodies"].append(fallback_id)

    # Create transitions
    for state in sm.states:
        source_id = state_id_map[state]
        for transition in state.transitions:
            # Skip transitions that have neither event nor conditions (auto transitions
            # without guards are not representable in the JSON format)
            if not transition.event and not transition.conditions:
                continue

            dest_id = state_id_map.get(transition.dest)
            if not dest_id:
                continue

            rel_id = str(uuid.uuid4())
            source_element = elements[source_id]
            target_element = elements[dest_id]

            source_dir, target_dir = determine_connection_direction(
                source_element["bounds"], target_element["bounds"]
            )
            source_point = calculate_connection_points(source_element["bounds"], source_dir)
            target_point = calculate_connection_points(target_element["bounds"], target_dir)
            path_points = calculate_path_points(source_point, target_point, source_dir, target_dir)
            rel_bounds = calculate_relationship_bounds(path_points)

            event_var = _sanitize_identifier(transition.event.name) if transition.event else ""

            relationships[rel_id] = {
                "id": rel_id,
                "name": event_var,
                "type": "StateTransition",
                "owner": None,
                "bounds": rel_bounds,
                "path": path_points,
                "source": {
                    "direction": source_dir,
                    "element": source_id,
                    "bounds": {
                        "x": source_point["x"],
                        "y": source_point["y"],
                        "width": 0,
                        "height": 0,
                    },
                },
                "target": {
                    "direction": target_dir,
                    "element": dest_id,
                    "bounds": {
                        "x": target_point["x"],
                        "y": target_point["y"],
                        "width": 0,
                        "height": 0,
                    },
                },
                "isManuallyLayouted": False,
            }

            # Add event_params if present
            event_params = getattr(transition, '_event_params', None)
            if event_params:
                relationships[rel_id]["params"] = str(event_params)

            # Add guard expression if conditions are present.
            # If the condition name matches a created code block, use the name
            # so the round-trip can re-associate it.  Otherwise, use the code
            # text directly (inline guard expression).
            if transition.conditions:
                cond = transition.conditions[0]  # primary guard
                if cond.name in created_code_blocks:
                    guard_value = cond.name
                elif cond.code:
                    guard_value = cond.code
                else:
                    guard_value = cond.name
                relationships[rel_id]["guard"] = guard_value

    # Position for comments
    comment_x = -970
    comment_y = -300

    # Create comment elements from metadata
    if sm.metadata and sm.metadata.description:
        comment_id = str(uuid.uuid4())
        elements[comment_id] = {
            "id": comment_id,
            "name": sm.metadata.description,
            "type": "Comments",
            "owner": None,
            "bounds": {
                "x": comment_x,
                "y": comment_y,
                "width": 200,
                "height": 100,
            },
        }
        comment_y += 130

    # State comments (linked to states)
    for state in sm.states:
        if state.metadata and state.metadata.description:
            comment_id = str(uuid.uuid4())
            state_id = state_id_map[state]

            elements[comment_id] = {
                "id": comment_id,
                "name": state.metadata.description,
                "type": "Comments",
                "owner": None,
                "bounds": {
                    "x": comment_x,
                    "y": comment_y,
                    "width": 200,
                    "height": 100,
                },
            }

            link_id = str(uuid.uuid4())
            source_element = elements[comment_id]
            target_element = elements[state_id]

            source_dir, target_dir = determine_connection_direction(
                source_element["bounds"], target_element["bounds"]
            )
            source_point = calculate_connection_points(source_element["bounds"], source_dir)
            target_point = calculate_connection_points(target_element["bounds"], target_dir)
            path_points = calculate_path_points(source_point, target_point, source_dir, target_dir)
            rel_bounds = calculate_relationship_bounds(path_points)

            relationships[link_id] = {
                "id": link_id,
                "name": "",
                "type": "Link",
                "owner": None,
                "bounds": rel_bounds,
                "path": path_points,
                "source": {
                    "direction": source_dir,
                    "element": comment_id,
                    "bounds": {
                        "x": source_point["x"],
                        "y": source_point["y"],
                        "width": 0,
                        "height": 0,
                    },
                },
                "target": {
                    "direction": target_dir,
                    "element": state_id,
                    "bounds": {
                        "x": target_point["x"],
                        "y": target_point["y"],
                        "width": 0,
                        "height": 0,
                    },
                },
                "isManuallyLayouted": False,
            }

            comment_y += 130

    v3_result = {
        "version": "3.0.0",
        "type": "StateMachineDiagram",
        "size": default_size,
        "interactive": {"elements": {}, "relationships": {}},
        "elements": elements,
        "relationships": relationships,
        "assessments": {},
    }
    from besser.utilities.web_modeling_editor.backend.services.converters._shape_normalizer import (
        v3_to_v4_model,
    )
    return v3_to_v4_model(
        v3_result, diagram_type="StateMachineDiagram",
        title=getattr(sm, "name", "") or "",
    )


def state_machine_to_json(content: str):
    """Convert a state machine Python file content to JSON format matching the frontend structure.

    This is the legacy entry point that AST-parses a Python code string. It is still
    used for file-import scenarios (e.g. ``/get-json-model`` endpoint). For converting
    a StateMachine metamodel object, use ``state_machine_object_to_json`` instead.
    """

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

    # Track metadata for comments
    state_comments = {}  # state_var -> comment_text
    sm_comment = None  # StateMachine metadata comment

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
                        if kw.arg == "metadata":
                            # Extract StateMachine metadata
                            if isinstance(kw.value, ast.Call):
                                for meta_kw in kw.value.keywords:
                                    if meta_kw.arg == "description":
                                        sm_comment = ast.literal_eval(meta_kw.value)

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
                    is_final = False

                    for kw in node.value.keywords:
                        if kw.arg == "name":
                            state_name = ast.literal_eval(kw.value)
                        elif kw.arg == "initial":
                            is_initial = ast.literal_eval(kw.value)
                        elif kw.arg == "final":
                            is_final = ast.literal_eval(kw.value)

                    if state_name:
                        states[node.targets[0].id] = {
                            "id": state_id,
                            "name": state_name,
                            "is_initial": is_initial,
                            "is_final": is_final,
                            "bodies": [],
                            "fallback_bodies": [],
                        }

                        # Always create as a regular State element
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

    # Create final nodes and transitions for final states
    for state_info in states.values():
        if state_info["is_final"]:
            final_node_id = str(uuid.uuid4())
            state_id = state_info["id"]
            state_bounds = elements[state_id]["bounds"]

            # Position final node to the right of the state
            final_node_x = state_bounds["x"] + state_bounds["width"] + 100
            final_node_y = state_bounds["y"] + (state_bounds["height"] / 2) - 22.5

            elements[final_node_id] = {
                "id": final_node_id,
                "name": "",
                "type": "StateFinalNode",
                "owner": None,
                "bounds": {
                    "x": final_node_x,
                    "y": final_node_y,
                    "width": 45,
                    "height": 45,
                },
            }

            # Create transition from state to final node
            source_point = {
                "x": state_bounds["x"] + state_bounds["width"],
                "y": state_bounds["y"] + state_bounds["height"] / 2,
            }
            target_point = {
                "x": final_node_x,
                "y": final_node_y + 22.5,
            }

            final_rel_id = str(uuid.uuid4())
            relationships[final_rel_id] = {
                "id": final_rel_id,
                "name": "",
                "type": "StateTransition",
                "owner": None,
                "source": {
                    "direction": "Right",
                    "element": state_id,
                    "bounds": {
                        "x": source_point["x"],
                        "y": source_point["y"],
                        "width": 0,
                        "height": 0,
                    },
                },
                "target": {
                    "direction": "Left",
                    "element": final_node_id,
                    "bounds": {
                        "x": target_point["x"],
                        "y": target_point["y"],
                        "width": 0,
                        "height": 0,
                    },
                },
                "bounds": {
                    "x": source_point["x"],
                    "y": source_point["y"],
                    "width": target_point["x"] - source_point["x"],
                    "height": 1,
                },
                "path": [source_point, target_point],
                "isManuallyLayouted": False,
            }

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

                    # Handle when_event(...).go_to(...)
                    elif (
                        node.value.func.attr == "go_to"
                        and isinstance(node.value.func.value, ast.Call)
                        and isinstance(node.value.func.value.func, ast.Attribute)
                        and node.value.func.value.func.attr == "when_event"
                    ):
                        source_state = node.value.func.value.func.value.id
                        rel_id = str(uuid.uuid4())

                        event_name = None
                        target_state = node.value.args[0].id if node.value.args and isinstance(node.value.args[0], ast.Name) else None
                        event_params = None

                        if node.value.func.value.args:
                            event_arg = node.value.func.value.args[0]
                            if isinstance(event_arg, ast.Name):
                                event_name = event_arg.id
                            elif isinstance(event_arg, ast.Call) and isinstance(event_arg.func, ast.Name):
                                event_name = event_arg.func.id

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

                    # Handle state metadata
                    elif node.value.func.attr == "metadata":
                        # This is an assignment like: state_var.metadata = Metadata(...)
                        pass  # Will be handled in Assign nodes

        elif isinstance(node, ast.Assign):
            # Check for state.metadata = Metadata(...) patterns
            if len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Attribute) and target.attr == "metadata":
                    state_var = target.value.id if isinstance(target.value, ast.Name) else None
                    if state_var and isinstance(node.value, ast.Call):
                        for kw in node.value.keywords:
                            if kw.arg == "description":
                                state_comments[state_var] = ast.literal_eval(kw.value)

    # Position for comments
    comment_x = -970
    comment_y = -300

    # Create comment elements from metadata
    # 1. State machine comment (unlinked)
    if sm_comment:
        comment_id = str(uuid.uuid4())
        elements[comment_id] = {
            "id": comment_id,
            "name": sm_comment,
            "type": "Comments",
            "owner": None,
            "bounds": {
                "x": comment_x,
                "y": comment_y,
                "width": 200,
                "height": 100,
            },
        }
        comment_y += 130

    # 2. State comments (linked to states)
    for state_var, comment_text in state_comments.items():
        if state_var in states:
            comment_id = str(uuid.uuid4())
            state_id = states[state_var]["id"]

            elements[comment_id] = {
                "id": comment_id,
                "name": comment_text,
                "type": "Comments",
                "owner": None,
                "bounds": {
                    "x": comment_x,
                    "y": comment_y,
                    "width": 200,
                    "height": 100,
                },
            }

            # Create Link relationship
            link_id = str(uuid.uuid4())
            source_element = elements[comment_id]
            target_element = elements[state_id]

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

            relationships[link_id] = {
                "id": link_id,
                "name": "",
                "type": "Link",
                "owner": None,
                "bounds": rel_bounds,
                "path": path_points,
                "source": {
                    "direction": source_dir,
                    "element": comment_id,
                    "bounds": {
                        "x": source_point["x"],
                        "y": source_point["y"],
                        "width": 0,
                        "height": 0,
                    },
                },
                "target": {
                    "direction": target_dir,
                    "element": state_id,
                    "bounds": {
                        "x": target_point["x"],
                        "y": target_point["y"],
                        "width": 0,
                        "height": 0,
                    },
                },
                "isManuallyLayouted": False,
            }

            comment_y += 130

    v3_result = {
        "version": "3.0.0",
        "type": "StateMachineDiagram",
        "size": default_size,
        "interactive": {"elements": {}, "relationships": {}},
        "elements": elements,
        "relationships": relationships,
        "assessments": {},
    }
    from besser.utilities.web_modeling_editor.backend.services.converters._shape_normalizer import (
        v3_to_v4_model,
    )
    return v3_to_v4_model(v3_result, diagram_type="StateMachineDiagram", title="")
