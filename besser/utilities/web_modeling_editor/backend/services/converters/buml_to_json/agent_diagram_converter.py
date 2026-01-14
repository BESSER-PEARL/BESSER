"""
Agent converter module for BUML to JSON conversion.
Handles agent diagram processing and function analysis.
"""

import uuid
import ast
from typing import Dict, Any, List

from ...utils.layout_calculator import (
    calculate_center_point,
    determine_connection_direction,
    calculate_connection_points,
    calculate_path_points,
    calculate_relationship_bounds,
)


def analyze_function_node(node: ast.FunctionDef, source_code: str) -> Dict[str, Any]:
    """
    Analyze a function node to determine its reply type and content.
    
    Args:
        node: AST function definition node
        source_code: Source code of the function
        
    Returns:
        Dictionary with reply type and content information
    """
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


def agent_buml_to_json(content: str) -> Dict[str, Any]:
    """
    Convert an agent Python file content to JSON format matching the frontend structure.
    
    Args:
        content: Agent model Python code as string
        
    Returns:
        Dictionary representing the agent diagram in JSON format
    """
    # Initialize structures
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

    
    # Track metadata for comments
    state_comments = {}  # state_var -> comment_text
    agent_comment = None  # Agent metadata comment
    
    try:
        # First pass: collect all intents and Agent metadata
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Call):
                    # Check for Agent instantiation with metadata
                    if isinstance(node.value.func, ast.Name) and node.value.func.id == "Agent":
                        for kw in node.value.keywords:
                            if kw.arg == "metadata":
                                if isinstance(kw.value, ast.Call):
                                    for meta_kw in kw.value.keywords:
                                        if meta_kw.arg == "description":
                                            agent_comment = ast.literal_eval(meta_kw.value)
                    elif (
                        isinstance(node.value.func, ast.Attribute)
                        and node.value.func.attr == "new_intent"
                    ):
                        intent_id = str(uuid.uuid4())
                        intent_name = None
                        sentences = []
                        intent_description = ""

                        args = node.value.args
                        intent_name = ast.literal_eval(args[0])

                        for kw in node.value.keywords:
                            if kw.arg == "description":
                                intent_description = ast.literal_eval(kw.value)

                        if len(args) >= 2 and isinstance(args[1], ast.List):
                            for elt in args[1].elts:
                                if isinstance(elt, ast.Constant) and isinstance(
                                    elt.value, str
                                ):
                                    sentence_id = str(uuid.uuid4())
                                    elements[sentence_id] = {
                                        "id": sentence_id,
                                        "name": elt.value.replace("\\'", "'"),
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
                                "intent_description": intent_description,
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
        custom_code_actions = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'CustomCodeAction':
                    # Check for 'callable' keyword argument
                    callable_name = None
                    for kw in node.value.keywords:
                        if kw.arg == 'callable' and isinstance(kw.value, ast.Name):
                            callable_name = kw.value.id
                    if callable_name:
                        custom_code_actions[var_name] = callable_name  # e.g., 'CustomCodeAction_initial' -> 'action_name'
        
        # Third pass collect all actions
        print("DEBUG: Collecting actions...")
        actions = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Expr) and
                isinstance(node.value, ast.Call) and
                isinstance(node.value.func, ast.Attribute) and
                node.value.func.attr == 'add_action' and
                isinstance(node.value.func.value, ast.Name) and
                len(node.value.args) == 1
                ):
                body_var = node.value.func.value.id  # e.g., 'initial_body'
                if (
                    isinstance(node.value.args[0], ast.Call) and
                    isinstance(node.value.args[0].func, ast.Name)
                    ):
                    if (node.value.args[0].func.id == 'AgentReply' and
                        len(node.value.args[0].args) == 1 and
                        isinstance(node.value.args[0].args[0], ast.Constant) and
                        isinstance(node.value.args[0].args[0].value, str)
                        ):
                        if body_var not in actions:
                            actions[body_var] = [node.value.args[0].args[0].value]
                        else:
                            actions[body_var].append(node.value.args[0].args[0].value)
                    elif node.value.args[0].func.id == 'LLMReply':
                        actions[body_var] = 'LLMReply'
                elif isinstance(node.value.args[0], ast.Name):
                    # Handle references to CustomCodeAction variables
                    action_var = node.value.args[0].id  # e.g., 'CustomCodeAction_initial'
                    if action_var in custom_code_actions:
                        function_name = custom_code_actions[action_var]  # e.g., 'action_name'
                        actions[body_var] = function_name  # Store the resolved function name
        
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

        print("DEBUG: Processing states and transitions...")
        
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
            # Check for state.metadata = Metadata(...) patterns
            elif isinstance(node, ast.Assign):
                if len(node.targets) == 1:
                    target = node.targets[0]
                    if isinstance(target, ast.Attribute) and target.attr == "metadata":
                        state_var = target.value.id if isinstance(target.value, ast.Name) else None
                        if state_var and isinstance(node.value, ast.Call):
                            for kw in node.value.keywords:
                                if kw.arg == "description":
                                    state_comments[state_var] = ast.literal_eval(kw.value)
                    
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
        print("DEBUG: Processing state bodies and transitions...")
        
        for node in ast.walk(tree):
            try:
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
                            "when_file_received",
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
                                "name": "",
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

                    
                    # Handle set_body
                    elif node.value.func.attr == "set_body":
                        try:
                            function_name = None
                            # Extract function name from Body('function_name', function_name) pattern
                            if isinstance(node.value.args[0], ast.Name):
                                function_name = node.value.args[0].id
                            else:
                                body_args = node.value.args[0].args
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
                            
                            if function_name in functions or (isinstance(actions.get(function_name), str) and actions.get(function_name) in functions):
                                if (isinstance(actions.get(function_name), str) and actions.get(function_name) in functions):
                                    function_name = actions[function_name]
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
                            elif function_name in actions:
                                if actions[function_name] == 'LLMReply':
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
                                    
                                else:
                                    for message in actions[function_name]:
                                        message = message.replace("\\'", "'")
                                        body_id = str(uuid.uuid4())
                                        elements[body_id] = {
                                            "id": body_id,
                                            "name": message,
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
                            import traceback
                            traceback.print_exc()
                            continue

                    # Add handling for fallback bodies
                    elif node.value.func.attr == "set_fallback_body":
                        try:
                            # Extract function name from Body('function_name', function_name) pattern
                            
                            function_name = None
                            # Extract function name from Body('function_name', function_name) pattern
                            if isinstance(node.value.args[0], ast.Name):
                                function_name = node.value.args[0].id
                            else:
                                body_args = node.value.args[0].args
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
                            
                            if function_name in functions or actions[function_name] in functions:
                                if actions[function_name] in functions:
                                    function_name = actions[function_name]
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
                            
                            elif function_name in actions:
                                if actions[function_name] == 'LLMReply':
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
                                else:
                                    body_id = str(uuid.uuid4())
                                    elements[body_id] = {
                                        "id": body_id,
                                        "name": actions[function_name].replace("\\'", "'"),
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
            except Exception as e:
                import traceback
                traceback.print_exc()
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

        # Position for comments
        comment_x = -970
        comment_y = -300

        # Create comment elements from metadata
        # 1. Agent comment (unlinked)
        if agent_comment:
            comment_id = str(uuid.uuid4())
            elements[comment_id] = {
                "id": comment_id,
                "name": agent_comment,
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
