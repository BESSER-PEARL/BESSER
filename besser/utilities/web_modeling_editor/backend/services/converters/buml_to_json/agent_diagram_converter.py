"""
Agent converter module for BUML to JSON conversion.
Handles agent diagram processing and function analysis.
"""

import logging
import uuid
import ast
from typing import Dict, Any

logger = logging.getLogger(__name__)

from ...utils.layout_calculator import (
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
    # Map Python variable identifier -> declared intent/state name. Needed because
    # ``agent_model_builder`` emits lowercased identifiers (``muscles_intent``) that
    # bind to the original PascalCase name passed as the first positional arg
    # (``agent.new_intent('Muscles_intent', ...)``). Transitions reference the
    # variable, so without these maps we'd round-trip the slug back to the
    # frontend as the canonical intent/state name and break casing.
    intent_var_to_name = {}
    state_var_to_name = {}


    # Track metadata for comments
    state_comments = {}  # state_var -> comment_text
    agent_comment = None  # Agent metadata comment

    def _add_action_elements_to_state(state_id: str, action_data: Any, fallback: bool = False) -> None:
        element_type = "AgentStateFallbackBody" if fallback else "AgentStateBody"
        state_key = "fallbackBodies" if fallback else "bodies"

        if not isinstance(action_data, list):
            return

        for action in action_data:
            if not isinstance(action, dict):
                continue

            action_type = action.get("type")
            if action_type == "text":
                message = (action.get("message") or "").replace("\\'", "'")
                body_id = str(uuid.uuid4())
                elements[body_id] = {
                    "id": body_id,
                    "name": message,
                    "type": element_type,
                    "owner": state_id,
                    "bounds": {
                        "x": elements[state_id]["bounds"]["x"],
                        "y": elements[state_id]["bounds"]["y"],
                        "width": 159,
                        "height": 30,
                    },
                    "replyType": "text",
                }
                elements[state_id][state_key].append(body_id)
            elif action_type == "llm":
                body_id = str(uuid.uuid4())
                elements[body_id] = {
                    "id": body_id,
                    "name": "AI response 🪄",
                    "type": element_type,
                    "owner": state_id,
                    "bounds": {
                        "x": elements[state_id]["bounds"]["x"],
                        "y": elements[state_id]["bounds"]["y"],
                        "width": 159,
                        "height": 30,
                    },
                    "replyType": "llm",
                }
                elements[state_id][state_key].append(body_id)
            elif action_type == "rag":
                rag_db_name = action.get("ragDatabaseName") or ""
                display_name = (
                    f"RAG reply using {rag_db_name} database"
                    if rag_db_name
                    else "RAG reply"
                )
                body_id = str(uuid.uuid4())
                elements[body_id] = {
                    "id": body_id,
                    "name": display_name,
                    "type": element_type,
                    "owner": state_id,
                    "bounds": {
                        "x": elements[state_id]["bounds"]["x"],
                        "y": elements[state_id]["bounds"]["y"],
                        "width": 159,
                        "height": 30,
                    },
                    "replyType": "rag",
                    "ragDatabaseName": rag_db_name,
                }
                elements[state_id][state_key].append(body_id)
            elif action_type == "db_reply":
                db_selection_type = action.get("dbSelectionType") or "default"
                db_custom_name = action.get("dbCustomName") or ""
                db_query_mode = action.get("dbQueryMode") or "llm_query"
                db_operation = action.get("dbOperation") or "any"
                db_sql_query = action.get("dbSqlQuery") or ""
                database_label = db_custom_name if db_selection_type == "custom" and db_custom_name else "Default database"
                mode_label = "SQL" if db_query_mode == "sql" else "LLM query"
                operation_label = "Any" if db_operation == "any" else db_operation.upper()
                body_id = str(uuid.uuid4())
                elements[body_id] = {
                    "id": body_id,
                    "name": f"DB action using {database_label} ({mode_label}, {operation_label})",
                    "type": element_type,
                    "owner": state_id,
                    "bounds": {
                        "x": elements[state_id]["bounds"]["x"],
                        "y": elements[state_id]["bounds"]["y"],
                        "width": 159,
                        "height": 30,
                    },
                    "replyType": "db_reply",
                    "dbSelectionType": db_selection_type,
                    "dbCustomName": db_custom_name,
                    "dbQueryMode": db_query_mode,
                    "dbOperation": db_operation,
                    "dbSqlQuery": db_sql_query,
                }
                elements[state_id][state_key].append(body_id)

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
                            if (
                                node.targets
                                and isinstance(node.targets[0], ast.Name)
                            ):
                                intent_var_to_name[node.targets[0].id] = intent_name

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
                    elif (
                        isinstance(node.value.func, ast.Attribute)
                        and node.value.func.attr == "new_rag"
                    ):
                        rag_name = None
                        if (
                            node.value.args
                            and isinstance(node.value.args[0], ast.Constant)
                            and isinstance(node.value.args[0].value, str)
                        ):
                            rag_name = node.value.args[0].value

                        for kw in node.value.keywords:
                            if (
                                kw.arg == "name"
                                and isinstance(kw.value, ast.Constant)
                                and isinstance(kw.value.value, str)
                            ):
                                rag_name = kw.value.value

                        if isinstance(rag_name, str) and rag_name.strip():
                            rag_id = str(uuid.uuid4())
                            elements[rag_id] = {
                                "id": rag_id,
                                "name": rag_name,
                                "type": "AgentRagElement",
                                "owner": None,
                                "bounds": {
                                    "x": states_x,
                                    "y": states_y,
                                    "width": 120,
                                    "height": 110,
                                },
                            }
                            if states_x < 200:
                                states_x += 300
                            else:
                                states_x = -280
                                states_y += 220

        # Collect Tool/Skill/Workspace primitives. The model_builder emits
        # them as bare expression statements (``agent.new_tool(...)``), not
        # assignments, so this scan is separate from the Assign-based loop
        # above and walks every Call node.
        primitive_factories = {"new_tool", "new_skill", "new_workspace"}
        seen_primitive_calls: set = set()
        for call_node in ast.walk(tree):
            if not (
                isinstance(call_node, ast.Call)
                and isinstance(call_node.func, ast.Attribute)
                and call_node.func.attr in primitive_factories
            ):
                continue
            if id(call_node) in seen_primitive_calls:
                continue
            seen_primitive_calls.add(id(call_node))

            builder_attr = call_node.func.attr
            primitive_kwargs: Dict[str, Any] = {}
            for kw in call_node.keywords:
                try:
                    primitive_kwargs[kw.arg] = ast.literal_eval(kw.value)
                except (ValueError, SyntaxError):
                    continue
            positional = call_node.args or []
            if positional and "name" not in primitive_kwargs:
                try:
                    primitive_kwargs["name"] = ast.literal_eval(positional[0])
                except (ValueError, SyntaxError):
                    pass

            primitive_name = primitive_kwargs.get("name")
            if not isinstance(primitive_name, str) or not primitive_name.strip():
                continue

            primitive_id = str(uuid.uuid4())
            base_bounds = {
                "x": states_x,
                "y": states_y,
                "width": 160,
                "height": 80,
            }
            if builder_attr == "new_tool":
                primitive_element = {
                    "id": primitive_id,
                    "name": primitive_name,
                    "type": "AgentTool",
                    "owner": None,
                    "bounds": base_bounds,
                    "description": primitive_kwargs.get("description", "") or "",
                    "code": primitive_kwargs.get("code", "") or "",
                }
            elif builder_attr == "new_skill":
                primitive_element = {
                    "id": primitive_id,
                    "name": primitive_name,
                    "type": "AgentSkill",
                    "owner": None,
                    "bounds": base_bounds,
                    "content": primitive_kwargs.get("content", "") or "",
                    "description": primitive_kwargs.get("description"),
                }
            else:  # new_workspace
                primitive_element = {
                    "id": primitive_id,
                    "name": primitive_name,
                    "type": "AgentWorkspace",
                    "owner": None,
                    "bounds": base_bounds,
                    "path": primitive_kwargs.get("path", "") or "",
                    "description": primitive_kwargs.get("description"),
                    "writable": bool(primitive_kwargs.get("writable", True)),
                    "max_read_bytes": int(primitive_kwargs.get("max_read_bytes", 200_000)),
                }
            elements[primitive_id] = primitive_element
            if states_x < 200:
                states_x += 300
            else:
                states_x = -280
                states_y += 220

        # Second pass: collect all functions
        states_x = -280
        states_y += 220
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                function_source = ast.get_source_segment(content, node)
                functions[function_name] = {
                    "node": node,
                    "source": function_source,
                }
        custom_code_actions = {}
        custom_condition_callables = {}
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
                elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'Condition':
                    # Map condition object variable names (e.g., condition_6_1) to their callable function names.
                    callable_name = None
                    for kw in node.value.keywords:
                        if kw.arg == 'callable' and isinstance(kw.value, ast.Name):
                            callable_name = kw.value.id
                    if (
                        callable_name is None
                        and len(node.value.args) >= 2
                        and isinstance(node.value.args[1], ast.Name)
                    ):
                        callable_name = node.value.args[1].id
                    if callable_name:
                        custom_condition_callables[var_name] = callable_name

        def _resolve_condition_source(condition_ref_name: str) -> str:
            condition_source = functions.get(condition_ref_name, {}).get("source")
            if condition_source:
                return condition_source
            callable_name = custom_condition_callables.get(condition_ref_name)
            if callable_name:
                callable_source = functions.get(callable_name, {}).get("source")
                if callable_source:
                    return callable_source
            return condition_ref_name

        # Third pass collect all actions
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
                            actions[body_var] = [{"type": "text", "message": node.value.args[0].args[0].value}]
                        else:
                            actions[body_var].append({"type": "text", "message": node.value.args[0].args[0].value})
                    elif node.value.args[0].func.id == 'LLMReply':
                        if body_var not in actions:
                            actions[body_var] = [{"type": "llm"}]
                        else:
                            actions[body_var].append({"type": "llm"})
                    elif node.value.args[0].func.id == 'RAGReply':
                        rag_db_name = ""
                        if (
                            len(node.value.args[0].args) >= 1
                            and isinstance(node.value.args[0].args[0], ast.Constant)
                            and isinstance(node.value.args[0].args[0].value, str)
                        ):
                            rag_db_name = node.value.args[0].args[0].value
                        for kw in node.value.args[0].keywords:
                            if (
                                kw.arg == 'rag_db_name'
                                and isinstance(kw.value, ast.Constant)
                                and isinstance(kw.value.value, str)
                            ):
                                rag_db_name = kw.value.value

                        if body_var not in actions:
                            actions[body_var] = [{"type": "rag", "ragDatabaseName": rag_db_name}]
                        else:
                            actions[body_var].append({"type": "rag", "ragDatabaseName": rag_db_name})
                    elif node.value.args[0].func.id == 'DBReply':
                        db_action = {
                            "type": "db_reply",
                            "dbSelectionType": "default",
                            "dbCustomName": "",
                            "dbQueryMode": "llm_query",
                            "dbOperation": "any",
                            "dbSqlQuery": "",
                        }
                        for kw in node.value.args[0].keywords:
                            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                if kw.arg == 'db_selection_type':
                                    db_action["dbSelectionType"] = kw.value.value
                                elif kw.arg == 'db_custom_name':
                                    db_action["dbCustomName"] = kw.value.value
                                elif kw.arg == 'db_query_mode':
                                    db_action["dbQueryMode"] = kw.value.value
                                elif kw.arg == 'db_operation':
                                    db_action["dbOperation"] = kw.value.value
                                elif kw.arg == 'db_sql_query':
                                    db_action["dbSqlQuery"] = kw.value.value

                        if body_var not in actions:
                            actions[body_var] = [db_action]
                        else:
                            actions[body_var].append(db_action)
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


        # Build a var-name -> LLM name map so reasoning states can resolve
        # their ``llm=...`` kwarg back to the registered LLM name.
        llm_var_to_name: Dict[str, str] = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id.startswith("LLM")
            ):
                resolved_name = None
                for kw in node.value.keywords:
                    if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        resolved_name = kw.value.value
                        break
                if resolved_name:
                    llm_var_to_name[node.targets[0].id] = resolved_name

        # Second pass: collect states and their configurations
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                if (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "new_reasoning_state"
                ):
                    state_id = str(uuid.uuid4())
                    state_name = var_name
                    is_initial = False
                    rs_kwargs: Dict[str, Any] = {}
                    llm_var_ref = None
                    llm_literal_name: str | None = None

                    if (
                        node.value.args
                        and isinstance(node.value.args[0], ast.Constant)
                        and isinstance(node.value.args[0].value, str)
                    ):
                        state_name = node.value.args[0].value

                    for kw in node.value.keywords:
                        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                            state_name = kw.value.value
                        elif kw.arg == "initial" and isinstance(kw.value, ast.Constant):
                            is_initial = bool(kw.value.value)
                        elif kw.arg == "llm":
                            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                llm_literal_name = kw.value.value
                            elif isinstance(kw.value, ast.Name):
                                llm_var_ref = kw.value.id
                        else:
                            try:
                                rs_kwargs[kw.arg] = ast.literal_eval(kw.value)
                            except (ValueError, SyntaxError):
                                continue

                    state_var_to_name[var_name] = state_name

                    state_obj = {
                        "id": state_id,
                        "name": state_name,
                        "is_initial": is_initial,
                        "bodies": [],
                        "fallback_bodies": [],
                    }
                    states[var_name] = state_obj
                    if state_name != var_name:
                        states[state_name] = state_obj

                    if llm_literal_name is not None:
                        llm_name_resolved = llm_literal_name
                    elif llm_var_ref:
                        llm_name_resolved = llm_var_to_name.get(llm_var_ref)
                    else:
                        llm_name_resolved = None

                    elements[state_id] = {
                        "id": state_id,
                        "name": state_name,
                        "type": "AgentReasoningState",
                        "owner": None,
                        "bounds": {
                            "x": states_x,
                            "y": states_y,
                            "width": 200,
                            "height": 110,
                        },
                        "llm_name": llm_name_resolved,
                        "max_steps": int(rs_kwargs.get("max_steps", 8)),
                        "enable_task_planning": bool(rs_kwargs.get("enable_task_planning", True)),
                        "stream_steps": bool(rs_kwargs.get("stream_steps", True)),
                        "system_prompt": rs_kwargs.get("system_prompt"),
                        "fallback_message": rs_kwargs.get("fallback_message"),
                    }

                    if states_x < 200:
                        states_x += 490
                    else:
                        states_x = -280
                        states_y += 220
                    continue

                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "new_state":
                    state_id = str(uuid.uuid4())
                    state_name = var_name  # Default to variable name
                    is_initial = False

                    # ``agent.new_state('Idle')`` passes the name as a positional
                    # arg; the builder emits this form, so check positional args
                    # before falling back to the variable name. Without this,
                    # round-trips lose the original casing because the builder's
                    # ``safe_var_name`` lowercases identifiers (``Idle`` -> ``idle``).
                    if (
                        node.value.args
                        and isinstance(node.value.args[0], ast.Constant)
                        and isinstance(node.value.args[0].value, str)
                    ):
                        state_name = node.value.args[0].value

                    # Try to extract state name and initial flag from keywords
                    for kw in node.value.keywords:
                        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                            state_name = kw.value.value
                        elif kw.arg == "initial" and isinstance(kw.value, ast.Constant):
                            is_initial = kw.value.value

                    state_var_to_name[var_name] = state_name

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

        for node in ast.walk(tree):
            try:
                if (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                ):
                    if node.value.func.attr == "go_to":
                        source_state = node.value.func.value.id if isinstance(node.value.func.value, ast.Name) else None
                        rel_id = str(uuid.uuid4())
                        condition_name = "auto"
                        transition_payload = ""
                        event_name = None
                        target_state = node.value.args[0].id if node.value.args and isinstance(node.value.args[0], ast.Name) else None

                        call_chain = node.value.func.value
                        custom_conditions: list[str] = []

                        while isinstance(call_chain, ast.Call) and isinstance(call_chain.func, ast.Attribute):
                            chain_attr = call_chain.func.attr

                            if chain_attr == "with_condition":
                                if call_chain.args and isinstance(call_chain.args[0], ast.Name):
                                    condition_func_name = call_chain.args[0].id
                                    custom_conditions.insert(0, _resolve_condition_source(condition_func_name))
                                call_chain = call_chain.func.value
                                continue

                            if isinstance(call_chain.func.value, ast.Name):
                                source_state = call_chain.func.value.id

                            if chain_attr == "when_intent_matched":
                                condition_name = "when_intent_matched"
                                if call_chain.args and isinstance(call_chain.args[0], ast.Name):
                                    intent_var = call_chain.args[0].id
                                    # Resolve the variable back to its declared
                                    # intent name. The builder emits lowercased
                                    # variable identifiers, so without this lookup
                                    # the round-tripped intentName would not match
                                    # the intent definition's name and downstream
                                    # generation would emit an undefined reference.
                                    transition_payload = intent_var_to_name.get(intent_var, intent_var)
                            elif chain_attr == "when_no_intent_matched":
                                condition_name = "when_no_intent_matched"
                            elif chain_attr == "when_variable_matches_operation":
                                condition_name = "when_variable_operation_matched"
                                transition_payload = {}
                                for kw in call_chain.keywords:
                                    if kw.arg == "operation" and isinstance(kw.value, ast.Attribute):
                                        operator_name = kw.value.attr
                                        operator_map = {
                                            "eq": "==",
                                            "lt": "<",
                                            "le": "<=",
                                            "ge": ">=",
                                            "gt": ">",
                                            "ne": "!=",
                                        }
                                        transition_payload["operator"] = operator_map.get(operator_name, operator_name)
                                    elif kw.arg == "var_name" and isinstance(kw.value, ast.Constant):
                                        transition_payload["variable"] = kw.value.value
                                    elif kw.arg == "target" and isinstance(kw.value, ast.Constant):
                                        transition_payload["targetValue"] = kw.value.value
                            elif chain_attr == "when_file_received":
                                condition_name = "when_file_received"
                                if call_chain.args and isinstance(call_chain.args[0], ast.Constant):
                                    transition_payload = call_chain.args[0].value
                            elif chain_attr == "when_event":
                                condition_name = "custom_transition"
                                selected_event = "None"
                                if call_chain.args:
                                    event_arg = call_chain.args[0]
                                    if isinstance(event_arg, ast.Call) and isinstance(event_arg.func, ast.Name):
                                        event_name = event_arg.func.id
                                        selected_event = event_name
                                    elif isinstance(event_arg, ast.Name):
                                        event_name = event_arg.id
                                        selected_event = event_name

                                transition_payload = {
                                    "event": selected_event,
                                    "conditions": custom_conditions,
                                }
                            elif chain_attr == "when_condition":
                                condition_name = "custom_transition"
                                if call_chain.args and isinstance(call_chain.args[0], ast.Name):
                                    condition_func_name = call_chain.args[0].id
                                    custom_conditions.insert(0, _resolve_condition_source(condition_func_name))

                                transition_payload = {
                                    "event": "None",
                                    "conditions": custom_conditions,
                                }

                            break

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

                            transition_type = "custom" if condition_name == "custom_transition" else "predefined"
                            predefined_block = {
                                "predefinedType": condition_name if transition_type == "predefined" else "",
                                "conditionValue": transition_payload if transition_type == "predefined" else "",
                            }
                            if transition_type == "predefined" and condition_name == "when_intent_matched":
                                predefined_block["intentName"] = transition_payload if isinstance(transition_payload, str) else ""
                                predefined_block.pop("conditionValue", None)
                            elif transition_type == "predefined" and condition_name == "when_file_received":
                                predefined_block["fileType"] = transition_payload if isinstance(transition_payload, str) else ""
                                predefined_block.pop("conditionValue", None)
                            custom_block = {
                                "event": (event_name or "None") if transition_type == "custom" else "None",
                                "condition": (
                                    transition_payload.get("conditions", [])
                                    if transition_type == "custom" and isinstance(transition_payload, dict)
                                    else []
                                ),
                            }

                            relationships[rel_id] = {
                                "id": rel_id,
                                "name": event_name or "",
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
                                "transitionType": transition_type,
                                "predefined": predefined_block,
                                "custom": custom_block,
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
                                elif isinstance(actions[function_name], list):
                                    _add_action_elements_to_state(state["id"], actions[function_name], fallback=False)
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
                            logger.error("Error processing agent body: %s", e, exc_info=True)
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

                            mapped_action = actions.get(function_name)
                            if function_name in functions or (isinstance(mapped_action, str) and mapped_action in functions):
                                if isinstance(mapped_action, str) and mapped_action in functions:
                                    function_name = mapped_action
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
                                elif isinstance(actions[function_name], list):
                                    _add_action_elements_to_state(state["id"], actions[function_name], fallback=True)
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
                            logger.warning("Error processing agent fallback body: %s", e, exc_info=True)
                            continue
            except Exception as e:
                logger.error("Error processing agent state machine: %s", e, exc_info=True)
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

    except Exception:
        logger.exception("Error converting agent BUML to JSON; returning partial diagram")
        return {
            "version": "3.0.0",
            "type": "AgentDiagram",
            "size": default_size,
            "interactive": {"elements": {}, "relationships": {}},
            "elements": elements,
            "relationships": relationships,
            "assessments": {},
        }
