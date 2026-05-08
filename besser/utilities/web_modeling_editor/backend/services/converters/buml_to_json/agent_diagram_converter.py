"""
Agent BUML -> v4 JSON converter.

Emits the v4 wire shape (``{nodes, edges}``) directly. ``AgentStateTransition``
edges are emitted in the canonical form (``transitionType +
predefined|custom``) — legacy variants are an *input* concern handled by
the agent processor's ``_normalise_agent_transitions``.
"""

import logging
import uuid
import ast
from typing import Dict, Any

logger = logging.getLogger(__name__)

from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json._node_builders import (
    make_node, make_edge,
)


def analyze_function_node(node: ast.FunctionDef, source_code: str) -> Dict[str, Any]:
    """Analyze a function node to determine its reply type and content."""
    body = node.body

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
        return {"replyType": "text", "replies": replies}

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
                    return {"replyType": "llm"}

    return {"replyType": "code", "code": source_code}


def _make_body_row(name: str, reply_type: str, **extras) -> dict:
    row: dict = {"id": str(uuid.uuid4()), "name": name, "replyType": reply_type}
    for k, v in extras.items():
        if v is not None:
            row[k] = v
    return row


def agent_buml_to_json(content: str) -> Dict[str, Any]:
    """Convert an agent Python file to the v4 ``{nodes, edges}`` wire shape."""
    nodes: list = []
    edges: list = []

    states_x = -550
    states_y = -300

    tree = ast.parse(content)
    states: dict = {}
    functions: dict = {}
    intents: dict = {}
    intent_var_to_name: dict = {}
    state_var_to_name: dict = {}
    state_comments: dict = {}
    agent_comment = None

    intent_id_by_name: dict = {}

    def _bodies_for_action(action_data, state_node_id: str, fallback: bool = False) -> list:
        result: list = []
        if not isinstance(action_data, list):
            return result
        for action in action_data:
            if not isinstance(action, dict):
                continue
            t = action.get("type")
            if t == "text":
                msg = (action.get("message") or "").replace("\\'", "'")
                result.append(_make_body_row(msg, "text"))
            elif t == "llm":
                result.append(_make_body_row("AI response 🪄", "llm"))
            elif t == "rag":
                rag_db_name = action.get("ragDatabaseName") or ""
                display_name = (
                    f"RAG reply using {rag_db_name} database"
                    if rag_db_name
                    else "RAG reply"
                )
                result.append(_make_body_row(display_name, "rag", ragDatabaseName=rag_db_name))
            elif t == "db_reply":
                db_selection_type = action.get("dbSelectionType") or "default"
                db_custom_name = action.get("dbCustomName") or ""
                db_query_mode = action.get("dbQueryMode") or "llm_query"
                db_operation = action.get("dbOperation") or "any"
                db_sql_query = action.get("dbSqlQuery") or ""
                database_label = db_custom_name if db_selection_type == "custom" and db_custom_name else "Default database"
                mode_label = "SQL" if db_query_mode == "sql" else "LLM query"
                operation_label = "Any" if db_operation == "any" else db_operation.upper()
                result.append(_make_body_row(
                    f"DB action using {database_label} ({mode_label}, {operation_label})",
                    "db_reply",
                    dbSelectionType=db_selection_type,
                    dbCustomName=db_custom_name,
                    dbQueryMode=db_query_mode,
                    dbOperation=db_operation,
                    dbSqlQuery=db_sql_query,
                ))
        return result

    try:
        # First pass: collect intents, RAG dbs, and Agent metadata.
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Call):
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
                        intent_description = ""
                        body_rows: list = []

                        args = node.value.args
                        intent_name = ast.literal_eval(args[0])
                        for kw in node.value.keywords:
                            if kw.arg == "description":
                                intent_description = ast.literal_eval(kw.value)

                        if len(args) >= 2 and isinstance(args[1], ast.List):
                            for elt in args[1].elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    body_rows.append({
                                        "id": str(uuid.uuid4()),
                                        "name": elt.value.replace("\\'", "'"),
                                    })

                        if intent_name:
                            intents[intent_name] = {"id": intent_id, "name": intent_name}
                            intent_id_by_name[intent_name] = intent_id
                            if node.targets and isinstance(node.targets[0], ast.Name):
                                intent_var_to_name[node.targets[0].id] = intent_name
                            nodes.append(make_node(
                                node_id=intent_id,
                                type_="AgentIntent",
                                data={
                                    "name": intent_name,
                                    "intent_description": intent_description,
                                    "bodies": body_rows,
                                },
                                position={"x": states_x, "y": states_y},
                                width=160,
                                height=100,
                            ))
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
                            nodes.append(make_node(
                                node_id=str(uuid.uuid4()),
                                type_="AgentRagElement",
                                data={"name": rag_name},
                                position={"x": states_x, "y": states_y},
                                width=120,
                                height=110,
                            ))
                            if states_x < 200:
                                states_x += 300
                            else:
                                states_x = -280
                                states_y += 220

        # Collect functions.
        states_x = -280
        states_y += 220
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = {
                    "node": node,
                    "source": ast.get_source_segment(content, node),
                }

        custom_code_actions: dict = {}
        custom_condition_callables: dict = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'CustomCodeAction':
                    callable_name = None
                    for kw in node.value.keywords:
                        if kw.arg == 'callable' and isinstance(kw.value, ast.Name):
                            callable_name = kw.value.id
                    if callable_name:
                        custom_code_actions[var_name] = callable_name
                elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'Condition':
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

        # Collect actions.
        actions: dict = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == 'add_action'
                and isinstance(node.value.func.value, ast.Name)
                and len(node.value.args) == 1
            ):
                body_var = node.value.func.value.id
                if (
                    isinstance(node.value.args[0], ast.Call)
                    and isinstance(node.value.args[0].func, ast.Name)
                ):
                    fn_id = node.value.args[0].func.id
                    if fn_id == 'AgentReply' and len(node.value.args[0].args) == 1 and isinstance(node.value.args[0].args[0], ast.Constant) and isinstance(node.value.args[0].args[0].value, str):
                        actions.setdefault(body_var, []).append({"type": "text", "message": node.value.args[0].args[0].value})
                    elif fn_id == 'LLMReply':
                        actions.setdefault(body_var, []).append({"type": "llm"})
                    elif fn_id == 'RAGReply':
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
                        actions.setdefault(body_var, []).append({"type": "rag", "ragDatabaseName": rag_db_name})
                    elif fn_id == 'DBReply':
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
                        actions.setdefault(body_var, []).append(db_action)
                elif isinstance(node.value.args[0], ast.Name):
                    action_var = node.value.args[0].id
                    if action_var in custom_code_actions:
                        function_name = custom_code_actions[action_var]
                        actions[body_var] = function_name

        # Initial node.
        initial_node_id = str(uuid.uuid4())
        nodes.append(make_node(
            node_id=initial_node_id,
            type_="StateInitialNode",
            data={"name": ""},
            position={"x": states_x - 300, "y": states_y + 20},
            width=45,
            height=45,
        ))

        # Collect states.
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "new_state":
                    state_id = str(uuid.uuid4())
                    state_name = var_name
                    is_initial = False

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
                            is_initial = kw.value.value

                    state_var_to_name[var_name] = state_name
                    state_obj = {
                        "id": state_id,
                        "name": state_name,
                        "is_initial": is_initial,
                        "bodies": [],
                        "fallbackBodies": [],
                    }
                    states[var_name] = state_obj
                    nodes.append(make_node(
                        node_id=state_id,
                        type_="AgentState",
                        data={
                            "name": state_name,
                            "bodies": [],
                            "fallbackBodies": [],
                        },
                        position={"x": states_x, "y": states_y},
                        width=160,
                        height=100,
                    ))
                    if states_x < 200:
                        states_x += 490
                    else:
                        states_x = -280
                        states_y += 220
                elif len(node.targets) == 1:
                    target = node.targets[0]
                    if isinstance(target, ast.Attribute) and target.attr == "metadata":
                        state_var = target.value.id if isinstance(target.value, ast.Name) else None
                        if state_var and isinstance(node.value, ast.Call):
                            for kw in node.value.keywords:
                                if kw.arg == "description":
                                    state_comments[state_var] = ast.literal_eval(kw.value)

        nodes_by_id = {n["id"]: n for n in nodes}

        # Initial state -> AgentStateTransitionInit.
        initial_state = None
        for state_info in states.values():
            if state_info["is_initial"]:
                initial_state = state_info
                break
        if not initial_state and states:
            initial_state = states[next(iter(states))]
        if initial_state:
            edges.append(make_edge(
                edge_id=str(uuid.uuid4()),
                source=initial_node_id,
                target=initial_state["id"],
                type_="AgentStateTransitionInit",
                data={"points": []},
            ))

        # Process bodies and transitions.
        for node in ast.walk(tree):
            try:
                if (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                ):
                    if node.value.func.attr == "go_to":
                        source_state = node.value.func.value.id if isinstance(node.value.func.value, ast.Name) else None
                        condition_name = "auto"
                        transition_payload: object = ""
                        event_name = None
                        target_state = node.value.args[0].id if node.value.args and isinstance(node.value.args[0], ast.Name) else None

                        call_chain = node.value.func.value
                        custom_conditions: list = []

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
                                transition_payload = {"event": selected_event, "conditions": custom_conditions}
                            elif chain_attr == "when_condition":
                                condition_name = "custom_transition"
                                if call_chain.args and isinstance(call_chain.args[0], ast.Name):
                                    condition_func_name = call_chain.args[0].id
                                    custom_conditions.insert(0, _resolve_condition_source(condition_func_name))
                                transition_payload = {"event": "None", "conditions": custom_conditions}
                            break

                        if source_state in states and target_state in states:
                            transition_type = "custom" if condition_name == "custom_transition" else "predefined"
                            edge_data: dict = {
                                "name": event_name or "",
                                "transitionType": transition_type,
                                "points": [],
                            }
                            if transition_type == "predefined":
                                pred = {"predefinedType": condition_name}
                                if condition_name == "when_intent_matched":
                                    pred["intentName"] = transition_payload if isinstance(transition_payload, str) else ""
                                elif condition_name == "when_file_received":
                                    pred["fileType"] = transition_payload if isinstance(transition_payload, str) else ""
                                elif transition_payload not in (None, ""):
                                    pred["conditionValue"] = transition_payload
                                edge_data["predefined"] = pred
                            else:
                                edge_data["custom"] = {
                                    "event": event_name or "None",
                                    "condition": (
                                        transition_payload.get("conditions", [])
                                        if isinstance(transition_payload, dict)
                                        else []
                                    ),
                                }
                            edges.append(make_edge(
                                edge_id=str(uuid.uuid4()),
                                source=states[source_state]["id"],
                                target=states[target_state]["id"],
                                type_="AgentStateTransition",
                                data=edge_data,
                            ))

                    elif node.value.func.attr == "set_body":
                        try:
                            function_name = None
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
                            state_node = nodes_by_id.get(state["id"])
                            if not state_node:
                                continue

                            if function_name in functions or (isinstance(actions.get(function_name), str) and actions.get(function_name) in functions):
                                if (isinstance(actions.get(function_name), str) and actions.get(function_name) in functions):
                                    function_name = actions[function_name]
                                result = analyze_function_node(functions[function_name]["node"], functions[function_name]["source"])
                                if result["replyType"] == "text":
                                    for reply in result["replies"]:
                                        state_node["data"]["bodies"].append(_make_body_row(reply, "text"))
                                elif result["replyType"] == "llm":
                                    state_node["data"]["bodies"].append(_make_body_row("AI response 🪄", "llm"))
                                elif result["replyType"] == "code":
                                    state_node["data"]["bodies"].append(_make_body_row(result["code"], "code"))
                            elif function_name in actions:
                                if actions[function_name] == 'LLMReply':
                                    state_node["data"]["bodies"].append(_make_body_row("AI response 🪄", "llm"))
                                elif isinstance(actions[function_name], list):
                                    state_node["data"]["bodies"].extend(_bodies_for_action(actions[function_name], state["id"], False))
                                else:
                                    for message in actions[function_name]:
                                        state_node["data"]["bodies"].append(_make_body_row(message.replace("\\'", "'"), "text"))
                            else:
                                state_node["data"]["bodies"].append({
                                    "id": str(uuid.uuid4()),
                                    "name": function_name,
                                })
                        except Exception as e:
                            logger.error("Error processing agent body: %s", e, exc_info=True)
                            continue

                    elif node.value.func.attr == "set_fallback_body":
                        try:
                            function_name = None
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
                            state_node = nodes_by_id.get(state["id"])
                            if not state_node:
                                continue

                            mapped_action = actions.get(function_name)
                            if function_name in functions or (isinstance(mapped_action, str) and mapped_action in functions):
                                if isinstance(mapped_action, str) and mapped_action in functions:
                                    function_name = mapped_action
                                result = analyze_function_node(functions[function_name]["node"], functions[function_name]["source"])
                                if result["replyType"] == "text":
                                    for reply in result["replies"]:
                                        state_node["data"]["fallbackBodies"].append(_make_body_row(reply, "text"))
                                elif result["replyType"] == "llm":
                                    state_node["data"]["fallbackBodies"].append(_make_body_row("AI response 🪄", "llm"))
                                elif result["replyType"] == "code":
                                    state_node["data"]["fallbackBodies"].append(_make_body_row(result["code"], "code"))
                            elif function_name in actions:
                                if actions[function_name] == 'LLMReply':
                                    state_node["data"]["fallbackBodies"].append(_make_body_row("AI response 🪄", "llm"))
                                elif isinstance(actions[function_name], list):
                                    state_node["data"]["fallbackBodies"].extend(_bodies_for_action(actions[function_name], state["id"], True))
                                else:
                                    state_node["data"]["fallbackBodies"].append(_make_body_row(actions[function_name].replace("\\'", "'"), "text"))
                            else:
                                state_node["data"]["fallbackBodies"].append({
                                    "id": str(uuid.uuid4()),
                                    "name": function_name,
                                })
                        except Exception as e:
                            logger.warning("Error processing agent fallback body: %s", e, exc_info=True)
                            continue
            except Exception as e:
                logger.error("Error processing agent state machine: %s", e, exc_info=True)
                continue

        comment_x = -970
        comment_y = -300
        if agent_comment:
            nodes.append(make_node(
                node_id=str(uuid.uuid4()),
                type_="Comments",
                data={"name": agent_comment},
                position={"x": comment_x, "y": comment_y},
                width=200,
                height=100,
            ))
            comment_y += 130

        for state_var, comment_text in state_comments.items():
            if state_var in states:
                comment_id = str(uuid.uuid4())
                nodes.append(make_node(
                    node_id=comment_id,
                    type_="Comments",
                    data={"name": comment_text},
                    position={"x": comment_x, "y": comment_y},
                    width=200,
                    height=100,
                ))
                edges.append(make_edge(
                    edge_id=str(uuid.uuid4()),
                    source=comment_id,
                    target=states[state_var]["id"],
                    type_="Link",
                    data={"points": []},
                ))
                comment_y += 130

        return {
            "version": "4.0.0",
            "type": "AgentDiagram",
            "title": "",
            "size": {"width": 1980, "height": 640},
            "nodes": nodes,
            "edges": edges,
            "interactive": {"elements": {}, "relationships": {}},
            "assessments": {},
        }

    except Exception:
        logger.exception("Error converting agent BUML to JSON; returning partial diagram")
        return {
            "version": "4.0.0",
            "type": "AgentDiagram",
            "title": "",
            "size": {"width": 1980, "height": 640},
            "nodes": nodes,
            "edges": edges,
            "interactive": {"elements": {}, "relationships": {}},
            "assessments": {},
        }
