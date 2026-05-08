"""
State machine conversion from BUML to v4 JSON.

Provides two entry points:
- ``state_machine_object_to_json(sm)`` — converts a StateMachine metamodel
  object directly.
- ``state_machine_to_json(content)`` — legacy AST-parses a Python code
  string (for file import).

Both emit the v4 wire shape directly (``{nodes, edges}``).
"""

import uuid
import re
import ast
from besser.BUML.metamodel.state_machine.state_machine import (
    StateMachine, CustomCodeAction,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json._node_builders import (
    make_node, make_edge,
)


def _sanitize_identifier(name: str) -> str:
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    sanitized = re.sub(r'^[^a-zA-Z_]+', '', sanitized)
    return sanitized or 'unnamed'


def state_machine_object_to_json(sm: StateMachine) -> dict:
    """Convert a StateMachine metamodel instance to the v4 wire shape."""
    nodes: list = []
    edges: list = []

    states_x = -550
    states_y = -300
    code_blocks_x = -970
    code_blocks_y = 80

    initial_node_id = str(uuid.uuid4())
    nodes.append(make_node(
        node_id=initial_node_id,
        type_="StateInitialNode",
        data={"name": ""},
        position={"x": states_x - 300, "y": states_y + 20},
        width=45,
        height=45,
    ))

    all_bodies = {}
    all_events = {}
    all_conditions = {}
    state_id_map: dict = {}

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

    created_code_blocks: dict = {}

    def _emit_code_block(name: str, source: str) -> None:
        nonlocal code_blocks_x
        code_block_id = str(uuid.uuid4())
        cleaned = "\n".join(line.rstrip() for line in source.splitlines() if line.strip())
        nodes.append(make_node(
            node_id=code_block_id,
            type_="StateCodeBlock",
            data={"name": name, "code": cleaned, "language": "python"},
            position={"x": code_blocks_x, "y": code_blocks_y},
            width=580,
            height=200,
        ))
        created_code_blocks[name] = code_block_id
        code_blocks_x += 610

    for name, body in all_bodies.items():
        if name in created_code_blocks:
            continue
        source_code = ""
        for action in body.actions:
            if isinstance(action, CustomCodeAction):
                source_code = action.code
                break
        if not source_code and getattr(body, "code", None):
            source_code = body.code
        if source_code:
            _emit_code_block(name, source_code)

    for name, event in all_events.items():
        if name in created_code_blocks:
            continue
        source_code = getattr(event, '_source_code', "") or ""
        if source_code:
            _emit_code_block(name, source_code)

    for name, condition in all_conditions.items():
        if name in created_code_blocks:
            continue
        source_code = getattr(condition, 'code', "") or ""
        if source_code and source_code.lstrip().startswith("def "):
            _emit_code_block(name, source_code)

    for state in sm.states:
        state_id = str(uuid.uuid4())
        state_id_map[state] = state_id
        bodies_data: list = []
        fallbacks_data: list = []
        if state.body and state.body.name in created_code_blocks:
            bodies_data.append({"id": str(uuid.uuid4()), "name": state.body.name})
        if state.fallback_body and state.fallback_body.name in created_code_blocks:
            fallbacks_data.append({"id": str(uuid.uuid4()), "name": state.fallback_body.name})

        nodes.append(make_node(
            node_id=state_id,
            type_="State",
            data={
                "name": state.name,
                "bodies": bodies_data,
                "fallbackBodies": fallbacks_data,
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

    # Initial transition.
    for state in sm.states:
        if state.initial:
            edges.append(make_edge(
                edge_id=str(uuid.uuid4()),
                source=initial_node_id,
                target=state_id_map[state],
                type_="StateTransition",
                data={"name": "", "points": []},
            ))
            break

    # State transitions.
    for state in sm.states:
        source_id = state_id_map[state]
        for transition in state.transitions:
            if not transition.event and not transition.conditions:
                continue
            dest_id = state_id_map.get(transition.dest)
            if not dest_id:
                continue

            event_var = _sanitize_identifier(transition.event.name) if transition.event else ""
            edge_data: dict = {"name": event_var, "points": []}
            event_params = getattr(transition, '_event_params', None)
            if event_params:
                edge_data["params"] = str(event_params)
            if transition.conditions:
                cond = transition.conditions[0]
                if cond.name in created_code_blocks:
                    guard_value = cond.name
                elif getattr(cond, "code", None):
                    guard_value = cond.code
                else:
                    guard_value = cond.name
                edge_data["guard"] = guard_value

            edges.append(make_edge(
                edge_id=str(uuid.uuid4()),
                source=source_id,
                target=dest_id,
                type_="StateTransition",
                data=edge_data,
            ))

    # Comments.
    comment_x = -970
    comment_y = -300
    if sm.metadata and sm.metadata.description:
        nodes.append(make_node(
            node_id=str(uuid.uuid4()),
            type_="Comments",
            data={"name": sm.metadata.description},
            position={"x": comment_x, "y": comment_y},
            width=200,
            height=100,
        ))
        comment_y += 130

    for state in sm.states:
        if state.metadata and state.metadata.description:
            comment_id = str(uuid.uuid4())
            nodes.append(make_node(
                node_id=comment_id,
                type_="Comments",
                data={"name": state.metadata.description},
                position={"x": comment_x, "y": comment_y},
                width=200,
                height=100,
            ))
            edges.append(make_edge(
                edge_id=str(uuid.uuid4()),
                source=comment_id,
                target=state_id_map[state],
                type_="Link",
                data={"points": []},
            ))
            comment_y += 130

    return {
        "version": "4.0.0",
        "type": "StateMachineDiagram",
        "title": getattr(sm, "name", "") or "",
        "size": {"width": 1980, "height": 640},
        "nodes": nodes,
        "edges": edges,
        "interactive": {"elements": {}, "relationships": {}},
        "assessments": {},
    }


def state_machine_to_json(content: str):
    """Convert a state machine Python file to the v4 wire shape (legacy)."""
    nodes: list = []
    edges: list = []

    states_x = -550
    states_y = -300
    code_blocks_x = -970
    code_blocks_y = 80

    tree = ast.parse(content)
    states: dict = {}
    functions: dict = {}
    state_comments: dict = {}
    sm_comment = None

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
                            if isinstance(kw.value, ast.Call):
                                for meta_kw in kw.value.keywords:
                                    if meta_kw.arg == "description":
                                        sm_comment = ast.literal_eval(meta_kw.value)

    initial_node_id = str(uuid.uuid4())
    nodes.append(make_node(
        node_id=initial_node_id,
        type_="StateInitialNode",
        data={"name": ""},
        position={"x": states_x - 300, "y": states_y + 20},
        width=45,
        height=45,
    ))

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
                            "bodies_data": [],
                            "fallback_data": [],
                        }
                        nodes.append(make_node(
                            node_id=state_id,
                            type_="State",
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

    nodes_by_id = {n["id"]: n for n in nodes}

    # Initial transition (only one initial state).
    for state_info in states.values():
        if state_info["is_initial"]:
            edges.append(make_edge(
                edge_id=str(uuid.uuid4()),
                source=initial_node_id,
                target=state_info["id"],
                type_="StateTransition",
                data={"name": "", "points": []},
            ))
            break

    for state_info in states.values():
        if state_info["is_final"]:
            final_node_id = str(uuid.uuid4())
            state_id = state_info["id"]
            state_node = nodes_by_id.get(state_id)
            if not state_node:
                continue
            sx = state_node["position"]["x"]
            sy = state_node["position"]["y"]
            sw = state_node["width"]
            sh = state_node["height"]
            nodes.append(make_node(
                node_id=final_node_id,
                type_="StateFinalNode",
                data={"name": ""},
                position={"x": sx + sw + 100, "y": sy + (sh / 2) - 22.5},
                width=45,
                height=45,
            ))
            edges.append(make_edge(
                edge_id=str(uuid.uuid4()),
                source=state_id,
                target=final_node_id,
                type_="StateTransition",
                data={"name": "", "points": []},
            ))

    created_code_blocks: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            if function_name in created_code_blocks:
                continue
            code_block_id = str(uuid.uuid4())
            function_source = ast.get_source_segment(content, node) or ""
            cleaned_source = "\n".join(
                line.rstrip()
                for line in function_source.splitlines()
                if line.strip()
            )
            nodes.append(make_node(
                node_id=code_block_id,
                type_="StateCodeBlock",
                data={"name": function_name, "code": cleaned_source, "language": "python"},
                position={"x": code_blocks_x, "y": code_blocks_y},
                width=580,
                height=200,
            ))
            created_code_blocks[function_name] = code_block_id
            code_blocks_x += 610

    # Process state bodies + transitions.
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Attribute):
                    if node.value.func.attr == "set_body":
                        state_var = node.value.func.value.id
                        body_func = node.value.keywords[0].value.id
                        if state_var in states and body_func in created_code_blocks:
                            state = states[state_var]
                            state_node = nodes_by_id.get(state["id"])
                            if state_node:
                                state_node["data"]["bodies"].append({
                                    "id": str(uuid.uuid4()),
                                    "name": body_func,
                                })

                    elif (
                        node.value.func.attr == "go_to"
                        and isinstance(node.value.func.value, ast.Call)
                        and isinstance(node.value.func.value.func, ast.Attribute)
                        and node.value.func.value.func.attr == "when_event"
                    ):
                        source_state = node.value.func.value.func.value.id
                        event_name = None
                        target_state = node.value.args[0].id if node.value.args and isinstance(node.value.args[0], ast.Name) else None
                        if node.value.func.value.args:
                            event_arg = node.value.func.value.args[0]
                            if isinstance(event_arg, ast.Name):
                                event_name = event_arg.id
                            elif isinstance(event_arg, ast.Call) and isinstance(event_arg.func, ast.Name):
                                event_name = event_arg.func.id

                        if source_state in states and target_state in states:
                            edges.append(make_edge(
                                edge_id=str(uuid.uuid4()),
                                source=states[source_state]["id"],
                                target=states[target_state]["id"],
                                type_="StateTransition",
                                data={"name": event_name, "points": []},
                            ))

                    elif node.value.func.attr == "set_fallback_body":
                        state_var = node.value.func.value.id
                        fallback_func = node.value.args[0].id
                        if state_var in states and fallback_func in functions:
                            state = states[state_var]
                            state_node = nodes_by_id.get(state["id"])
                            if state_node:
                                state_node["data"]["fallbackBodies"].append({
                                    "id": str(uuid.uuid4()),
                                    "name": fallback_func,
                                })

        elif isinstance(node, ast.Assign):
            if len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Attribute) and target.attr == "metadata":
                    state_var = target.value.id if isinstance(target.value, ast.Name) else None
                    if state_var and isinstance(node.value, ast.Call):
                        for kw in node.value.keywords:
                            if kw.arg == "description":
                                state_comments[state_var] = ast.literal_eval(kw.value)

    comment_x = -970
    comment_y = -300

    if sm_comment:
        nodes.append(make_node(
            node_id=str(uuid.uuid4()),
            type_="Comments",
            data={"name": sm_comment},
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
        "type": "StateMachineDiagram",
        "title": "",
        "size": {"width": 1980, "height": 640},
        "nodes": nodes,
        "edges": edges,
        "interactive": {"elements": {}, "relationships": {}},
        "assessments": {},
    }
