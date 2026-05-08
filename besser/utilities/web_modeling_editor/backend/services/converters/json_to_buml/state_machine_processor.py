"""
State machine processing for converting v4 JSON to a BUML StateMachine.

Reads the v4 wire shape (``{nodes, edges}``) natively. See
``docs/source/migrations/uml-v4-shape.md`` for the spec.
"""

import logging
import re

from besser.BUML.metamodel.state_machine.state_machine import (
    StateMachine, Body, Event, Condition, CustomCodeAction,
)
from besser.BUML.metamodel.structural import Metadata
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml._node_helpers import (
    node_data,
)

logger = logging.getLogger(__name__)


def _sanitize_identifier(name: str) -> str:
    """Sanitize a string to be a valid Python identifier."""
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    sanitized = re.sub(r'^[^a-zA-Z_]+', '', sanitized)
    return sanitized or 'unnamed'


def process_state_machine(json_data):
    """Process a v4 State Machine Diagram and return a ``StateMachine``.

    Args:
        json_data: Dictionary containing the state machine diagram JSON
            in the v4 wire shape (``model.nodes`` / ``model.edges``).
    """
    sm_name = json_data.get("title", "Generated_State_Machine")
    if ' ' in sm_name:
        sm_name = sm_name.replace(' ', '_')

    sm = StateMachine(name=sm_name)

    model_data = json_data.get('model')
    if not model_data:
        raise ConversionError("State machine JSON is missing the 'model' key.")
    nodes = model_data.get('nodes') or []
    edges = model_data.get('edges') or []
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []

    nodes_by_id = {n.get("id"): n for n in nodes if n.get("id")}

    states_by_id = {}  # node_id -> State
    body_names = set()
    event_names = set()

    comment_nodes = {}  # comment_id -> text
    comment_links = {}  # comment_id -> [linked_node_ids]

    # First pass: collect body / fallback body names from State data, and
    # comments.
    for node in nodes:
        node_type = node.get("type")
        node_id = node.get("id")
        data = node_data(node)
        if node_type == "Comments":
            comment_nodes[node_id] = data.get("name", "")
            continue
        if node_type == "State":
            for body in data.get("bodies") or []:
                if body.get("name"):
                    body_names.add(body["name"])
            for body in data.get("fallbackBodies") or []:
                if body.get("name"):
                    body_names.add(body["name"])

    guard_names = set()

    # Collect event names + guards from edges, and comment links.
    for edge in edges:
        edge_type = edge.get("type")
        edge_data = edge.get("data") or {}
        if edge_type == "StateTransition":
            if edge_data.get("name"):
                event_names.add(edge_data["name"])
            if edge_data.get("guard"):
                guard_names.add(edge_data["guard"])
        elif edge_type == "Link":
            source_id = edge.get("source")
            target_id = edge.get("target")
            comment_id = None
            target = None
            if source_id in comment_nodes:
                comment_id = source_id
                target = target_id
            elif target_id in comment_nodes:
                comment_id = target_id
                target = source_id
            if comment_id and target:
                comment_links.setdefault(comment_id, []).append(target)

    # Build Body / Event / Condition objects from StateCodeBlock nodes.
    body_objects = {}
    event_objects = {}
    condition_objects = {}
    for node in nodes:
        if node.get("type") != "StateCodeBlock":
            continue
        data = node_data(node)
        name = data.get("name") or ""
        code_content = data.get("code", "") or ""
        if not name:
            function_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_content)
            if function_match:
                name = function_match.group(1)
        if not name:
            continue
        cleaned_code = "\n".join(line for line in code_content.splitlines() if line.strip())
        if name in body_names:
            body = Body(name=name, actions=[CustomCodeAction(source=cleaned_code)])
            body_objects[name] = body
        if name in event_names:
            event = Event(name=name)
            event._source_code = cleaned_code
            event_objects[name] = event
        if name in guard_names:
            condition = Condition(name=name, source=cleaned_code)
            condition_objects[name] = condition

    # Determine initial / final state ids from edges.
    initial_state_ids = set()
    final_state_ids = set()
    for edge in edges:
        if edge.get("type") != "StateTransition":
            continue
        source_id = edge.get("source")
        target_id = edge.get("target")
        if (nodes_by_id.get(source_id) or {}).get("type") == "StateInitialNode":
            initial_state_ids.add(target_id)
        if (nodes_by_id.get(target_id) or {}).get("type") == "StateFinalNode":
            final_state_ids.add(source_id)

    # Create states; initial state(s) first to satisfy ordering constraint.
    state_nodes = [n for n in nodes if n.get("type") == "State"]
    state_nodes.sort(key=lambda n: n.get("id") not in initial_state_ids)

    for node in state_nodes:
        node_id = node.get("id")
        data = node_data(node)
        raw_name = data.get("name", "") or ""
        if not raw_name.strip():
            logger.warning("State node '%s' has an empty name, using 'unnamed'.", node_id)
            raw_name = "unnamed"
        is_initial = node_id in initial_state_ids
        is_final = node_id in final_state_ids
        try:
            state = sm.new_state(name=raw_name, initial=is_initial, final=is_final)
        except ValueError as e:
            logger.warning("Could not create state '%s': %s", raw_name, e)
            continue
        states_by_id[node_id] = state

    # Assign bodies / fallback bodies.
    for node in state_nodes:
        node_id = node.get("id")
        state = states_by_id.get(node_id)
        if not state:
            continue
        data = node_data(node)
        for body in data.get("bodies") or []:
            body_name = body.get("name")
            if body_name in body_objects:
                state.set_body(body=body_objects[body_name])
        for body in data.get("fallbackBodies") or []:
            body_name = body.get("name")
            if body_name in body_objects:
                state.set_fallback_body(body=body_objects[body_name])

    # Create transitions.
    for edge in edges:
        if edge.get("type") != "StateTransition":
            continue
        source_id = edge.get("source")
        target_id = edge.get("target")

        if (nodes_by_id.get(source_id) or {}).get("type") == "StateInitialNode":
            continue
        if (nodes_by_id.get(target_id) or {}).get("type") == "StateFinalNode":
            continue

        source_state = states_by_id.get(source_id)
        target_state = states_by_id.get(target_id)
        if not source_state or not target_state:
            logger.warning(
                "Skipping transition: source '%s' or target '%s' state not found.",
                source_id, target_id,
            )
            continue

        edge_data = edge.get("data") or {}
        event_name = edge_data.get("name", "")
        params = edge_data.get("params")
        guard_expr = edge_data.get("guard", "")

        guard_condition = None
        if guard_expr:
            guard_condition = condition_objects.get(guard_expr)
            if not guard_condition:
                guard_condition = Condition(
                    name=_sanitize_identifier(guard_expr),
                    source=guard_expr,
                )
                condition_objects[guard_expr] = guard_condition

        if event_name:
            event = event_objects.get(event_name)
            if not event:
                event = Event(name=event_name)
                event_objects[event_name] = event
            try:
                builder = source_state.when_event(event)
                if guard_condition:
                    builder = builder.with_condition(guard_condition)
                builder.go_to(target_state)
            except ValueError as e:
                logger.warning(
                    "Could not create transition from '%s' to '%s': %s",
                    source_state.name, target_state.name, e,
                )
            if params and source_state.transitions:
                last_transition = source_state.transitions[-1]
                last_transition._event_params = params
        elif guard_condition:
            try:
                source_state.when_condition(guard_condition).go_to(target_state)
            except ValueError as e:
                logger.warning(
                    "Could not create guard-only transition from '%s' to '%s': %s",
                    source_state.name, target_state.name, e,
                )

    # Apply comments.
    for comment_id, comment_text in comment_nodes.items():
        if comment_id in comment_links:
            for linked_id in comment_links[comment_id]:
                if linked_id in states_by_id:
                    states_by_id[linked_id].metadata = Metadata(description=comment_text)
        else:
            sm.metadata = Metadata(description=comment_text)

    return sm
