"""
State machine processing for converting JSON to BUML format.

Returns a StateMachine metamodel instance (not a code string).
"""

import logging
import re

from besser.BUML.metamodel.state_machine.state_machine import (
    StateMachine, State, Body, Event, Transition, CustomCodeAction,
)
from besser.BUML.metamodel.structural import Metadata
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError

logger = logging.getLogger(__name__)


def _sanitize_identifier(name: str) -> str:
    """Sanitize a string to be a valid Python identifier.

    Replaces any non-alphanumeric character (except underscore) with underscore,
    and strips leading digits.
    """
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    sanitized = re.sub(r'^[^a-zA-Z_]+', '', sanitized)
    return sanitized or 'unnamed'


def process_state_machine(json_data):
    """Process State Machine Diagram specific elements and return a StateMachine instance.

    Args:
        json_data: Dictionary containing the state machine diagram JSON data.

    Returns:
        StateMachine: A BUML StateMachine metamodel instance.

    Raises:
        ConversionError: If the JSON data is invalid or missing required fields.
    """
    sm_name = json_data.get("title", "Generated_State_Machine")
    if ' ' in sm_name:
        sm_name = sm_name.replace(' ', '_')

    sm = StateMachine(name=sm_name)

    model_data = json_data.get('model')
    if not model_data:
        raise ConversionError("State machine JSON is missing the 'model' key.")
    elements = model_data.get('elements') or {}
    relationships = model_data.get('relationships') or {}

    # Track states by element ID for later reference
    states_by_id = {}  # element_id -> State object
    body_names = set()
    event_names = set()

    # Store comments for later processing
    comment_elements = {}  # {comment_id: comment_text}
    comment_links = {}  # {comment_id: [linked_element_ids]}

    # Collect all body and event names first
    for element_id, element in elements.items():
        if element.get("type") == "Comments":
            comment_text = element.get("name", "")
            comment_elements[element_id] = comment_text
            continue
        elif element.get("type") == "StateBody":
            body_names.add(element.get("name"))
        elif element.get("type") == "StateFallbackBody":
            body_names.add(element.get("name"))

    # Collect event names from transitions
    for rel in relationships.values():
        if rel.get("type") == "StateTransition" and rel.get("name"):
            event_names.add(rel.get("name"))
        elif rel.get("type") == "Link":
            # Handle comment links
            source_element_id = rel.get("source", {}).get("element")
            target_element_id = rel.get("target", {}).get("element")

            comment_id = None
            target_id = None

            if source_element_id in comment_elements:
                comment_id = source_element_id
                target_id = target_element_id
            elif target_element_id in comment_elements:
                comment_id = target_element_id
                target_id = source_element_id

            if comment_id and target_id:
                if comment_id not in comment_links:
                    comment_links[comment_id] = []
                comment_links[comment_id].append(target_id)

    # Build Body and Event objects from code blocks
    body_objects = {}  # function name -> Body instance
    event_objects = {}  # function name -> Event instance

    for element in elements.values():
        if element.get("type") == "StateCodeBlock":
            name = element.get("name", "")
            code_content = element.get("code", "")

            # If name is empty, try to extract function name from code content
            if not name:
                function_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_content)
                if function_match:
                    name = function_match.group(1)

            if not name:
                continue

            # Clean up the code content by removing extra newlines
            cleaned_code = "\n".join(line for line in code_content.splitlines() if line.strip())

            if name in body_names:
                # Create a Body with a CustomCodeAction containing the source code.
                # We cannot pass a live callable here since we only have source text,
                # so we use the actions-based constructor with CustomCodeAction.
                body = Body(name=name, actions=[CustomCodeAction(source=cleaned_code)])
                body_objects[name] = body

            if name in event_names:
                # The Event metamodel class only takes a name (no callable).
                # The code content is stored separately on the code block element
                # and will be preserved through round-trip via code blocks.
                event = Event(name=name)
                # Attach the source code for round-trip fidelity (not part of
                # Event's formal metamodel, but needed for BUML export).
                event._source_code = cleaned_code
                event_objects[name] = event

    # Determine which element IDs are initial states (targets of transitions from StateInitialNode)
    initial_state_ids = set()
    for rel in relationships.values():
        if rel.get("type") == "StateTransition":
            source_id = rel.get("source", {}).get("element")
            target_id = rel.get("target", {}).get("element")
            if elements.get(source_id, {}).get("type") == "StateInitialNode":
                initial_state_ids.add(target_id)

    # Create states - initial state(s) first to satisfy StateMachine ordering constraint
    state_elements = [
        (eid, elem) for eid, elem in elements.items()
        if elem.get("type") == "State"
    ]

    # Sort so initial states come first
    state_elements.sort(key=lambda pair: pair[0] not in initial_state_ids)

    for element_id, element in state_elements:
        raw_name = element.get("name", "")
        if not raw_name.strip():
            logger.warning("State element '%s' has an empty name, using 'unnamed'.", element_id)
            raw_name = "unnamed"

        is_initial = element_id in initial_state_ids

        try:
            state = sm.new_state(name=raw_name, initial=is_initial)
        except ValueError as e:
            # Handle duplicate state names or other validation errors gracefully
            logger.warning("Could not create state '%s': %s", raw_name, e)
            continue

        states_by_id[element_id] = state

    # Assign bodies and fallback bodies to states
    for element_id, element in elements.items():
        if element.get("type") == "State":
            state = states_by_id.get(element_id)
            if not state:
                continue

            for body_id in element.get("bodies", []):
                body_element = elements.get(body_id)
                if body_element:
                    body_name = body_element.get("name")
                    if body_name in body_objects:
                        state.set_body(body=body_objects[body_name])

            for fallback_id in element.get("fallbackBodies", []):
                fallback_element = elements.get(fallback_id)
                if fallback_element:
                    fallback_name = fallback_element.get("name")
                    if fallback_name in body_objects:
                        state.set_fallback_body(body=body_objects[fallback_name])

    # Create transitions
    for relationship in relationships.values():
        if relationship.get("type") == "StateTransition":
            source_id = relationship.get("source", {}).get("element")
            target_id = relationship.get("target", {}).get("element")

            # Skip transitions from initial node (already handled by is_initial flag)
            if elements.get(source_id, {}).get("type") == "StateInitialNode":
                continue

            source_state = states_by_id.get(source_id)
            target_state = states_by_id.get(target_id)

            if not source_state or not target_state:
                logger.warning(
                    "Skipping transition: source state '%s' or target state '%s' not found.",
                    source_id, target_id
                )
                continue

            event_name = relationship.get("name", "")
            params = relationship.get("params")

            if event_name:
                event = event_objects.get(event_name)
                if not event:
                    # Create a simple event if no code block was associated
                    event = Event(name=event_name)
                    event_objects[event_name] = event

                # Use the TransitionBuilder API: state.when_event(event).go_to(dest)
                try:
                    source_state.when_event(event).go_to(target_state)
                except ValueError as e:
                    logger.warning(
                        "Could not create transition from '%s' to '%s': %s",
                        source_state.name, target_state.name, e
                    )

                # Store event_params on the transition for round-trip fidelity.
                # The metamodel Transition class does not have a formal event_params
                # attribute, so we attach it as an informal attribute.
                if params and source_state.transitions:
                    last_transition = source_state.transitions[-1]
                    last_transition._event_params = params

    # Process comments - apply as metadata
    for comment_id, comment_text in comment_elements.items():
        if comment_id in comment_links:
            # Comment is linked to one or more elements
            for linked_element_id in comment_links[comment_id]:
                if linked_element_id in states_by_id:
                    state = states_by_id[linked_element_id]
                    state.metadata = Metadata(description=comment_text)
        else:
            # Unlinked comment - add to StateMachine metadata
            sm.metadata = Metadata(description=comment_text)

    return sm
