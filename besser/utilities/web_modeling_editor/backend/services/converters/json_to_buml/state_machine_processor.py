"""
State machine processing for converting JSON to BUML format.
"""

import re


def process_state_machine(json_data):
    """Process State Machine Diagram specific elements and return Python code as string."""
    code_lines = []
    code_lines.append("#######################")
    code_lines.append("# STATE MACHINE MODEL #")
    code_lines.append("#######################")
    code_lines.append("")
    code_lines.append("import datetime")
    code_lines.append("from besser.BUML.metamodel.state_machine.state_machine import StateMachine, Session, Body, Event")
    code_lines.append("from besser.BUML.metamodel.structural import Metadata\n")
    sm_name = json_data.get("title", "Generated_State_Machine")
    if ' ' in sm_name:
        sm_name = sm_name.replace(' ', '_')
    code_lines.append(f"sm = StateMachine(name='{sm_name}')\n")

    elements = json_data.get('model', {}).get('elements', {})
    relationships = json_data.get('model', {}).get('relationships', {})

    # Track states by ID for later reference
    states_by_id = {}
    body_names = set()
    event_names = set()
    
    # Store comments for later processing
    comment_elements = {}  # {comment_id: comment_text}
    comment_links = {}  # {comment_id: [linked_element_ids]}

    # Collect all body and event names first
    for element in elements.values():
        if element.get("type") == "Comments":
            comment_text = element.get("name", "")
            comment_elements[element.get("id")] = comment_text
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

    # Write function definitions first
    for element in elements.values():
        if element.get("type") == "StateCodeBlock":
            name = element.get("name", "")
            code_content = element.get("code", {})

            # If name is empty, try to extract function name from code content
            if not name:
                # Look for "def function_name(" pattern in the code
                function_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_content)
                if function_match:
                    name = function_match.group(1)

            # Clean up the code content by removing extra newlines
            cleaned_code = "\n".join(line for line in code_content.splitlines() if line.strip())
            # Write the function definition with its code content
            code_lines.append(cleaned_code)  # Write the actual function code
            code_lines.append("")  # Add single blank line after function

            if name in body_names:
                code_lines.append(f"{name} = Body(name='{name}', callable={name})")
            if name in event_names:
                code_lines.append(f"{name} = Event(name='{name}', callable={name})")
            code_lines.append("")  # Add blank line after Body/Event creation

    # Create states
    for element_id, element in elements.items():
        if element.get("type") == "State":
            is_initial = False
            for rel in relationships.values():
                if (rel.get("type") == "StateTransition" and
                    rel.get("target", {}).get("element") == element_id and
                    elements.get(rel.get("source", {}).get("element", ""), {}).get("type") == "StateInitialNode"):
                    is_initial = True
                    break

            state_name = element.get("name", "")
            code_lines.append(f"{state_name}_state = sm.new_state(name='{state_name}', initial={str(is_initial)})")
            states_by_id[element_id] = state_name
    code_lines.append("")

    # Assign bodies to states
    for element_id, element in elements.items():
        if element.get("type") == "State":
            state_name = element.get("name", "")
            for body_id in element.get("bodies", []):
                body_element = elements.get(body_id)
                if body_element:
                    body_name = body_element.get("name")
                    if body_name in body_names:
                        code_lines.append(f"{state_name}_state.set_body(body={body_name})")

            for fallback_id in element.get("fallbackBodies", []):
                fallback_element = elements.get(fallback_id)
                if fallback_element:
                    fallback_name = fallback_element.get("name")
                    if fallback_name in body_names:
                        code_lines.append(f"{state_name}_state.set_fallback_body({fallback_name})")
    code_lines.append("")

    # Write transitions
    for relationship in relationships.values():
        if relationship.get("type") == "StateTransition":
            source_id = relationship.get("source", {}).get("element")
            target_id = relationship.get("target", {}).get("element")

            if elements.get(source_id, {}).get("type") == "StateInitialNode":
                continue

            source_name = states_by_id.get(source_id)
            target_name = states_by_id.get(target_id)

            if source_name and target_name:
                event_name = relationship.get("name", "")
                params = relationship.get("params")

                if event_name:
                    event_params = f"event_params={{ {params} }}" if params else "event_params={}"
                    code_lines.append(f"{source_name}_state.when_event_go_to(")
                    code_lines.append(f"    event={event_name},")
                    code_lines.append(f"    dest={target_name}_state,")
                    code_lines.append(f"    {event_params}")
                    code_lines.append(")")

    # Process comments
    state_id_to_name = {state_id: state_name for state_id, state_name in states_by_id.items()}
    
    for comment_id, comment_text in comment_elements.items():
        if comment_id in comment_links:
            # Comment is linked to one or more elements
            for linked_element_id in comment_links[comment_id]:
                if linked_element_id in state_id_to_name:
                    # Apply comment to state's metadata
                    state_name = state_id_to_name[linked_element_id]
                    escaped_comment = comment_text.replace("'", "\\'").replace("\n", "\\n")
                    code_lines.append(f"{state_name}_state.metadata = Metadata(description='{escaped_comment}')")
        else:
            # Unlinked comment - add to StateMachine metadata
            escaped_comment = comment_text.replace("'", "\\'").replace("\n", "\\n")
            code_lines.append(f"sm.metadata = Metadata(description='{escaped_comment}')")
    
    if comment_elements:
        code_lines.append("")

    return "\n".join(code_lines)
