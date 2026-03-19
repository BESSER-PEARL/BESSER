"""
State Machine Builder

Generates Python code from a StateMachine metamodel instance.
The generated code can be exec()'d to recreate the StateMachine.
"""

import re
from besser.BUML.metamodel.state_machine.state_machine import (
    StateMachine, CustomCodeAction,
)


def _escape_python_string(value: str) -> str:
    """Escape a string for safe interpolation into generated Python source code.

    Prevents code injection when user-controlled values (names, labels, etc.)
    are embedded inside string literals in generated Python files that may
    later be executed with ``exec()``.
    """
    return (value
            .replace('\\', '\\\\')
            .replace("'", "\\'")
            .replace('"', '\\"')
            .replace('\n', '\\n')
            .replace('\r', '\\r'))

def _sanitize_identifier(name: str) -> str:
    """Sanitize a string to be a valid Python identifier."""
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    sanitized = re.sub(r'^[^a-zA-Z_]+', '', sanitized)
    return sanitized or 'unnamed'


def state_machine_to_code(model: StateMachine, file_path: str = None,
                          model_var_name: str = "sm") -> str:
    """
    Generate Python code from a StateMachine metamodel instance.

    Args:
        model: The StateMachine instance to convert.
        file_path: Optional path to write the generated code to.
        model_var_name: Variable name to use for the state machine (default: "sm").

    Returns:
        The generated Python code as a string.
    """
    code_lines = []
    code_lines.append("#######################")
    code_lines.append("# STATE MACHINE MODEL #")
    code_lines.append("#######################")
    code_lines.append("")
    code_lines.append("import datetime")
    code_lines.append("from besser.BUML.metamodel.state_machine.state_machine import StateMachine, Session, Body, Event, Condition")
    code_lines.append("from besser.BUML.metamodel.structural import Metadata\n")

    sm_name_safe = _escape_python_string(model.name)
    code_lines.append(f"{model_var_name} = StateMachine(name='{sm_name_safe}')\n")

    # Collect all unique Body, Event, and Condition objects across all states
    all_bodies = {}  # name -> Body
    all_events = {}  # name -> Event
    all_conditions = {}  # name -> Condition

    for state in model.states:
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

    # Write function definitions and Body/Event objects
    written_code_blocks = set()

    for name, body in all_bodies.items():
        safe_name = _sanitize_identifier(name)
        # Extract source code from CustomCodeAction if present
        source_code = ""
        for action in body.actions:
            if isinstance(action, CustomCodeAction):
                source_code = action.code
                break
        if not source_code and body.code:
            source_code = body.code

        if source_code and name not in written_code_blocks:
            cleaned_code = "\n".join(line for line in source_code.splitlines() if line.strip())
            code_lines.append(cleaned_code)
            code_lines.append("")
            written_code_blocks.add(name)

        code_lines.append(f"{safe_name} = Body(name='{_escape_python_string(name)}', callable={safe_name})")
        code_lines.append("")

    for name, event in all_events.items():
        safe_name = _sanitize_identifier(name)
        # Check for attached source code (from round-trip)
        source_code = getattr(event, '_source_code', "")

        if source_code and name not in written_code_blocks:
            cleaned_code = "\n".join(line for line in source_code.splitlines() if line.strip())
            code_lines.append(cleaned_code)
            code_lines.append("")
            written_code_blocks.add(name)
            code_lines.append(f"{safe_name} = Event(name='{_escape_python_string(name)}')")
        else:
            code_lines.append(f"{safe_name} = Event(name='{_escape_python_string(name)}')")
        code_lines.append("")

    for name, condition in all_conditions.items():
        safe_name = _sanitize_identifier(name)
        source_code = getattr(condition, 'code', "") or ""

        if source_code and name not in written_code_blocks:
            cleaned_code = "\n".join(line for line in source_code.splitlines() if line.strip())
            code_lines.append(cleaned_code)
            code_lines.append("")
            written_code_blocks.add(name)
            code_lines.append(f"{safe_name} = Condition(name='{_escape_python_string(name)}', callable={safe_name})")
        else:
            code_lines.append(f"{safe_name} = Condition(name='{_escape_python_string(name)}', callable=lambda session, params: True)")
        code_lines.append("")

    # Create states with initial and final flags
    for state in model.states:
        safe_state = _sanitize_identifier(state.name)
        state_name_safe = _escape_python_string(state.name)
        code_lines.append(
            f"{safe_state}_state = {model_var_name}.new_state("
            f"name='{state_name_safe}', initial={state.initial}, final={state.final})"
        )
    code_lines.append("")

    # Assign bodies to states (skip for final states as they shouldn't have bodies)
    for state in model.states:
        if not state.final:  # Don't add bodies to final states
            safe_state = _sanitize_identifier(state.name)
            if state.body:
                safe_body = _sanitize_identifier(state.body.name)
                code_lines.append(f"{safe_state}_state.set_body(body={safe_body})")
            if state.fallback_body:
                safe_fallback = _sanitize_identifier(state.fallback_body.name)
                code_lines.append(f"{safe_state}_state.set_fallback_body({safe_fallback})")
    code_lines.append("")

    # Write transitions
    for state in model.states:
        safe_source = _sanitize_identifier(state.name)
        for transition in state.transitions:
            safe_dest = _sanitize_identifier(transition.dest.name)

            if transition.event and not transition.conditions:
                # Event-only transition
                safe_event = _sanitize_identifier(transition.event.name)
                code_lines.append(f"{safe_source}_state.when_event({safe_event}).go_to({safe_dest}_state)")

            elif transition.event and transition.conditions:
                # Event + guard(s) transition: use fluent API
                safe_event = _sanitize_identifier(transition.event.name)
                builder_expr = f"{safe_source}_state.when_event({safe_event})"
                for cond in transition.conditions:
                    safe_cond = _sanitize_identifier(cond.name)
                    builder_expr += f".with_condition({safe_cond})"
                builder_expr += f".go_to({safe_dest}_state)"
                code_lines.append(builder_expr)

            elif transition.conditions and not transition.event:
                # Guard-only transition (no event)
                first_cond = transition.conditions[0]
                safe_cond = _sanitize_identifier(first_cond.name)
                builder_expr = f"{safe_source}_state.when_condition({safe_cond})"
                for cond in transition.conditions[1:]:
                    safe_extra = _sanitize_identifier(cond.name)
                    builder_expr += f".with_condition({safe_extra})"
                builder_expr += f".go_to({safe_dest}_state)"
                code_lines.append(builder_expr)

    # Process metadata/comments
    for state in model.states:
        if state.metadata and state.metadata.description:
            safe_state = _sanitize_identifier(state.name)
            escaped_comment = _escape_python_string(state.metadata.description)
            code_lines.append(f"{safe_state}_state.metadata = Metadata(description='{escaped_comment}')")

    if model.metadata and model.metadata.description:
        escaped_comment = _escape_python_string(model.metadata.description)
        code_lines.append(f"{model_var_name}.metadata = Metadata(description='{escaped_comment}')")

    code_lines.append("")

    result = "\n".join(code_lines)

    if file_path:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(result)

    return result
