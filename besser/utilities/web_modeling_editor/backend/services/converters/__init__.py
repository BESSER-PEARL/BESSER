"""
Converters module for handling JSON to BUML and BUML to JSON conversions.
"""

from .json_to_buml import (
    process_class_diagram,
    process_state_machine,
    process_agent_diagram,
    process_object_diagram,
    json_to_buml_project,
    process_gui_diagram,
)
from .buml_to_json import (
    class_buml_to_json,
    parse_buml_content,
    state_machine_to_json,
    agent_buml_to_json,
    analyze_function_node,
    object_buml_to_json,
    gui_buml_to_json,
    parse_gui_buml_content,
    project_to_json,
    empty_model,
)

__all__ = [
    "process_class_diagram",
    "process_state_machine",
    "process_agent_diagram",
    "process_object_diagram",
    "json_to_buml_project",
    "process_gui_diagram",
    "class_buml_to_json",
    "parse_buml_content",
    "state_machine_to_json",
    "agent_buml_to_json",
    "analyze_function_node",
    "object_buml_to_json",
    "gui_buml_to_json",
    "parse_gui_buml_content",
    "project_to_json",
    "empty_model",
]
