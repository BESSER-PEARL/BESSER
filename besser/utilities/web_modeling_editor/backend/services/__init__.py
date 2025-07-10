"""
Services module for the web modeling editor backend.
Organized into submodules for better maintainability.
"""

# Import from converters module
from .converters.json_to_buml import (
    process_class_diagram,
    process_state_machine,
    process_agent_diagram,
    process_object_diagram,
    json_to_buml_project,
)
from .converters.buml_to_json import (
    class_buml_to_json,
    parse_buml_content,
    state_machine_to_json,
    agent_buml_to_json,
    object_buml_to_json,
    project_to_json,
)

# Import from validators module
from .validators import (
    check_ocl_constraint,
)

# Import from deployment module
from .deployment import (
    run_docker_compose,
)

# Import from utils module
from .utils import (
    calculate_center_point,
    determine_connection_direction,
    calculate_connection_points,
    calculate_path_points,
    calculate_relationship_bounds,
)

__all__ = [
    # Converters
    "process_class_diagram",
    "process_state_machine",
    "process_agent_diagram",
    "process_object_diagram",
    "class_buml_to_json",
    "parse_buml_content",
    "state_machine_to_json",
    "agent_buml_to_json",
    "object_buml_to_json",
    "project_to_json",
    "json_to_buml_project",
    
    # Validators
    "check_ocl_constraint",
    
    # Deployment
    "run_docker_compose",
    
    # Utils
    "calculate_center_point",
    "determine_connection_direction",
    "calculate_connection_points",
    "calculate_path_points",
    "calculate_relationship_bounds",
]
