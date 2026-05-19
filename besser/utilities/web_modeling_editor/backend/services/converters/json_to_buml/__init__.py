"""
JSON to BUML conversion modules.
"""

from .class_diagram_processor import process_class_diagram
from .object_diagram_processor import process_object_diagram
from .state_machine_processor import process_state_machine
from .agent_diagram_processor import process_agent_diagram
from .gui_diagram_processor import process_gui_diagram
from .quantum_diagram_processor import process_quantum_diagram
from .nn_diagram_processor import process_nn_diagram
from .component_diagram_processor import process_component_diagram
from .deployment_diagram_processor import process_deployment_diagram
# Importing project_converter last so it can pick up every processor above
# via `from . import (...)` without hitting a partially-initialised package.
from .project_converter import json_to_buml_project

__all__ = [
    'process_class_diagram',
    'process_object_diagram',
    'process_state_machine',
    'process_agent_diagram',
    'json_to_buml_project',
    'process_gui_diagram',
    'process_quantum_diagram',
    'process_nn_diagram',
    'process_component_diagram',
    'process_deployment_diagram',
]
