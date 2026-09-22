"""
JSON to BUML conversion modules.
"""

from .class_diagram_processor import process_class_diagram
from .object_diagram_processor import process_object_diagram
from .state_machine_processor import process_state_machine
from .agent_diagram_processor import process_agent_diagram
# bpmn_diagram_processor must be imported before project_converter — the latter does
# `from . import (..., process_bpmn_diagram)` to wire BPMN into project conversion, so
# the name has to be present in this package's namespace when project_converter loads.
from .bpmn_diagram_processor import process_bpmn_diagram
from .gui_diagram_processor import process_gui_diagram
from .quantum_diagram_processor import process_quantum_diagram
from .nn_diagram_processor import process_nn_diagram
# Importing project_converter last so it can pick up every processor above
# (BPMN, …) via `from . import (...)` without hitting a partially-initialised package.
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
    'process_bpmn_diagram',
]
