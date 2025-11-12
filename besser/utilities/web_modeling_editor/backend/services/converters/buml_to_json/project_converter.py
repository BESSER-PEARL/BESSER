"""
Project converter module for BUML to JSON conversion.
Handles project structure processing and diagram coordination.
"""

import uuid
import re
from datetime import datetime, timezone
from typing import Dict, Any

from .class_diagram_converter import parse_buml_content, class_buml_to_json
from .state_machine_converter import state_machine_to_json
from .agent_diagram_converter import agent_buml_to_json
from .object_diagram_converter import object_buml_to_json
from .gui_diagram_converter import gui_buml_to_json


def empty_model(diagram_type: str) -> Dict[str, Any]:
    """
    Create an empty model template for the specified diagram type.
    
    Args:
        diagram_type: Type of diagram to create empty model for
        
    Returns:
        Dictionary representing empty model structure
    """
    # GUINoCodeDiagram has a different structure with pages instead of elements
    if diagram_type == "GUINoCodeDiagram":
        return {
            "version": "3.0.0",
            "type": diagram_type,
            "size": {"width": 1400, "height": 740},
            "pages": {}
        }
    
    return {
        "version": "3.0.0",
        "type": diagram_type,
        "size": {"width": 1400, "height": 740},
        "elements": {},
        "relationships": {},
        "interactive": {"elements": {}, "relationships": {}},
        "assessments": {}
    }


def project_to_json(content: str) -> Dict[str, Any]:
    """
    Convert a BUML project content to JSON format matching the frontend structure.
    
    Args:
        content: Project Python code as string
        
    Returns:
        Dictionary representing the complete project with all diagrams
    """
    def extract_section(name: str, next_headers: list) -> str:
        """
        Extract a section from the project content.
        
        Args:
            name: Section name to extract
            next_headers: List of headers that might follow this section
            
        Returns:
            Extracted section content
        """
        pattern = rf"# {name.upper()} MODEL #(.*?)# ({'|'.join(next_headers)})"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()

        pattern = rf"# {name.upper()} MODEL #(.*?)(# PROJECT DEFINITION|$)"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""

    # Detect models included in the project
    model_match = re.search(r"models\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if not model_match:
        raise ValueError("No models defined in 'models=[...]'")

    model_names = re.findall(r'\b(\w+)\b', model_match.group(1))

    # Extract project metadata
    project_name = re.search(r'Project\s*\(\s*name\s*=\s*"(.*?)"', content)
    project_description = re.search(r'Metadata\s*\(\s*description\s*=\s*"(.*?)"', content)
    project_owner = re.search(r'owner\s*=\s*"(.*?)"', content)

    project_name = project_name.group(1) if project_name else "Unnamed Project"
    project_description = project_description.group(1) if project_description else "No description"
    project_owner = project_owner.group(1) if project_owner else "Unknown"

    section_extractors = {
        'domain_model': ('STRUCTURAL', ['OBJECT', 'AGENT', 'GUI', 'STATE MACHINE']),
        'object_model': ('OBJECT', ['AGENT', 'GUI', 'STATE MACHINE']),
        'agent': ('AGENT', ['GUI', 'STATE MACHINE']),
        'gui_model': ('GUI', ['STATE MACHINE']),
        'sm': ('STATE MACHINE', []),
    }

    diagram_jsons = {}

    # Extract domain_model first if exists (because we'll use it for object_model)
    domain_code = ""
    if "domain_model" in model_names:
        domain_code = extract_section(*section_extractors["domain_model"])

    for model_name in model_names:
        if model_name not in section_extractors:
            continue

        header, next_headers = section_extractors[model_name]
        section_code = extract_section(header, next_headers)

        diagram_id = str(uuid.uuid4())
        last_update = datetime.now(timezone.utc).isoformat()

        if model_name == "domain_model":
            parsed_domain_model = parse_buml_content(section_code)
            model = class_buml_to_json(parsed_domain_model)
            diagram_jsons["ClassDiagram"] = {
                "id": diagram_id,
                "title": "Class Diagram",
                "model": model,
                "lastUpdate": last_update
            }

        elif model_name == "object_model":
            combined_code = domain_code + "\n" + section_code
            model = object_buml_to_json(combined_code, diagram_jsons["ClassDiagram"].get("model", {}))
            diagram_jsons["ObjectDiagram"] = {
                "id": diagram_id,
                "title": "Object Diagram",
                "model": model,
                "lastUpdate": last_update
            }

        elif model_name == "agent":
            model = agent_buml_to_json(section_code)
            diagram_jsons["AgentDiagram"] = {
                "id": diagram_id,
                "title": "Agent Diagram",
                "model": model,
                "lastUpdate": last_update
            }

        elif model_name == "gui_model":
            model = gui_buml_to_json(section_code)
            diagram_jsons["GUINoCodeDiagram"] = {
                "id": diagram_id,
                "title": "GUI Diagram",
                "model": model,
                "lastUpdate": last_update
            }

        elif model_name == "sm":
            model = state_machine_to_json(section_code)
            diagram_jsons["StateMachineDiagram"] = {
                "id": diagram_id,
                "title": "State Machine Diagram",
                "model": model,
                "lastUpdate": last_update
            }

    project_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    diagram_defaults = {
        "ClassDiagram": "ClassDiagram",
        "ObjectDiagram": "ObjectDiagram",
        "AgentDiagram": "AgentDiagram",
        "StateMachineDiagram": "StateMachineDiagram",
        "GUINoCodeDiagram": "GUINoCodeDiagram"
    }

    for diagram_type, model_type in diagram_defaults.items():
        if diagram_type not in diagram_jsons:
            diagram_jsons[diagram_type] = {
                "id": str(uuid.uuid4()),
                "title": diagram_type.replace("Diagram", " Diagram"),
                "model": empty_model(model_type),
                "lastUpdate": datetime.now(timezone.utc).isoformat()
            }

    return {
        "id": project_id,
        "type": "Project",
        "name": project_name,
        "description": project_description,
        "owner": project_owner,
        "createdAt": created_at,
        "currentDiagramType": "ClassDiagram",
        "diagrams": diagram_jsons,
        "settings": {
            "defaultDiagramType": "ClassDiagram",
            "autoSave": True,
            "collaborationEnabled": False
        }
    }
