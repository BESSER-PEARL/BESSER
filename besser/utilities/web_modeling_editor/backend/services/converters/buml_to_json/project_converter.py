"""
Project converter module for BUML to JSON conversion.
Handles project structure processing and diagram coordination.
Supports both single-diagram and multi-diagram per type formats.
"""

import logging
import uuid
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

from .class_diagram_converter import parse_buml_content, class_buml_to_json
from .state_machine_converter import state_machine_to_json
from .agent_diagram_converter import agent_buml_to_json
from .object_diagram_converter import object_buml_to_json
from .gui_diagram_converter import gui_buml_to_json
from .quantum_diagram_converter import quantum_buml_to_json
from .nn_diagram_converter import nn_buml_to_json

logger = logging.getLogger(__name__)

# Maps model variable name prefixes to (section header keyword, diagram type, default title)
SECTION_CONFIG = {
    'domain_model': ('STRUCTURAL', 'ClassDiagram', 'Class Diagram'),
    'object_model': ('OBJECT', 'ObjectDiagram', 'Object Diagram'),
    'agent': ('AGENT', 'AgentDiagram', 'Agent Diagram'),
    'gui_model': ('GUI', 'GUINoCodeDiagram', 'GUI Diagram'),
    'quantum_model': ('QUANTUM', 'QuantumCircuitDiagram', 'Quantum Circuit Diagram'),
    'sm': ('STATE MACHINE', 'StateMachineDiagram', 'State Machine Diagram'),
    'nn_model': ('NN', 'NNDiagram', 'NN Diagram'),
}

# All known section header keywords used as boundary markers
ALL_SECTION_KEYWORDS = ['STRUCTURAL', 'OBJECT', 'AGENT', 'GUI', 'QUANTUM', 'STATE MACHINE', 'NN']


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
            "pages": [],
            "styles": [],
            "assets": [],
            "symbols": []
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


def _build_section_boundary_pattern() -> str:
    """
    Build a regex alternation matching any section header or the project definition marker.

    Returns:
        Regex pattern string matching known section boundaries
    """
    keyword_alts = '|'.join(re.escape(kw) for kw in ALL_SECTION_KEYWORDS)
    # Match both old format: # KEYWORD MODEL #
    # and new numbered format: # KEYWORD MODEL 1: "Title" #
    return rf'#\s*(?:{keyword_alts})\s+MODEL(?:\s+\d+)?(?:\s*:\s*"[^"]*")?\s*#|# PROJECT DEFINITION'


def _extract_all_sections(content: str, keyword: str) -> List[Tuple[str, str]]:
    """
    Extract ALL sections matching a given type keyword from the project content.

    Supports both the old single-section format:
        # STRUCTURAL MODEL #
    and the new multi-section numbered format:
        # STRUCTURAL MODEL 1: "User Model" #
        # STRUCTURAL MODEL 2: "Product Model" #

    Args:
        content: Full project Python code as string
        keyword: Section keyword to search for (e.g. 'STRUCTURAL', 'AGENT')

    Returns:
        List of (title, section_code) tuples. Title is extracted from the header
        if present, otherwise a default is used based on the keyword.
    """
    # Pattern matches:
    #   # KEYWORD MODEL #                          (old format, no number, no title)
    #   # KEYWORD MODEL 1 #                        (numbered, no title)
    #   # KEYWORD MODEL 1: "Some Title" #          (numbered with title)
    #   #  KEYWORD MODEL  #                        (extra whitespace, e.g. GUI)
    header_pattern = re.compile(
        rf'#\s*{re.escape(keyword)}\s+MODEL(?:\s+(\d+))?(?:\s*:\s*"([^"]*)")?\s*#',
        re.IGNORECASE
    )

    boundary_pattern = _build_section_boundary_pattern()

    sections = []
    for match in header_pattern.finditer(content):
        number = match.group(1)  # e.g. "1", "2", or None
        title = match.group(2)   # e.g. "User Model" or None
        section_start = match.end()

        # Find the next section boundary after this header
        next_boundary = re.search(boundary_pattern, content[section_start:])
        if next_boundary:
            section_code = content[section_start:section_start + next_boundary.start()].strip()
        else:
            # Last section in the file: take everything until end
            section_code = content[section_start:].strip()

        if not section_code:
            logger.debug("Empty section found for keyword '%s' (number=%s)", keyword, number)
            continue

        # Build a descriptive title
        if title:
            resolved_title = title
        elif number:
            resolved_title = f"{keyword.title()} Model {number}"
        else:
            resolved_title = None  # Caller will use default

        sections.append((resolved_title, section_code))

    return sections


def _convert_section(
    model_name: str,
    section_code: str,
    title: str,
    domain_sections: List[Tuple[str, str]],
    class_diagram_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Convert a single section's code into a diagram JSON entry.

    Args:
        model_name: Model variable name prefix (e.g. 'domain_model', 'agent')
        section_code: The extracted Python code for this section
        title: Display title for the diagram
        domain_sections: All structural/domain sections (needed for object diagram context)
        class_diagram_list: Already-converted class diagrams (needed for object diagram references)

    Returns:
        Dictionary with 'id', 'title', 'model', and 'lastUpdate' keys
    """
    diagram_id = str(uuid.uuid4())
    last_update = datetime.now(timezone.utc).isoformat()

    try:
        if model_name == "domain_model":
            parsed = parse_buml_content(section_code)
            model = class_buml_to_json(parsed)

        elif model_name == "object_model":
            # Object diagrams need the domain model code prepended for context
            # Combine all domain sections as context prefix
            domain_code = "\n".join(code for _, code in domain_sections)
            combined_code = domain_code + "\n" + section_code if domain_code else section_code
            # Use the first class diagram's model as reference, or empty dict
            class_model_ref = class_diagram_list[0]["model"] if class_diagram_list else {}
            model = object_buml_to_json(combined_code, class_model_ref)

        elif model_name == "agent":
            model = agent_buml_to_json(section_code)

        elif model_name == "gui_model":
            model = gui_buml_to_json(section_code)

        elif model_name == "quantum_model":
            model = quantum_buml_to_json(section_code)

        elif model_name == "nn_model":
            model = nn_buml_to_json(section_code)

        elif model_name == "sm":
            model = state_machine_to_json(section_code)

        else:
            logger.warning("Unknown model name '%s', skipping conversion", model_name)
            return None

    except (SyntaxError, ValueError, TypeError) as e:
        logger.error(
            "Failed to convert section '%s' (type: %s): %s",
            title, model_name, e, exc_info=True,
        )
        raise ValueError(
            f"Failed to convert '{title}' section ({model_name}): {e}"
        ) from e

    return {
        "id": diagram_id,
        "title": title,
        "model": model,
        "lastUpdate": last_update,
    }


def project_to_json(content: str) -> Dict[str, Any]:
    """
    Convert a BUML project content to JSON format matching the frontend structure.

    Supports both the legacy single-diagram-per-type format and the new
    multi-diagram-per-type format with numbered/titled section headers.

    Args:
        content: Project Python code as string

    Returns:
        Dictionary representing the complete project with all diagrams.
        Each diagram type maps to a list of diagram entries.
    """
    # Detect models included in the project
    model_match = re.search(r"models\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if not model_match:
        raise ValueError("No models defined in 'models=[...]'")

    model_names = re.findall(r'\b(\w+)\b', model_match.group(1))
    logger.debug("Detected model names in project: %s", model_names)

    # Extract project metadata
    project_name_match = re.search(r'Project\s*\(\s*name\s*=\s*"(.*?)"', content)
    project_desc_match = re.search(r'Metadata\s*\(\s*description\s*=\s*"(.*?)"', content)
    project_owner_match = re.search(r'owner\s*=\s*"(.*?)"', content)

    project_name = project_name_match.group(1) if project_name_match else "Unnamed Project"
    project_description = project_desc_match.group(1) if project_desc_match else "No description"
    project_owner = project_owner_match.group(1) if project_owner_match else "Unknown"

    # diagram_jsons maps diagram type -> list of diagram entries
    diagram_jsons: Dict[str, List[Dict[str, Any]]] = {}

    # First pass: extract all structural/domain sections (needed as context for object diagrams)
    # Check for both exact "domain_model" and suffixed variants like "domain_model_1"
    domain_sections: List[Tuple[str, str]] = []
    has_domain_model = any(re.sub(r'_\d+$', '', name) == "domain_model" for name in model_names)
    if has_domain_model:
        keyword = SECTION_CONFIG["domain_model"][0]
        domain_sections = _extract_all_sections(content, keyword)
        logger.debug("Found %d structural model section(s)", len(domain_sections))

    # Process each model type referenced in the project
    # Use a deduplicated ordered list to process each base model name only once.
    # Variable names may carry numeric suffixes (e.g. domain_model_1, agent_2)
    # generated by project_builder._suffixed_name for multi-diagram projects.
    # Strip trailing _<digits> to recover the base name used in SECTION_CONFIG.
    seen_base_names = set()
    for model_name in model_names:
        # Strip numeric suffix: "domain_model_1" -> "domain_model", "agent_2" -> "agent"
        base_name = re.sub(r'_\d+$', '', model_name)

        if base_name in seen_base_names:
            continue
        seen_base_names.add(base_name)

        if base_name not in SECTION_CONFIG:
            logger.debug("Skipping unknown model name '%s' (base: '%s')", model_name, base_name)
            continue

        keyword, diagram_type, default_title = SECTION_CONFIG[base_name]
        sections = _extract_all_sections(content, keyword)

        if not sections:
            logger.info("No sections found for '%s' (keyword: %s), skipping", base_name, keyword)
            continue

        diagram_list = []
        for idx, (title, section_code) in enumerate(sections):
            resolved_title = title if title else default_title
            # Append index suffix for multi-diagram types (only when more than one)
            if len(sections) > 1 and not title:
                resolved_title = f"{default_title} {idx + 1}"

            logger.debug(
                "Converting section %d/%d for %s: '%s'",
                idx + 1, len(sections), diagram_type, resolved_title
            )

            # class_diagram_list is only needed for object_model; pass current ClassDiagram list
            class_diagrams = diagram_jsons.get("ClassDiagram", [])
            entry = _convert_section(
                base_name, section_code, resolved_title,
                domain_sections, class_diagrams,
            )
            if entry is not None:
                diagram_list.append(entry)

        if diagram_list:
            diagram_jsons[diagram_type] = diagram_list

    project_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    # Ensure every diagram type has at least one entry (empty default)
    diagram_defaults = {
        "ClassDiagram": "ClassDiagram",
        "ObjectDiagram": "ObjectDiagram",
        "AgentDiagram": "AgentDiagram",
        "StateMachineDiagram": "StateMachineDiagram",
        "GUINoCodeDiagram": "GUINoCodeDiagram",
        "QuantumCircuitDiagram": "QuantumCircuitDiagram",
        "NNDiagram": "NNDiagram",
    }

    for diagram_type, model_type in diagram_defaults.items():
        if diagram_type not in diagram_jsons:
            diagram_jsons[diagram_type] = [{
                "id": str(uuid.uuid4()),
                "title": diagram_type.replace("Diagram", " Diagram"),
                "model": empty_model(model_type),
                "lastUpdate": datetime.now(timezone.utc).isoformat(),
            }]

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
            "collaborationEnabled": False,
        },
    }
