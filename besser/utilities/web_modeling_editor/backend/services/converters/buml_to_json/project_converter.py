"""
Project converter module for BUML to JSON conversion.
Handles project structure processing and diagram coordination.
Supports both single-diagram and multi-diagram per type formats.
"""

import ast
import hashlib
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple

from besser.utilities.web_modeling_editor.backend.services.converters.minted_values import (
    fixed_minted_values,
    new_id,
    next_id_scope,
    now_iso,
)
from besser.BUML.metamodel.structural import DomainModel
from besser.utilities.buml_code_builder.project_builder import is_user_reference_domain_model

from .class_diagram_converter import parse_buml_content, class_buml_to_json
from .state_machine_converter import state_machine_to_json
from .agent_diagram_converter import agent_buml_to_json
from .object_diagram_converter import object_buml_to_json
from .gui_diagram_converter import gui_buml_to_json
from .quantum_diagram_converter import quantum_buml_to_json
from .nn_diagram_converter import nn_buml_to_json
from .bpmn_diagram_converter import bpmn_buml_to_json
from .kg_diagram_converter import kg_buml_to_json
from .component_diagram_converter import component_buml_to_json
from .deployment_diagram_converter import deployment_buml_to_json

logger = logging.getLogger(__name__)

# Maps model variable name prefixes to (section header keyword, diagram type, default title)
SECTION_CONFIG = {
    'domain_model': ('STRUCTURAL', 'ClassDiagram', 'Class Diagram'),
    'object_model': ('OBJECT', 'ObjectDiagram', 'Object Diagram'),
    'agent': ('AGENT', 'AgentDiagram', 'Agent Diagram'),
    'gui_model': ('GUI', 'GUINoCodeDiagram', 'GUI Diagram'),
    'quantum_model': ('QUANTUM', 'QuantumCircuitDiagram', 'Quantum Circuit Diagram'),
    'kg_model': ('KNOWLEDGE_GRAPH', 'KnowledgeGraphDiagram', 'Knowledge Graph Diagram'),
    'sm': ('STATE MACHINE', 'StateMachineDiagram', 'State Machine Diagram'),
    'nn_model': ('NN', 'NNDiagram', 'NN Diagram'),
    'bpmn_model': ('BPMN', 'BPMNDiagram', 'BPMN Diagram'),
    'component_model': ('COMPONENT', 'ComponentDiagram', 'Component Diagram'),
    'deployment_model': ('DEPLOYMENT', 'DeploymentDiagram', 'Deployment Diagram'),
}

# A project export writes a UserDiagram as two ordinary sections (see
# project_builder): the user reference domain model under a STRUCTURAL header,
# bound to ``user_model``, and the diagram's objects under an OBJECT header,
# bound to ``user_object_model``. Those bindings tell them apart from class and
# object diagrams. Numbered (``_<n>``) variants appear when there are several.
_USER_MODEL_ASSIGN = re.compile(r'^user_model(?:_\d+)?\s*(?::[^=\n]*)?=(?!=)', re.MULTILINE)
_USER_OBJECT_MODEL_ASSIGN = re.compile(
    r'^user_object_model(?:_\d+)?\s*(?::[^=\n]*)?=(?!=)', re.MULTILINE,
)
_OBJECT_MODEL_NAME = re.compile(r'ObjectModel\(\s*name\s*=\s*"((?:[^"\\]|\\.)*)"')
def _parse_user_reference_section(section_code: str) -> Optional[DomainModel]:
    """Return the parsed model if a STRUCTURAL section is the user reference domain model, else ``None``.

    Older exports also bound an ordinary class diagram that merely has a class
    named ``User`` to ``user_model``; such a section has other classes and is
    imported as a class diagram.
    """
    if not _USER_MODEL_ASSIGN.search(section_code):
        return None
    try:
        domain_model = parse_buml_content(section_code)
    except Exception:  # noqa: BLE001 -- unparsable: let the class-diagram path report it
        return None
    return domain_model if is_user_reference_domain_model(domain_model) else None


# The name json_to_buml gives a model whose diagram has no title, by model
# variable prefix: such a name says nothing about the model, so its diagram
# gets the default title back rather than this name.
_UNTITLED_MODEL_NAMES = {
    'domain_model': 'unnamed',
    'object_model': 'Generated_Object_Model',
    'agent': 'Generated_Agent',
    'gui_model': 'GUI',
    'quantum_model': 'Quantum_Circuit',
    'kg_model': 'knowledge_graph',
    'sm': 'Generated_State_Machine',
    'nn_model': 'Neural_Network',
    'bpmn_model': 'Generated_BPMN_Model',
    'component_model': 'Generated_Component_Model',
    'deployment_model': 'Generated_Deployment_Model',
}


def _assigned_model_name(section_code: str, variables: List[str]) -> Optional[str]:
    """The name a section gives the model it binds to one of ``variables``, or ``None``.

    Reads ``var = Kind(name="...")`` or ``var = Kind("...")`` (the code builder
    writes one of these for every model); a section that does not parse, or
    builds the model some other way, has no name here.
    """
    try:
        tree = ast.parse(section_code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not isinstance(value, ast.Call):
            continue
        if not any(isinstance(target, ast.Name) and target.id in variables for target in targets):
            continue
        arguments = [kw.value for kw in value.keywords if kw.arg == "name"] or value.args[:1]
        if arguments and isinstance(arguments[0], ast.Constant) and isinstance(arguments[0].value, str):
            return arguments[0].value or None
    return None


def _model_title(name: Optional[str], base_name: str, default_title: str) -> Optional[str]:
    """The diagram title that gives a model back ``name`` on its way into B-UML, or ``None``.

    A diagram's title is what json_to_buml names its model, so a model keeps its
    name through the editor's JSON. A name that is the default title as json_to_buml
    writes it ("Class_Diagram"), or the name it gives an untitled diagram, keeps the
    default title.
    """
    if not name:
        return None
    if name in (default_title, default_title.replace(" ", "_"), _UNTITLED_MODEL_NAMES.get(base_name)):
        return default_title
    return name


def _object_model_title(section_code: str) -> str:
    """Return the ``ObjectModel(name=...)`` of an object section, or ''."""
    match = _OBJECT_MODEL_NAME.search(section_code)
    if not match:
        return ""
    try:
        return ast.literal_eval(f'"{match.group(1)}"')
    except (ValueError, SyntaxError):
        return match.group(1)


# All known section header keywords used as boundary markers
ALL_SECTION_KEYWORDS = [
    'STRUCTURAL', 'OBJECT', 'AGENT', 'GUI', 'QUANTUM', 'STATE MACHINE', 'NN',
    'BPMN', 'KNOWLEDGE_GRAPH', 'COMPONENT', 'DEPLOYMENT',
]


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

    # KnowledgeGraphDiagram stores nodes/edges rather than elements/relationships.
    if diagram_type == "KnowledgeGraphDiagram":
        return {
            "version": "1.0.0",
            "type": diagram_type,
            "nodes": [],
            "edges": [],
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


def _has_code(section_code: str) -> bool:
    """True when ``section_code`` holds a line that is neither blank nor a comment."""
    return any(line.strip() and not line.lstrip().startswith("#") for line in section_code.splitlines())


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
    # (number, title) of a header with no code under it, waiting for the header
    # that follows it straight away.
    carried = (None, None)
    for match in header_pattern.finditer(content):
        number = match.group(1) or carried[0]  # e.g. "1", "2", or None
        title = match.group(2) or carried[1]   # e.g. "User Model" or None
        carried = (None, None)
        section_start = match.end()

        # Find the next section boundary after this header
        next_boundary = re.search(boundary_pattern, content[section_start:])
        if next_boundary:
            section_code = content[section_start:section_start + next_boundary.start()].strip()
        else:
            # Last section in the file: take everything until end
            section_code = content[section_start:].strip()

        if not _has_code(section_code):
            # A header with nothing but comments (the ``####`` border of a
            # builder's own banner) under it. ``project_to_code`` writes a
            # numbered, titled header for each of several models right above
            # the banner its builder writes, so the title belongs to the
            # section that banner opens.
            logger.debug("Empty section found for keyword '%s' (number=%s)", keyword, number)
            if next_boundary and header_pattern.match(content, section_start + next_boundary.start()):
                carried = (number, title)
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
    user_reference_json: Dict[str, Any] = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convert a single section's code into a diagram JSON entry.

    Args:
        model_name: Model variable name prefix (e.g. 'domain_model', 'agent')
        section_code: The extracted Python code for this section
        title: Display title for the diagram
        domain_sections: All structural/domain sections (needed for object diagram context)
        class_diagram_list: Already-converted class diagrams (needed for object diagram references)
        user_reference_json: The user reference domain model as class-diagram
            JSON (needed for ``user_object_model`` sections)
        warnings: Receives a message for every repair made to load the section

    Returns:
        Dictionary with 'id', 'title', 'model', and 'lastUpdate' keys
    """
    next_id_scope()
    diagram_id = new_id()
    last_update = now_iso()

    try:
        if model_name == "domain_model":
            parsed = parse_buml_content(section_code, warnings)
            model = class_buml_to_json(parsed)

        elif model_name == "object_model":
            # Object diagrams need the domain model code prepended for context
            # Combine all domain sections as context prefix
            domain_code = "\n".join(code for _, code in domain_sections)
            combined_code = domain_code + "\n" + section_code if domain_code else section_code
            # Use the first class diagram's model as reference, or empty dict
            class_model_ref = class_diagram_list[0]["model"] if class_diagram_list else {}
            model = object_buml_to_json(combined_code, class_model_ref)

        elif model_name == "user_object_model":
            model = object_buml_to_json(
                section_code, user_reference_json or {"elements": {}, "relationships": {}},
                diagram_type="UserDiagram",
            )

        elif model_name == "agent":
            model = agent_buml_to_json(section_code)

        elif model_name == "gui_model":
            # A GUI section names the domain elements it binds to, so the
            # structural code is passed as context. The converter executes it
            # first and binds the ClassName_attributeName aliases the GUI
            # section uses -- including inherited attributes, which older
            # exports spell after the binding class rather than the declaring
            # one (a Guest table showing Person.id wrote Guest_id).
            domain_code = "\n".join(code for _, code in domain_sections)
            model = gui_buml_to_json(section_code, context_code=domain_code or None)

        elif model_name == "quantum_model":
            model = quantum_buml_to_json(section_code)

        elif model_name == "nn_model":
            model = nn_buml_to_json(section_code)

        elif model_name == "bpmn_model":
            model = bpmn_buml_to_json(section_code)

        elif model_name == "kg_model":
            model = kg_buml_to_json(section_code)

        elif model_name == "sm":
            model = state_machine_to_json(section_code)

        elif model_name == "component_model":
            model = component_buml_to_json(section_code)

        elif model_name == "deployment_model":
            model = deployment_buml_to_json(section_code)

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


# Detection heuristics for single-diagram BUML files that omit the Project(...)
# wrapper. Keep these in sync with the per-type checks in
# routers.conversion_router.get_single_json_model.
SINGLE_DIAGRAM_KEYWORDS: List[Tuple[str, Tuple[str, ...]]] = [
    # Order matters: 'agent' is checked before 'state_machine' because agent
    # files also contain StateMachine-style constructs.
    ('AgentDiagram', (
        'agent(', '.new_intent(', '.new_state(',
        'when_intent_matched', 'session.reply',
    )),
    ('StateMachineDiagram', (
        'statemachine(', 'when_event_go_to', '.add_transition', '.new_body',
    )),
    ('ClassDiagram', (
        'domainmodel(', '.create_class(', '.add_attribute(',
        '.create_association',
    )),
    ('GUINoCodeDiagram', (
        'guimodel(', '.new_screen(', '.new_module(',
        'viewcomponent', 'viewcontainer',
    )),
    ('NNDiagram', (
        '.add_layer(', '.add_tensor_op(', '.add_sub_nn(',
        '.add_configuration(', '.add_train_data(', '.add_test_data(',
    )),
    ('BPMNDiagram', (
        'bpmnmodel(', '.add_process(', '.add_flow_node(',
        '.add_sequence_flow(',
    )),
    ('ComponentDiagram', (
        'componentmodel(', '.add_component(', '.add_relationship(',
        'agentcategory.', 'agenticedge(',
    )),
    ('DeploymentDiagram', (
        'deploymentmodel(', '.add_artifact(', 'deploymentrelation(',
        'communicationpath(',
    )),
]

# Single source of truth for the diagram types every imported project carries
# (one slot per type, with an empty placeholder when the source file has none)
# and their default titles. Insertion order is the order of the ``diagrams`` /
# ``currentDiagramIndices`` keys in the returned project. Keys are diagram
# *model* types; ``_fill_diagram_defaults`` re-keys the BPMN bucket for WME.
DIAGRAM_DEFAULT_TITLES: Dict[str, str] = {
    'ClassDiagram': 'Class Diagram',
    'ObjectDiagram': 'Object Diagram',
    'AgentDiagram': 'Agent Diagram',
    'StateMachineDiagram': 'State Machine Diagram',
    'GUINoCodeDiagram': 'GUI Diagram',
    'QuantumCircuitDiagram': 'Quantum Circuit Diagram',
    'NNDiagram': 'NN Diagram',
    'BPMNDiagram': 'BPMN Diagram',
    'ComponentDiagram': 'Component Diagram',
    'DeploymentDiagram': 'Deployment Diagram',
    'UserDiagram': 'User Diagram',
    'KnowledgeGraphDiagram': 'Knowledge Graph Diagram',
}


def _fill_diagram_defaults(
    diagram_jsons: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    """Give every known diagram type a slot and an active index.

    Types missing from *diagram_jsons* get one empty placeholder diagram. Every
    bucket in the result -- known type or not -- gets a
    ``currentDiagramIndices`` entry, so the frontend never sees a bucket
    without an active index. The BPMN bucket is re-keyed as ``"BPMN"`` (WME's
    key; ``model.type`` stays ``"BPMNDiagram"``).

    Returns:
        ``(diagrams, current_diagram_indices)``
    """
    filled: Dict[str, List[Dict[str, Any]]] = {}
    for diagram_type, title in DIAGRAM_DEFAULT_TITLES.items():
        if diagram_jsons.get(diagram_type):
            filled[diagram_type] = diagram_jsons[diagram_type]
            continue
        next_id_scope()
        filled[diagram_type] = [{
            "id": new_id(),
            "title": title,
            "model": empty_model(diagram_type),
            "lastUpdate": now_iso(),
        }]
    # Keep any bucket of a type not listed above (canonical types first).
    for diagram_type, entries in diagram_jsons.items():
        filled.setdefault(diagram_type, entries)

    filled["BPMN"] = filled.pop("BPMNDiagram")

    current_diagram_indices = {diagram_type: 0 for diagram_type in filled}
    return filled, current_diagram_indices


def _detect_single_diagram_type(content: str) -> str:
    """Return the diagram type for a single-diagram BUML file, or '' if unknown."""
    content_lower = content.lower()
    # AgentDiagram wins over StateMachineDiagram because agent files share
    # state-machine vocabulary; this matches the precedence used in
    # get_single_json_model.
    agent_match = any(kw in content_lower for kw in SINGLE_DIAGRAM_KEYWORDS[0][1])
    if agent_match:
        return 'AgentDiagram'
    for diagram_type, keywords in SINGLE_DIAGRAM_KEYWORDS[1:]:
        if any(kw in content_lower for kw in keywords):
            return diagram_type
    return ''


def _build_project_from_single_diagram(
    content: str, warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Wrap a single-diagram BUML file (no Project(...)) into a project structure."""
    diagram_type = _detect_single_diagram_type(content)
    if not diagram_type:
        raise ValueError(
            "No models defined in 'models=[...]' and the file was not recognized "
            "as a single-diagram BUML file. Supported single-diagram types: "
            "ClassDiagram, AgentDiagram, StateMachineDiagram, GUINoCodeDiagram, "
            "NNDiagram, BPMNDiagram, ComponentDiagram, DeploymentDiagram."
        )

    title = DIAGRAM_DEFAULT_TITLES[diagram_type]
    next_id_scope()
    try:
        if diagram_type == 'ClassDiagram':
            parsed = parse_buml_content(content, warnings)
            model = class_buml_to_json(parsed)
        elif diagram_type == 'AgentDiagram':
            model = agent_buml_to_json(content)
        elif diagram_type == 'StateMachineDiagram':
            model = state_machine_to_json(content)
        elif diagram_type == 'GUINoCodeDiagram':
            model = gui_buml_to_json(content)
        elif diagram_type == 'NNDiagram':
            model = nn_buml_to_json(content)
        elif diagram_type == 'BPMNDiagram':
            model = bpmn_buml_to_json(content)
        elif diagram_type == 'ComponentDiagram':
            model = component_buml_to_json(content)
        elif diagram_type == 'DeploymentDiagram':
            model = deployment_buml_to_json(content)
        else:
            raise ValueError(f"Unsupported single-diagram type: {diagram_type}")
    except (SyntaxError, ValueError, TypeError) as e:
        raise ValueError(
            f"Failed to parse single-diagram BUML file (detected as {diagram_type}): {e}"
        ) from e

    filled_entry = {
        "id": new_id(),
        "title": title,
        "model": model,
        "lastUpdate": now_iso(),
    }

    diagram_jsons, current_diagram_indices = _fill_diagram_defaults(
        {diagram_type: [filled_entry]}
    )
    current_diagram_type = "BPMN" if diagram_type == "BPMNDiagram" else diagram_type

    next_id_scope()
    return {
        "id": new_id(),
        "type": "Project",
        "schemaVersion": 3,
        "name": "Imported Project",
        "description": "Imported from single-diagram BUML file",
        "owner": "Unknown",
        "createdAt": now_iso(),
        "currentDiagramType": current_diagram_type,
        "currentDiagramIndices": current_diagram_indices,
        "diagrams": diagram_jsons,
        "settings": {
            "defaultDiagramType": diagram_type,
            "autoSave": True,
            "collaborationEnabled": False,
        },
    }


def project_to_json(
    content: str,
    warnings: Optional[List[str]] = None,
    *,
    id_seed: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert a BUML project content to JSON format matching the frontend structure.

    Supports both the legacy single-diagram-per-type format and the new
    multi-diagram-per-type format with numbered/titled section headers.
    Also accepts plain single-diagram BUML files that omit the Project(...)
    wrapper — they are wrapped into a one-diagram project so that the
    Project Import flow works with any well-formed BUML file.

    The ids (project, diagrams, elements) and timestamps (``createdAt``,
    ``lastUpdate``) the conversion makes up are fresh by default, as the editor
    wants on import. With ``id_seed`` every such id is derived from it, and with
    ``timestamp`` every such time is that value (ISO 8601), so the same content
    and the same arguments give the same JSON; pass both for that.

    Args:
        content: Project Python code as string
        warnings: Receives a message for every repair made to load a model
        id_seed: Derive the minted ids from this string instead of minting fresh uuids
        timestamp: Record this time instead of the current one

    Returns:
        Dictionary representing the complete project with all diagrams.
        Each diagram type maps to a list of diagram entries.
    """
    if id_seed is None and timestamp is None:
        return _project_to_json(content, warnings)
    if id_seed is None or timestamp is None:
        raise ValueError("project_to_json fixes the ids and the timestamps together: pass both or neither")
    with fixed_minted_values(id_seed, timestamp):
        return _project_to_json(content, warnings)


# An editor id: a uuid, or what the GUI converter mints ("page-0123456789").
_EDITOR_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|\b[a-z-]+-[0-9a-f]{10}\b")


def reproducible_project_to_json(
    content: str, warnings: Optional[List[str]] = None, *, timestamp: str,
) -> Dict[str, Any]:
    """``project_to_json`` whose output depends on what it holds alone: the same models give the same JSON.

    The ids are derived from a hash of the project without its ids (a first
    conversion with a constant seed), so two B-UML files that differ only in
    form (variable names, comments, a JSON -> B-UML round trip) give the same
    ids, and different projects get different ones. Every time is ``timestamp``.
    """
    draft = project_to_json(content, [], id_seed="", timestamp=timestamp)
    # In the converters' own order: sorting the keys would order elements by their ids.
    held = _EDITOR_ID.sub("", json.dumps(draft))
    seed = hashlib.sha256(held.encode("utf-8")).hexdigest()
    return project_to_json(content, warnings, id_seed=seed, timestamp=timestamp)


def _project_to_json(content: str, warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    """``project_to_json`` with the ids and timestamps ``minted_values`` gives."""
    # Detect models included in the project
    model_match = re.search(r"models\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if not model_match:
        # No Project(...) wrapper — fall back to single-diagram detection so
        # plain class/agent/state-machine/GUI/NN files can still be imported.
        logger.info(
            "No 'models=[...]' declaration found; using single-diagram fallback."
        )
        return _build_project_from_single_diagram(content, warnings)

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
    base_names = {re.sub(r'_\d+$', '', name) for name in model_names}
    if base_names & {"domain_model", "user_model"}:
        keyword = SECTION_CONFIG["domain_model"][0]
        domain_sections = _extract_all_sections(content, keyword)
        logger.debug("Found %d structural model section(s)", len(domain_sections))

    # UserDiagrams: the user reference domain model is context, not a class
    # diagram, and each user_object_model section becomes a UserDiagram.
    user_reference_models: List[DomainModel] = []
    class_sections: List[Tuple[str, str]] = []
    for section in domain_sections:
        reference_model = _parse_user_reference_section(section[1])
        if reference_model is None:
            class_sections.append(section)
        else:
            user_reference_models.append(reference_model)
    domain_sections = class_sections
    user_object_sections: List[Tuple[str, str]] = []
    user_reference_json: Dict[str, Any] = {}
    if base_names & {"object_model", "user_object_model"}:
        object_sections = _extract_all_sections(content, SECTION_CONFIG["object_model"][0])
        if user_reference_models:
            user_object_sections = [
                section for section in object_sections
                if _USER_OBJECT_MODEL_ASSIGN.search(section[1])
            ]
            user_reference_json = class_buml_to_json(user_reference_models[0])
    # Without the reference model, a user_object_model section instantiates an
    # ordinary class diagram (older exports) and is imported as an object diagram.
    model_aliases = {"user_model": "domain_model"}
    if not user_object_sections:
        model_aliases["user_object_model"] = "object_model"

    # Process each model type referenced in the project
    # Use a deduplicated ordered list to process each base model name only once.
    # Variable names may carry numeric suffixes (e.g. domain_model_1, agent_2)
    # generated by project_builder._suffixed_name for multi-diagram projects.
    # Strip trailing _<digits> to recover the base name used in SECTION_CONFIG.
    seen_base_names = set()
    for model_name in model_names:
        # Strip numeric suffix: "domain_model_1" -> "domain_model", "agent_2" -> "agent"
        base_name = re.sub(r'_\d+$', '', model_name)
        base_name = model_aliases.get(base_name, base_name)

        if base_name in seen_base_names:
            continue
        seen_base_names.add(base_name)

        if base_name == "user_object_model":
            diagram_jsons["UserDiagram"] = [
                _convert_section(
                    base_name, section_code,
                    _object_model_title(section_code) or title
                    or DIAGRAM_DEFAULT_TITLES["UserDiagram"],
                    domain_sections, [], user_reference_json,
                )
                for title, section_code in user_object_sections
            ]
            continue

        if base_name not in SECTION_CONFIG:
            logger.debug("Skipping unknown model name '%s' (base: '%s')", model_name, base_name)
            continue

        keyword, diagram_type, default_title = SECTION_CONFIG[base_name]
        if base_name == "domain_model":
            sections = domain_sections
        elif base_name == "object_model":
            sections = [
                section for section in _extract_all_sections(content, keyword)
                if section not in user_object_sections
            ]
        else:
            sections = _extract_all_sections(content, keyword)

        if not sections:
            logger.info("No sections found for '%s' (keyword: %s), skipping", base_name, keyword)
            continue

        variables = [name for name in model_names if model_aliases.get(re.sub(r'_\d+$', '', name),
                                                                          re.sub(r'_\d+$', '', name)) == base_name]
        diagram_list = []
        for idx, (title, section_code) in enumerate(sections):
            # A header's title, else the model's own name, else the default title
            # (numbered when there are several).
            resolved_title = title or _model_title(
                _assigned_model_name(section_code, variables), base_name, default_title,
            )
            if not resolved_title:
                resolved_title = f"{default_title} {idx + 1}" if len(sections) > 1 else default_title

            logger.debug(
                "Converting section %d/%d for %s: '%s'",
                idx + 1, len(sections), diagram_type, resolved_title
            )

            # class_diagram_list is only needed for object_model; pass current ClassDiagram list
            class_diagrams = diagram_jsons.get("ClassDiagram", [])
            entry = _convert_section(
                base_name, section_code, resolved_title,
                domain_sections, class_diagrams, warnings=warnings,
            )
            if entry is not None:
                diagram_list.append(entry)

        if diagram_list:
            diagram_jsons[diagram_type] = diagram_list

    # Section-header fallback: some project files declare `models=[domain_model]`
    # but omit the `# STRUCTURAL MODEL #` (or sibling) section headers entirely —
    # all the model code lives in one flat block above the project definition.
    # When this happens, `diagram_jsons` ends up empty and every diagram type
    # gets an empty placeholder, which the frontend renders as a blank canvas.
    # Reuse the single-diagram detector to recover the actual model, then
    # restore the project name/description/owner from the Project(...) wrapper.
    # A detected model that fails to parse raises: swallowing it here imported
    # the file as a blank canvas with no error.
    if not diagram_jsons and _detect_single_diagram_type(content):
        logger.info(
            "Project declares models=%s but no section headers were found; "
            "falling back to single-diagram detection.", model_names,
        )
        result = _build_project_from_single_diagram(content, warnings)
        result["name"] = project_name
        result["description"] = project_description
        result["owner"] = project_owner
        return result

    next_id_scope()
    project_id = new_id()
    created_at = now_iso()

    # Ensure every diagram type has at least one entry (empty default)
    diagram_jsons, current_diagram_indices = _fill_diagram_defaults(diagram_jsons)

    return {
        "id": project_id,
        "type": "Project",
        "schemaVersion": 3,
        "name": project_name,
        "description": project_description,
        "owner": project_owner,
        "createdAt": created_at,
        "currentDiagramType": "ClassDiagram",
        "currentDiagramIndices": current_diagram_indices,
        "diagrams": diagram_jsons,
        "settings": {
            "defaultDiagramType": "ClassDiagram",
            "autoSave": True,
            "collaborationEnabled": False,
        },
    }
