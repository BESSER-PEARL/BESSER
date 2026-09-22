"""
Project converter module for BUML to JSON conversion.
Handles project structure processing and diagram coordination.
Supports both single-diagram and multi-diagram per type formats.
"""

import ast
import logging
import uuid
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from .class_diagram_converter import parse_buml_content, class_buml_to_json
from .state_machine_converter import state_machine_to_json
from .agent_diagram_converter import agent_buml_to_json
from .object_diagram_converter import object_buml_to_json
from .gui_diagram_converter import gui_buml_to_json
from .quantum_diagram_converter import quantum_buml_to_json
from .nn_diagram_converter import nn_buml_to_json
from .bpmn_diagram_converter import bpmn_buml_to_json

logger = logging.getLogger(__name__)

# Maps the *constructor* used in a model's assignment (e.g. ``ObjectModel`` in
# ``object_model_1 = ObjectModel(...)``) to (diagram type, default title).
#
# This is the authoritative source of a model's diagram type: it is derived from
# the actual Python assignment recovered from the AST, NOT from a comment banner.
# A stray/duplicate/reformatted ``# ... MODEL #`` banner therefore cannot change
# the diagram set — only the ``Project(models=[...])`` list and the constructors
# of the referenced variables can.
CONSTRUCTOR_TO_DIAGRAM: Dict[str, Tuple[str, str]] = {
    'DomainModel': ('ClassDiagram', 'Class Diagram'),
    'ObjectModel': ('ObjectDiagram', 'Object Diagram'),
    'Agent': ('AgentDiagram', 'Agent Diagram'),
    'GUIModel': ('GUINoCodeDiagram', 'GUI Diagram'),
    'QuantumCircuit': ('QuantumCircuitDiagram', 'Quantum Circuit Diagram'),
    'StateMachine': ('StateMachineDiagram', 'State Machine Diagram'),
    'NN': ('NNDiagram', 'NN Diagram'),
    'BPMNModel': ('BPMNDiagram', 'BPMN Diagram'),
}


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


# ---------------------------------------------------------------------------
# AST-based sectioning
#
# Instead of counting comment banners, we recover each model's Python source
# chunk from the module AST: the model's own assignment plus the transitive
# closure of module-level definitions it references. The diagram set is driven
# entirely by the ``Project(models=[...])`` list, so stray/duplicate banners are
# irrelevant.
# ---------------------------------------------------------------------------


def _call_func_name(call: ast.Call) -> Optional[str]:
    """Return the simple callable name of a Call (``Foo(...)`` -> 'Foo', ``x.Foo(...)`` -> 'Foo')."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _expr_root_name(node: ast.AST) -> Optional[str]:
    """Walk the value/func chain of an Attribute/Call/Subscript down to its root Name id."""
    while True:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Subscript):
            node = node.value
        else:
            return None


def _target_base_names(target: ast.AST) -> set:
    """Base variable names bound or mutated by an assignment target.

    ``x`` -> {'x'}; ``x.attr`` / ``x.attr = ...`` -> {'x'} (mutation of x);
    ``x[i]`` -> {'x'}; tuple/list targets -> union of their elements.
    """
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Attribute, ast.Subscript)):
        root = _expr_root_name(target)
        return {root} if root else set()
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set = set()
        for elt in target.elts:
            names |= _target_base_names(elt)
        return names
    return set()


def _statement_defines(stmt: ast.stmt) -> set:
    """Names a top-level statement binds *or* mutates.

    Mutations count as definitions so that e.g. ``Book.attributes = {...}`` is
    pulled into any chunk that already needs ``Book``.
    """
    names: set = set()
    if isinstance(stmt, ast.Import):
        for alias in stmt.names:
            names.add((alias.asname or alias.name).split('.')[0])
    elif isinstance(stmt, ast.ImportFrom):
        for alias in stmt.names:
            names.add(alias.asname or alias.name)
    elif isinstance(stmt, ast.Assign):
        for tgt in stmt.targets:
            names |= _target_base_names(tgt)
    elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        names |= _target_base_names(stmt.target)
    elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.add(stmt.name)
    elif isinstance(stmt, ast.Expr):
        # Bare expression such as ``book.add_constraint(c)`` mutates ``book``.
        root = _expr_root_name(stmt.value)
        if root:
            names.add(root)
    return names


def _statement_uses(stmt: ast.stmt) -> set:
    """Names a statement references (over-approximated as every Name it contains).

    Imports reference nothing at module scope. Over-approximating is safe: it can
    only pull *more* definitions into a chunk, never fewer, and generated
    statements never mention two different model variables at once.
    """
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return set()
    return {node.id for node in ast.walk(stmt) if isinstance(node, ast.Name)}


def _assignment_name_and_call(stmt: ast.stmt) -> Tuple[Optional[str], Optional[ast.Call]]:
    """For ``name = Call(...)`` or ``name: Ann = Call(...)`` return (name, call)."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
            and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Call):
        return stmt.targets[0].id, stmt.value
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) \
            and isinstance(stmt.value, ast.Call):
        return stmt.target.id, stmt.value
    return None, None


def _extract_name_kwarg(call: ast.Call) -> Optional[str]:
    """Return the string value of the ``name=`` keyword of a constructor call, if any."""
    for kw in call.keywords:
        if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def _project_model_names(tree: ast.Module) -> List[str]:
    """Return the ordered variable names in ``Project(models=[...])``.

    Returns an empty list when no ``Project(...)`` call carrying a ``models=``
    list/tuple of plain names is present.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_func_name(node) == "Project":
            for kw in node.keywords:
                if kw.arg == "models" and isinstance(kw.value, (ast.List, ast.Tuple)):
                    return [elt.id for elt in kw.value.elts if isinstance(elt, ast.Name)]
    return []


def _closure_indices(root_name: str, name_to_def_indices: Dict[str, List[int]],
                     uses_by_index: Dict[int, set]) -> set:
    """Transitive closure of statement indices needed to define ``root_name``.

    Every statement that defines or mutates a needed name is pulled in, and each
    pulled-in statement contributes its own referenced names to the frontier.
    """
    included: set = set()
    needed = {root_name}
    processed: set = set()
    while needed - processed:
        name = (needed - processed).pop()
        processed.add(name)
        for idx in name_to_def_indices.get(name, ()):
            if idx not in included:
                included.add(idx)
                needed |= uses_by_index[idx]
    return included


def _source_of(content: str, tree: ast.Module, indices: set) -> str:
    """Join the original source segments of the given top-level statement indices, in order."""
    segments = []
    for idx in sorted(indices):
        segment = ast.get_source_segment(content, tree.body[idx])
        if segment:
            segments.append(segment)
    return "\n\n".join(segments)


def _convert_model(
    diagram_type: str,
    section_code: str,
    title: str,
    domain_prefix_code: str,
    class_diagram_list: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Convert a single model's source chunk into a diagram JSON entry.

    Args:
        diagram_type: Frontend diagram type (e.g. 'ClassDiagram', 'ObjectDiagram').
        section_code: The Python source chunk for this model (see _build_ast_sections).
        title: Display title for the diagram.
        domain_prefix_code: Concatenated source of all class-diagram models. Object
            diagrams need the domain-model code prepended for context (preserved
            from the previous banner-based behaviour).
        class_diagram_list: Already-converted class diagrams (object diagrams use the
            first one as their class reference).

    Returns:
        Dictionary with 'id', 'title', 'model', and 'lastUpdate' keys, or None.
    """
    diagram_id = str(uuid.uuid4())
    last_update = datetime.now(timezone.utc).isoformat()

    try:
        if diagram_type == "ClassDiagram":
            parsed = parse_buml_content(section_code)
            model = class_buml_to_json(parsed)

        elif diagram_type == "ObjectDiagram":
            # Object diagrams need the domain-model code prepended for context.
            combined_code = (
                domain_prefix_code + "\n" + section_code if domain_prefix_code else section_code
            )
            # Use the first class diagram's model as reference, or empty dict.
            class_model_ref = class_diagram_list[0]["model"] if class_diagram_list else {}
            model = object_buml_to_json(combined_code, class_model_ref)

        elif diagram_type == "AgentDiagram":
            model = agent_buml_to_json(section_code)

        elif diagram_type == "GUINoCodeDiagram":
            model = gui_buml_to_json(section_code)

        elif diagram_type == "QuantumCircuitDiagram":
            model = quantum_buml_to_json(section_code)

        elif diagram_type == "NNDiagram":
            model = nn_buml_to_json(section_code)

        elif diagram_type == "BPMNDiagram":
            model = bpmn_buml_to_json(section_code)

        elif diagram_type == "StateMachineDiagram":
            model = state_machine_to_json(section_code)

        else:
            logger.warning("Unknown diagram type '%s', skipping conversion", diagram_type)
            return None

    except (SyntaxError, ValueError, TypeError) as e:
        logger.error(
            "Failed to convert '%s' (type: %s): %s",
            title, diagram_type, e, exc_info=True,
        )
        raise ValueError(
            f"Failed to convert '{title}' ({diagram_type}): {e}"
        ) from e

    return {
        "id": diagram_id,
        "title": title,
        "model": model,
        "lastUpdate": last_update,
    }


def _build_ast_sections(content: str) -> Optional[List[Dict[str, Any]]]:
    """Derive per-model source chunks from the AST, driven by ``Project(models=[...])``.

    Returns an ordered list (matching the ``models=[...]`` order) of dicts with
    keys ``var``, ``diagram_type``, ``title`` and ``section_code``. Object-diagram
    chunks have the domain-model statements removed (they are supplied separately
    via the prepended domain prefix). The domain prefix itself is attached to every
    object entry under ``domain_prefix_code``.

    Returns ``None`` when the file has no ``Project(models=[...])`` wrapper (the
    caller then uses the single-diagram fallback).
    """
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        logger.warning("Could not AST-parse project content: %s", e)
        return None

    model_names = _project_model_names(tree)
    if not model_names:
        return None

    logger.debug("Project models=[...] order: %s", model_names)

    # Index every top-level statement's defines/uses.
    name_to_def_indices: Dict[str, List[int]] = {}
    uses_by_index: Dict[int, set] = {}
    ctor_by_name: Dict[str, Tuple[str, str, Optional[str]]] = {}
    for idx, stmt in enumerate(tree.body):
        uses_by_index[idx] = _statement_uses(stmt)
        for name in _statement_defines(stmt):
            name_to_def_indices.setdefault(name, []).append(idx)

        target_name, call = _assignment_name_and_call(stmt)
        if target_name is not None and call is not None:
            ctor = _call_func_name(call)
            if ctor in CONSTRUCTOR_TO_DIAGRAM and target_name not in ctor_by_name:
                diagram_type, default_title = CONSTRUCTOR_TO_DIAGRAM[ctor]
                ctor_by_name[target_name] = (diagram_type, default_title, _extract_name_kwarg(call))

    # Resolve each model variable (in list order) to its diagram type.
    resolved: List[Dict[str, Any]] = []
    for var in model_names:
        if var not in ctor_by_name:
            logger.debug("Model var '%s' has no recognised constructor; skipping", var)
            continue
        diagram_type, default_title, name_kwarg = ctor_by_name[var]
        resolved.append({
            "var": var,
            "diagram_type": diagram_type,
            "default_title": default_title,
            "name_kwarg": name_kwarg,
            "closure": _closure_indices(var, name_to_def_indices, uses_by_index),
        })

    if not resolved:
        return None

    # Domain (class-diagram) statements are the shared context for object diagrams.
    domain_union: set = set()
    for entry in resolved:
        if entry["diagram_type"] == "ClassDiagram":
            domain_union |= entry["closure"]
    domain_prefix_code = _source_of(content, tree, domain_union)

    # Count per-type occurrences so we can disambiguate default titles.
    type_counts: Dict[str, int] = {}
    for entry in resolved:
        type_counts[entry["diagram_type"]] = type_counts.get(entry["diagram_type"], 0) + 1
    type_running: Dict[str, int] = {}

    sections: List[Dict[str, Any]] = []
    for entry in resolved:
        diagram_type = entry["diagram_type"]
        type_running[diagram_type] = type_running.get(diagram_type, 0) + 1

        # Title comes from the model's name= argument; fall back to the default,
        # numbering it when several models of the same type share the default.
        if entry["name_kwarg"]:
            title = entry["name_kwarg"]
        elif type_counts[diagram_type] > 1:
            title = f"{entry['default_title']} {type_running[diagram_type]}"
        else:
            title = entry["default_title"]

        # Object chunks exclude the shared domain statements (prepended separately);
        # every other type keeps its self-contained closure.
        if diagram_type == "ObjectDiagram":
            indices = entry["closure"] - domain_union
        else:
            indices = entry["closure"]

        sections.append({
            "var": entry["var"],
            "diagram_type": diagram_type,
            "title": title,
            "section_code": _source_of(content, tree, indices),
            "domain_prefix_code": domain_prefix_code,
        })

    return sections


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
]

_SINGLE_DIAGRAM_DEFAULT_TITLES = {
    'ClassDiagram': 'Class Diagram',
    'ObjectDiagram': 'Object Diagram',
    'AgentDiagram': 'Agent Diagram',
    'StateMachineDiagram': 'State Machine Diagram',
    'GUINoCodeDiagram': 'GUI Diagram',
    'QuantumCircuitDiagram': 'Quantum Circuit Diagram',
    'NNDiagram': 'NN Diagram',
    'BPMNDiagram': 'BPMN Diagram',
}


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


def _build_project_from_single_diagram(content: str) -> Dict[str, Any]:
    """Wrap a single-diagram BUML file (no Project(...)) into a project structure."""
    diagram_type = _detect_single_diagram_type(content)
    if not diagram_type:
        raise ValueError(
            "No models defined in 'models=[...]' and the file was not recognized "
            "as a single-diagram BUML file. Supported single-diagram types: "
            "ClassDiagram, AgentDiagram, StateMachineDiagram, GUINoCodeDiagram, "
            "NNDiagram, BPMNDiagram."
        )

    title = _SINGLE_DIAGRAM_DEFAULT_TITLES[diagram_type]
    try:
        if diagram_type == 'ClassDiagram':
            parsed = parse_buml_content(content)
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
        else:
            raise ValueError(f"Unsupported single-diagram type: {diagram_type}")
    except (SyntaxError, ValueError, TypeError) as e:
        raise ValueError(
            f"Failed to parse single-diagram BUML file (detected as {diagram_type}): {e}"
        ) from e

    filled_entry = {
        "id": str(uuid.uuid4()),
        "title": title,
        "model": model,
        "lastUpdate": datetime.now(timezone.utc).isoformat(),
    }

    diagram_defaults = {
        "ClassDiagram": "ClassDiagram",
        "ObjectDiagram": "ObjectDiagram",
        "AgentDiagram": "AgentDiagram",
        "StateMachineDiagram": "StateMachineDiagram",
        "GUINoCodeDiagram": "GUINoCodeDiagram",
        "QuantumCircuitDiagram": "QuantumCircuitDiagram",
        "NNDiagram": "NNDiagram",
        "BPMNDiagram": "BPMNDiagram",
    }

    diagram_jsons: Dict[str, List[Dict[str, Any]]] = {}
    for dt, mt in diagram_defaults.items():
        if dt == diagram_type:
            diagram_jsons[dt] = [filled_entry]
        else:
            diagram_jsons[dt] = [{
                "id": str(uuid.uuid4()),
                "title": _SINGLE_DIAGRAM_DEFAULT_TITLES[dt],
                "model": empty_model(mt),
                "lastUpdate": datetime.now(timezone.utc).isoformat(),
            }]

    current_diagram_indices = {dt: 0 for dt in diagram_defaults}

    # WME keys the BPMN bucket as "BPMN" (model.type stays "BPMNDiagram").
    if "BPMNDiagram" in diagram_jsons:
        diagram_jsons["BPMN"] = diagram_jsons.pop("BPMNDiagram")
        current_diagram_indices["BPMN"] = current_diagram_indices.pop("BPMNDiagram")
    current_diagram_type = "BPMN" if diagram_type == "BPMNDiagram" else diagram_type

    return {
        "id": str(uuid.uuid4()),
        "type": "Project",
        "schemaVersion": 3,
        "name": "Imported Project",
        "description": "Imported from single-diagram BUML file",
        "owner": "Unknown",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "currentDiagramType": current_diagram_type,
        "currentDiagramIndices": current_diagram_indices,
        "diagrams": diagram_jsons,
        "settings": {
            "defaultDiagramType": diagram_type,
            "autoSave": True,
            "collaborationEnabled": False,
        },
    }


def project_to_json(content: str) -> Dict[str, Any]:
    """
    Convert a BUML project content to JSON format matching the frontend structure.

    The diagram set is driven by the authoritative ``Project(models=[...])`` list
    recovered from the module AST: each variable in that list yields exactly one
    diagram, whose type is read from its assignment's constructor
    (``ObjectModel(...)`` -> object diagram, ``DomainModel(...)`` -> class diagram,
    ...). Comment banners are ignored, so a stray/duplicate ``# ... MODEL #`` line
    cannot change the number or identity of diagrams (WME issue #161).

    Also accepts plain single-diagram BUML files that omit the ``Project(...)``
    wrapper — they are wrapped into a one-diagram project so that the Project
    Import flow works with any well-formed BUML file.

    Args:
        content: Project Python code as string

    Returns:
        Dictionary representing the complete project with all diagrams.
        Each diagram type maps to a list of diagram entries.
    """
    # Derive per-model source chunks from the AST, driven by Project(models=[...]).
    sections = _build_ast_sections(content)
    if sections is None:
        # No Project(...) wrapper — fall back to single-diagram detection so
        # plain class/agent/state-machine/GUI/NN files can still be imported.
        logger.info(
            "No 'models=[...]' declaration found; using single-diagram fallback."
        )
        return _build_project_from_single_diagram(content)

    # Extract project metadata (orthogonal to sectioning; kept as lightweight regex).
    project_name_match = re.search(r'Project\s*\(\s*name\s*=\s*"(.*?)"', content)
    project_desc_match = re.search(r'Metadata\s*\(\s*description\s*=\s*"(.*?)"', content)
    project_owner_match = re.search(r'owner\s*=\s*"(.*?)"', content)

    project_name = project_name_match.group(1) if project_name_match else "Unnamed Project"
    project_description = project_desc_match.group(1) if project_desc_match else "No description"
    project_owner = project_owner_match.group(1) if project_owner_match else "Unknown"

    # diagram_jsons maps diagram type -> list of diagram entries.
    diagram_jsons: Dict[str, List[Dict[str, Any]]] = {}

    # Convert class diagrams first so object diagrams can reference them, then the
    # rest. Within each pass, sections keep their Project(models=[...]) order, so
    # per-type diagram lists are ordered by the models list.
    def _emit(section: Dict[str, Any]) -> None:
        entry = _convert_model(
            section["diagram_type"],
            section["section_code"],
            section["title"],
            section["domain_prefix_code"],
            diagram_jsons.get("ClassDiagram", []),
        )
        if entry is not None:
            diagram_jsons.setdefault(section["diagram_type"], []).append(entry)

    for section in sections:
        if section["diagram_type"] == "ClassDiagram":
            _emit(section)
    for section in sections:
        if section["diagram_type"] != "ClassDiagram":
            _emit(section)

    # Safety net: a Project(...) wrapper whose models resolved to nothing usable
    # (e.g. all constructors unrecognised, or a flat file the AST closure could not
    # section). Recover the embedded model via single-diagram detection and restore
    # the project metadata from the wrapper.
    if not diagram_jsons:
        logger.info(
            "Project wrapper present but no diagrams were produced; "
            "falling back to single-diagram detection."
        )
        try:
            result = _build_project_from_single_diagram(content)
        except ValueError:
            # No detectable single-diagram type either — fall through to the
            # all-empty default below so the caller sees a structured project.
            result = None
        if result is not None:
            result["name"] = project_name
            result["description"] = project_description
            result["owner"] = project_owner
            return result

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
        "BPMNDiagram": "BPMNDiagram",
    }

    for diagram_type, model_type in diagram_defaults.items():
        if diagram_type not in diagram_jsons:
            diagram_jsons[diagram_type] = [{
                "id": str(uuid.uuid4()),
                "title": diagram_type.replace("Diagram", " Diagram"),
                "model": empty_model(model_type),
                "lastUpdate": datetime.now(timezone.utc).isoformat(),
            }]

    current_diagram_indices = {diagram_type: 0 for diagram_type in diagram_defaults}

    # WME keys the BPMN bucket as "BPMN" (model.type stays "BPMNDiagram").
    if "BPMNDiagram" in diagram_jsons:
        diagram_jsons["BPMN"] = diagram_jsons.pop("BPMNDiagram")
        current_diagram_indices["BPMN"] = current_diagram_indices.pop("BPMNDiagram")

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
