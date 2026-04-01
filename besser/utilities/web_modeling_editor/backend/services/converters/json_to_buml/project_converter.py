"""
Project conversion from JSON to BUML format.

Processes ALL diagrams of each type (up to 5 per type) in a project,
not just the active one.
"""

import logging

from . import (
    process_class_diagram,
    process_object_diagram,
    process_agent_diagram,
    process_state_machine,
)
from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural.structural import Metadata
from besser.utilities.web_modeling_editor.backend.constants.user_buml_model import (
    domain_model as user_reference_domain_model,
)

logger = logging.getLogger(__name__)


def _is_valid_diagram(diag, diagram_type):
    """Check whether a single diagram has meaningful content worth processing.

    Returns True if the diagram contains elements/pages/cols depending on its type.
    """
    if diag is None or not hasattr(diag, "model") or not diag.model:
        return False

    model = diag.model
    data = model if isinstance(model, dict) else vars(model) if hasattr(model, "__dict__") else {}

    if diagram_type == "GUINoCodeDiagram":
        pages = data.get("pages") if isinstance(data, dict) else getattr(model, "pages", None)
        return bool(pages)

    if diagram_type == "QuantumCircuitDiagram":
        cols = data.get("cols") if isinstance(data, dict) else getattr(model, "cols", None)
        return cols is not None  # Empty list is valid

    # Standard element-based diagrams
    elements = data.get("elements") if isinstance(data, dict) else getattr(model, "elements", None)
    return bool(elements)


def _collect_valid_diagrams(project):
    """Return a dict mapping each diagram type to a list of valid DiagramInput objects.

    Iterates through ALL diagrams in the project's arrays, filtering out empty ones.
    """
    diagram_types = [
        "ClassDiagram",
        "ObjectDiagram",
        "StateMachineDiagram",
        "AgentDiagram",
        "GUINoCodeDiagram",
        "QuantumCircuitDiagram",
        "UserDiagram",
    ]

    result = {}
    for dtype in diagram_types:
        all_diags = project.diagrams.get(dtype, [])
        valid = [d for d in all_diags if _is_valid_diagram(d, dtype)]
        result[dtype] = valid

    return result


def json_to_buml_project(project):
    """
    Generates a B-UML Project instance from a Pydantic Project object.

    Processes ALL diagrams of each type (not just the active one), so that
    every model is included in the resulting Project.
    """
    name = project.name
    if name and "-" in name:
        name = name.replace("-", "_")
    description = project.description or ""

    # Collect all valid diagrams, grouped by type
    diagrams = _collect_valid_diagrams(project)

    model_list = []

    # ── ClassDiagram caching ──────────────────────────────────────────
    # Cache processed ClassDiagrams by ID so we never process the same one twice
    # (e.g. when an ObjectDiagram and a GUINoCodeDiagram both reference it).
    processed_class_diagrams = {}  # diagram ID → BUML DomainModel

    def _get_or_process_class_diagram(class_diag):
        """Process a ClassDiagram to BUML, caching by diagram ID."""
        diag_id = class_diag.id if class_diag.id else id(class_diag)
        if diag_id not in processed_class_diagrams:
            processed_class_diagrams[diag_id] = process_class_diagram(class_diag.model_dump())
        return processed_class_diagrams[diag_id]

    # ── Process ALL ClassDiagrams first ───────────────────────────────
    for class_diag in diagrams.get("ClassDiagram", []):
        domain_model = _get_or_process_class_diagram(class_diag)
        model_list.append(domain_model)

    # ── Process ALL ObjectDiagrams ────────────────────────────────────
    # Each ObjectDiagram can reference its own ClassDiagram via per-diagram references.
    for obj_diag in diagrams.get("ObjectDiagram", []):
        obj_class_diag = None
        if hasattr(project, "get_referenced_diagram"):
            obj_class_diag = project.get_referenced_diagram(obj_diag, "ClassDiagram")
        # Fallback: use the first valid ClassDiagram in the project
        if not obj_class_diag and diagrams.get("ClassDiagram"):
            obj_class_diag = diagrams["ClassDiagram"][0]
        if obj_class_diag:
            obj_domain_model = _get_or_process_class_diagram(obj_class_diag)
            object_model = process_object_diagram(obj_diag.model_dump(), obj_domain_model)
            model_list.append(object_model)
        else:
            logger.warning(
                "ObjectDiagram '%s' skipped: no ClassDiagram reference found.",
                getattr(obj_diag, "title", "unknown"),
            )

    # ── Process ALL UserDiagrams ──────────────────────────────────────
    # User diagrams behave like object diagrams but always use the dedicated
    # user reference domain model so they resolve classes even when the
    # project does not include the corresponding class diagram.
    for user_diag in diagrams.get("UserDiagram", []):
        user_model = process_object_diagram(
            user_diag.model_dump(), user_reference_domain_model
        )
        model_list.append(user_model)

    # ── Process ALL AgentDiagrams ─────────────────────────────────────
    for agent_diag in diagrams.get("AgentDiagram", []):
        agent_model = process_agent_diagram(agent_diag.model_dump())
        model_list.append(agent_model)

    # ── Process ALL GUINoCodeDiagrams ─────────────────────────────────
    # Each GUINoCodeDiagram can reference its own ClassDiagram.
    from .gui_diagram_processor import process_gui_diagram

    for gui_diag in diagrams.get("GUINoCodeDiagram", []):
        gui_ref_class = None
        if hasattr(project, "get_referenced_diagram"):
            gui_ref_class = project.get_referenced_diagram(gui_diag, "ClassDiagram")
        if not gui_ref_class and diagrams.get("ClassDiagram"):
            gui_ref_class = diagrams["ClassDiagram"][0]

        if gui_ref_class:
            gui_domain_model = _get_or_process_class_diagram(gui_ref_class)

            class_diagram_json = gui_ref_class.model_dump()
            gui_diagram_data = gui_diag.model_dump()

            # Extract the actual model data (not the whole diagram wrapper)
            if isinstance(gui_diagram_data, dict) and "model" in gui_diagram_data:
                gui_json = gui_diagram_data["model"]
            else:
                gui_json = gui_diagram_data

            # Extract class diagram model data
            if isinstance(class_diagram_json, dict) and "model" in class_diagram_json:
                class_json = class_diagram_json["model"]
            else:
                class_json = class_diagram_json

            gui_model = process_gui_diagram(gui_json, class_json, gui_domain_model)
            model_list.append(gui_model)
        else:
            logger.warning(
                "GUINoCodeDiagram '%s' skipped: no ClassDiagram reference found.",
                getattr(gui_diag, "title", "unknown"),
            )

    # ── Process ALL QuantumCircuitDiagrams ────────────────────────────
    from .quantum_diagram_processor import process_quantum_diagram

    for quantum_diag in diagrams.get("QuantumCircuitDiagram", []):
        quantum_model = process_quantum_diagram(quantum_diag.model_dump())
        model_list.append(quantum_model)

    # ── Process ALL StateMachineDiagrams ────────────────────────────────
    for sm_diag in diagrams.get("StateMachineDiagram", []):
        try:
            sm_model = process_state_machine(sm_diag.model_dump())
            model_list.append(sm_model)
        except Exception as e:
            logger.warning(
                "StateMachineDiagram '%s' could not be processed: %s",
                getattr(sm_diag, "title", "unknown"), e,
            )

    # Ensure ALL processed ClassDiagrams are in model_list.
    # Object/GUI diagrams may reference ClassDiagrams that were not in the
    # project's own ClassDiagram array (edge case with per-diagram references).
    for cd_model in processed_class_diagrams.values():
        if cd_model not in model_list:
            model_list.append(cd_model)

    metadata = Metadata(description=description)

    project_instance = Project(
        name=name,
        models=model_list,
        owner=project.owner or "",
        metadata=metadata
    )

    return project_instance
