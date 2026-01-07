"""
Project conversion from JSON to BUML format.
"""

from . import (
    process_class_diagram,
    process_object_diagram,
    process_agent_diagram,
)
from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural.structural import Metadata
from besser.utilities.web_modeling_editor.backend.constants.user_buml_model import (
    domain_model as user_reference_domain_model,
)


def json_to_buml_project(project):
    """
    Generates a B-UML Project instance from a Pydantic Project object.
    """
    name = project.name
    description = project.description or ""

    # List of diagram names to check
    diagram_names = [
        "ClassDiagram",
        "ObjectDiagram",
        "StateMachineDiagram",
        "AgentDiagram",
        "GUINoCodeDiagram",
        "UserDiagram",
    ]
    diagrams = {}

    # Filter out empty diagrams (those without elements)
    for d_name in diagram_names:
        diag = project.diagrams.get(d_name)
        elements = None
        
        # Special handling for GUINoCodeDiagram - it doesn't have "elements", it has "pages"
        if d_name == "GUINoCodeDiagram":
            if diag and hasattr(diag, "model"):
                if isinstance(diag.model, dict):
                    pages = diag.model.get("pages")
                else:
                    pages = getattr(diag.model, "pages", None)
                # GUI diagram is valid if it has pages
                if diag and pages:
                    diagrams[d_name] = diag
                else:
                    diagrams[d_name] = None
            else:
                diagrams[d_name] = None
        else:
            # Standard element-based diagram handling
            if diag and hasattr(diag, "model"):
                if isinstance(diag.model, dict):
                    elements = diag.model.get("elements")
                else:
                    elements = getattr(diag.model, "elements", None)
            if diag and elements:
                diagrams[d_name] = diag
            else:
                diagrams[d_name] = None

    model_list = []

    # Process ClassDiagram first (domain_model)
    domain_model_py = diagrams.get("ClassDiagram")
    if domain_model_py:
        domain_model = process_class_diagram(domain_model_py.model_dump())
        model_list.append(domain_model)
    else:
        domain_model = None

    # Process ObjectDiagram only if it exists and domain_model is available
    object_model_py = diagrams.get("ObjectDiagram")
    if object_model_py and domain_model:
        object_model = process_object_diagram(object_model_py.model_dump(), domain_model)
        model_list.append(object_model)

    # User diagrams behave like object diagrams for conversion purposes
    user_model_py = diagrams.get("UserDiagram")
    if user_model_py:
        # Use the dedicated user reference domain so user diagrams resolve classes even
        # when the project does not include the corresponding class diagram.
        user_model = process_object_diagram(
            user_model_py.model_dump(), user_reference_domain_model
        )
        model_list.append(user_model)

    # Process AgentDiagram if it exists
    agent_model_py = diagrams.get("AgentDiagram")
    if agent_model_py:
        agent_model = process_agent_diagram(agent_model_py.model_dump())
        model_list.append(agent_model)

    # Process GUINoCodeDiagram if it exists
    gui_model_py = diagrams.get("GUINoCodeDiagram")
    if gui_model_py:
        from .gui_diagram_processor import process_gui_diagram
        
        # GUI processing requires class_model (JSON) and domain_model (BUML)
        # Get the class diagram JSON for GUI processing
        class_diagram_py = diagrams.get("ClassDiagram")
        if class_diagram_py and domain_model:
            class_diagram_json = class_diagram_py.model_dump()
            gui_diagram_data = gui_model_py.model_dump()
            
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
            
            print(f"[GUI Processing] GUI diagram has pages: {bool(gui_json.get('pages') if isinstance(gui_json, dict) else False)}")
            gui_model = process_gui_diagram(gui_json, class_json, domain_model)
            model_list.append(gui_model)
        else:
            # GUI diagram exists but no ClassDiagram - skip GUI processing
            print("Warning: GUINoCodeDiagram found but ClassDiagram is missing. Skipping GUI processing.")


    metadata = Metadata(description=description)

    project_instance = Project(
        name=name,
        models=model_list,
        owner=project.owner or "",
        metadata=metadata
    )

    return project_instance
