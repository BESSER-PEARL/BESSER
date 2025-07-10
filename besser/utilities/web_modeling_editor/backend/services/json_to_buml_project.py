from besser.utilities.web_modeling_editor.backend.services import (
    process_class_diagram,
    process_object_diagram,
    process_agent_diagram,
)
from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural.structural import Metadata


def json_to_buml_project(project):
    """
    Generates a B-UML Project instance from a Pydantic Project object.
    """
    name = project.name
    description = project.description or ""

    # List of diagram names to check
    diagram_names = ["ClassDiagram", "ObjectDiagram", "StateMachineDiagram", "AgentDiagram"]
    diagrams = {}

    # Filter out empty diagrams (those without elements)
    for d_name in diagram_names:
        diag = project.diagrams.get(d_name)
        elements = None
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
        object_model = process_object_diagram(object_model_py.model, domain_model)
        model_list.append(object_model)

    # Process AgentDiagram if it exists
    agent_model_py = diagrams.get("AgentDiagram")
    if agent_model_py:
        agent_model = process_agent_diagram(agent_model_py.model_dump())
        model_list.append(agent_model)

    metadata = Metadata(description=description)

    project_instance = Project(
        name=name,
        models=model_list,
        owner=project.owner or "",
        metadata=metadata
    )

    return project_instance
