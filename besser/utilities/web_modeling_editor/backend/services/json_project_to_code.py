from besser.BUML.metamodel.project import Project

def json_to_project_code(json_obj):
    """
    Generates a string with the Python code to instantiate a Project object
    from the given JSON object.
    """
    data = json_obj.get('elements', {})
    project_id = data.get("id", "")
    name = data.get("name", "")
    description = data.get("description", "")
    owner = data.get("owner", "")
    created_at = data.get("createdAt", "")
    models = data.get("models", [])
    print(data)

    # Format the models list as a Python list of strings
    models_str = "[" + ", ".join(f'"{m}"' for m in models) + "]"

    code = (
        f'from besser.BUML.metamodel.project import Project\n'
        f'from ClassDiagram import domain_model\n'
        f'from StateMachineDiagram import sm\n'
        f'from AgentDiagram import agent\n\n'
        f'project = Project(\n'
        f'    name="{name}",\n'
        f'    models=[domain_model, sm, agent],\n'
        f')'
    )
    return code
