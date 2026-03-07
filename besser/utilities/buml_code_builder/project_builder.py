"""
Project Builder

This module generates Python code for complete BUML projects including all models.
"""

import os
import tempfile
import uuid
import shutil
from besser.BUML.metamodel.structural.structural import DomainModel
from besser.BUML.metamodel.object.object import ObjectModel
from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.state_machine.agent import Agent
from besser.utilities.buml_code_builder.domain_model_builder import (
    domain_model_to_code,
    contains_user_class,
    is_user_object_model,
)
from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code
from besser.utilities.buml_code_builder.quantum_model_builder import quantum_model_to_code
from besser.utilities.buml_code_builder.nn_model_builder import nn_model_to_code

try:
    from besser.utilities.web_modeling_editor.backend.constants.user_buml_model import (
        domain_model as reference_user_domain_model,
    )
except ImportError:
    reference_user_domain_model = None


def project_to_code(project: Project, file_path: str, sm: str = ""):
    """
    Generates Python code for a complete B-UML project and writes it to a specified file.
    
    Parameters:
        project (Project): The B-UML project containing multiple models
        file_path (str): The path where the generated code will be saved
        sm (str): Optional state machine code to include
        
    Outputs:
        - A Python file containing all models in the project
    """
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not file_path.endswith('.py'):
        file_path += '.py'

    domain_model = None
    objectmodel = None
    agent_model = None
    user_domain_model = None
    user_objectmodel = None
    gui_model = None

    # Import GUIModel locally to avoid circular imports
    try:
        from besser.BUML.metamodel.gui import GUIModel
    except ImportError:
        GUIModel = None

    for model in project.models:
        if isinstance(model, DomainModel):
            if contains_user_class(model):
                user_domain_model = user_domain_model or model
            elif domain_model is None:
                domain_model = model
        if isinstance(model, ObjectModel):
            if is_user_object_model(model):
                user_objectmodel = user_objectmodel or model
            elif objectmodel is None:
                objectmodel = model
        if isinstance(model, Agent):
            agent_model = model
        if GUIModel and isinstance(model, GUIModel):
            gui_model = model
            
    # Check for QuantumCircuit model (import locally to avoid circular deps if needed, though less likely here)
    try:
        from besser.BUML.metamodel.quantum import QuantumCircuit
        quantum_model = next((m for m in project.models if isinstance(m, QuantumCircuit)), None)
    except ImportError:
        quantum_model = None

    if user_objectmodel and not user_domain_model and reference_user_domain_model:
        user_domain_model = reference_user_domain_model

    # Check for NN model
    try:
        from besser.BUML.metamodel.nn import NN
        nn_model = next((m for m in project.models if isinstance(m, NN)), None)
    except ImportError:
        nn_model = None

    models = []
    with open(file_path, 'w', encoding='utf-8') as f:
        # Models
        temp_dir = tempfile.mkdtemp(prefix=f"besser_{uuid.uuid4().hex}_")

        if objectmodel and domain_model:
            output_file_path = os.path.join(temp_dir, "domain_model.py")
            domain_model_to_code(model=domain_model, file_path=output_file_path, objectmodel=objectmodel)
            with open(output_file_path, "r", encoding='utf-8') as m:
                file_content = m.read()
            content_str = file_content
            f.write(content_str)
            f.write("\n\n")
            models.append("domain_model")
            models.append("object_model")
        elif domain_model:
            output_file_path = os.path.join(temp_dir, "domain_model.py")
            domain_model_to_code(model=domain_model, file_path=output_file_path)
            with open(output_file_path, "r", encoding='utf-8') as m:
                file_content = m.read()
            content_str = file_content
            f.write(content_str)
            f.write("\n\n")
            models.append("domain_model")

        if user_domain_model:
            output_file_path = os.path.join(temp_dir, "user_model.py")
            domain_model_to_code(
                model=user_domain_model,
                file_path=output_file_path,
                objectmodel=user_objectmodel,
                model_var_name="user_model",
                metadata_var_name="user_metadata",
                object_model_var_name="user_object_model",
            )
            with open(output_file_path, "r", encoding='utf-8') as m:
                file_content = m.read()
            content_str = file_content
            f.write(content_str)
            f.write("\n\n")
            models.append("user_model")
            if user_objectmodel:
                models.append("user_object_model")
        
        if agent_model:
            output_file_path = os.path.join(temp_dir, "agent_model.py")
            agent_model_to_code(model=agent_model, file_path=output_file_path)
            with open(output_file_path, "r", encoding='utf-8') as m:
                file_content = m.read()
            content_str = file_content
            f.write(content_str)
            f.write("\n\n")
            models.append("agent")
        
        if gui_model:
            # Use gui_code_builder to generate GUI model code
            # gui_model_to_code appends to file, but we're already writing, so write to temp file first
            output_file_path = os.path.join(temp_dir, "gui_model.py")
            from besser.utilities.buml_code_builder.gui_model_builder import gui_model_to_code
            # Pass domain_model=None because we already wrote it above
            gui_model_to_code(model=gui_model, file_path=output_file_path, domain_model=None)
            with open(output_file_path, "r", encoding='utf-8') as m:
                file_content = m.read()
            content_str = file_content
            f.write(content_str)
            f.write("\n\n")
            models.append("gui_model")

        if quantum_model:
            output_file_path = os.path.join(temp_dir, "quantum_model.py")
            quantum_model_to_code(model=quantum_model, file_path=output_file_path)
            with open(output_file_path, "r", encoding='utf-8') as m:
                file_content = m.read()
            content_str = file_content
            f.write(content_str)
            f.write("\n\n")
            models.append("quantum_model")

        if nn_model:
            output_file_path = os.path.join(temp_dir, "nn_model.py")
            nn_model_to_code(model=nn_model, file_path=output_file_path)
            with open(output_file_path, "r", encoding='utf-8') as m:
                file_content = m.read()
            content_str = file_content
            f.write(content_str)
            f.write("\n\n")
            # Use the same variable naming logic as nn_model_builder
            nn_var_name = nn_model.name.replace(' ', '_').replace('-', '_').lower()
            if nn_var_name and nn_var_name[0].isdigit():
                nn_var_name = 'nn_' + nn_var_name
            models.append(nn_var_name if nn_var_name else 'nn_model')

        if sm != "":
            f.write(sm)
            models.append("sm")
            f.write("\n\n")

        shutil.rmtree(temp_dir, ignore_errors=True)

        # Project
        # Write imports
        f.write("######################\n")
        f.write("# PROJECT DEFINITION #\n")
        f.write("######################\n\n")
        f.write("from besser.BUML.metamodel.project import Project\n")
        f.write("from besser.BUML.metamodel.structural.structural import Metadata\n\n")

        # Write metadata and project
        f.write(f'metadata = Metadata(description="{project.metadata.description}")\n')
        f.write("project = Project(\n")
        f.write(f'    name="{project.name}",\n')
        f.write(f'    models=[{", ".join(models)}],\n')
        f.write(f'    owner="{project.owner}",\n')
        f.write("    metadata=metadata\n")
        f.write(")\n")
    
    print(f"Project saved to {file_path}")
