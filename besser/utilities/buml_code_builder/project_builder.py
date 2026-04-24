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
from besser.BUML.metamodel.state_machine.state_machine import StateMachine
from besser.utilities.buml_code_builder.common import _escape_python_string
from besser.utilities.buml_code_builder.domain_model_builder import (
    domain_model_to_code,
    contains_user_class,
    is_user_object_model,
)
from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code
from besser.utilities.buml_code_builder.state_machine_builder import state_machine_to_code
from besser.utilities.buml_code_builder.quantum_model_builder import quantum_model_to_code
from besser.utilities.buml_code_builder.nn_model_builder import nn_model_to_code, _name_to_var

try:
    from besser.utilities.web_modeling_editor.backend.constants.user_buml_model import (
        domain_model as reference_user_domain_model,
    )
except ImportError:
    reference_user_domain_model = None


def _suffixed_name(base_name: str, index: int, total: int) -> str:
    """Return *base_name* with a ``_<index>`` suffix when *total* > 1."""
    if total <= 1:
        return base_name
    return f"{base_name}_{index}"


def _write_temp_to_output(temp_path: str, out_file, section_header: str = ""):
    """Read a temp file and append its content (with optional header) to *out_file*."""
    with open(temp_path, "r", encoding="utf-8") as tmp:
        content = tmp.read()
    if section_header:
        out_file.write(section_header)
    out_file.write(content)
    out_file.write("\n\n")


def project_to_code(project: Project, file_path: str, sm: str = ""):
    """
    Generates Python code for a complete B-UML project and writes it to a specified file.

    This function handles projects with multiple models of the same type.  When a
    project contains more than one model of a given kind (e.g. two DomainModels),
    each generated variable receives a numeric suffix (``domain_model_1``,
    ``domain_model_2``, ...).  When there is only one model of a given kind the
    output is identical to the previous single-model behaviour (no suffix).

    Parameters:
        project (Project): The B-UML project containing multiple models
        file_path (str): The path where the generated code will be saved
        sm (str): Deprecated. Legacy state machine code string (kept for backward
                  compatibility). StateMachine objects in the project's model list
                  are now handled automatically.

    Outputs:
        - A Python file containing all models in the project
    """
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not file_path.endswith('.py'):
        file_path += '.py'

    # ------------------------------------------------------------------ #
    # Classify every model in the project into typed buckets              #
    # ------------------------------------------------------------------ #
    domain_models = []          # regular DomainModels (no User class)
    object_models = []          # regular ObjectModels (not user-related)
    user_domain_models = []     # DomainModels containing a User class
    user_object_models = []     # ObjectModels for user reference domain
    agent_models = []
    gui_models = []
    quantum_models = []
    state_machine_models = []   # StateMachine models
    nn_models = []

    # Import GUIModel locally to avoid circular imports
    try:
        from besser.BUML.metamodel.gui import GUIModel
    except ImportError:
        GUIModel = None

    # Import QuantumCircuit locally
    try:
        from besser.BUML.metamodel.quantum import QuantumCircuit
    except ImportError:
        QuantumCircuit = None

    # Import NN locally
    try:
        from besser.BUML.metamodel.nn import NN
    except ImportError:
        NN = None

    for model in project.models:
        if isinstance(model, DomainModel):
            if contains_user_class(model):
                user_domain_models.append(model)
            else:
                domain_models.append(model)
        elif isinstance(model, ObjectModel):
            if is_user_object_model(model):
                user_object_models.append(model)
            else:
                object_models.append(model)
        elif isinstance(model, Agent):
            agent_models.append(model)
        elif isinstance(model, StateMachine):
            state_machine_models.append(model)
        elif GUIModel and isinstance(model, GUIModel):
            gui_models.append(model)
        elif QuantumCircuit and isinstance(model, QuantumCircuit):
            quantum_models.append(model)
        elif NN and isinstance(model, NN):
            nn_models.append(model)

    # If we have user object models but no user domain model, use the
    # reference one shipped with the editor backend (when available).
    if user_object_models and not user_domain_models and reference_user_domain_model:
        user_domain_models.append(reference_user_domain_model)

    # ------------------------------------------------------------------ #
    # Pair regular domain models with their object models.                #
    # When there's exactly one of each, pair them together (old behavior).#
    # Otherwise, keep them separate.                                      #
    # ------------------------------------------------------------------ #
    # Build (domain_model, object_model_or_None) pairs
    domain_pairs = []
    if len(domain_models) == 1 and len(object_models) == 1:
        domain_pairs.append((domain_models[0], object_models[0]))
    else:
        for dm in domain_models:
            domain_pairs.append((dm, None))
        # Standalone object models that were not paired
        # (handled below as extra domain_model_to_code calls with no domain model)

    # Similarly pair user domain models with user object models
    user_pairs = []
    if len(user_domain_models) == 1 and len(user_object_models) <= 1:
        user_pairs.append((user_domain_models[0], user_object_models[0] if user_object_models else None))
    else:
        for udm in user_domain_models:
            user_pairs.append((udm, None))

    # ------------------------------------------------------------------ #
    # Helper: count totals so we know whether to add numeric suffixes     #
    # ------------------------------------------------------------------ #
    n_domain = len(domain_pairs)
    n_user = len(user_pairs)
    n_agent = len(agent_models)
    n_gui = len(gui_models)
    n_quantum = len(quantum_models)
    n_sm = len(state_machine_models)
    n_nn = len(nn_models)

    # Variable names collected for the final Project(...) definition
    model_vars = []

    with open(file_path, 'w', encoding='utf-8') as f:
        temp_dir = tempfile.mkdtemp(prefix=f"besser_{uuid.uuid4().hex}_")

        try:
            # ---------------------------------------------------------- #
            # DOMAIN MODELS (regular, non-user)                          #
            # ---------------------------------------------------------- #
            for idx, (dm, om) in enumerate(domain_pairs, start=1):
                var_name = _suffixed_name("domain_model", idx, n_domain)
                obj_var_name = _suffixed_name("object_model", idx, n_domain)
                meta_var_name = _suffixed_name("domain_model_metadata", idx, n_domain) if n_domain > 1 else None

                section = ""
                if n_domain > 1:
                    label = getattr(dm, "name", f"Model {idx}")
                    section = f"# STRUCTURAL MODEL {idx}: \"{label}\" #\n\n"

                tmp_path = os.path.join(temp_dir, f"domain_model_{idx}.py")

                if om:
                    domain_model_to_code(
                        model=dm,
                        file_path=tmp_path,
                        objectmodel=om,
                        model_var_name=var_name,
                        metadata_var_name=meta_var_name,
                        object_model_var_name=obj_var_name,
                    )
                    _write_temp_to_output(tmp_path, f, section_header=section)
                    model_vars.append(var_name)
                    model_vars.append(obj_var_name)
                else:
                    domain_model_to_code(
                        model=dm,
                        file_path=tmp_path,
                        model_var_name=var_name,
                        metadata_var_name=meta_var_name,
                    )
                    _write_temp_to_output(tmp_path, f, section_header=section)
                    model_vars.append(var_name)

            # Standalone object models (when not paired 1:1 with domain models)
            if not (len(domain_models) == 1 and len(object_models) == 1):
                n_standalone_obj = len(object_models)
                for idx, om in enumerate(object_models, start=1):
                    obj_var_name = _suffixed_name("object_model", idx, n_standalone_obj)
                    # Object models need a domain_model to reference; pass None and
                    # generate just the object portion via domain_model_to_code.
                    # For standalone objects without a paired domain model we find
                    # the domain_model attribute on the ObjectModel itself.
                    paired_dm = getattr(om, "domain_model", None)
                    if paired_dm:
                        dm_var = _suffixed_name("object_domain_model", idx, n_standalone_obj)
                        tmp_path = os.path.join(temp_dir, f"object_model_{idx}.py")
                        domain_model_to_code(
                            model=paired_dm,
                            file_path=tmp_path,
                            objectmodel=om,
                            model_var_name=dm_var,
                            object_model_var_name=obj_var_name,
                        )
                        _write_temp_to_output(tmp_path, f)
                        model_vars.append(obj_var_name)

            # ---------------------------------------------------------- #
            # USER DOMAIN MODELS                                         #
            # ---------------------------------------------------------- #
            for idx, (udm, uom) in enumerate(user_pairs, start=1):
                var_name = _suffixed_name("user_model", idx, n_user)
                meta_var = _suffixed_name("user_metadata", idx, n_user)
                obj_var = _suffixed_name("user_object_model", idx, n_user)

                section = ""
                if n_user > 1:
                    label = getattr(udm, "name", f"User Model {idx}")
                    section = f"# USER MODEL {idx}: \"{label}\" #\n\n"

                tmp_path = os.path.join(temp_dir, f"user_model_{idx}.py")
                domain_model_to_code(
                    model=udm,
                    file_path=tmp_path,
                    objectmodel=uom,
                    model_var_name=var_name,
                    metadata_var_name=meta_var,
                    object_model_var_name=obj_var,
                )
                _write_temp_to_output(tmp_path, f, section_header=section)
                model_vars.append(var_name)
                if uom:
                    model_vars.append(obj_var)

            # ---------------------------------------------------------- #
            # AGENT MODELS                                               #
            # ---------------------------------------------------------- #
            for idx, am in enumerate(agent_models, start=1):
                var_name = _suffixed_name("agent", idx, n_agent)

                section = ""
                if n_agent > 1:
                    label = getattr(am, "name", f"Agent {idx}")
                    section = f"# AGENT MODEL {idx}: \"{label}\" #\n\n"

                tmp_path = os.path.join(temp_dir, f"agent_model_{idx}.py")
                agent_model_to_code(
                    model=am,
                    file_path=tmp_path,
                    model_var_name=var_name,
                )
                _write_temp_to_output(tmp_path, f, section_header=section)
                model_vars.append(var_name)

            # ---------------------------------------------------------- #
            # GUI MODELS                                                 #
            # ---------------------------------------------------------- #
            if gui_models:
                from besser.utilities.buml_code_builder.gui_model_builder import gui_model_to_code

                for idx, gm in enumerate(gui_models, start=1):
                    var_name = _suffixed_name("gui_model", idx, n_gui)

                    section = ""
                    if n_gui > 1:
                        label = getattr(gm, "name", f"GUI {idx}")
                        section = f"# GUI MODEL {idx}: \"{label}\" #\n\n"

                    tmp_path = os.path.join(temp_dir, f"gui_model_{idx}.py")
                    # domain_model=None because we already wrote structural models above
                    gui_model_to_code(
                        model=gm,
                        file_path=tmp_path,
                        domain_model=None,
                        model_var_name=var_name,
                    )
                    _write_temp_to_output(tmp_path, f, section_header=section)
                    model_vars.append(var_name)

            # ---------------------------------------------------------- #
            # QUANTUM MODELS                                             #
            # ---------------------------------------------------------- #
            for idx, qm in enumerate(quantum_models, start=1):
                var_name = _suffixed_name("quantum_model", idx, n_quantum)

                section = ""
                if n_quantum > 1:
                    label = getattr(qm, "name", f"Quantum {idx}")
                    section = f"# QUANTUM MODEL {idx}: \"{label}\" #\n\n"

                tmp_path = os.path.join(temp_dir, f"quantum_model_{idx}.py")
                quantum_model_to_code(
                    model=qm,
                    file_path=tmp_path,
                    model_var_name=var_name,
                )
                _write_temp_to_output(tmp_path, f, section_header=section)
                model_vars.append(var_name)

            # ---------------------------------------------------------- #
            # STATE MACHINE MODELS                                       #
            # ---------------------------------------------------------- #
            for idx, smm in enumerate(state_machine_models, start=1):
                var_name = _suffixed_name("sm", idx, n_sm)

                section = ""
                if n_sm > 1:
                    label = getattr(smm, "name", f"State Machine {idx}")
                    section = f"# STATE MACHINE MODEL {idx}: \"{label}\" #\n\n"

                tmp_path = os.path.join(temp_dir, f"state_machine_{idx}.py")
                state_machine_to_code(
                    model=smm,
                    file_path=tmp_path,
                    model_var_name=var_name,
                )
                _write_temp_to_output(tmp_path, f, section_header=section)
                model_vars.append(var_name)

            # ---------------------------------------------------------- #
            # NN MODELS                                                  #
            # ---------------------------------------------------------- #
            for idx, nm in enumerate(nn_models, start=1):
                var_name = _suffixed_name(_name_to_var(nm.name), idx, n_nn)

                section = ""
                if n_nn > 1:
                    label = getattr(nm, "name", f"NN {idx}")
                    section = f"# NN MODEL {idx}: \"{label}\" #\n\n"

                tmp_path = os.path.join(temp_dir, f"nn_model_{idx}.py")
                # Thread the suffixed var_name through so the NN builder
                # writes ``my_nn_1 = NN(...)`` instead of ``my_nn = NN(...)``.
                # Previously the Project(...) line referenced ``my_nn_1`` but
                # the actual binding was ``my_nn`` → NameError on exec() when
                # a project held more than one NN.
                nn_model_to_code(model=nm, file_path=tmp_path, model_var_name=var_name)
                _write_temp_to_output(tmp_path, f, section_header=section)
                model_vars.append(var_name)

            # Legacy: if a raw code string was passed, include it as-is
            if sm != "" and not state_machine_models:
                f.write(sm)
                model_vars.append("sm")
                f.write("\n\n")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # -------------------------------------------------------------- #
        # PROJECT DEFINITION                                             #
        # -------------------------------------------------------------- #
        f.write("######################\n")
        f.write("# PROJECT DEFINITION #\n")
        f.write("######################\n\n")
        f.write("from besser.BUML.metamodel.project import Project\n")
        f.write("from besser.BUML.metamodel.structural.structural import Metadata\n\n")

        # Write metadata and project
        f.write(f'metadata = Metadata(description="{_escape_python_string(project.metadata.description)}")\n')
        f.write("project = Project(\n")
        f.write(f'    name="{_escape_python_string(project.name)}",\n')
        f.write(f'    models=[{", ".join(model_vars)}],\n')
        f.write(f'    owner="{_escape_python_string(project.owner)}",\n')
        f.write("    metadata=metadata\n")
        f.write(")\n")

    print(f"Project saved to {file_path}")
