"""
BESSER Backend API

This module provides FastAPI endpoints for the BESSER web modeling editor backend.
It handles code generation from UML diagrams using various generators.

The editor is available at: https://editor.besser-pearl.org
"""

# Standard library imports
import os
import io
import uuid
import zipfile
import shutil
import tempfile
import importlib.util
import sys
import asyncio
import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, File, UploadFile, Body, Form




# BESSER image-to-UML and BUML utilities
from besser.utilities.image_to_buml import image_to_buml
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import parse_buml_content, class_buml_to_json
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response

# BESSER utilities
from besser.utilities.buml_code_builder.domain_model_builder import domain_model_to_code
from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code
from besser.utilities.buml_code_builder.project_builder import project_to_code

# Backend models
from besser.utilities.web_modeling_editor.backend.models import (
    DiagramInput,
    ProjectInput,
    FeedbackSubmission
)

# Backend services - Converters
from besser.utilities.web_modeling_editor.backend.services.converters import (
    # JSON to BUML converters
    process_class_diagram,
    process_state_machine,
    process_agent_diagram,
    process_object_diagram,
    json_to_buml_project,
    process_gui_diagram,
    process_quantum_diagram,
    # BUML to JSON converters
    class_buml_to_json,
    parse_buml_content,
    state_machine_to_json,
    agent_buml_to_json,
    object_buml_to_json,
    gui_buml_to_json,
    project_to_json,
)

# Backend services - Other services
from besser.utilities.web_modeling_editor.backend.services.validators import (
    check_ocl_constraint,
)
from besser.utilities.web_modeling_editor.backend.services.feedback_service import (
    submit_feedback,
)
from besser.utilities.web_modeling_editor.backend.services.deployment import (
    run_docker_compose,
    github_oauth_router,
    github_deploy_router,
)
from besser.utilities.web_modeling_editor.backend.services.utils import (
    cleanup_temp_resources,
    validate_generator,
)
from besser.utilities.web_modeling_editor.backend.services.reverse_engineering import (
    csv_to_domain_model,
)

# Backend configuration
from besser.utilities.web_modeling_editor.backend.config import (
    SUPPORTED_GENERATORS,
    get_generator_info,
    get_filename_for_generator,
    is_generator_supported,
)

# Initialize FastAPI application
app = FastAPI(
    title="BESSER Backend API",
    description="Backend services for web modeling editor",
    version="1.0.0",
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],  # Expose Content-Disposition header to frontend
)

# Include GitHub OAuth and deployment routers
app.include_router(github_oauth_router, prefix="/besser_api")
app.include_router(github_deploy_router, prefix="/besser_api")

# Constants
API_VERSION = "1.0.0"
TEMP_DIR_PREFIX = "besser_agent_"
OUTPUT_DIR = "output"
AGENT_MODEL_FILENAME = "agent_model.py"
AGENT_OUTPUT_FILENAME = "agent_output.zip"


# API Endpoints
@app.get("/besser_api/")
def get_api_root():
    """
    Get API root information.
    
    Returns:
        dict: Basic API information including available generators
    """
    return {
        "message": "BESSER Backend API",
        "version": API_VERSION,
        "supported_generators": list(SUPPORTED_GENERATORS.keys()),
        "endpoints": {
            "generate": "/besser_api/generate-output",
            "deploy": "/besser_api/deploy-app", 
            "export_buml": "/besser_api/export-buml",
            "export_project": "/besser_api/export-project_as_buml",
            "get_project_json_model": "/besser_api/get-project-json-model",
            "get_single_json_model": "/besser_api/get-single-json-model",
            "validate_diagram": "/besser_api/validate-diagram",
            "check_ocl": "/besser_api/check-ocl (deprecated, use validate_diagram)"
        }
    }


def generate_agent_files(agent_model):
    """
    Generate agent files from an agent model.
    
    Args:
        agent_model: The agent model to generate files from
        
    Returns:
        tuple: (zip_buffer, file_name) containing the zip file buffer and filename
        
    Raises:
        Exception: If agent file generation fails
    """
    temp_dir = None
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"{TEMP_DIR_PREFIX}{uuid.uuid4().hex}_")
        
        # Generate the agent model to a file
        agent_file = os.path.join(temp_dir, AGENT_MODEL_FILENAME)
        agent_model_to_code(agent_model, agent_file)
        
        # Execute the generated agent model file
        sys.path.insert(0, temp_dir)
        
        # Import the generated module
        spec = importlib.util.spec_from_file_location("agent_model", agent_file)
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        
        # Get the BAFGenerator from the supported generators
        generator_info = get_generator_info("agent")
        generator_class = generator_info.generator_class
        
        # Use the BAFGenerator with the agent model from the module
        if hasattr(agent_module, 'agent'):
            generator = generator_class(agent_module.agent)
        else:
            # Fall back to the original agent model
            generator = generator_class(agent_model)
        
        generator.generate()
        
        # Create a zip file with all the generated files
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Add the agent model file
            zip_file.write(agent_file, os.path.basename(agent_file))
            
            # Add all files from the output directory
            if os.path.exists(OUTPUT_DIR):
                for file_name in os.listdir(OUTPUT_DIR):
                    file_path = os.path.join(OUTPUT_DIR, file_name)
                    if os.path.isfile(file_path):
                        zip_file.write(file_path, file_name)
        
        zip_buffer.seek(0)
        return zip_buffer, AGENT_OUTPUT_FILENAME
        
    except Exception as e:
        print(f"Error generating agent files: {str(e)}")
        raise e
    finally:
        # Clean up resources
        cleanup_temp_resources(temp_dir)


@app.post("/besser_api/generate-output-from-project")
async def generate_code_output_from_project(input_data: ProjectInput):
    """
    Generate code output from a complete project.
    This endpoint handles generators that require multiple diagrams (e.g., Web App).
    """
    try:
        generator_type = input_data.settings.get("generator") if input_data.settings else None

        if not generator_type:
            raise HTTPException(
                status_code=400, 
                detail="Generator type is required in project settings"
            )

        # Validate generator
        if not is_generator_supported(generator_type):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported generator type: {generator_type}. Supported types: {list(SUPPORTED_GENERATORS.keys())}"
            )

        generator_info = get_generator_info(generator_type)

        # Get configuration from project settings
        config = input_data.settings.get("config", {}) if input_data.settings else {}

        # Handle Web App generator (requires both ClassDiagram and GUINoCodeDiagram)
        if generator_type == "web_app":
            return await _handle_web_app_project_generation(input_data, generator_info, config)

        # Handle Qiskit generator (requires QuantumCircuitDiagram)
        if generator_type == "qiskit":
            quantum_diagram = input_data.diagrams.get("QuantumCircuitDiagram")
            if not quantum_diagram:
                raise HTTPException(
                    status_code=400,
                    detail="QuantumCircuitDiagram is required for Qiskit generator"
                )
            # Convert to DiagramInput with the quantum diagram data
            diagram_input = DiagramInput(
                id=quantum_diagram.id,
                title=quantum_diagram.title,
                model=quantum_diagram.model,
                lastUpdate=quantum_diagram.lastUpdate,
                generator=generator_type,
                config=config,
                referenceDiagramData=quantum_diagram.referenceDiagramData if hasattr(quantum_diagram, 'referenceDiagramData') else None
            )
            return await generate_code_output(diagram_input)

        # For other generators, use the current diagram
        current_diagram = input_data.diagrams.get(input_data.currentDiagramType)
        if not current_diagram:
            raise HTTPException(
                status_code=400,
                detail=f"No diagram found for type: {input_data.currentDiagramType}"
            )

        # Convert to DiagramInput and use existing logic
        diagram_input = DiagramInput(
            id=current_diagram.id,
            title=current_diagram.title,
            model=current_diagram.model,
            lastUpdate=current_diagram.lastUpdate,
            generator=generator_type,
            config=config,
            referenceDiagramData=current_diagram.referenceDiagramData
        )

        return await generate_code_output(diagram_input)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/besser_api/generate-output")
async def generate_code_output(input_data: DiagramInput):
    """
    Generate code from UML diagram data using specified generator.
    
    Args:
        input_data: DiagramInput containing diagram data and generator type
        
    Returns:
        StreamingResponse or Response: Generated code file or ZIP
        
    Raises:
        HTTPException: If generator is not supported or generation fails
    """
    temp_dir = tempfile.mkdtemp(prefix=f"besser_{uuid.uuid4().hex}_")

    try:
        json_data = input_data.model_dump()
        generator_type = input_data.generator

        # Validate generator type
        if not is_generator_supported(generator_type):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid generator type: {generator_type}. Supported types: {list(SUPPORTED_GENERATORS.keys())}"
            )

        generator_info = get_generator_info(generator_type)

        # Handle agent generators (different diagram type)
        if generator_info.category == "ai_agent":
            return await _handle_agent_generation(json_data)
            
        # Handle quantum generators
        if generator_info.category == "quantum":
            return await _generate_qiskit(json_data, generator_info.generator_class, input_data.config, temp_dir)

        # Handle class diagram based generators
        return await _handle_class_diagram_generation(
            json_data, generator_type, generator_info, input_data.config, temp_dir
        )

    except HTTPException as e:
        cleanup_temp_resources(temp_dir)
        raise e
    except Exception as e:
        cleanup_temp_resources(temp_dir)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


async def _handle_web_app_project_generation(input_data: ProjectInput, generator_info, config: dict):
    """Handle Web App generation from a complete project with both ClassDiagram and GUINoCodeDiagram.
    Optionally includes AgentDiagram if agent components are present in the GUI."""
    try:
        # Extract ClassDiagram
        class_diagram = input_data.diagrams.get("ClassDiagram")
        if not class_diagram:
            raise HTTPException(
                status_code=400,
                detail="ClassDiagram is required for Web App generator"
            )

        # Extract GUINoCodeDiagram
        gui_diagram = input_data.diagrams.get("GUINoCodeDiagram")
        if not gui_diagram:
            raise HTTPException(
                status_code=400,
                detail="GUINoCodeDiagram is required for Web App generator"
            )

        temp_dir = tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)

        # Process class diagram to BUML
        buml_model = process_class_diagram(class_diagram.model_dump())

        gui_model = process_gui_diagram(gui_diagram.model, class_diagram.model, buml_model)

        # Check if GUI model contains agent components
        agent_diagram = None
        agent_model = None
        has_agent_components = _check_for_agent_components(gui_model)
        
        if has_agent_components:
            # Extract AgentDiagram if present
            agent_diagram = input_data.diagrams.get("AgentDiagram")
            if agent_diagram and agent_diagram.model:
                # Process agent diagram to BUML
                # process_agent_diagram expects the full diagram structure with title, config, and model
                agent_diagram_dict = agent_diagram.model_dump()
                if agent_diagram_dict and isinstance(agent_diagram_dict, dict):
                    agent_model = process_agent_diagram(agent_diagram_dict)
                else:
                    print("Warning: AgentDiagram data is invalid. Agent components will not be functional.")
            else:
                print("Warning: GUI contains agent components but no AgentDiagram found. Agent components will not be functional.")

        # Generate Web App TypeScript project
        generator_class = generator_info.generator_class

        return await _generate_web_app(buml_model, gui_model, generator_class, config, temp_dir, agent_model)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Web App generation failed: {str(e)}")


async def _handle_agent_generation(json_data: dict):
    """Handle agent diagram generation."""
    try:
        # Get languages from config if present
        config = json_data.get('config', {})
        languages = config.get('languages') if config else None

        # New format: languages is a dict with 'source' and 'target'
        if languages and isinstance(languages, dict):
            source_lang = languages.get('source')
            target_langs = languages.get('target', None)
            # If no target languages, fall back to single agent generation
            agent_models = []

            # Default agent (no language)
            default_json_data = {**json_data}
            if 'config' in default_json_data and isinstance(default_json_data['config'], dict):
                default_json_data['config'] = {k: v for k, v in default_json_data['config'].items() if k != 'languages'}
            agent_models.append((process_agent_diagram(default_json_data), 'default'))

            # Agents for each target language
            for lang in target_langs:
                new_config = dict(config) if config else {}
                new_config['language'] = lang
                # Pass source language to process_agent_diagram via config
                if source_lang:
                    new_config['source_language'] = source_lang
                new_json_data = dict(json_data)
                new_json_data['config'] = new_config
                agent_models.append((process_agent_diagram(new_json_data), lang))

            # Generate files for each agent model
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for agent_model, lang in agent_models:
                    temp_dir = tempfile.mkdtemp(prefix=f"besser_agent_{lang}_")
                    try:
                        # Generate agent files in temp dir
                        agent_file = os.path.join(temp_dir, f"agent_model_{lang}.py")
                        agent_model_to_code(agent_model, agent_file)
                        # Import and generate output files
                        sys.path.insert(0, temp_dir)
                        spec = importlib.util.spec_from_file_location(f"agent_model_{lang}", agent_file)
                        agent_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(agent_module)
                        generator_info = get_generator_info("agent")
                        generator_class = generator_info.generator_class
                        if hasattr(agent_module, 'agent'):
                            generator = generator_class(agent_module.agent)
                        else:
                            generator = generator_class(agent_model)
                        generator.generate()
                        # Add agent model file inside language folder
                        zip_file.write(agent_file, f"{lang}/agent_model_{lang}.py")
                        # Add all files from output dir inside language folder
                        if os.path.exists(OUTPUT_DIR):
                            for file_name in os.listdir(OUTPUT_DIR):
                                file_path = os.path.join(OUTPUT_DIR, file_name)
                                if os.path.isfile(file_path):
                                    zip_file.write(file_path, f"{lang}/{file_name}")
                    finally:
                        cleanup_temp_resources(temp_dir)
            zip_buffer.seek(0)
            return StreamingResponse(
                zip_buffer,
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=agents_multi_lang.zip"},
            )
        else:
            # Single agent (default behavior)
            agent_model = process_agent_diagram(json_data)
            zip_buffer, file_name = generate_agent_files(agent_model)
            return StreamingResponse(
                zip_buffer,
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={file_name}"},
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing agent diagram: {str(e)}"
        )


async def _handle_class_diagram_generation(
    json_data: dict,
    generator_type: str,
    generator_info,
    config: dict,
    temp_dir: str
):
    """Handle class diagram based generation."""
    # Process the class diagram JSON data
    buml_model = process_class_diagram(json_data)
    generator_class = generator_info.generator_class
    print("Generator type " + generator_type)

    # Generate based on generator type
    if generator_type == "django":
        return await _generate_django(buml_model, generator_class, config, temp_dir)
    if generator_type == "sql":
        return await _generate_sql(buml_model, generator_class, config, temp_dir)
    if generator_type == "sqlalchemy":
        return await _generate_sqlalchemy(buml_model, generator_class, config, temp_dir)
    if generator_type == "jsonschema":
        return await _generate_jsonschema(buml_model, generator_class, config, temp_dir)

    return await _generate_standard(buml_model, generator_class, generator_type, generator_info, temp_dir)


async def _generate_django(buml_model, generator_class, config: dict, temp_dir: str):
    """Generate Django project."""
    if not config:
        raise HTTPException(status_code=400, detail="Django configuration is required")

    project_dir = os.path.join(temp_dir, config["project_name"])

    # Clean up any existing project directory
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)

    os.makedirs(temp_dir, exist_ok=True)
    original_cwd = os.getcwd()
    os.chdir(temp_dir)

    try:
        generator_instance = generator_class(
            model=buml_model,
            project_name=config["project_name"],
            app_name=config["app_name"],
            containerization=config["containerization"],
            output_dir=temp_dir,
        )
        generator_instance.generate()

        # Wait for file system operations
        await asyncio.sleep(1)

        # Validate generation
        if not os.path.exists(project_dir) or not os.listdir(project_dir):
            raise ValueError("Django project generation failed: Output directory is empty")

        # Create ZIP file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, _, files in os.walk(project_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, project_dir)
                    zip_file.write(file_path, arc_name)

        zip_buffer.seek(0)
        file_name = get_filename_for_generator("django")

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={file_name}"},
        )
    finally:
        os.chdir(original_cwd)
        cleanup_temp_resources(temp_dir)


async def _generate_sql(buml_model, generator_class, config: dict, temp_dir: str):
    """Generate SQL files."""
    dialect = "standard"
    if config and "dialect" in config:
        dialect = config["dialect"]

    generator_instance = generator_class(
        buml_model, output_dir=temp_dir, sql_dialect=dialect
    )
    generator_instance.generate()

    return _create_file_response(temp_dir, "sql")


async def _generate_sqlalchemy(buml_model, generator_class, config: dict, temp_dir: str):
    """Generate SQLAlchemy files."""
    dbms = "sqlite"
    if config and "dbms" in config:
        dbms = config["dbms"]
    
    generator_instance = generator_class(buml_model, output_dir=temp_dir)
    generator_instance.generate(dbms=dbms)
    
    return _create_file_response(temp_dir, "sqlalchemy")


async def _generate_jsonschema(buml_model, generator_class, config: dict, temp_dir: str):
    """Generate JSON Schema files."""
    mode = "regular"
    if config and "mode" in config:
        mode = config["mode"]
    
    generator_instance = generator_class(buml_model, output_dir=temp_dir, mode=mode)
    generator_instance.generate()
    
    if mode == "smart_data":
        # For smart_data mode, we create a zip file with a specific name
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, temp_dir)
                    zip_file.write(file_path, arc_name)
        
        zip_buffer.seek(0)
        cleanup_temp_resources(temp_dir)
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=smart_data_schemas.zip"},
        )
    else:
        return _create_file_response(temp_dir, "jsonschema")

async def _generate_qiskit(json_data: dict, generator_class, config: dict, temp_dir: str):
    """Generate Qiskit code."""
    # Process quantum diagram
    # Note: json_data here is the DiagramInput model dump, so it has 'model' field
    # process_quantum_diagram expects the diagram data structure
    
    # Validate that this is a quantum diagram (has 'cols' in model)
    model_data = json_data.get('model', {})
    if not isinstance(model_data, dict) or 'cols' not in model_data:
        # Check if this looks like a ClassDiagram or other diagram type
        diagram_type = model_data.get('type', 'unknown') if isinstance(model_data, dict) else 'unknown'
        raise HTTPException(
            status_code=400,
            detail=f"Invalid diagram type for Qiskit generator. Expected QuantumCircuitDiagram but received '{diagram_type}'. "
                   f"Please use the 'generate-output-from-project' endpoint or select the Quantum Circuit diagram."
        )
    
    # Debug logging
    # print(f"[Qiskit Gen] json_data keys: {json_data.keys()}")
    # print(f"[Qiskit Gen] title: {json_data.get('title')}")
    # print(f"[Qiskit Gen] model keys: {model_data.keys() if isinstance(model_data, dict) else type(model_data)}")
    # cols = model_data.get('cols', []) if isinstance(model_data, dict) else []
    # print(f"[Qiskit Gen] cols length: {len(cols)}")
    # if cols:
    #     print(f"[Qiskit Gen] first col: {cols[0]}")
    
    quantum_model = process_quantum_diagram(json_data)
    
    # print(f"[Qiskit Gen] quantum_model: {quantum_model}")
    # print(f"[Qiskit Gen] qregs: {quantum_model.qregs}")
    # print(f"[Qiskit Gen] operations count: {len(quantum_model.operations)}")
    
    # Extract Qiskit config
    backend_type = config.get('backend', 'aer_simulator') if config else 'aer_simulator'
    shots = config.get('shots', 1024) if config else 1024
    
    generator_instance = generator_class(
        quantum_model, 
        output_dir=temp_dir,
        backend_type=backend_type,
        shots=shots
    )
    generator_instance.generate()
    
    return _create_file_response(temp_dir, "qiskit")

def _check_for_agent_components(gui_model):
    """Check if the GUI model contains any agent components."""
    from besser.BUML.metamodel.gui.dashboard import AgentComponent
    
    if not gui_model or not gui_model.modules:
        return False
    
    for module in gui_model.modules:
        if not module.screens:
            continue
        for screen in module.screens:
            if not screen.view_elements:
                continue
            for element in screen.view_elements:
                if isinstance(element, AgentComponent):
                    return True
                # Check recursively in containers
                from besser.BUML.metamodel.gui import ViewContainer
                if isinstance(element, ViewContainer):
                    if _check_container_for_agent_components(element):
                        return True
    return False

def _check_container_for_agent_components(container):
    """Recursively check a container for agent components."""
    from besser.BUML.metamodel.gui.dashboard import AgentComponent
    
    if not container.view_elements:
        return False
    
    for element in container.view_elements:
        if isinstance(element, AgentComponent):
            return True
        from besser.BUML.metamodel.gui import ViewContainer
        if isinstance(element, ViewContainer):
            if _check_container_for_agent_components(element):
                return True
    return False

async def _generate_web_app(buml_model, gui_model, generator_class, config: dict, temp_dir: str, agent_model=None):
    """Generate web application files. Optionally includes agent model if agent components are present."""
    generator_instance = generator_class(buml_model, gui_model, output_dir=temp_dir, agent_model=agent_model)
    generator_instance.generate()
    return _create_zip_response(temp_dir, "web_app")

async def _generate_standard(buml_model, generator_class, generator_type: str, generator_info, temp_dir: str):
    """Generate standard files (non-Django, non-SQL)."""
    generator_instance = generator_class(buml_model, output_dir=temp_dir)
    
    try:
        generator_instance.generate()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Code generation failed.")

    # Return ZIP or single file based on generator info
    if generator_info.output_type == "zip":
        return _create_zip_response(temp_dir, generator_type)
    else:
        return _create_file_response(temp_dir, generator_type)


def _create_zip_response(temp_dir: str, generator_type: str):
    """Create a ZIP file response."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_name = os.path.relpath(file_path, temp_dir)
                zip_file.write(file_path, arc_name)

    zip_buffer.seek(0)
    file_name = get_filename_for_generator(generator_type)
    cleanup_temp_resources(temp_dir)
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={file_name}"},
    )


def _create_file_response(temp_dir: str, generator_type: str):
    """Create a single file response."""
    files = os.listdir(temp_dir)
    if not files:
        raise ValueError(f"{generator_type} generation failed: No output files were created.")

    file_name = files[0]
    output_file_path = os.path.join(temp_dir, file_name)
    
    with open(output_file_path, "rb") as f:
        file_content = f.read()

    # Use the proper filename from config
    proper_filename = get_filename_for_generator(generator_type)
    cleanup_temp_resources(temp_dir)
    
    return Response(
        content=file_content,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{proper_filename}"'},
    )


@app.post("/besser_api/deploy-app")
async def deploy_app(input_data: DiagramInput):
    """
    Deploy application using Docker Compose.
    
    Args:
        input_data: DiagramInput containing diagram data and generator type
        
    Returns:
        dict: Deployment status and information
        
    Raises:
        HTTPException: If deployment fails
    """
    temp_dir = tempfile.mkdtemp(prefix=f'besser_{uuid.uuid4().hex}_')
    try:
        json_data = input_data.model_dump()
        generator = input_data.generator

        if not validate_generator(generator, SUPPORTED_GENERATORS):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid generator type: {generator}. Supported types: {list(SUPPORTED_GENERATORS.keys())}"
            )

        buml_model = process_class_diagram(json_data)
        generator_class = SUPPORTED_GENERATORS[generator].generator_class

        # Handle Django generator with config
        if generator == "django":
            if not input_data.config:
                raise HTTPException(status_code=400, detail="Django configuration is required")

            # Clean up any existing project directory first
            project_dir = os.path.join(temp_dir, input_data.config['project_name'])
            if os.path.exists(project_dir):
                shutil.rmtree(project_dir)

            # Create a fresh project directory
            os.makedirs(temp_dir, exist_ok=True)

            # Change to temp directory before running django-admin
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                generator_instance = generator_class(
                    model=buml_model,
                    project_name=input_data.config['project_name'],
                    app_name=input_data.config['app_name'],
                    containerization=True,
                    output_dir=temp_dir
                )

                # Generate the Django project
                generator_instance.generate()

                # Wait a moment for file system operations to complete
                await asyncio.sleep(1)

                # Check if the project directory exists and has content
                if not os.path.exists(project_dir) or not os.listdir(project_dir):
                    raise ValueError("Django project generation failed: Output directory is empty")

                full_path = os.path.join(temp_dir, input_data.config['project_name'])
                run_docker_compose(directory=full_path, project_name=input_data.config['project_name'])

            finally:
                # Always restore the original working directory
                os.chdir(original_cwd)

            cleanup_temp_resources(temp_dir)

        else:
            raise ValueError("Deployment only possible for Django projects")
    except HTTPException as e:
        # Handle known exceptions with specific status codes
        cleanup_temp_resources(temp_dir)
        raise e
    except Exception as e:
        cleanup_temp_resources(temp_dir)
        print(f"Error during file generation or django deployment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")



@app.post("/besser_api/export-project_as_buml")
async def export_project_as_buml(input_data: ProjectInput = Body(...)):
    try:
        buml_project = json_to_buml_project(input_data)
        state_machine = input_data.diagrams.get("StateMachineDiagram", None)

        state_machine_code = ""
        if (
            state_machine is not None
            and hasattr(state_machine, "model")
            and state_machine.model
            and state_machine.model.get("elements")
        ):
            state_machine_code = process_state_machine(state_machine.model_dump())

        temp_dir = tempfile.mkdtemp(prefix=f"besser_{uuid.uuid4().hex}_")
        output_file_path = os.path.join(temp_dir, "project.py")
        project_to_code(project=buml_project, file_path=output_file_path, sm=state_machine_code)
        with open(output_file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
        shutil.rmtree(temp_dir, ignore_errors=True)
        return Response(
            content=file_content,
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=project.py"},
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/besser_api/export-buml")
async def export_buml(input_data: DiagramInput):
    # Create unique temporary directory for this request
    temp_dir = tempfile.mkdtemp(prefix=f"besser_{uuid.uuid4().hex}_")
    try:
        json_data = input_data.model_dump()
        elements_data = input_data.elements
        if elements_data.get("type") == "StateMachineDiagram":
            state_machine_code = process_state_machine(elements_data)
            output_file_path = os.path.join(temp_dir, "state_machine.py")
            with open(output_file_path, "w") as f:
                f.write(state_machine_code)
            with open(output_file_path, "rb") as f:
                file_content = f.read()
            shutil.rmtree(temp_dir, ignore_errors=True)
            return Response(
                content=file_content,
                media_type="text/plain",
                headers={
                    "Content-Disposition": "attachment; filename=state_machine.py"
                },
            )

        elif elements_data.get("type") == "ClassDiagram":
            buml_model = process_class_diagram(json_data)
            output_file_path = os.path.join(temp_dir, "domain_model.py")
            domain_model_to_code(model=buml_model, file_path=output_file_path)
            with open(output_file_path, "rb") as f:
                file_content = f.read()
            shutil.rmtree(temp_dir, ignore_errors=True)
            return Response(
                content=file_content,
                media_type="text/plain",
                headers={"Content-Disposition": "attachment; filename=domain_model.py"},
            )

        elif elements_data.get("type") == "ObjectDiagram":
            # Old referencing without project
            # Handle object diagram - need both class model and object model in one file
            # reference_data = elements_data.get("referenceDiagramData", {})
            # if not reference_data:
            #     raise ValueError("Object diagram requires reference class diagram data")

            # Process the reference class diagram first
            # reference_json = {"elements": reference_data, "diagramTitle": reference_data.get("title", "Reference Classes")}
            # domain_model = process_class_diagram(reference_json)

            # Process the object diagram with the domain model
            object_model = process_object_diagram(json_data, buml_model)

            # Generate a single file with both domain model and object model
            output_file_path = os.path.join(temp_dir, "complete_model.py")
            domain_model_to_code(model=buml_model, file_path=output_file_path, objectmodel=object_model)
            with open(output_file_path, "rb") as f:
                file_content = f.read()
            shutil.rmtree(temp_dir, ignore_errors=True)
            return Response(
                content=file_content,
                media_type="text/plain",
                headers={"Content-Disposition": "attachment; filename=complete_model.py"},
            )

        elif elements_data.get("type") == "AgentDiagram":
            agent_model = process_agent_diagram(json_data)
            output_file_path = os.path.join(temp_dir, "agent_buml.py")
            agent_model_to_code(agent_model, output_file_path)
            with open(output_file_path, "rb") as f:
                file_content = f.read()
            shutil.rmtree(temp_dir, ignore_errors=True)
            return Response(
                content=file_content,
                media_type="text/plain",
                headers={
                    "Content-Disposition": "attachment; filename=agent_buml.py"
                },
            )

        else:
            raise ValueError(
                f"Unsupported or missing diagram type: {elements_data.get('type')}"
            )

    except HTTPException as e:
        # Handle known exceptions with specific status codes
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise e
    except Exception as e:
        # Handle unexpected exceptions
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/besser_api/get-project-json-model")
async def get_project_json_model(buml_file: UploadFile = File(...)):
    try:
        content = await buml_file.read()
        buml_content = content.decode("utf-8")

        parsed_project = project_to_json(buml_content)

        return {
            "project": parsed_project,
            "exportedAt": datetime.utcnow().isoformat(),
            "version": "2.0.0"
        }

    except Exception as e:
        print(f"Error in get_project_json_model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/besser_api/get-json-model")
async def get_single_json_model(buml_file: UploadFile = File(...)):
    """
    Convert a BUML Python file to JSON format for single diagram import.
    This endpoint handles both full project files and single diagram files.
    """
    try:
        content = await buml_file.read()
        buml_content = content.decode("utf-8")

        # First, try to detect what type of file this is by analyzing the content
        diagram_data = None
        diagram_type = None
        diagram_title = buml_file.filename.rsplit('.', 1)[0] if buml_file.filename else "Imported Diagram"

        # Analyze content to determine the type
        content_lower = buml_content.lower()
        
        # Check for specific BUML patterns to determine the diagram type
        is_agent = any(keyword in content_lower for keyword in [
            'agent(', '.new_intent(', '.new_state(', 'when_intent_matched', 'session.reply'
        ])
        
        is_state_machine = any(keyword in content_lower for keyword in [
            'statemachine(', 'when_event_go_to', '.add_transition', '.new_body'
        ]) and not is_agent
        
        is_domain_model = any(keyword in content_lower for keyword in [
            'domainmodel(', '.create_class(', '.add_attribute(', '.create_association'
        ])
        
        is_gui_model = any(keyword in content_lower for keyword in [
            'guimodel(', '.new_screen(', '.new_module(', 'viewcomponent', 'viewcontainer'
        ])
        
        is_project = 'project(' in content_lower or 'def create_project' in content_lower

        # Try to parse based on detected type
        if is_project:
            try:
                parsed_project = project_to_json(buml_content)
                
                # Find the first available diagram in the project
                if parsed_project.get("ClassDiagram") and parsed_project["ClassDiagram"].get("model"):
                    diagram_data = parsed_project["ClassDiagram"]
                    diagram_type = "ClassDiagram"
                elif parsed_project.get("ObjectDiagram") and parsed_project["ObjectDiagram"].get("model"):
                    diagram_data = parsed_project["ObjectDiagram"]
                    diagram_type = "ObjectDiagram"
                elif parsed_project.get("StateMachineDiagram") and parsed_project["StateMachineDiagram"].get("model"):
                    diagram_data = parsed_project["StateMachineDiagram"]
                    diagram_type = "StateMachineDiagram"
                elif parsed_project.get("AgentDiagram") and parsed_project["AgentDiagram"].get("model"):
                    diagram_data = parsed_project["AgentDiagram"]
                    diagram_type = "AgentDiagram"
                elif parsed_project.get("GUINoCodeDiagram") and parsed_project["GUINoCodeDiagram"].get("model"):
                    diagram_data = parsed_project["GUINoCodeDiagram"]
                    diagram_type = "GUINoCodeDiagram"
                    
                if diagram_data and diagram_data.get("title"):
                    diagram_title = diagram_data["title"]
                    
            except Exception as project_error:
                print(f"Project parsing failed: {str(project_error)}")
                
        elif is_agent:
            try:
                print("Detected Agent diagram, parsing...")
                agent_json = agent_buml_to_json(buml_content)
                diagram_data = {
                    "title": diagram_title,
                    "model": agent_json
                }
                diagram_type = "AgentDiagram"
            except Exception as agent_error:
                print(f"Agent diagram parsing failed: {str(agent_error)}")
                
        elif is_state_machine:
            try:
                print("Detected State Machine diagram, parsing...")
                state_machine_json = state_machine_to_json(buml_content)
                diagram_data = {
                    "title": diagram_title,
                    "model": state_machine_json
                }
                diagram_type = "StateMachineDiagram"
            except Exception as sm_error:
                print(f"State machine parsing failed: {str(sm_error)}")
                
        elif is_domain_model:
            try:
                print("Detected Domain Model (Class diagram), parsing...")
                domain_model = parse_buml_content(buml_content)
                if domain_model and len(domain_model.types) > 0:
                    diagram_json = class_buml_to_json(domain_model)
                    diagram_data = {
                        "title": diagram_title,
                        "model": diagram_json
                    }
                    diagram_type = "ClassDiagram"
                else:
                    raise ValueError("No types found in domain model")
            except Exception as class_error:
                print(f"Class diagram parsing failed: {str(class_error)}")
                
        elif is_gui_model:
            try:
                print("Detected GUI Model diagram, parsing...")
                gui_json = gui_buml_to_json(buml_content)
                diagram_data = {
                    "title": diagram_title,
                    "model": gui_json
                }
                diagram_type = "GUINoCodeDiagram"
            except Exception as gui_error:
                print(f"GUI diagram parsing failed: {str(gui_error)}")

        # Return the diagram in the format expected by the frontend
        return {
            "title": diagram_title,
            "model": {
                **diagram_data.get("model", {}),
                "type": diagram_type
            },
            "diagramType": diagram_type,
            "exportedAt": datetime.utcnow().isoformat(),
            "version": "2.0.0"
        }

    except HTTPException as e:
        # Re-raise HTTP exceptions to preserve status codes
        raise e
    except Exception as e:
        print(f"Error in get_single_json_model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process the uploaded file: {str(e)}")

@app.post("/besser_api/csv-to-domain-model")
async def csv_to_domain_model_endpoint(files: list[UploadFile] = File(...)):
    """
    Accepts one or more CSV files and returns a B-UML domain model as JSON.
    """
    temp_dir = tempfile.mkdtemp(prefix="besser_csv_")
    file_paths = []
    try:
        # Save uploaded files to temp dir
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
            file_paths.append(file_path)

        # Generate DomainModel from CSVs
        domain_model = csv_to_domain_model(file_paths, model_name="ClassDiagram")

        # Parse domain model to JSON using the same logic as get_single_json_model
        diagram_title = "ClassDiagram"
        diagram_type = "ClassDiagram"
        try:
            # If not, parse_buml_content can be used if needed
            if domain_model and hasattr(domain_model, "types") and len(domain_model.types) > 0:
                diagram_json = class_buml_to_json(domain_model)
                diagram_data = {
                    "title": diagram_title,
                    "model": diagram_json
                }
            else:
                raise ValueError("No types found in domain model")
        except Exception as class_error:
            print(f"Class diagram parsing failed: {str(class_error)}")
            raise HTTPException(status_code=500, detail=f"Class diagram parsing failed: {str(class_error)}")

        # Return the diagram in the format expected by the frontend
        return {
            "title": diagram_title,
            "model": {
                **diagram_data.get("model", {}),
                "type": diagram_type
            },
            "diagramType": diagram_type,
            "exportedAt": datetime.utcnow().isoformat(),
            "version": "2.0.0"
        }
    except Exception as e:
        print(f"Error in csv_to_domain_model_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process CSV files: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/besser_api/get-json-model-from-image")
async def get_json_model_from_image(
    image_file: UploadFile = File(...),
    api_key: str = Form(...)
):
    """
    Accepts a PNG or JPEG image and an OpenAI API key, uses BESSER's imagetouml feature to transform the image into a BUML class diagram, then converts the BUML to JSON and returns the JSON object.
    """
    import os
    import tempfile

    try:
        # Save uploaded image to a temp folder
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, image_file.filename)
            with open(image_path, "wb") as f:
                f.write(await image_file.read())

            # Create a folder for the image as expected by mockup_to_buml
            image_folder = os.path.join(temp_dir, "images")
            os.makedirs(image_folder, exist_ok=True)
            os.rename(image_path, os.path.join(image_folder, image_file.filename))

            # Use image_to_buml to generate DomainModel from image and API key
            # Find the image file path
            image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                raise HTTPException(status_code=400, detail="No valid image file found.")
            image_path = os.path.join(image_folder, image_files[0])

            domain_model = image_to_buml(image_path=image_path, openai_token=api_key)
            diagram_json = class_buml_to_json(domain_model)

            diagram_title = diagram_json.get("title", "Imported Class Diagram")
            diagram_type = "ClassDiagram"
            return {
                "title": diagram_title,
                "model": {**diagram_json, "type": diagram_type},
                "diagramType": diagram_type,
                "exportedAt": datetime.utcnow().isoformat(),
                "version": "2.0.0",
            }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in get_json_model_from_image: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process the uploaded image: {str(e)}"
        )


@app.post("/besser_api/validate-diagram")
async def validate_diagram(input_data: DiagramInput):
    """
    Validate diagram by converting to BUML and running metamodel validation.
    
    This is the unified validation endpoint that:
    1. Converts JSON to BUML (construction validation)
    2. Calls .validate() method on the model for structured metamodel validation
    3. For ClassDiagram/ObjectDiagram: runs OCL constraint checks if conversion succeeded
    4. Returns unified validation results with errors, warnings, and OCL results
    """
    try:
        diagram_type = input_data.model.get("type") if input_data.model else None
        validation_errors = []
        validation_warnings = []
        buml_model = None
        object_model = None
        
        # Step 1: Convert to BUML (construction validation)
        try:
            if diagram_type == "ClassDiagram":
                json_data = {
                    "title": input_data.title,
                    "model": input_data.model
                }
                buml_model = process_class_diagram(json_data)
                
                # Run structured metamodel validation
                validation_result = buml_model.validate(raise_exception=False)
                validation_errors.extend(validation_result.get("errors", []))
                validation_warnings.extend(validation_result.get("warnings", []))
                
            elif diagram_type == "ObjectDiagram":
                reference_data = input_data.model.get("referenceDiagramData", {})
                if not reference_data:
                    return {
                        "isValid": False,
                        "errors": ["Object diagram requires reference class diagram data"],
                        "warnings": [],
                        "message": "❌ Validation failed"
                    }
                
                # Process the reference class diagram first
                reference_json = {
                    "title": reference_data.get("title", "Reference Classes"),
                    "model": {
                        "elements": reference_data.get("elements", {}),
                        "relationships": reference_data.get("relationships", {})
                    }
                }
                buml_model = process_class_diagram(reference_json)
                
                # Validate the domain model first
                domain_validation = buml_model.validate(raise_exception=False)
                validation_errors.extend(domain_validation.get("errors", []))
                validation_warnings.extend(domain_validation.get("warnings", []))
                
                # If domain model is valid, process and validate object model
                if domain_validation.get("success", False):
                    object_model = process_object_diagram(input_data.model_dump(), buml_model)
                    object_validation = object_model.validate(raise_exception=False)
                    validation_errors.extend(object_validation.get("errors", []))
                    validation_warnings.extend(object_validation.get("warnings", []))
                
            elif diagram_type == "StateMachineDiagram":
                state_machine_code = process_state_machine(input_data.model_dump())
                return {
                    "isValid": True,
                    "message": "✅ State machine diagram is valid",
                    "errors": [],
                    "warnings": []
                }
                
            elif diagram_type == "AgentDiagram":
                agent_model = process_agent_diagram(input_data.model_dump())
                return {
                    "isValid": True,
                    "message": "✅ Agent diagram is valid",
                    "errors": [],
                    "warnings": []
                }
                
            elif diagram_type == "GUINoCodeDiagram":
                return {
                    "isValid": True,
                    "message": "✅ GUI diagram is valid",
                    "errors": [],
                    "warnings": []
                }
            else:
                return {
                    "isValid": False,
                    "errors": [f"Unsupported diagram type: {diagram_type}"],
                    "warnings": [],
                    "message": "❌ Validation failed"
                }
                
        except ValueError as e:
            # Construction validation errors (from BUML creation setters)
            error_msg = str(e)
            validation_errors.append(error_msg)
        except Exception as e:
            validation_errors.append(f"Validation error: {str(e)}")
        
        # Step 2: If BUML model created successfully AND it's a diagram with OCL support
        ocl_results = None
        if buml_model and diagram_type in ["ClassDiagram", "ObjectDiagram"] and len(validation_errors) == 0:
            try:
                if diagram_type == "ObjectDiagram" and object_model:
                    ocl_results = check_ocl_constraint(buml_model, object_model)
                else:
                    ocl_results = check_ocl_constraint(buml_model)
                    
                # Add OCL warnings if present
                if hasattr(buml_model, "ocl_warnings") and buml_model.ocl_warnings:
                    validation_warnings.extend(buml_model.ocl_warnings)
                    
            except Exception as e:
                validation_warnings.append(f"OCL check warning: {str(e)}")
        
        # Step 3: Build unified response
        is_valid = len(validation_errors) == 0
        response = {
            "isValid": is_valid,
            "errors": validation_errors,
            "warnings": validation_warnings,
            "message": "✅ Diagram is valid" if is_valid else "❌ Validation failed"
        }
        
        # Add OCL-specific results if available
        if ocl_results:
            response["valid_constraints"] = ocl_results.get("valid_constraints", [])
            response["invalid_constraints"] = ocl_results.get("invalid_constraints", [])
            if ocl_results.get("message"):
                response["ocl_message"] = ocl_results["message"]
        
        return response
        
    except Exception as e:
        print(f"Error in validate_diagram: {str(e)}")
        return {
            "isValid": False,
            "errors": [f"Unexpected error: {str(e)}"],
            "warnings": [],
            "message": "❌ Validation failed"
        }


@app.post("/besser_api/check-ocl")
async def check_ocl(input_data: DiagramInput):
    """
    Deprecated: Use /validate-diagram instead.
    This endpoint is kept for backwards compatibility and redirects to the new unified validation.
    """
    print("Warning: /check-ocl is deprecated. Use /validate-diagram instead.")
    return await validate_diagram(input_data)


@app.post("/besser_api/feedback")
async def feedback_endpoint(feedback: FeedbackSubmission):
    """
    Feedback submission endpoint.
    
    Receives user feedback and processes it via the feedback service.
    See feedback_service.py for implementation details.
    
    Args:
        feedback: FeedbackSubmission model containing user feedback data
        
    Returns:
        dict: Status message confirming feedback receipt
        
    Raises:
        HTTPException: If feedback processing fails
    """
    try:
        return submit_feedback(feedback)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )



# Main application entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=9000,
        log_level="info"
    )
#The editor is available at: https://editor.besser-pearl.org