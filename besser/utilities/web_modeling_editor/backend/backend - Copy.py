"""
BESSER Backend API

This module provides FastAPI endpoints for the BESSER web modeling editor backend.
It handles code generation from UML diagrams using various generators.
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
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, File, UploadFile, Body, Form


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# BESSER image-to-UML and BUML utilities
from besser.utilities.image_to_buml import image_to_buml
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import parse_buml_content, class_buml_to_json
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response

# BESSER utilities
from besser.utilities.buml_code_builder import (
    domain_model_to_code, 
    agent_model_to_code, 
    project_to_code
)

# Backend models
from besser.utilities.web_modeling_editor.backend.models import (
    DiagramInput,
    ProjectInput
)

# Backend services - Converters
from besser.utilities.web_modeling_editor.backend.services.converters import (
    # JSON to BUML converters
    process_class_diagram,
    process_state_machine,
    process_agent_diagram,
    process_object_diagram,
    json_to_buml_project,
    # BUML to JSON converters
    class_buml_to_json,
    parse_buml_content,
    state_machine_to_json,
    agent_buml_to_json,
    object_buml_to_json,
    project_to_json,
)

# Backend services - Other services
from besser.utilities.web_modeling_editor.backend.services.validators import (
    check_ocl_constraint,
)
from besser.utilities.web_modeling_editor.backend.services.deployment import (
    run_docker_compose,
)
from besser.utilities.web_modeling_editor.backend.services.utils import (
    cleanup_temp_resources,
    validate_generator,
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
            "validate_ocl": "/besser_api/check-ocl"
        }
    }


def generate_agent_files(agent_model, config: Dict[str, Any] = None):
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
        print(agent_model)
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
        # If config is provided and not empty, save it as a JSON file
        config_file = None
        if config and config != {}:
            config_file = os.path.join(temp_dir, "config.json")
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

        # Use the BAFGenerator with the agent model from the module
       
        if hasattr(agent_module, 'agent'):
            generator = generator_class(agent_module.agent, config_path=config_file)
        else:
            # Fall back to the original agent model
            generator = generator_class(agent_model, config_path=config_file)
        
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


async def _handle_agent_generation(json_data: dict):
    """Handle agent diagram generation."""
    try:
        # Get languages from config if present
        config = json_data.get('config', {})
        print("Generating default agent...")
        languages = config.get('languages') if config else None

        # New format: languages is a dict with 'source' and 'target'
        if languages and isinstance(languages, dict):
            print("mnpte jerje")
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
                        agent_file = os.path.join(temp_dir, f"agent_model_{lang}.py")
                        agent_model_to_code(agent_model, agent_file)
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
                        zip_file.write(agent_file, f"{lang}/agent_model_{lang}.py")
                        if os.path.exists(OUTPUT_DIR):
                            for file_name in os.listdir(OUTPUT_DIR):
                                file_path = os.path.join(OUTPUT_DIR, file_name)
                                if os.path.isfile(file_path):
                                    zip_file.write(file_path, f"{lang}/{file_name}")
                    finally:
                        print("dsds")
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
            zip_buffer, file_name = generate_agent_files(agent_model, config)
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
    
    # Generate based on generator type
    if generator_type == "django":
        return await _generate_django(buml_model, generator_class, config, temp_dir)
    elif generator_type == "sql":
        return await _generate_sql(buml_model, generator_class, config, temp_dir)
    elif generator_type == "sqlalchemy":
        return await _generate_sqlalchemy(buml_model, generator_class, config, temp_dir)
    elif generator_type == "jsonschema":
        return await _generate_jsonschema(buml_model, generator_class, config, temp_dir)
    else:
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
        if state_machine.model.get("elements"):
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
            # Handle object diagram - need both class model and object model in one file
            reference_data = elements_data.get("referenceDiagramData", {})
            if not reference_data:
                raise ValueError("Object diagram requires reference class diagram data")

            # Process the reference class diagram first
            reference_json = {"elements": reference_data, "diagramTitle": reference_data.get("title", "Reference Classes")}
            domain_model = process_class_diagram(reference_json)

            # Process the object diagram with the domain model
            object_model = process_object_diagram(json_data, domain_model)

            # Generate a single file with both domain model and object model
            output_file_path = os.path.join(temp_dir, "complete_model.py")
            domain_model_to_code(model=domain_model, file_path=output_file_path, objectmodel=object_model)
            with open(output_file_path, "rb") as f:
                file_content = f.read()
            shutil.rmtree(temp_dir, ignore_errors=True)
            return Response(
                content=file_content,
                media_type="text/plain",
                headers={"Content-Disposition": "attachment; filename=complete_model.py"},
            )

        elif elements_data.get("type") == "AgentDiagram":
            print("lelelel")
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
    
@app.post("/besser_api/get-single-json-model")
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


@app.post("/besser_api/check-ocl")
async def check_ocl(input_data: DiagramInput):
    try:
        # Check if this is an ObjectDiagram by looking at the elements type
        diagram_type = input_data.model.get("type") if input_data.model else None

        if diagram_type == "ObjectDiagram":
            # Handle object diagram - need both class model and object model in one file
            reference_data = input_data.model.get("referenceDiagramData", {})
            if not reference_data:
                raise ValueError("Object diagram requires reference class diagram data")
            
            # Process the reference class diagram first
            reference_json = {
                "title": reference_data.get("title", "Reference Classes"),
                "model": {
                    "elements": reference_data.get("elements", {}),
                    "relationships": reference_data.get("relationships", {})
                }
            }
            buml_model = process_class_diagram(reference_json)
            # Process the object diagram with the domain model
            object_model = process_object_diagram(input_data.model_dump(), buml_model)
            result = check_ocl_constraint(buml_model, object_model)

        else:
            # Convert diagram to BUML model
            json_data = {
                    "title": input_data.title,
                    "model": input_data.model
                }
            buml_model = process_class_diagram(json_data)
            if not buml_model:
                return {"success": False, "message": "Failed to create BUML model"}

            # Check OCL constraints
            result = check_ocl_constraint(buml_model)

        # Add warnings to the message if they exist
        if hasattr(buml_model, "ocl_warnings") and buml_model.ocl_warnings:
            warnings_text = "\n\nWarnings:\n" + "\n".join(buml_model.ocl_warnings)
            result["message"] = result["message"] + warnings_text

        return result

    except Exception as e:
        print(f"Error in check_ocl: {str(e)}")
        return {"success": False, "message": f"{str(e)}"}


# Main application entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=9000,
        log_level="info"
    )
