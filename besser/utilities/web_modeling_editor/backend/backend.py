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
from typing import List, Dict, Any

# Core FastAPI imports
from fastapi import FastAPI, HTTPException, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response

# BESSER utilities
from besser.utilities.buml_code_builder import (
    domain_model_to_code, 
    agent_model_to_code, 
    project_to_code
)

# BESSER generators
from besser.generators.django import DjangoGenerator
from besser.generators.python_classes import PythonGenerator
from besser.generators.java_classes import JavaGenerator
from besser.generators.pydantic_classes import PydanticGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.generators.sql import SQLGenerator
from besser.generators.backend import BackendGenerator
from besser.generators.json import JSONSchemaGenerator
from besser.generators.agents.baf_generator import BAFGenerator

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

# Initialize FastAPI applications
app = FastAPI(
    title="BESSER Backend API",
    description="API for generating code from UML class diagrams using various generators",
    version="1.0.0",
)

# Create API sub-application
api = FastAPI(
    title="BESSER Backend API",
    description="Backend services for web modeling editor",
    version="1.0.0",
)

# Configure CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
API_VERSION = "1.0.0"
TEMP_DIR_PREFIX = "besser_agent_"
OUTPUT_DIR = "output"
AGENT_MODEL_FILENAME = "agent_model.py"
AGENT_OUTPUT_FILENAME = "agent_output.zip"

# Generator configuration mapping
SUPPORTED_GENERATORS = {
    # Object-oriented generators
    "python": PythonGenerator,
    "java": JavaGenerator,
    "pydantic": PydanticGenerator,
    
    # Web framework generators
    "django": DjangoGenerator,
    "backend": BackendGenerator,
    
    # Database generators
    "sqlalchemy": SQLAlchemyGenerator,
    "sql": SQLGenerator,
    
    # Data format generators
    "jsonschema": JSONSchemaGenerator,
    
    # AI/Agent generators
    "agent": BAFGenerator
}


# Utility functions
def cleanup_temp_resources(temp_dir: str = None, output_dir: str = OUTPUT_DIR):
    """
    Clean up temporary resources.
    
    Args:
        temp_dir: Temporary directory to clean up
        output_dir: Output directory to clean up
    """
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    
    # Remove temp directory from sys.path if it exists
    if temp_dir in sys.path:
        sys.path.remove(temp_dir)


def validate_generator(generator_type: str) -> bool:
    """
    Validate if the generator type is supported.
    
    Args:
        generator_type: The type of generator to validate
        
    Returns:
        bool: True if generator is supported, False otherwise
    """
    return generator_type in SUPPORTED_GENERATORS


# API Endpoints
@api.get("/")
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
            "generate": "/generate-output",
            "deploy": "/deploy-app", 
            "export_buml": "/export-buml",
            "export_project": "/export-project_as_buml",
            "get_model": "/get-json-model",
            "validate_ocl": "/check-ocl"
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
        
        # Use the BAFGenerator with the agent model from the module
        if hasattr(agent_module, 'agent'):
            generator = BAFGenerator(agent_module.agent)
        else:
            # Fall back to the original agent model
            generator = BAFGenerator(agent_model)
        
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


@api.post("/generate-output")
async def generate_code_output(input_data: DiagramInput):
    """
    Generate code from UML diagram data using specified generator.
    
    Args:
        input_data: DiagramInput containing diagram data and generator type
        
    Returns:
        StreamingResponse: ZIP file containing generated code
        
    Raises:
        HTTPException: If generator is not supported or generation fails
    """
    temp_dir = tempfile.mkdtemp(prefix=f"besser_{uuid.uuid4().hex}_")
    try:
        json_data = input_data.model_dump()
        generator = input_data.generator

        if not validate_generator(generator):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid generator type: {generator}. Supported types: {list(SUPPORTED_GENERATORS.keys())}"
            )

        buml_model = None

        if generator == "agent":
            try:
                # Create the agent model from the elements
                agent_model = process_agent_diagram(json_data)
                zip_buffer, file_name = generate_agent_files(agent_model)
                
                return StreamingResponse(
                    zip_buffer,
                    media_type="application/zip",
                    headers={"Content-Disposition": f"attachment; filename={file_name}"},
                )
            except Exception as e:
                print(f"Error processing agent diagram: {str(e)}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error processing agent diagram: {str(e)}"
                )
            
        else:
            # Process the class diagram JSON data
            buml_model = process_class_diagram(json_data)

        generator_class = SUPPORTED_GENERATORS[generator]

        # Handle Django generator with config
        if generator == "django":
            if not input_data.config:
                raise HTTPException(
                    status_code=400, detail="Django configuration is required"
                )

            # Clean up any existing project directory first
            project_dir = os.path.join(temp_dir, input_data.config["project_name"])
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
                    project_name=input_data.config["project_name"],
                    app_name=input_data.config["app_name"],
                    containerization=input_data.config["containerization"],
                    output_dir=temp_dir,
                )

                # Generate the Django project
                generator_instance.generate()

                # Wait a moment for file system operations to complete
                await asyncio.sleep(1)

                # Check if the project directory exists and has content
                if not os.path.exists(project_dir) or not os.listdir(project_dir):
                    raise ValueError("Django project generation failed: Output directory is empty")

            finally:
                # Always restore the original working directory
                os.chdir(original_cwd)

        # Handle SQL generator with config
        elif generator == "sql":
            dialect = "standard"
            if input_data.config and "dialect" in input_data.config:
                dialect = input_data.config["dialect"]
            generator_instance = generator_class(
                buml_model, output_dir=temp_dir, sql_dialect=dialect
            )
            generator_instance.generate()

        # Handle SQLAlchemy generator with config
        elif generator == "sqlalchemy":
            dbms = "sqlite"
            if input_data.config and "dbms" in input_data.config:
                dbms = input_data.config["dbms"]
            generator_instance = generator_class(
                buml_model, 
                output_dir=temp_dir
            )
            generator_instance.generate(dbms=dbms)

            generator_instance = generator_class(buml_model, output_dir=temp_dir)
            generator_instance.generate(dbms=dbms)
        else:
            # Save buml_model to a file for inspection
            generator_instance = generator_class(buml_model, output_dir=temp_dir)
            try:
                # Generate the code
                generator_instance.generate()
            except Exception as e:
                raise HTTPException(status_code=500, detail="Code generation failed.")

        # Handle zip file generators
        if generator in ["java", "backend", "django"]:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                # For Django, start from the project directory
                base_dir = project_dir if generator == "django" else temp_dir
                for root, _, files in os.walk(base_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path, base_dir)
                        zip_file.write(file_path, arc_name)

            zip_buffer.seek(0)
            file_name = f"{generator}_output.zip"
            shutil.rmtree(temp_dir, ignore_errors=True)
            return StreamingResponse(
                zip_buffer,
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={file_name}"},
            )
        if generator == "agent":
            # Create a zip file containing the two files in the output folder
            try:
                output_dir = os.path.join("output")
                if not os.path.exists(output_dir) or len(os.listdir(output_dir)) < 2:
                    raise HTTPException(
                        status_code=500, detail="Output folder does not contain the required files."
                    )
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for file_name in os.listdir(output_dir):
                        file_path = os.path.join(output_dir, file_name)
                        zip_file.write(file_path, file_name)
                zip_buffer.seek(0)
                file_name = "agent_output.zip"
                shutil.rmtree("output", ignore_errors=True)
                return StreamingResponse(
                    zip_buffer,
                    media_type="application/zip",
                    headers={"Content-Disposition": f"attachment; filename={file_name}"},
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error creating zip file: {str(e)}")
        # For non-zip generators, find the generated file
        files = os.listdir(temp_dir)
        if not files:
            raise ValueError(f"{generator} generation failed: No output files were created.")


        file_name = files[0]
        output_file_path = os.path.join(temp_dir, file_name)
        with open(output_file_path, "rb") as f:
            file_content = f.read()

        shutil.rmtree(temp_dir, ignore_errors=True)
        return Response(
            content=file_content,
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{file_name}"'},
        )
    except HTTPException as e:
        # Handle known exceptions with specific status codes
        cleanup_temp_resources(temp_dir)
        raise e
    except Exception as e:
        cleanup_temp_resources(temp_dir)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@api.post("/deploy-app")
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

        if not validate_generator(generator):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid generator type: {generator}. Supported types: {list(SUPPORTED_GENERATORS.keys())}"
            )

        buml_model = process_class_diagram(json_data)
        generator_class = SUPPORTED_GENERATORS[generator]

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



@api.post("/export-project_as_buml")
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


@api.post("/export-buml")
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


from fastapi import UploadFile, File, HTTPException
from datetime import datetime

@api.post("/get-json-model")
async def get_json_model(buml_file: UploadFile = File(...)):
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
        print(f"Error in get_json_model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/check-ocl")
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
            json_data = {"elements": input_data.model}
            object_model = process_object_diagram(json_data, buml_model)
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


# Mount the API sub-application
app.mount("/besser_api", api)

# Main application entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=9000,
        log_level="info"
    )
