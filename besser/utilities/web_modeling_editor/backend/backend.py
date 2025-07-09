import os
import io
import zipfile
import shutil
import tempfile
import uuid
import asyncio
from fastapi import FastAPI, HTTPException, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response

from besser.utilities.buml_code_builder import domain_model_to_code, agent_model_to_code, project_to_code
from besser.utilities.web_modeling_editor.backend.services import run_docker_compose
from besser.generators.django import DjangoGenerator
from besser.generators.python_classes import PythonGenerator
from besser.generators.java_classes import JavaGenerator
from besser.generators.pydantic_classes import PydanticGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.generators.sql import SQLGenerator
from besser.generators.backend import BackendGenerator
from besser.generators.json import JSONSchemaGenerator
from besser.generators.agents.baf_generator import BAFGenerator

from besser.utilities.web_modeling_editor.backend.models import (
    DiagramInput,
    ProjectInput
)
from besser.utilities.web_modeling_editor.backend.services.json_to_buml import (
    process_class_diagram,
    process_state_machine,
    process_agent_diagram,
    process_object_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.buml_to_json import (
    domain_model_to_json,
    parse_buml_content,
    state_machine_to_json,
    agent_buml_to_json,
    object_buml_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.ocl_checker import (
    check_ocl_constraint,
)

from besser.utilities.web_modeling_editor.backend.services import (
    json_to_buml_project
)

app = FastAPI(
    title="Besser Backend API",
    description="API for generating code from UML class diagrams using various generators",
    version="1.0.0",
)

# API sub-application
api = FastAPI()

# Set up CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define generator mappings
GENERATOR_CONFIG = {
    "python": PythonGenerator,
    "java": JavaGenerator,
    "django": DjangoGenerator,
    "pydantic": PydanticGenerator,
    "sqlalchemy": SQLAlchemyGenerator,
    "sql": SQLGenerator,
    "backend": BackendGenerator,
    "jsonschema": JSONSchemaGenerator,
    "agent": BAFGenerator
}


@api.get("/")
def read_api_root():
    return {"message": "BESSER API is running."}


def generate_agent_files(agent_model):
    """
    Generate agent files from an agent model.
    
    Args:
        agent_model: The agent model to generate files from
        
    Returns:
        tuple: (zip_buffer, file_name) containing the zip file buffer and filename
    """
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"besser_agent_{uuid.uuid4().hex}_")
        
        # Generate the agent model to a file
        agent_file = os.path.join(temp_dir, "agent_model.py")
        agent_model_to_code(agent_model, agent_file)
        
        # Execute the generated agent model file
        import sys
        import importlib.util
        
        # Add the temp directory to sys.path so we can import the module
        sys.path.insert(0, temp_dir)
        
        # Import the generated module
        spec = importlib.util.spec_from_file_location("agent_model", agent_file)
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        
        # Now use the BAFGenerator with the agent model from the module
        if hasattr(agent_module, 'agent'):
            # Use the agent from the module
            x = BAFGenerator(agent_module.agent)
        else:
            # Fall back to the original agent model
            x = BAFGenerator(agent_model)
        
        x.generate()
        
        # Create a zip file with all the generated files
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Add the agent model file
            zip_file.write(agent_file, os.path.basename(agent_file))
            
            # Add all files from the output directory
            output_dir = "output"
            if os.path.exists(output_dir):
                for file_name in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, file_name)
                    if os.path.isfile(file_path):
                        zip_file.write(file_path, file_name)
        
        zip_buffer.seek(0)
        file_name = "agent_output.zip"
        
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        
        # Remove the temp directory from sys.path
        if temp_dir in sys.path:
            sys.path.remove(temp_dir)
        
        return zip_buffer, file_name
    except Exception as e:
        print(f"Error generating agent files: {str(e)}")
        raise e


@api.post("/generate-output")
async def generate_output(input_data: DiagramInput):
    temp_dir = tempfile.mkdtemp(prefix=f"besser_{uuid.uuid4().hex}_")
    try:
        json_data = input_data.model_dump()
        generator = input_data.generator

        if generator not in GENERATOR_CONFIG:
            raise HTTPException(
                status_code=400, detail="Invalid generator type specified."
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

        generator_class = GENERATOR_CONFIG[generator]

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
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise e
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@api.post("/deploy-app")
async def deploy_app(input_data: DiagramInput):
    temp_dir = tempfile.mkdtemp(prefix=f'besser_{uuid.uuid4().hex}_')
    try:
        json_data = input_data.model_dump()
        generator = input_data.generator

        if generator not in GENERATOR_CONFIG:
            raise HTTPException(status_code=400, detail="Invalid generator type specified.")

        buml_model = process_class_diagram(json_data)
        generator_class = GENERATOR_CONFIG[generator]

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

            shutil.rmtree(temp_dir, ignore_errors=True)

        else:
            raise ValueError("Deployment only possible for Django projects")
    except HTTPException as e:
        # Handle known exceptions with specific status codes
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise e
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
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


@api.post("/get-json-model")
async def get_json_model(buml_file: UploadFile = File(...)):
    try:
        content = await buml_file.read()
        buml_content = content.decode("utf-8")      # Try to determine what type of model this is
        is_state_machine = "StateMachine" in buml_content and "Session" in buml_content
        is_agent = "Agent" in buml_content and "Session" in buml_content
        is_object_model = "ObjectModel" in buml_content

        if is_state_machine:
            # Convert the state machine Python code directly to JSON
            json_model = state_machine_to_json(buml_content)
            model_name = buml_file.filename
        
        elif is_agent:
            # Convert the agent Python code directly to JSON
            json_model = agent_buml_to_json(buml_content)
            model_name = buml_file.filename
            
        elif is_object_model:
            # Convert the object model Python code directly to JSON
            json_model = object_buml_to_json(buml_content)
            model_name = buml_file.filename
  
        else:
            # Parse the BUML content into a domain model and get OCL constraints
            domain_model = parse_buml_content(buml_content)
            # Convert the domain model to JSON format
            json_model = domain_model_to_json(domain_model)
            model_name = domain_model.name

        wrapped_response = {
            "title": model_name,
            "model": json_model,
        }

        return wrapped_response

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
        print(f"Error in check_ocl: {str(e)}")  # Debug print
        return {"success": False, "message": f"{str(e)}"}


# Mount the `api` app under `/besser_api`
app.mount("/besser_api", api)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
