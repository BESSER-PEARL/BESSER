from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
import os, io, zipfile, shutil
import tempfile
import uuid
import asyncio

from besser.utilities.buml_code_builder import domain_model_to_code
from besser.generators.django import DjangoGenerator
from besser.generators.python_classes import PythonGenerator
from besser.generators.java_classes import JavaGenerator
from besser.generators.pydantic_classes import PydanticGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.generators.sql import SQLGenerator
from besser.generators.backend import BackendGenerator
from besser.generators.json import JSONSchemaGenerator

from besser.utilities.web_modeling_editor.backend.models.class_diagram import ClassDiagramInput
from besser.utilities.web_modeling_editor.backend.services.json_to_buml import process_class_diagram, process_state_machine
from besser.utilities.web_modeling_editor.backend.services.buml_to_json import domain_model_to_json, parse_buml_content, state_machine_to_json
from besser.utilities.web_modeling_editor.backend.services.ocl_checker import check_ocl_constraint

app = FastAPI(
    title="Besser Backend API",
    description="API for generating code from UML class diagrams using various generators",
    version="1.0.0"
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
    "jsonschema": JSONSchemaGenerator
}

@api.get("/")
def read_api_root():
    return {"message": "BESSER API is running"}

@api.post("/generate-output")
async def generate_output(input_data: ClassDiagramInput):
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
                    containerization=input_data.config['containerization'],
                    output_dir=temp_dir
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
                buml_model, 
                output_dir=temp_dir,
                sql_dialect=dialect
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

        else:
            generator_instance = generator_class(buml_model, output_dir=temp_dir)
            generator_instance.generate()

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
                headers={"Content-Disposition": f"attachment; filename={file_name}"}
            )

        # For non-zip generators, find the generated file
        files = os.listdir(temp_dir)
        if not files:
            raise ValueError(f"{generator} generation failed: No output files were created.")
        
        file_name = files[0]
        output_file_path = os.path.join(temp_dir, file_name)

        with open(output_file_path, 'rb') as f:
            file_content = f.read()
        
        shutil.rmtree(temp_dir, ignore_errors=True)
        return Response(
            content=file_content, 
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={file_name}"}
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

@api.post("/export-buml")
async def export_buml(input_data: ClassDiagramInput):
    # Create unique temporary directory for this request
    temp_dir = tempfile.mkdtemp(prefix=f'besser_{uuid.uuid4().hex}_')
    try:
        json_data = input_data.dict()
        elements_data = input_data.elements

        if elements_data.get("type") == "StateMachineDiagram":
            state_machine_code = process_state_machine(elements_data)
            output_file_path = os.path.join(temp_dir, "state_machine.py")
            with open(output_file_path, "w") as f:
                f.write(state_machine_code)
            with open(output_file_path, "rb") as f:
                file_content = f.read()
            shutil.rmtree(temp_dir, ignore_errors=True)
            return Response(content=file_content, media_type="text/plain",
                          headers={"Content-Disposition": "attachment; filename=state_machine.py"})

        elif elements_data.get("type") == "ClassDiagram":
            buml_model = process_class_diagram(json_data)
            output_file_path = os.path.join(temp_dir, "domain_model.py")
            domain_model_to_code(model=buml_model, file_path=output_file_path)
            with open(output_file_path, "rb") as f:
                file_content = f.read()
            shutil.rmtree(temp_dir, ignore_errors=True)
            return Response(content=file_content, media_type="text/plain",
                          headers={"Content-Disposition": "attachment; filename=domain_model.py"})

        else:
            raise ValueError(f"Unsupported or missing diagram type: {elements_data.get('type')}")

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
        buml_content = content.decode('utf-8')
        
        # Try to determine if it's a state machine or domain model
        is_state_machine = 'StateMachine' in buml_content and 'Session' in buml_content
        
        if is_state_machine:
            # Convert the state machine Python code directly to JSON
            json_model = state_machine_to_json(buml_content)
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
async def check_ocl(input_data: ClassDiagramInput):
    try:
        # Convert diagram to BUML model
        json_data = {"elements": input_data.elements}
        buml_model = process_class_diagram(json_data)
        
        if not buml_model:
            return {
                "success": False,
                "message": "Failed to create BUML model"
            }
        
        # Check OCL constraints
        result = check_ocl_constraint(buml_model)
        
        # Add warnings to the message if they exist
        if hasattr(buml_model, 'ocl_warnings') and buml_model.ocl_warnings:
            warnings_text = "\n\nWarnings:\n" + "\n".join(buml_model.ocl_warnings)
            result['message'] = result['message'] + warnings_text
            
        return result
        
    except Exception as e:
        print(f"Error in check_ocl: {str(e)}")  # Debug print
        return {
            "success": False,
            "message": f"{str(e)}"
        }

#Mount the `api` app under `/besser_api` 
app.mount("/besser_api", api)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
