from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from besser.utilities.buml_code_builder import domain_model_to_code
from besser.generators.django import DjangoGenerator
from besser.generators.python_classes import PythonGenerator
from besser.generators.java_classes import JavaGenerator
from besser.generators.pydantic_classes import PydanticGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.generators.sql import SQLGenerator

from besser_backend.models.class_diagram import ClassDiagramInput
from besser_backend.services.json_to_buml import process_class_diagram, process_state_machine
from besser_backend.services.buml_to_json import domain_model_to_json, parse_buml_content

app = FastAPI(
    title="Besser Backend API",
    description="API for generating code from UML class diagrams using various generators",
    version="1.0.0"
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define generator mappings
GENERATOR_CONFIG = {
    "python": (PythonGenerator, "classes.py"),
    "java": (JavaGenerator, "Class.java"),
    "django": (DjangoGenerator, "models.py"),
    "pydantic": (PydanticGenerator, "pydantic_classes.py"),
    "sqlalchemy": (SQLAlchemyGenerator, "sql_alchemy.py"),
    "sql": (SQLGenerator, "tables.sql"),
}

@app.post("/generate-output")
async def generate_output(input_data: ClassDiagramInput):
    try:
        json_data = input_data.dict()
        print("Input data received:", json_data)

        # Create and clear output directory
        os.makedirs("output", exist_ok=True)
        for file in os.listdir("output"):
            file_path = os.path.join("output", file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

        # Handle class diagram generation
        generator = input_data.generator
        if generator not in GENERATOR_CONFIG:
            raise HTTPException(status_code=400, detail="Invalid generator type specified.")

        buml_model = process_class_diagram(json_data)
        generator_class, file_name = GENERATOR_CONFIG[generator]
        generator_instance = generator_class(buml_model, output_dir="output")

        output_file_path = os.path.join("output", file_name)
        generator_instance.generate()

        if not os.path.exists(output_file_path):
            raise ValueError(f"{generator} generation failed: Output file was not created.")

        return FileResponse(output_file_path, filename=file_name, media_type="text/plain")

    except ValueError as ve:
        print(f"ValueError: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error during file generation or response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/export-buml")
async def export_buml(input_data: dict):
    try:
        print("Input data received for BUML export:", input_data)

        # Create and clear output directory
        os.makedirs("output", exist_ok=True)
        for file in os.listdir("output"):
            file_path = os.path.join("output", file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

        # Get the elements data which contains the actual diagram content
        elements_data = input_data.get("elements", {})
        
        # Check diagram type from the elements data
        if elements_data.get("type") == "StateMachineDiagram":
            print("Processing State Machine Diagram")
            # Handle state machine diagram
            state_machine_code = process_state_machine(elements_data)
            output_file_path = "output/state_machine.py"
            with open(output_file_path, "w") as f:
                f.write(state_machine_code)
            return FileResponse(output_file_path, filename="state_machine.py", media_type="text/plain")
        elif elements_data.get("type") == "ClassDiagram":
            # Handle class diagram
            buml_model = process_class_diagram(elements_data)
            print(f"BUML model created: {buml_model}")
            output_file_path = "output/domain_model.py"
            domain_model_to_code(model=buml_model, file_path=output_file_path)
            return FileResponse(output_file_path, filename="domain_model.py", media_type="text/plain")
        else:
            raise ValueError(f"Unsupported or missing diagram type: {elements_data.get('type')}")

    except ValueError as ve:
        print(f"ValueError: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error during BUML export: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/get-json-model")
async def get_json_model(buml_file: UploadFile = File(...)):
    try:
        content = await buml_file.read()
        buml_content = content.decode('utf-8')
        
        # Parse the BUML content into a domain model
        domain_model = parse_buml_content(buml_content)
        print("Domain model created:", domain_model)
        # Convert the domain model to JSON format using the renamed function
        json_model = domain_model_to_json(domain_model)
        
        wrapped_response = {
            "title": buml_file.filename,
            "model": json_model
        }
        
        return wrapped_response
            
    except Exception as e:
        print(f"Error in get_json_model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
