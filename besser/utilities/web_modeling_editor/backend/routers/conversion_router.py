"""
Conversion Router

Handles all conversion, export, and import endpoints for the BESSER web modeling editor backend.
"""

import logging
import os
import shutil
import tempfile
import uuid
import importlib.util
import json
from copy import deepcopy
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, File, UploadFile, Body, Form
from fastapi.responses import Response

# Backend response models
from besser.utilities.web_modeling_editor.backend.models.responses import (
    DiagramExportResponse,
    ProjectExportResponse,
)

# BESSER image-to-UML and BUML utilities
from besser.utilities.image_to_buml import image_to_buml
from besser.utilities.kg_to_buml import kg_to_buml

# BESSER utilities
from besser.utilities.buml_code_builder.domain_model_builder import domain_model_to_code
from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code
from besser.utilities.buml_code_builder.project_builder import project_to_code

# Backend models
from besser.utilities.web_modeling_editor.backend.models import (
    DiagramInput,
    ProjectInput,
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
    gui_buml_to_json,
    project_to_json,
)

# Backend services - Other services
from besser.utilities.web_modeling_editor.backend.services.utils import (
    cleanup_temp_resources,
)
from besser.utilities.web_modeling_editor.backend.services.utils.agent_generation_utils import (
    extract_openai_api_key,
)
from besser.utilities.web_modeling_editor.backend.services.reverse_engineering import (
    csv_to_domain_model,
)

# Backend configuration
from besser.utilities.web_modeling_editor.backend.config import (
    get_generator_info,
)

# Backend constants
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    API_VERSION,
    TEMP_DIR_PREFIX,
    AGENT_TEMP_DIR_PREFIX,
    CSV_TEMP_DIR_PREFIX,
    OUTPUT_DIR_NAME,
    AGENT_MODEL_FILENAME,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/besser_api", tags=["conversion"])


@router.post("/export-project_as_buml")
async def export_project_as_buml(input_data: ProjectInput = Body(...)):
    try:
        buml_project = json_to_buml_project(input_data)
        state_machine = input_data.get_active_diagram("StateMachineDiagram")

        state_machine_code = ""
        if (
            state_machine is not None
            and hasattr(state_machine, "model")
            and state_machine.model
            and state_machine.model.get("elements")
        ):
            state_machine_code = process_state_machine(state_machine.model_dump())

        temp_dir = tempfile.mkdtemp(prefix=f"{TEMP_DIR_PREFIX}{uuid.uuid4().hex}_")
        try:
            output_file_path = os.path.join(temp_dir, "project.py")
            project_to_code(project=buml_project, file_path=output_file_path, sm=state_machine_code)
            with open(output_file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
            return Response(
                content=file_content,
                media_type="text/plain",
                headers={"Content-Disposition": 'attachment; filename="project.py"'},
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@router.post("/export-buml")
async def export_buml(input_data: DiagramInput):
    # Create unique temporary directory for this request
    temp_dir = tempfile.mkdtemp(prefix=f"{TEMP_DIR_PREFIX}{uuid.uuid4().hex}_")
    try:
        json_data = input_data.model_dump()
        elements_data = input_data.model
        if elements_data.get("type") == "StateMachineDiagram":
            state_machine_code = process_state_machine(json_data)
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
                    "Content-Disposition": 'attachment; filename="state_machine.py"'
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
                headers={"Content-Disposition": 'attachment; filename="domain_model.py"'},
            )

        elif elements_data.get("type") == "ObjectDiagram":
            # Build the reference class diagram model from referenceDiagramData
            reference_data = input_data.referenceDiagramData or elements_data.get("referenceDiagramData", {})
            if not reference_data:
                raise ValueError("Object diagram requires reference class diagram data")

            reference_json = {"title": "Reference Classes", "model": reference_data}
            buml_model = process_class_diagram(reference_json)

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
                headers={"Content-Disposition": 'attachment; filename="complete_model.py"'},
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
                    "Content-Disposition": 'attachment; filename="agent_buml.py"'
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


@router.post("/get-project-json-model", response_model=ProjectExportResponse)
async def get_project_json_model(buml_file: UploadFile = File(...)):
    try:
        content = await buml_file.read()
        buml_content = content.decode("utf-8")

        parsed_project = project_to_json(buml_content)

        return {
            "project": parsed_project,
            "exportedAt": datetime.now(timezone.utc).isoformat(),
            "version": API_VERSION
        }

    except Exception as e:
        logger.error("Error in get_project_json_model: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/get-json-model", response_model=DiagramExportResponse)
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
        safe_filename = os.path.basename(buml_file.filename) if buml_file.filename else ""
        diagram_title = safe_filename.rsplit('.', 1)[0] if safe_filename else "Imported Diagram"

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
                logger.error("Project parsing failed: %s", str(project_error))

        elif is_agent:
            try:
                logger.info("Detected Agent diagram, parsing...")
                agent_json = agent_buml_to_json(buml_content)
                diagram_data = {
                    "title": diagram_title,
                    "model": agent_json
                }
                diagram_type = "AgentDiagram"
            except Exception as agent_error:
                logger.error("Agent diagram parsing failed: %s", str(agent_error))

        elif is_state_machine:
            try:
                logger.info("Detected State Machine diagram, parsing...")
                state_machine_json = state_machine_to_json(buml_content)
                diagram_data = {
                    "title": diagram_title,
                    "model": state_machine_json
                }
                diagram_type = "StateMachineDiagram"
            except Exception as sm_error:
                logger.error("State machine parsing failed: %s", str(sm_error))

        elif is_domain_model:
            try:
                logger.info("Detected Domain Model (Class diagram), parsing...")
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
                logger.error("Class diagram parsing failed: %s", str(class_error))

        elif is_gui_model:
            try:
                logger.info("Detected GUI Model diagram, parsing...")
                gui_json = gui_buml_to_json(buml_content)
                diagram_data = {
                    "title": diagram_title,
                    "model": gui_json
                }
                diagram_type = "GUINoCodeDiagram"
            except Exception as gui_error:
                logger.error("GUI diagram parsing failed: %s", str(gui_error))

        # Check if we successfully parsed any diagram
        if diagram_data is None or diagram_type is None:
            raise ValueError(
                "Could not parse BUML file. The file format was not recognized as a valid BUML diagram or project. "
                "Supported formats: ClassDiagram, ObjectDiagram, StateMachineDiagram, AgentDiagram, GUINoCodeDiagram, or Project."
            )

        # Return the diagram in the format expected by the frontend
        return {
            "title": diagram_title,
            "model": {
                **diagram_data.get("model", {}),
                "type": diagram_type
            },
            "diagramType": diagram_type,
            "exportedAt": datetime.now(timezone.utc).isoformat(),
            "version": API_VERSION
        }

    except HTTPException as e:
        # Re-raise HTTP exceptions to preserve status codes
        raise e
    except Exception as e:
        logger.error("Error in get_single_json_model: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to process the uploaded file: {str(e)}")

@router.post("/csv-to-domain-model", response_model=DiagramExportResponse)
async def csv_to_domain_model_endpoint(files: list[UploadFile] = File(...)):
    """
    Accepts one or more CSV files and returns a B-UML domain model as JSON.
    """
    temp_dir = tempfile.mkdtemp(prefix=CSV_TEMP_DIR_PREFIX)
    file_paths = []
    try:
        # Save uploaded files to temp dir
        for file in files:
            safe_filename = os.path.basename(file.filename)
            file_path = os.path.join(temp_dir, safe_filename)
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
            logger.error("Class diagram parsing failed: %s", str(class_error))
            raise HTTPException(status_code=500, detail=f"Class diagram parsing failed: {str(class_error)}")

        # Check if we successfully parsed any diagram
        if diagram_data is None or diagram_type is None:
            raise ValueError(
                "Could not parse BUML file. The file format was not recognized as a valid BUML diagram or project. "
                "Supported formats: ClassDiagram, ObjectDiagram, StateMachineDiagram, AgentDiagram, GUINoCodeDiagram, or Project."
            )

        # Return the diagram in the format expected by the frontend
        return {
            "title": diagram_title,
            "model": {
                **diagram_data.get("model", {}),
                "type": diagram_type
            },
            "diagramType": diagram_type,
            "exportedAt": datetime.now(timezone.utc).isoformat(),
            "version": API_VERSION
        }
    except Exception as e:
        logger.error("Error in csv_to_domain_model_endpoint: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to process CSV files: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@router.post("/get-json-model-from-image", response_model=DiagramExportResponse)
async def get_json_model_from_image(
    image_file: UploadFile = File(...),
    api_key: str = Form(...)
):
    """
    Accepts a PNG or JPEG image and an OpenAI API key, uses BESSER's imagetouml feature to transform the image into a BUML class diagram, then converts the BUML to JSON and returns the JSON object.
    """
    try:
        # Save uploaded image to a temp folder
        with tempfile.TemporaryDirectory() as temp_dir:
            safe_filename = os.path.basename(image_file.filename)
            image_path = os.path.join(temp_dir, safe_filename)
            with open(image_path, "wb") as f:
                f.write(await image_file.read())

            # Create a folder for the image as expected by mockup_to_buml
            image_folder = os.path.join(temp_dir, "images")
            os.makedirs(image_folder, exist_ok=True)
            os.rename(image_path, os.path.join(image_folder, safe_filename))

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
                "exportedAt": datetime.now(timezone.utc).isoformat(),
                "version": API_VERSION,
            }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("Error in get_json_model_from_image: %s", str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to process the uploaded image: {str(e)}"
        )

@router.post("/get-json-model-from-kg", response_model=DiagramExportResponse)
async def get_json_model_from_kg(
    kg_file: UploadFile = File(...),
    api_key: str = Form(...)
):
    """
    Accepts an .TTL, .RDF, or .JSON knowledge graphs and an OpenAI API key, uses BESSER's kgtouml feature to transform the kg into a BUML class diagram in JSON.
    """
    try:
        # Save uploaded kg to a temp folder
        with tempfile.TemporaryDirectory() as temp_dir:
            safe_filename = os.path.basename(kg_file.filename)
            kg_path = os.path.join(temp_dir, safe_filename)
            with open(kg_path, "wb") as f:
                f.write(await kg_file.read())
            # Create a folder for the kg as expected by mockup_to_buml
            kg_folder = os.path.join(temp_dir, "kgs")
            os.makedirs(kg_folder, exist_ok=True)
            os.rename(kg_path, os.path.join(kg_folder, safe_filename))
            # Use kg_to_buml to generate DomainModel from kg and API key
            # Find the kg file path
            kg_files = [f for f in os.listdir(kg_folder) if f.lower().endswith(('.ttl', '.json', '.rdf'))]
            if not kg_files:
                raise HTTPException(status_code=400, detail="No valid KG file found.")
            kg_path = os.path.join(kg_folder, kg_files[0])
            domain_model = kg_to_buml(kg_path=kg_path, openai_token=api_key)
            diagram_json = class_buml_to_json(domain_model)

            diagram_title = diagram_json.get("title", "Imported Class Diagram")
            diagram_type = "ClassDiagram"
            return {
                "title": diagram_title,
                "model": {**diagram_json, "type": diagram_type},
                "diagramType": diagram_type,
                "exportedAt": datetime.now(timezone.utc).isoformat(),
                "version": API_VERSION,
            }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("Error in get-json-model-from-kg: %s", str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to process the uploaded KG: {str(e)}"
        )


@router.post("/transform_agent_model_json")
async def transform_agent_model_json(input_data: DiagramInput):
    """Transform an agent JSON model using the generator and return personalized_agent_model as JSON."""
    temp_dir = tempfile.mkdtemp(prefix=f"{AGENT_TEMP_DIR_PREFIX}{uuid.uuid4().hex}_")

    try:
        json_data = input_data.model_dump()
        raw_config = json_data.get("config") if isinstance(json_data.get("config"), dict) else None
        fallback_config = input_data.config if isinstance(input_data.config, dict) else None
        base_config = raw_config if raw_config is not None else fallback_config
        if base_config is None:
            raise HTTPException(status_code=400, detail="Config is required for transformation")

        logger.debug("[Agent transform] config: %s", json.dumps(base_config, indent=2, default=str))
        config = deepcopy(base_config)
        user_profile_payload = config.get("userProfileModel") if isinstance(config, dict) else None
        if isinstance(user_profile_payload, dict):
            config["userProfileModel"] = _generate_user_profile_document(user_profile_payload)
        elif isinstance(config, dict) and "userProfileModel" in config:
            config.pop("userProfileModel", None)

        json_data["config"] = config
        # Convert to BUML model
        agent_model = process_agent_diagram(json_data)

        # Persist BUML to file for generator import
        agent_file = os.path.join(temp_dir, AGENT_MODEL_FILENAME)
        agent_model_to_code(agent_model, agent_file)

        # Import the generated module (spec_from_file_location uses absolute path, no sys.path needed)
        spec = importlib.util.spec_from_file_location("agent_model", agent_file)
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)

        generator_info = get_generator_info("agent")
        generator_class = generator_info.generator_class
        generator_instance = generator_class(
            getattr(agent_module, "agent", agent_model),
            config=config,
            openai_api_key=extract_openai_api_key(config),
            output_dir=temp_dir,
        )
        generator_instance.generate()

        personalized_json_path = os.path.join(temp_dir, OUTPUT_DIR_NAME, "personalized_agent_model.json")
        if not os.path.isfile(personalized_json_path):
            raise HTTPException(status_code=500, detail="personalized_agent_model.json not found after generation")

        with open(personalized_json_path, "r", encoding="utf-8") as f:
            personalized_json = json.load(f)

        return {
            "model": personalized_json,
            "diagramType": "AgentDiagram",
            "exportedAt": datetime.now(timezone.utc).isoformat(),
            "version": API_VERSION,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transform failed: {str(e)}")
    finally:
        cleanup_temp_resources(temp_dir)


def _generate_user_profile_document(user_profile_model: dict) -> dict:
    """Generate the normalized JSON document for a stored user profile diagram.

    This is a local helper used by transform_agent_model_json. It delegates to
    the generation router's implementation for the actual user profile generation.
    """
    # Import here to avoid circular imports at module level
    from besser.utilities.web_modeling_editor.backend.routers.generation_router import (
        _generate_user_profile_document as _gen_user_profile_doc,
    )
    return _gen_user_profile_doc(user_profile_model)
