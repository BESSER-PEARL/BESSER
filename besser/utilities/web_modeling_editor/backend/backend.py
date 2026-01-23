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
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
import sys
import json
sys.path.append("C:/Users/conrardy/Desktop/git/BESSER")
from fastapi import FastAPI, HTTPException, File, UploadFile, Body, Form



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
from besser.generators.agents.baf_generator import GenerationMode

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
    process_gui_diagram,
    # BUML to JSON converters
    class_buml_to_json,
    parse_buml_content,
    state_machine_to_json,
    agent_buml_to_json,
    object_buml_to_json,
    gui_buml_to_json,
    project_to_json,
)
from besser.utilities.web_modeling_editor.backend.constants.user_buml_model import (
    domain_model as user_reference_domain_model,
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
from besser.utilities.web_modeling_editor.backend.services.utils.agent_generation_utils import (
    build_configurations_package,
    append_zip_contents,
    slugify_name,
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
            "validate_diagram": "/besser_api/validate-diagram",
            "check_ocl": "/besser_api/check-ocl (deprecated, use validate_diagram)"
        }
    }


def generate_agent_files(agent_model, config, generation_mode: GenerationMode = GenerationMode.FULL):
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
            generator = generator_class(agent_module.agent, config=config, generation_mode=generation_mode)
        else:
            # Fall back to the original agent model
            generator = generator_class(agent_model, config=config, generation_mode=generation_mode)
        
        generator.generate()

        # Prepare user profile bundle when personalization mappings are provided
        user_profiles_path = None
        if isinstance(config, dict) and isinstance(config.get('personalizationMapping'), list):
            user_profiles = []
            for idx, entry in enumerate(config.get('personalizationMapping')):
                if not isinstance(entry, dict):
                    continue
                profile_data = entry.get('user_profile')
                profile_name = entry.get('name') or f"variant_{idx + 1}"
                if profile_data is not None:
                    user_profiles.append({
                        "name": profile_name,
                        "user_profile": profile_data,
                    })

            if user_profiles:
                user_profiles_path = os.path.join(temp_dir, "user_profiles.json")
                with open(user_profiles_path, "w", encoding="utf-8") as profiles_file:
                    json.dump(user_profiles, profiles_file, indent=2)
        
        # Create a zip file with all the generated files
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Add the agent model file
            zip_file.write(agent_file, os.path.basename(agent_file))

            # Add user profiles if generated
            if user_profiles_path and os.path.exists(user_profiles_path):
                zip_file.write(user_profiles_path, os.path.basename(user_profiles_path))
            
            # Add all files from the output directory
            if os.path.exists(OUTPUT_DIR):
                for root, _, files in os.walk(OUTPUT_DIR):
                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        arcname = os.path.relpath(file_path, OUTPUT_DIR)
                        zip_file.write(file_path, arcname)
        
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

        if generator_info.category == "object_model":
            diagram_type = _get_diagram_type(json_data)
            if diagram_type == "UserDiagram":
                return await _handle_user_diagram_generation(
                    json_data,
                    generator_type,
                    generator_info,
                    input_data.config,
                    temp_dir,
                )

            return await _handle_object_diagram_generation(
                json_data,
                generator_type,
                generator_info,
                input_data.config,
                temp_dir,
            )

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
    """Handle Web App generation from a complete project with both ClassDiagram and GUINoCodeDiagram."""
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

        # Generate Web App TypeScript project
        generator_class = generator_info.generator_class

        return await _generate_web_app(buml_model, gui_model, generator_class, config, temp_dir)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Web App generation failed: {str(e)}")


async def _handle_agent_generation(json_data: dict):
    """Handle agent diagram generation."""
    try:
        # Get languages from config if present
        config = json_data.get('config', {})

        is_config_dict = isinstance(config, dict)
        if is_config_dict and isinstance(config.get('personalizationMapping'), list):
            normalized_mappings = []
            for index, entry in enumerate(config.get('personalizationMapping') or []):
                if not isinstance(entry, dict):
                    raise HTTPException(status_code=400, detail=f"personalizationMapping entry at index {index} must be an object")

                user_profile_payload = entry.get('user_profile')
                if user_profile_payload is None:
                    raise HTTPException(status_code=400, detail=f"personalizationMapping entry at index {index} is missing 'user_profile'")

                simplified_profile = _generate_user_profile_document(user_profile_payload)

                agent_model_json = entry.get('agent_model')
                agent_model_buml = None
                if isinstance(agent_model_json, dict):
                    mapping_payload = dict(json_data)
                    mapping_payload['model'] = deepcopy(agent_model_json)
                    try:
                        agent_model_buml = process_agent_diagram(mapping_payload)
                    except Exception as conversion_error:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to convert personalizationMapping agent_model at index {index} to BUML: {conversion_error}",
                        )

                normalized_entry = dict(entry)
                normalized_entry['user_profile'] = simplified_profile
                if agent_model_buml is not None:
                    normalized_entry['agent_model'] = agent_model_buml
                normalized_mappings.append(normalized_entry)

            config['personalizationMapping'] = normalized_mappings
            json_data['config'] = config


        languages = config.get('languages') if is_config_dict else None
        configuration_variants = config.get('configurations') if is_config_dict else None
        base_model_snapshot = config.get('baseModel') if is_config_dict else None
        variation_entries = config.get('variations') if is_config_dict else None

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
        elif base_model_snapshot is not None and isinstance(variation_entries, list):
            def to_agent_model(model_snapshot: Dict[str, Any]):
                variant_payload = dict(json_data)
                variant_payload['model'] = model_snapshot
                return process_agent_diagram(variant_payload)

            combined_zip = io.BytesIO()
            with zipfile.ZipFile(combined_zip, "w", zipfile.ZIP_DEFLATED) as zip_file:
                base_agent_model = to_agent_model(base_model_snapshot)
                base_buffer, _ = generate_agent_files(
                    base_agent_model,
                    config,
                    generation_mode=GenerationMode.CODE_ONLY,
                )
                append_zip_contents(zip_file, base_buffer)

                for index, entry in enumerate(variation_entries):
                    if not isinstance(entry, dict):
                        continue
                    variant_snapshot = entry.get('model')
                    if not variant_snapshot:
                        continue
                    try:
                        variant_agent_model = to_agent_model(variant_snapshot)
                    except Exception as conversion_error:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid agent model provided for variation '{entry.get('name', index)}': {conversion_error}",
                        ) from conversion_error

                    folder_name = slugify_name(entry.get('name'), f"variation_{index + 1}")
                    variant_buffer, _ = generate_agent_files(
                        variant_agent_model,
                        entry.get('config'),
                        generation_mode=GenerationMode.CODE_ONLY,
                    )
                    append_zip_contents(zip_file, variant_buffer, prefix=folder_name or f"variation_{index + 1}")

            combined_zip.seek(0)
            return StreamingResponse(
                combined_zip,
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={AGENT_OUTPUT_FILENAME}"},
            )
        elif configuration_variants and isinstance(configuration_variants, list):
            agent_model = process_agent_diagram(json_data)
            bundle_buffer = build_configurations_package(
                agent_model,
                configuration_variants,
                generate_agent_files,
            )
            return StreamingResponse(
                bundle_buffer,
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={AGENT_OUTPUT_FILENAME}"},
            )
        elif 'personalizationMapping' in config:
            agent_model = process_agent_diagram(json_data)
            zip_buffer, file_name = generate_agent_files(agent_model, config, generation_mode=GenerationMode.CODE_ONLY)
            return StreamingResponse(
                zip_buffer,
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={file_name}"},
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


@app.post("/besser_api/transform_agent_model_json")
async def transform_agent_model_json(input_data: DiagramInput):
    """Transform an agent JSON model using the generator and return personalized_agent_model as JSON."""
    temp_dir = tempfile.mkdtemp(prefix=f"{TEMP_DIR_PREFIX}{uuid.uuid4().hex}_")
    original_cwd = os.getcwd()
    sys_path_added = False

    try:
        json_data = input_data.model_dump()
        raw_config = json_data.get("config") if isinstance(json_data.get("config"), dict) else None
        fallback_config = input_data.config if isinstance(input_data.config, dict) else None
        base_config = raw_config if raw_config is not None else fallback_config
        if not base_config:
            raise HTTPException(status_code=400, detail="Config is required for transformation")

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

        # Prepare import context
        sys.path.insert(0, temp_dir)
        sys_path_added = True
        os.chdir(temp_dir)

        spec = importlib.util.spec_from_file_location("agent_model", agent_file)
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)

        generator_info = get_generator_info("agent")
        generator_class = generator_info.generator_class
        generator_instance = generator_class(getattr(agent_module, "agent", agent_model), config=config)
        generator_instance.generate()

        personalized_json_path = os.path.join(temp_dir, OUTPUT_DIR, "personalized_agent_model.json")
        if not os.path.isfile(personalized_json_path):
            raise HTTPException(status_code=500, detail="personalized_agent_model.json not found after generation")

        with open(personalized_json_path, "r", encoding="utf-8") as f:
            personalized_json = json.load(f)

        return {
            "model": personalized_json,
            "diagramType": "AgentDiagram",
            "exportedAt": datetime.utcnow().isoformat(),
            "version": "2.0.0",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transform failed: {str(e)}")
    finally:
        os.chdir(original_cwd)
        if sys_path_added:
            try:
                sys.path.remove(temp_dir)
            except ValueError:
                pass
        cleanup_temp_resources(temp_dir)


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


async def _handle_object_diagram_generation(
    json_data: dict,
    generator_type: str,
    generator_info,
    config: dict,
    temp_dir: str,
):
    """Handle generators that operate on object diagrams."""
    reference_payload = _extract_reference_class_diagram(json_data)
    if not reference_payload:
        raise HTTPException(
            status_code=400,
            detail="Object diagram generation requires reference class diagram data."
        )

    domain_model = process_class_diagram(reference_payload)
    object_model = process_object_diagram(json_data, domain_model)

    generator_class = generator_info.generator_class
    generator_instance = generator_class(object_model, output_dir=temp_dir)
    generator_instance.generate()

    return _create_file_response(temp_dir, generator_type)


async def _handle_user_diagram_generation(
    json_data: dict,
    generator_type: str,
    generator_info,
    config: dict,
    temp_dir: str,
):
    """Handle user diagram generation using the preset user reference domain."""
    try:
        object_model = process_object_diagram(json_data, user_reference_domain_model)
    except Exception as conversion_error:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to convert user diagram to BUML: {conversion_error}",
        ) from conversion_error

    generator_class = generator_info.generator_class
    generator_instance = generator_class(object_model, output_dir=temp_dir)
    generator_instance.generate()

    _normalize_user_model_output(object_model, temp_dir)

    return _create_file_response(temp_dir, generator_type)


def _generate_user_profile_document(user_profile_model: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the normalized JSON document for a stored user profile diagram."""
    if not isinstance(user_profile_model, dict):
        raise HTTPException(status_code=400, detail="userProfileModel must contain a serialized UserDiagram")

    diagram_title = (
        user_profile_model.get("title")
        or user_profile_model.get("name")
        or user_profile_model.get("id")
        or "UserProfile"
    )
    prepared_payload = {
        "title": diagram_title,
        "diagramType": "UserDiagram",
        "model": deepcopy(user_profile_model),
        "generator": "json",
    }

    model_section = prepared_payload["model"]
    if isinstance(model_section, dict):
        model_section.setdefault("type", "UserDiagram")

    temp_dir = tempfile.mkdtemp(prefix=f"user_profile_{uuid.uuid4().hex}_")
    try:
        object_model = process_object_diagram(prepared_payload, user_reference_domain_model)
        generator_info = get_generator_info("json")
        if not generator_info:
            raise HTTPException(status_code=500, detail="JSON generator is not configured")
        generator_class = generator_info.generator_class
        generator_instance = generator_class(object_model, output_dir=temp_dir)
        generator_instance.generate()

        _normalize_user_model_output(object_model, temp_dir)

        file_name = _sanitize_object_model_filename(getattr(object_model, "name", None))
        json_path = os.path.join(temp_dir, f"{file_name}.json")
        if not os.path.isfile(json_path):
            raise HTTPException(status_code=500, detail="Failed to render user profile JSON document")

        with open(json_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to convert user profile model: {exc}") from exc
    finally:
        cleanup_temp_resources(temp_dir)


def _normalize_user_model_output(object_model, temp_dir: str) -> None:
    file_name = _sanitize_object_model_filename(getattr(object_model, "name", None))
    json_path = os.path.join(temp_dir, f"{file_name}.json")
    if not os.path.isfile(json_path):
        return

    try:
        with open(json_path, "r", encoding="utf-8") as source:
            document = json.load(source)
    except (OSError, json.JSONDecodeError):
        return

    normalized_document = _build_user_model_hierarchy(document)
    if not normalized_document:
        return

    with open(json_path, "w", encoding="utf-8") as target:
        json.dump(normalized_document, target, indent=2, ensure_ascii=False)


def _build_user_model_hierarchy(document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    objects = document.get("objects")
    if not isinstance(objects, list):
        return None

    objects_by_id: Dict[str, Dict[str, Any]] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        object_id = obj.get("id")
        if object_id:
            objects_by_id[object_id] = obj

    if not objects_by_id:
        return None

    root_id = next(
        (obj_id for obj_id, obj in objects_by_id.items() if obj.get("class") == "User"),
        None,
    )
    if not root_id:
        return None

    root_model = _build_user_model_node(root_id, objects_by_id, include_identity=True, path=set())
    if root_model is None:
        return None

    normalized_document = {key: value for key, value in document.items() if key != "objects"}
    normalized_document["model"] = root_model
    return normalized_document


def _build_user_model_node(
    object_id: str,
    objects_by_id: Dict[str, Dict[str, Any]],
    include_identity: bool,
    path: Set[str],
) -> Optional[Dict[str, Any]]:
    if object_id in path:
        return None

    obj = objects_by_id.get(object_id)
    if not obj:
        return None

    path.add(object_id)
    try:
        node: Dict[str, Any] = {}
        if include_identity:
            node["id"] = obj.get("id")
            node["class"] = obj.get("class")

        attributes = obj.get("attributes")
        if isinstance(attributes, dict):
            node.update(attributes)

        child_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        relationships = obj.get("relationships")
        if isinstance(relationships, dict):
            for target_ids in relationships.values():
                if not isinstance(target_ids, list):
                    continue
                for target_id in target_ids:
                    child_obj = objects_by_id.get(target_id)
                    if not child_obj:
                        continue
                    child_node = _build_user_model_node(
                        target_id,
                        objects_by_id,
                        include_identity=False,
                        path=path,
                    )
                    if child_node is None:
                        continue
                    key = child_obj.get("class") or child_obj.get("id")
                    if not key:
                        continue
                    child_groups[key].append(child_node)

        for child_key, children in child_groups.items():
            if not children:
                continue
            if len(children) == 1:
                node[child_key] = children[0]
            else:
                node[child_key] = children

        return node
    finally:
        path.remove(object_id)


def _sanitize_object_model_filename(name: Optional[str]) -> str:
    cleaned = (name or "object_model").strip().replace(" ", "_")
    return cleaned or "object_model"


def _extract_reference_class_diagram(json_data: dict):
    reference_data = json_data.get("referenceDiagramData")
    if not reference_data:
        reference_data = json_data.get("model", {}).get("referenceDiagramData")

    if not reference_data:
        return None

    if isinstance(reference_data, dict) and isinstance(reference_data.get("model"), dict):
        model_payload = reference_data["model"]
        title = reference_data.get("title") or reference_data["model"].get("title")
    else:
        model_payload = reference_data
        title = reference_data.get("title")

    if not isinstance(model_payload, dict) or "elements" not in model_payload:
        return None

    return {
        "title": title or "ReferenceClassDiagram",
        "model": model_payload,
    }


def _get_diagram_type(json_data: dict) -> Optional[str]:
    """Safely extract the diagram type label from the payload."""
    explicit_type = json_data.get("diagramType")
    if isinstance(explicit_type, str):
        return explicit_type

    model_section = json_data.get("model")
    if isinstance(model_section, dict):
        model_type = model_section.get("type")
        if isinstance(model_type, str):
            return model_type

    return None


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

async def _generate_web_app(buml_model, gui_model, generator_class, config: dict, temp_dir: str):
    """Generate web application files."""
    generator_instance = generator_class(buml_model, gui_model, output_dir=temp_dir)
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
                
            elif diagram_type == "UserDiagram":
                buml_model = user_reference_domain_model
                domain_validation = buml_model.validate(raise_exception=False)
                validation_errors.extend(domain_validation.get("errors", []))
                validation_warnings.extend(domain_validation.get("warnings", []))

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


# Main application entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=9000,
        log_level="info"
    )
