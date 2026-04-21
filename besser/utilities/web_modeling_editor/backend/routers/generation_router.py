"""
Generation Router

Handles all code generation endpoints for the BESSER web modeling editor backend.
"""

import ast
import logging
import os
import io
import uuid
import zipfile
import shutil
import tempfile
import importlib.util
import asyncio
import json
import re
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response

# BESSER utilities
from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code
from besser.generators.agents.baf_generator import GenerationMode

# Backend models
from besser.utilities.web_modeling_editor.backend.models import (
    DiagramInput,
    ProjectInput,
)

# Backend services - Converters
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_class_diagram,
    process_agent_diagram,
    process_object_diagram,
    process_gui_diagram,
    process_quantum_diagram,
)
from besser.utilities.web_modeling_editor.backend.constants.user_buml_model import (
    domain_model as user_reference_domain_model,
)

# Backend services - Other services
from besser.utilities.web_modeling_editor.backend.services.utils.agent_generation_utils import (
    extract_openai_api_key,
    normalize_personalization_mapping,
    handle_multi_language_generation,
    handle_variation_generation,
    handle_configuration_variants,
    handle_personalized_agent,
)

# Backend configuration
from besser.utilities.web_modeling_editor.backend.config import (
    SUPPORTED_GENERATORS,
    get_generator_info,
    get_filename_for_generator,
    is_generator_supported,
)

# Backend constants
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    TEMP_DIR_PREFIX,
    AGENT_TEMP_DIR_PREFIX,
    OUTPUT_DIR_NAME,
    AGENT_MODEL_FILENAME,
    AGENT_OUTPUT_FILENAME,
    DEFAULT_SQL_DIALECT,
    DEFAULT_DBMS,
    DEFAULT_JSONSCHEMA_MODE,
    DEFAULT_QISKIT_BACKEND,
    DEFAULT_QISKIT_SHOTS,
    DEFAULT_DJANGO_PROJECT_NAME,
    DEFAULT_DJANGO_APP_NAME,
)

# Backend exceptions

# Centralized error handling
from besser.utilities.web_modeling_editor.backend.routers.error_handler import (
    handle_endpoint_errors,
)

logger = logging.getLogger(__name__)

# Key-name fragments that mark a value as sensitive. Substring match
# (case-insensitive) so this catches camelCase, kebab-case, snake_case,
# screaming-snake, and weird vendor names alike: ``openaiApiKey``,
# ``OPENAI_API_KEY``, ``api-key``, ``X-Auth-Token`` all hit one of these.
SENSITIVE_KEYS = frozenset({
    "api_key", "apikey", "api-key",
    "secret", "password", "passwd",
    "token",                              # access_token, id_token, refresh_token
    "credential",
    "private",                            # private_key, private-key
    "auth",                               # bearer_auth, basic_auth, x-auth-*
    "cert",                               # cert, certificate
    "session",                            # session_id, session_token
    "key",                                # catch-all (last so longer matches dominate)
})


def _safe_path(base_dir: str, user_filename: str) -> str:
    """Resolve a user-provided filename safely within base_dir."""
    safe_name = os.path.basename(user_filename)
    full_path = os.path.realpath(os.path.join(base_dir, safe_name))
    if not full_path.startswith(os.path.realpath(base_dir)):
        raise ValueError("Invalid path")
    return full_path


def _key_is_sensitive(key: str) -> bool:
    """True if a config key name looks like it holds a secret."""
    if not isinstance(key, str):
        return False
    lowered = key.lower()
    return any(token in lowered for token in SENSITIVE_KEYS)


def sanitize_config(config):
    """Return a deep-copied structure with sensitive values masked.

    Walks nested dicts, lists, and tuples so a credential nested under
    ``config["llm"]["openai"]["api_key"]`` is still masked. Non-dict
    leaves are returned untouched (we only redact when the *key name*
    looks sensitive — value-level heuristics are too prone to false
    positives for now).
    """
    if isinstance(config, dict):
        return {
            k: ("***" if _key_is_sensitive(k) else sanitize_config(v))
            for k, v in config.items()
        }
    if isinstance(config, list):
        return [sanitize_config(item) for item in config]
    if isinstance(config, tuple):
        return tuple(sanitize_config(item) for item in config)
    return config


router = APIRouter(prefix="/besser_api", tags=["generation"])


def generate_agent_files(
    agent_model,
    config,
    generation_mode: GenerationMode = GenerationMode.FULL,
):
    """
    Generate agent files from an agent model.

    Args:
        agent_model: The agent model to generate files from
        config: Configuration dictionary for the agent generation
        generation_mode: The generation mode to use

    Returns:
        tuple: (zip_buffer, file_name) containing the zip file buffer and filename

    Raises:
        Exception: If agent file generation fails
    """
    try:
        with tempfile.TemporaryDirectory(prefix=f"{AGENT_TEMP_DIR_PREFIX}{uuid.uuid4().hex}_") as temp_dir:
            # Generate the agent model to a file
            agent_file = os.path.join(temp_dir, AGENT_MODEL_FILENAME)
            agent_model_to_code(agent_model, agent_file)

            # Validate generated code with ast.parse() before executing
            with open(agent_file, "r", encoding="utf-8") as f:
                agent_file_content = f.read()
            try:
                ast.parse(agent_file_content, filename=agent_file)
            except SyntaxError as syntax_err:
                logger.error("Generated agent code contains syntax errors: %s", syntax_err)
                raise HTTPException(
                    status_code=400,
                    detail="Generated code contains syntax errors"
                ) from syntax_err

            # Import the generated module (spec_from_file_location uses absolute path, no sys.path needed)
            spec = importlib.util.spec_from_file_location("agent_model", agent_file)
            agent_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(agent_module)

            # Get the BAFGenerator from the supported generators
            generator_info = get_generator_info("agent")
            generator_class = generator_info.generator_class
            openai_api_key = extract_openai_api_key(config)

            # Use the BAFGenerator with the agent model from the module
            if hasattr(agent_module, 'agent'):
                generator = generator_class(
                    agent_module.agent,
                    config=config,
                    openai_api_key=openai_api_key,
                    generation_mode=generation_mode,
                )
            else:
                # Fall back to the original agent model
                generator = generator_class(
                    agent_model,
                    config=config,
                    openai_api_key=openai_api_key,
                    generation_mode=generation_mode,
                )

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
                output_dir = os.path.join(temp_dir, OUTPUT_DIR_NAME)
                if os.path.exists(output_dir):
                    for root, _, files in os.walk(output_dir):
                        for file_name in files:
                            file_path = os.path.join(root, file_name)
                            arcname = os.path.relpath(file_path, output_dir)
                            zip_file.write(file_path, arcname)

            zip_buffer.seek(0)
            return zip_buffer, AGENT_OUTPUT_FILENAME

    except Exception:
        logger.exception("Error generating agent files")
        raise


@router.post("/generate-output-from-project")
@handle_endpoint_errors("generate_code_output_from_project")
async def generate_code_output_from_project(input_data: ProjectInput):
    """
    Generate code output from a complete project.
    This endpoint handles generators that require multiple diagrams (e.g., Web App).
    """
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
        quantum_diagram = input_data.get_active_diagram("QuantumCircuitDiagram")
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
    current_diagram = input_data.get_active_diagram(input_data.currentDiagramType)
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


@router.post("/generate-output")
@handle_endpoint_errors("generate_code_output")
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
    with tempfile.TemporaryDirectory(prefix=f"{TEMP_DIR_PREFIX}{uuid.uuid4().hex}_") as temp_dir:
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


@handle_endpoint_errors("_handle_web_app_project_generation")
async def _handle_web_app_project_generation(input_data: ProjectInput, generator_info, config: dict):
    """Handle Web App generation from a complete project with both ClassDiagram and GUINoCodeDiagram.
    Optionally includes AgentDiagram if agent components are present in the GUI."""
    # Extract active GUINoCodeDiagram first — it drives which other diagrams are used
    gui_diagram = input_data.get_active_diagram("GUINoCodeDiagram")
    if not gui_diagram:
        raise HTTPException(
            status_code=400,
            detail="GUINoCodeDiagram is required for Web App generator"
        )

    # Use the GUI diagram's per-diagram references to find the correct ClassDiagram
    class_diagram = input_data.get_referenced_diagram(gui_diagram, "ClassDiagram")
    if not class_diagram:
        raise HTTPException(
            status_code=400,
            detail="ClassDiagram is required for Web App generator"
        )

    with tempfile.TemporaryDirectory(prefix=TEMP_DIR_PREFIX) as temp_dir:
        # Process class diagram to BUML
        buml_model = process_class_diagram(class_diagram.model_dump())

        gui_model = process_gui_diagram(gui_diagram.model, class_diagram.model, buml_model)

        # Check if GUI model contains agent components
        agent_diagram = None
        agent_model = None
        has_agent_components = _check_for_agent_components(gui_model)

        agent_config = None
        if has_agent_components:
            # Use the GUI diagram's per-diagram reference for AgentDiagram
            agent_diagram = input_data.get_referenced_diagram(gui_diagram, "AgentDiagram")
            if agent_diagram and agent_diagram.model:
                # Process agent diagram to BUML
                # process_agent_diagram expects the full diagram structure with title, config, and model
                agent_diagram_dict = agent_diagram.model_dump()
                if agent_diagram_dict and isinstance(agent_diagram_dict, dict):
                    agent_model = process_agent_diagram(agent_diagram_dict)
                    # Try diagram-level config first, then project-level agentConfig, then project config
                    project_agent_config = config.get('agentConfig') if isinstance(config, dict) else None
                    agent_config = agent_diagram_dict.get('config') or agent_diagram.config or project_agent_config or config
                    logger.debug("[WebApp agent] resolved agent_config: %s", json.dumps(sanitize_config(agent_config), indent=2, default=str) if agent_config else 'None')
                else:
                    logger.warning("AgentDiagram data is invalid. Agent components will not be functional.")
            else:
                logger.warning("GUI contains agent components but no AgentDiagram found. Agent components will not be functional.")

        # Generate Web App TypeScript project
        generator_class = generator_info.generator_class

        return await _generate_web_app(buml_model, gui_model, generator_class, config, temp_dir, agent_model, agent_config)


def _streaming_zip(zip_buffer: io.BytesIO, file_name: str) -> StreamingResponse:
    """Return a StreamingResponse wrapping a ZIP buffer."""
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{file_name}"'},
    )


@handle_endpoint_errors("_handle_agent_generation")
async def _handle_agent_generation(json_data: dict):
    """Handle agent diagram generation by dispatching to specialized helpers."""
    config = json_data.get('config', {})
    logger.debug("[Agent generation] config: %s", json.dumps(sanitize_config(config), indent=2, default=str) if config else 'None')

    if config is None:
        agent_model = process_agent_diagram(json_data)
        zip_buffer, file_name = await asyncio.to_thread(generate_agent_files, agent_model, config)
        return _streaming_zip(zip_buffer, file_name)

    is_config_dict = isinstance(config, dict)

    # Normalize personalization mappings in-place
    if is_config_dict and isinstance(config.get('personalizationMapping'), list):
        normalize_personalization_mapping(config, json_data, _generate_user_profile_document)

    languages = config.get('languages') if is_config_dict else None
    configuration_variants = config.get('configurations') if is_config_dict else None
    base_model_snapshot = config.get('baseModel') if is_config_dict else None
    variation_entries = config.get('variations') if is_config_dict else None

    if languages and isinstance(languages, dict):
        buf, name = await handle_multi_language_generation(
            json_data, config, languages, generate_agent_files, output_dir=OUTPUT_DIR_NAME,
        )
        return _streaming_zip(buf, name)

    if base_model_snapshot is not None and isinstance(variation_entries, list):
        buf, name = handle_variation_generation(
            json_data, config, base_model_snapshot, variation_entries, generate_agent_files,
        )
        return _streaming_zip(buf, name)

    if configuration_variants and isinstance(configuration_variants, list):
        buf, name = handle_configuration_variants(
            json_data, configuration_variants, generate_agent_files,
        )
        return _streaming_zip(buf, name)

    if is_config_dict and 'personalizationMapping' in config:
        buf, name = handle_personalized_agent(json_data, config, generate_agent_files)
        return _streaming_zip(buf, name)

    # Single agent fallback
    agent_model = process_agent_diagram(json_data)
    zip_buffer, file_name = await asyncio.to_thread(generate_agent_files, agent_model, config)
    return _streaming_zip(zip_buffer, file_name)


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
    logger.info("Generator type %s", generator_type)

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
            detail="Object diagram generation requires reference class diagram data.",
        )

    domain_model = process_class_diagram(reference_payload)
    object_model = process_object_diagram(json_data, domain_model)

    generator_class = generator_info.generator_class
    generator_instance = generator_class(object_model, output_dir=temp_dir)
    await asyncio.to_thread(generator_instance.generate)

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
        logger.exception("Failed to convert user diagram to BUML")
        raise HTTPException(
            status_code=400,
            detail="Failed to convert user diagram to BUML. Please check the diagram data.",
        ) from conversion_error

    generator_class = generator_info.generator_class
    generator_instance = generator_class(object_model, output_dir=temp_dir)
    await asyncio.to_thread(generator_instance.generate)

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
        "generator": "jsonobject",
    }

    model_section = prepared_payload["model"]
    if isinstance(model_section, dict):
        model_section.setdefault("type", "UserDiagram")

    try:
        with tempfile.TemporaryDirectory(prefix=f"user_profile_{uuid.uuid4().hex}_") as temp_dir:
            object_model = process_object_diagram(prepared_payload, user_reference_domain_model)
            generator_info = get_generator_info("jsonobject")
            if not generator_info:
                raise HTTPException(status_code=500, detail="JSONObject generator is not configured")
            generator_class = generator_info.generator_class
            generator_instance = generator_class(object_model, output_dir=temp_dir)
            generator_instance.generate()

            _normalize_user_model_output(object_model, temp_dir)

            file_name = _sanitize_object_model_filename(getattr(object_model, "name", None))
            json_path = _safe_path(temp_dir, f"{file_name}.json")
            if not os.path.isfile(json_path):
                raise HTTPException(status_code=500, detail="Failed to render user profile JSON document")

            with open(json_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to convert user profile model")
        raise HTTPException(status_code=400, detail="Failed to convert user profile model. Please check the input data.") from exc


def _normalize_user_model_output(object_model, temp_dir: str) -> None:
    file_name = _sanitize_object_model_filename(getattr(object_model, "name", None))
    json_path = _safe_path(temp_dir, f"{file_name}.json")
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
    cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', (name or "object_model").strip())
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
    project_name = config.get("project_name", DEFAULT_DJANGO_PROJECT_NAME) if config else DEFAULT_DJANGO_PROJECT_NAME
    app_name = config.get("app_name", DEFAULT_DJANGO_APP_NAME) if config else DEFAULT_DJANGO_APP_NAME
    containerization = config.get("containerization", False) if config else False

    # Sanitize project_name to prevent path traversal
    project_name = os.path.basename(project_name)
    if not project_name:
        project_name = DEFAULT_DJANGO_PROJECT_NAME

    project_dir = _safe_path(temp_dir, project_name)

    # Clean up any existing project directory
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)

    os.makedirs(temp_dir, exist_ok=True)

    generator_instance = generator_class(
        model=buml_model,
        project_name=project_name,
        app_name=app_name,
        containerization=containerization,
        output_dir=temp_dir,
    )
    await asyncio.to_thread(generator_instance.generate)

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
        headers={"Content-Disposition": f'attachment; filename="{file_name}"'},
    )


async def _generate_sql(buml_model, generator_class, config: dict, temp_dir: str):
    """Generate SQL files."""
    dialect = DEFAULT_SQL_DIALECT
    if config and "dialect" in config:
        dialect = config["dialect"]

    generator_instance = generator_class(
        buml_model, output_dir=temp_dir, sql_dialect=dialect
    )
    await asyncio.to_thread(generator_instance.generate)

    return _create_file_response(temp_dir, "sql")


async def _generate_sqlalchemy(buml_model, generator_class, config: dict, temp_dir: str):
    """Generate SQLAlchemy files."""
    dbms = DEFAULT_DBMS
    if config and "dbms" in config:
        dbms = config["dbms"]

    generator_instance = generator_class(buml_model, output_dir=temp_dir)
    await asyncio.to_thread(generator_instance.generate, dbms=dbms)

    return _create_file_response(temp_dir, "sqlalchemy")


async def _generate_jsonschema(buml_model, generator_class, config: dict, temp_dir: str):
    """Generate JSON Schema files."""
    mode = DEFAULT_JSONSCHEMA_MODE
    if config and "mode" in config:
        mode = config["mode"]

    generator_instance = generator_class(buml_model, output_dir=temp_dir, mode=mode)
    await asyncio.to_thread(generator_instance.generate)

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

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="smart_data_schemas.zip"'},
        )
    else:
        return _create_file_response(temp_dir, "jsonschema")

async def _generate_qiskit(json_data: dict, generator_class, config: dict, temp_dir: str):
    """Generate Qiskit code."""
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

    quantum_model = process_quantum_diagram(json_data)

    # Extract Qiskit config
    backend_type = config.get('backend', DEFAULT_QISKIT_BACKEND) if config else DEFAULT_QISKIT_BACKEND
    shots = config.get('shots', DEFAULT_QISKIT_SHOTS) if config else DEFAULT_QISKIT_SHOTS

    generator_instance = generator_class(
        quantum_model,
        output_dir=temp_dir,
        backend_type=backend_type,
        shots=shots
    )
    await asyncio.to_thread(generator_instance.generate)

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

async def _generate_web_app(buml_model, gui_model, generator_class, config: dict, temp_dir: str, agent_model=None, agent_config=None):
    """Generate web application files. Optionally includes agent model if agent components are present."""
    generator_instance = generator_class(buml_model, gui_model, output_dir=temp_dir, agent_model=agent_model, agent_config=agent_config)
    await asyncio.to_thread(generator_instance.generate)
    return _create_zip_response(temp_dir, "web_app")

@handle_endpoint_errors("_generate_standard")
async def _generate_standard(buml_model, generator_class, generator_type: str, generator_info, temp_dir: str):
    """Generate standard files (non-Django, non-SQL)."""
    generator_instance = generator_class(buml_model, output_dir=temp_dir)
    await asyncio.to_thread(generator_instance.generate)

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

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{file_name}"'},
    )


def _create_file_response(temp_dir: str, generator_type: str):
    """Create a single file response."""
    files = os.listdir(temp_dir)
    if not files:
        raise ValueError(f"{generator_type} generation failed: No output files were created.")

    # Match against the expected filename from generator config
    expected_name = get_filename_for_generator(generator_type)
    if expected_name in files:
        file_name = expected_name
    else:
        # Fall back to matching by extension, then first file
        generator_info = get_generator_info(generator_type)
        ext = generator_info.file_extension if generator_info else ""
        matching = [f for f in sorted(files) if f.endswith(ext)] if ext else []
        file_name = matching[0] if matching else sorted(files)[0]
    output_file_path = _safe_path(temp_dir, file_name)

    with open(output_file_path, "rb") as f:
        file_content = f.read()

    # Use the proper filename from config
    proper_filename = get_filename_for_generator(generator_type)

    return Response(
        content=file_content,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{proper_filename}"'},
    )
