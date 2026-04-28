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
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Header, HTTPException
from fastapi.responses import StreamingResponse, Response

from besser.utilities.web_modeling_editor.backend.services.deployment.github_oauth import (
    get_user_token,
)

# BESSER utilities
from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code
from besser.generators.agents.baf_generator import GenerationMode
from besser.generators.agents.agent_personalization import call_openai_chat

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
    collect_agents_from_diagrams,
    extract_openai_api_key,
    normalize_personalization_mapping,
    handle_multi_language_generation,
    handle_variation_generation,
    handle_configuration_variants,
    handle_personalized_agent,
)
from besser.utilities.web_modeling_editor.backend.services.utils.agent_config_recommendation_utils import (
    RECOMMENDATION_ALLOWED_VALUES,
    load_default_agent_recommendation_config,
    extract_json_object,
    normalize_recommended_agent_config,
)
from besser.utilities.web_modeling_editor.backend.services.utils.agent_config_manual_mapping_utils import (
    get_manual_agent_config_mapping,
    build_manual_mapping_recommendation,
)
from besser.utilities.web_modeling_editor.backend.services.utils.user_profile_utils import (
    generate_user_profile_document as _generate_user_profile_document,
    normalize_user_model_output as _normalize_user_model_output,
    safe_path as _safe_path,
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

# Centralized error handling
from besser.utilities.web_modeling_editor.backend.routers.error_handler import (
    handle_endpoint_errors,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    GenerationError,
    ValidationError,
)

logger = logging.getLogger(__name__)

SENSITIVE_KEYS = {'api_key', 'openai_api_key', 'secret', 'password', 'token', 'apikey', 'api-key'}


def sanitize_config(config: dict) -> dict:
    """Return a shallow copy of config with sensitive values masked."""
    return {k: '***' if any(s in k.lower() for s in SENSITIVE_KEYS) else v for k, v in config.items()}


router = APIRouter(prefix="/besser_api", tags=["generation"])


def _require_github_session(github_session: Optional[str]) -> None:
    """Verify a GitHub OAuth session is present and active.

    Raises HTTPException(401) when the session header is missing or expired.
    Mirrors the auth gate used by the deploy endpoints in github_deploy_api.py.
    """
    if not github_session:
        raise HTTPException(
            status_code=401,
            detail="GitHub authentication required. Please sign in with GitHub first.",
        )
    if not get_user_token(github_session):
        raise HTTPException(
            status_code=401,
            detail="GitHub session expired. Please sign in again.",
        )


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@router.post("/recommend-agent-config-llm")
@handle_endpoint_errors("recommend_agent_config_llm")
async def recommend_agent_config_llm(
    payload: Dict[str, Any] = Body(...),
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
):
    """Recommend a structured agent configuration from a user profile using an LLM.

    Requires authenticated GitHub session.
    """
    _require_github_session(github_session)
    user_profile_model = payload.get("userProfileModel")
    if not isinstance(user_profile_model, dict):
        raise ValidationError("userProfileModel is required and must be a JSON object")

    user_profile_name = payload.get("userProfileName") if isinstance(payload.get("userProfileName"), str) else None
    current_config = payload.get("currentConfig") if isinstance(payload.get("currentConfig"), dict) else {}
    requested_model = payload.get("model")
    llm_model = (
        requested_model
        if isinstance(requested_model, str) and requested_model.strip()
        else "gpt-5.5"
    )

    profile_document = _generate_user_profile_document(user_profile_model)
    default_config = load_default_agent_recommendation_config()

    allowed_values_payload = {
        key: value
        for key, value in RECOMMENDATION_ALLOWED_VALUES.items()
        if key not in {"llmProvider", "openaiModels"}
    }

    system_prompt = (
        "You are an assistant that recommends a valid agent configuration JSON for a user profile. "
        "Return ONLY a JSON object with this exact shape: "
        "{presentation:{...}, modality:{...}, behavior:{...}, content:{...}, system:{...}}. "
        "Use only allowed values and keep output concise and deterministic. "
        "Do not include markdown, comments, or explanatory text."
    )
    user_prompt = (
        "Create a recommended configuration adapted to this user profile.\n\n"
        f"Selected profile name: {user_profile_name or 'N/A'}\n\n"
        f"Default config baseline:\n{json.dumps(default_config, ensure_ascii=False, indent=2)}\n\n"
        f"Allowed values:\n{json.dumps(allowed_values_payload, ensure_ascii=False, indent=2)}\n\n"
        f"Current config context (optional):\n{json.dumps(current_config, ensure_ascii=False, indent=2)}\n\n"
        f"User profile document:\n{json.dumps(profile_document, ensure_ascii=False, indent=2)}\n\n"
        "Return only the JSON object."
    )

    try:
        recommendation_text = await asyncio.to_thread(
            call_openai_chat,
            system_prompt,
            user_prompt,
            model=llm_model,
            openai_api_key=extract_openai_api_key(payload if isinstance(payload, dict) else {}),
            config=payload,
        )
        parsed_recommendation = extract_json_object(recommendation_text)
        normalized_config = normalize_recommended_agent_config(parsed_recommendation, user_profile_name)

        return {
            "config": normalized_config,
            "source": "openai",
            "model": llm_model,
            "generatedAt": _utc_now_iso(),
        }
    except RuntimeError as runtime_error:
        raise ValidationError(str(runtime_error)) from runtime_error
    except ValueError as parse_error:
        logger.exception("Failed to parse LLM recommendation response")
        raise GenerationError("Failed to parse LLM recommendation response") from parse_error
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to generate LLM recommendation")
        raise GenerationError("Failed to generate LLM recommendation") from exc


@router.get("/agent-config-manual-mapping")
@handle_endpoint_errors("get_agent_config_manual_mapping")
async def get_agent_config_manual_mapping(
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
):
    """Return the complete rule mapping used for deterministic recommendations.

    Requires authenticated GitHub session.
    """
    _require_github_session(github_session)
    return {
        "mapping": get_manual_agent_config_mapping(),
        "source": "manual_mapping",
        "generatedAt": _utc_now_iso(),
    }


@router.post("/recommend-agent-config-mapping")
@handle_endpoint_errors("recommend_agent_config_mapping")
async def recommend_agent_config_mapping(
    payload: Dict[str, Any] = Body(...),
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
):
    """Recommend a structured agent configuration using deterministic mapping rules.

    Requires authenticated GitHub session.
    """
    _require_github_session(github_session)
    user_profile_model = payload.get("userProfileModel")
    if not isinstance(user_profile_model, dict):
        raise ValidationError("userProfileModel is required and must be a JSON object")

    user_profile_name = payload.get("userProfileName") if isinstance(payload.get("userProfileName"), str) else None
    current_config = payload.get("currentConfig") if isinstance(payload.get("currentConfig"), dict) else {}

    profile_document = _generate_user_profile_document(user_profile_model)
    recommendation = build_manual_mapping_recommendation(
        user_profile_document=profile_document,
        user_profile_name=user_profile_name,
        current_config=current_config,
    )

    return {
        "config": recommendation["config"],
        "matchedRules": recommendation.get("matchedRules", []),
        "signals": recommendation.get("signals", {}),
        "source": "manual_mapping",
        "generatedAt": _utc_now_iso(),
    }


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

            # Pin the generator to this request's temp dir. Without this, the
            # base GeneratorInterface falls back to "<CWD>/output", which (a)
            # leaks artifacts across concurrent requests and (b) means the
            # harvester below walks an empty temp_dir/output and ships a zip
            # containing only the raw agent_model.py.
            generator_output_dir = os.path.join(temp_dir, OUTPUT_DIR_NAME)

            # Use the BAFGenerator with the agent model from the module
            if hasattr(agent_module, 'agent'):
                generator = generator_class(
                    agent_module.agent,
                    output_dir=generator_output_dir,
                    config=config,
                    openai_api_key=openai_api_key,
                    generation_mode=generation_mode,
                )
            else:
                # Fall back to the original agent model
                generator = generator_class(
                    agent_model,
                    output_dir=generator_output_dir,
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
            referenceDiagramData=(
                quantum_diagram.referenceDiagramData
                if hasattr(quantum_diagram, 'referenceDiagramData')
                else None
            ),
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

        # Collect every AgentDiagram in the project if the GUI uses agent components.
        # The frontend dropdown enumerates all agents, so the generator must
        # satisfy any binding — we don't filter to the active reference here.
        agent_models = []
        agent_configs = {}
        has_agent_components = _check_for_agent_components(gui_model)

        if has_agent_components:
            project_agent_config = config.get('agentConfig') if isinstance(config, dict) else None
            default_cfg = project_agent_config or config
            agent_models, agent_configs = collect_agents_from_diagrams(
                input_data.diagrams.get("AgentDiagram", []),
                default_config=default_cfg,
            )
            for name, cfg in agent_configs.items():
                logger.debug("[WebApp agent] resolved config for %s: %s",
                             name,
                             json.dumps(sanitize_config(cfg), indent=2, default=str) if cfg else 'None')
            if not agent_models:
                logger.warning(
                    "GUI contains agent components but no AgentDiagram was found. "
                    "Agent components will not be functional."
                )

        # Generate Web App TypeScript project
        generator_class = generator_info.generator_class

        return await _generate_web_app(
            buml_model, gui_model, generator_class, config, temp_dir,
            agent_models=agent_models, agent_configs=agent_configs,
        )


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
    sanitized_config_log = (
        json.dumps(sanitize_config(config), indent=2, default=str) if config else 'None'
    )
    logger.debug("[Agent generation] config: %s", sanitized_config_log)

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
        detail_message = (
            f"Invalid diagram type for Qiskit generator. Expected QuantumCircuitDiagram "
            f"but received '{diagram_type}'. Please use the 'generate-output-from-project' "
            f"endpoint or select the Quantum Circuit diagram."
        )
        raise HTTPException(
            status_code=400,
            detail=detail_message,
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

async def _generate_web_app(buml_model, gui_model, generator_class, config: dict, temp_dir: str,
                            agent_models=None, agent_configs=None):
    """Generate web application files.

    Supports multi-agent projects: ``agent_models`` is a list and each is emitted
    under ``agents/<slug>/`` in the generated output.
    """
    generator_instance = generator_class(
        buml_model, gui_model, output_dir=temp_dir,
        agent_models=agent_models, agent_configs=agent_configs,
    )
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
