"""Utility helpers for building agent configuration bundles."""

from __future__ import annotations

import importlib.util
import io
import logging
import os
import re
import sys
import tempfile
import zipfile

logger = logging.getLogger(__name__)
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from fastapi import HTTPException

from besser.generators.agents.baf_generator import GenerationMode
from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code
from besser.utilities.web_modeling_editor.backend.config import get_generator_info
from besser.utilities.web_modeling_editor.backend.services.converters import process_agent_diagram
AgentZipGenerator = Callable[[Any, Optional[Dict[str, Any]]], Tuple[io.BytesIO, str]]


def extract_openai_api_key(config: Any) -> Optional[str]:
    if not isinstance(config, dict):
        return None

    for candidate_key in ("openai_api_key", "openaiApiKey", "OPENAI_API_KEY", "apiKey"):
        candidate_value = config.get(candidate_key)
        if isinstance(candidate_value, str) and candidate_value.strip():
            return candidate_value.strip()

    system_section = config.get("system")
    if isinstance(system_section, dict):
        for candidate_key in ("openai_api_key", "openaiApiKey", "OPENAI_API_KEY", "apiKey"):
            candidate_value = system_section.get(candidate_key)
            if isinstance(candidate_value, str) and candidate_value.strip():
                return candidate_value.strip()

    return None


def slugify_name(value: Optional[str], fallback: str) -> str:
    """Sanitize folder names while keeping them human readable."""
    name = (value or fallback or "").strip()
    safe_name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    safe_name = re.sub(r"_+", "_", safe_name).strip("_ ")
    return safe_name or fallback


def append_zip_contents(
    target_zip: zipfile.ZipFile,
    source_buffer: io.BytesIO,
    prefix: Optional[str] = None,
) -> None:
    """Append contents of an in-memory zip archive into another archive."""
    source_buffer.seek(0)
    with zipfile.ZipFile(source_buffer, "r") as source_zip:
        for info in source_zip.infolist():
            arcname = info.filename
            if prefix:
                arcname = f"{prefix}/{arcname}" if arcname else prefix
            if info.is_dir():
                if not arcname.endswith('/'):
                    arcname += '/'
                target_zip.writestr(arcname, b"")
                continue
            target_zip.writestr(arcname, source_zip.read(info.filename))


def build_configurations_package(
    agent_model: Any,
    configurations: List[Dict[str, Any]],
    generate_agent_zip_fn: AgentZipGenerator,
) -> io.BytesIO:
    """Create a zip containing the base agent plus variants for each configuration."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as combined_zip:
        base_buffer, _ = generate_agent_zip_fn(agent_model, None)
        append_zip_contents(combined_zip, base_buffer)

        used_prefixes: Set[str] = set()
        for index, entry in enumerate(configurations):
            configuration_payload = entry.get("configuration")
            if not isinstance(configuration_payload, dict):
                continue

            sanitized_payload = dict(configuration_payload)
            sanitized_payload.pop("configurations", None)

            fallback_name = f"configuration_{index + 1}"
            folder_name = slugify_name(entry.get("name"), fallback_name)
            unique_name = folder_name
            suffix = 1
            while unique_name in used_prefixes:
                unique_name = f"{folder_name}_{suffix}"
                suffix += 1
            used_prefixes.add(unique_name)

            config_buffer, _ = generate_agent_zip_fn(agent_model, sanitized_payload)
            append_zip_contents(combined_zip, config_buffer, prefix=unique_name)

    zip_buffer.seek(0)
    return zip_buffer


def normalize_personalization_mapping(
    config: dict,
    json_data: dict,
    profile_generator_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> None:
    """Validate and normalize personalizationMapping entries in-place.

    Args:
        config: The configuration dict containing ``personalizationMapping``.
        json_data: The full diagram payload (mutated with the updated config).
        profile_generator_fn: Callback that converts a raw user-profile payload
            into a simplified profile document.
    """
    normalized_mappings: List[Dict[str, Any]] = []
    for index, entry in enumerate(config.get('personalizationMapping') or []):
        if not isinstance(entry, dict):
            raise HTTPException(
                status_code=400,
                detail=f"personalizationMapping entry at index {index} must be an object",
            )

        user_profile_payload = entry.get('user_profile')
        if user_profile_payload is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "personalizationMapping entry "
                    f"at index {index} is missing 'user_profile'"
                ),
            )

        simplified_profile = profile_generator_fn(user_profile_payload)

        agent_model_json = entry.get('agent_model')
        agent_model_buml = None
        if isinstance(agent_model_json, dict):
            mapping_payload = dict(json_data)
            mapping_payload['model'] = deepcopy(agent_model_json)
            try:
                agent_model_buml = process_agent_diagram(mapping_payload)
            except Exception as conversion_error:
                logger.exception("Failed to convert personalizationMapping agent_model at index %d", index)
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Failed to convert personalizationMapping agent_model at index {index}. "
                        "Please check the agent model data."
                    ),
                ) from conversion_error

        normalized_entry = dict(entry)
        normalized_entry['user_profile'] = simplified_profile
        if agent_model_buml is not None:
            normalized_entry['agent_model'] = agent_model_buml
        normalized_mappings.append(normalized_entry)

    config['personalizationMapping'] = normalized_mappings
    json_data['config'] = config


async def handle_multi_language_generation(
    json_data: dict,
    config: dict,
    languages: dict,
    generate_agent_files_fn: Callable,
    output_dir: str = "output",
) -> Tuple[io.BytesIO, str]:
    """Generate a ZIP with one agent variant per target language.

    Returns:
        ``(zip_buffer, filename)`` tuple ready to be wrapped in a response.
    """
    source_lang = languages.get('source')
    target_langs = languages.get('target', None)
    agent_models: List[Tuple[Any, str]] = []

    # Default agent (no language)
    default_json_data = {**json_data}
    if 'config' in default_json_data and isinstance(default_json_data['config'], dict):
        default_json_data['config'] = {
            k: v for k, v in default_json_data['config'].items() if k != 'languages'
        }
    agent_models.append((process_agent_diagram(default_json_data), 'default'))

    # Agents for each target language
    for lang in target_langs:
        new_config = dict(config) if config else {}
        new_config['language'] = lang
        if source_lang:
            new_config['source_language'] = source_lang
        new_json_data = dict(json_data)
        new_json_data['config'] = new_config
        agent_models.append((process_agent_diagram(new_json_data), lang))

    # Generate files for each agent model
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for agent_model, lang in agent_models:
            with tempfile.TemporaryDirectory(prefix=f"besser_agent_{lang}_") as temp_dir:
                agent_file = os.path.join(temp_dir, f"agent_model_{lang}.py")
                agent_model_to_code(agent_model, agent_file)
                sys.path.insert(0, temp_dir)
                spec = importlib.util.spec_from_file_location(
                    f"agent_model_{lang}", agent_file
                )
                agent_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(agent_module)
                generator_info = get_generator_info("agent")
                generator_class = generator_info.generator_class
                variant_openai_api_key = extract_openai_api_key(new_config)
                if hasattr(agent_module, 'agent'):
                    generator = generator_class(
                        agent_module.agent,
                        config=new_config,
                        openai_api_key=variant_openai_api_key,
                    )
                else:
                    generator = generator_class(
                        agent_model,
                        config=new_config,
                        openai_api_key=variant_openai_api_key,
                    )
                generator.generate()
                zip_file.write(agent_file, f"{lang}/agent_model_{lang}.py")
                if os.path.exists(output_dir):
                    for file_name in os.listdir(output_dir):
                        file_path = os.path.join(output_dir, file_name)
                        if os.path.isfile(file_path):
                            zip_file.write(file_path, f"{lang}/{file_name}")
    zip_buffer.seek(0)
    return zip_buffer, "agents_multi_lang.zip"


def handle_variation_generation(
    json_data: dict,
    config: dict,
    base_model_snapshot: Dict[str, Any],
    variation_entries: list,
    generate_agent_files_fn: Callable,
) -> Tuple[io.BytesIO, str]:
    """Generate a ZIP containing the base model plus each variation.

    Returns:
        ``(zip_buffer, filename)`` tuple.
    """
    def to_agent_model(model_snapshot: Dict[str, Any]):
        variant_payload = dict(json_data)
        variant_payload['model'] = model_snapshot
        return process_agent_diagram(variant_payload)

    combined_zip = io.BytesIO()
    with zipfile.ZipFile(combined_zip, "w", zipfile.ZIP_DEFLATED) as zip_file:
        base_agent_model = to_agent_model(base_model_snapshot)
        base_buffer, _ = generate_agent_files_fn(
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
                logger.exception("Invalid agent model for variation '%s'", entry.get('name', index))
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid agent model provided for variation "
                        f"'{entry.get('name', index)}'. Please check the agent model data."
                    ),
                ) from conversion_error

            folder_name = slugify_name(entry.get('name'), f"variation_{index + 1}")
            variant_buffer, _ = generate_agent_files_fn(
                variant_agent_model,
                entry.get('config'),
                generation_mode=GenerationMode.CODE_ONLY,
            )
            append_zip_contents(
                zip_file,
                variant_buffer,
                prefix=folder_name or f"variation_{index + 1}",
            )

    combined_zip.seek(0)
    return combined_zip, "agent_output.zip"


def handle_configuration_variants(
    json_data: dict,
    configuration_variants: list,
    generate_agent_files_fn: Callable,
) -> Tuple[io.BytesIO, str]:
    """Generate a ZIP bundle for each configuration variant.

    Returns:
        ``(zip_buffer, filename)`` tuple.
    """
    agent_model = process_agent_diagram(json_data)
    bundle_buffer = build_configurations_package(
        agent_model,
        configuration_variants,
        generate_agent_files_fn,
    )
    return bundle_buffer, "agent_output.zip"


def handle_personalized_agent(
    json_data: dict,
    config: dict,
    generate_agent_files_fn: Callable,
) -> Tuple[io.BytesIO, str]:
    """Generate agent files with personalization mapping applied.

    Returns:
        ``(zip_buffer, filename)`` tuple.
    """
    agent_model = process_agent_diagram(json_data)
    zip_buffer, file_name = generate_agent_files_fn(
        agent_model,
        config,
        generation_mode=GenerationMode.CODE_ONLY,
    )
    return zip_buffer, file_name
