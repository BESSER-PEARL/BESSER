"""Agent (BAF) generation behind ``/generate-output``."""

import ast
import asyncio
import io
import json
import logging
import os
import tempfile
import uuid
import zipfile
from copy import deepcopy
from typing import Optional

from besser.generators.agents.baf_generator import GenerationMode
from besser.utilities.secret_redaction import sanitize_config
from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code
from besser.utilities.web_modeling_editor.backend.config import get_generator_info
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    AGENT_MODEL_FILENAME,
    AGENT_OUTPUT_FILENAME,
    AGENT_TEMP_DIR_PREFIX,
    OUTPUT_DIR_NAME,
)
from besser.utilities.web_modeling_editor.backend.services.converters import process_agent_diagram
from besser.utilities.web_modeling_editor.backend.services.exceptions import ValidationError
from besser.utilities.web_modeling_editor.backend.services.generation.downloads import (
    GeneratedOutput,
    agent_zip_output,
)
from besser.utilities.web_modeling_editor.backend.services.utils.agent_generation_utils import (
    extract_openai_api_key,
    handle_configuration_variants,
    handle_multi_language_generation,
    handle_personalized_agent,
    handle_variation_generation,
    normalize_personalization_mapping,
)
from besser.utilities.web_modeling_editor.backend.services.utils.user_profile_utils import (
    generate_user_profile_document as _generate_user_profile_document,
)

logger = logging.getLogger(__name__)


def generate_agent_files(
    agent_model,
    config,
    generation_mode: GenerationMode = GenerationMode.FULL,
    config_yaml: Optional[str] = None,
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
        ValidationError: If the generated agent code does not parse
        Exception: If agent file generation fails
    """
    try:
        with tempfile.TemporaryDirectory(prefix=f"{AGENT_TEMP_DIR_PREFIX}{uuid.uuid4().hex}_") as temp_dir:
            # Generate the agent model to a file
            agent_file = os.path.join(temp_dir, AGENT_MODEL_FILENAME)
            agent_model_to_code(agent_model, agent_file)

            # The file is shipped in the ZIP, never imported: it carries the user's
            # code actions and tool code verbatim, and importing it would run them
            # in this process. A syntax error still means the shipped model is broken.
            with open(agent_file, "r", encoding="utf-8") as f:
                agent_file_content = f.read()
            try:
                ast.parse(agent_file_content, filename=agent_file)
            except SyntaxError as syntax_err:
                logger.error("Generated agent code contains syntax errors: %s", syntax_err)
                raise ValidationError("Generated code contains syntax errors") from syntax_err

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

            # A copy, because personalization (configure_agent) mutates the model
            # and callers reuse one model across configuration variants.
            generator = generator_class(
                deepcopy(agent_model),
                output_dir=generator_output_dir,
                config=config,
                openai_api_key=openai_api_key,
                generation_mode=generation_mode,
                config_yaml=config_yaml,
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


async def generate_agent_output(json_data: dict) -> GeneratedOutput:
    """Generate an agent diagram's ZIP, dispatching on its config (languages, variations, ...)."""
    config = json_data.get('config', {})
    config_yaml: Optional[str] = json_data.get('configYaml') if isinstance(json_data.get('configYaml'), str) else None
    sanitized_config_log = (
        json.dumps(sanitize_config(config), indent=2, default=str) if config else 'None'
    )
    logger.debug("[Agent generation] config: %s", sanitized_config_log)

    if config is None:
        agent_model = process_agent_diagram(json_data)
        zip_buffer, file_name = await asyncio.to_thread(
            generate_agent_files, agent_model, config, config_yaml=config_yaml
        )
        return agent_zip_output(zip_buffer, file_name)

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
        return agent_zip_output(buf, name)

    if base_model_snapshot is not None and isinstance(variation_entries, list):
        buf, name = handle_variation_generation(
            json_data, config, base_model_snapshot, variation_entries, generate_agent_files,
            config_yaml=config_yaml,
        )
        return agent_zip_output(buf, name)

    if configuration_variants and isinstance(configuration_variants, list):
        buf, name = handle_configuration_variants(
            json_data, configuration_variants, generate_agent_files,
            config_yaml=config_yaml,
        )
        return agent_zip_output(buf, name)

    if is_config_dict and 'personalizationMapping' in config:
        buf, name = handle_personalized_agent(json_data, config, generate_agent_files, config_yaml=config_yaml)
        return agent_zip_output(buf, name)

    # Single agent fallback
    agent_model = process_agent_diagram(json_data)
    zip_buffer, file_name = await asyncio.to_thread(
        generate_agent_files, agent_model, config, config_yaml=config_yaml
    )
    return agent_zip_output(zip_buffer, file_name)


def has_agent_components(gui_model) -> bool:
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
                    if _container_has_agent_components(element):
                        return True
    return False


def _container_has_agent_components(container) -> bool:
    """Recursively check a container for agent components."""
    from besser.BUML.metamodel.gui.dashboard import AgentComponent

    if not container.view_elements:
        return False

    for element in container.view_elements:
        if isinstance(element, AgentComponent):
            return True
        from besser.BUML.metamodel.gui import ViewContainer
        if isinstance(element, ViewContainer):
            if _container_has_agent_components(element):
                return True
    return False
