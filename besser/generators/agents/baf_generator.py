import logging
import os
import textwrap
from enum import Enum

from jinja2 import Environment, FileSystemLoader
import json
import re

from besser.BUML.metamodel.state_machine.agent import Agent
from besser.BUML.metamodel.structural import Method
from besser.generators import GeneratorInterface

from besser.generators.agents.agent_personalization import configure_agent, flatten_agent_config_structure

# BESSER utilities
from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters import agent_buml_to_json

logger = logging.getLogger(__name__)


# Keys that represent personalization content. These mirror the structured
# sections in ``default_config.json`` (``presentation`` + ``modality``) plus
# their flattened equivalents produced by ``flatten_agent_config_structure``.
# Presence of any of these in the config warrants running the personalization
# pipeline (``configure_agent`` + ``personalized_agent_model.py`` emission).
# System-level runtime settings (platform, LLM, intent recognition, languages,
# API key) do NOT appear here — they are applied via normal template rendering.
# Inverting the check (allowlist personalization keys instead of denying
# system keys) prevents typo'd system keys from silently triggering the
# personalization pass.
_PERSONALIZATION_CONFIG_KEYS = frozenset({
    # Structured top-level sections (post-flatten still kept as keys on the
    # original dict by ``flatten_agent_config_structure``).
    "presentation",
    "modality",
    "content",
    # Flattened presentation fields.
    "agentLanguage",
    "agentStyle",
    "languageComplexity",
    "sentenceLength",
    "interfaceStyle",
    "voiceStyle",
    "avatar",
    "useAbbreviations",
    # Flattened modality fields.
    "inputModalities",
    "outputModalities",
    # Flattened content fields (user-profile-driven reply rewriting).
    "adaptContentToUserProfile",
    "userProfileName",
    "userProfileModel",
})


def _config_has_personalization_content(config) -> bool:
    """Return True when ``config`` contains fields that warrant running the
    personalization pass (``configure_agent`` + ``personalized_agent_model.py``
    emission). A config with only system-level runtime settings does not.
    """
    if not config or not isinstance(config, dict):
        return False
    if config.get("personalizationMapping"):
        return True
    if config.get("personalizationrules"):
        return True
    for key in config:
        if key in _PERSONALIZATION_CONFIG_KEYS:
            return True
    return False


class GenerationMode(Enum):
    FULL = "full"
    PERSONALIZED_ONLY = "personalized_only"
    CODE_ONLY = "code_only"


class BAFGenerator(GeneratorInterface):
    """
    BAFGenerator is a class that implements the GeneratorInterface and is responsible for generating
    the agent code, using the BESSER Agent Framework (BAF), based on an input agent model.

    Args:
        model (Agent): A agent model.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
        generation_mode (GenerationMode | str, optional): Controls which pipeline stages run.
            - GenerationMode.FULL (default): personalization (if config) + templated code.
            - GenerationMode.PERSONALIZED_ONLY: run personalization JSON/model export only.
            - GenerationMode.CODE_ONLY: skip personalization helpers, render templates immediately.
    """
    def __init__(
        self,
        model: Agent,
        output_dir: str = None,
        config_path: str = None,
        config: dict = None,
        openai_api_key: str = None,
        generation_mode: GenerationMode | str = GenerationMode.FULL,
    ):
        super().__init__(model, output_dir)
        self.config = flatten_agent_config_structure(config) if isinstance(config, dict) else config
        self.openai_api_key = openai_api_key
        if isinstance(generation_mode, GenerationMode):
            self.generation_mode = generation_mode
        elif isinstance(generation_mode, str):
            normalized_mode = generation_mode.strip().lower()
            self.generation_mode = next(
                (mode for mode in GenerationMode if mode.value == normalized_mode),
                GenerationMode.FULL,
            )
        else:
            self.generation_mode = GenerationMode.FULL
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                self.config = (
                    flatten_agent_config_structure(loaded_config)
                    if isinstance(loaded_config, dict)
                    else loaded_config
                )

    def generate(self):
        """
        Generates the BAF agent code and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output
        folder.
        """

        # TODO: TelegramPlatform.add_handler() not implemented in generator
        # TODO: Verify imports are added when necessary
        # TODO: Platform name not safe (hardcoded 'websocket_platform' and 'telegram_platform' in
        #       jinja template), when accessed from body can be different name
        # TODO: Global variables?
        # -->   (OPTION 1) agent.create_global_var(name: str, type: type, value: Any) --> not
        #       supports "custom" values (e.g. x = agent.get_name() )
        # -->   (OPTION 2) agent.add_code_line('x = agent.get_name()')

        def is_class(obj, name):
            return obj.__class__.__name__ == name

        def is_type(obj, type_name: str):
            return type(obj).__name__ == type_name

        def replace_agent_session_with_session_in_signature(func: Method) -> str:
            if func:
                # Replace 'AgentSession' with 'Session' in the code
                code = func.code.replace('AgentSession', 'Session')
                # Extract function name using regex
                match = re.search(r'def\s+(\w+)\s*\(.*session\s*:\s*Session.*\)', code)
                if match:
                    code = code + "\n" + f"{func.name} = {match.group(1)}\n"
                return textwrap.dedent(code)
            else:
                return None

        def rag_slug(name: str, index: int) -> str:
            slug = (name or '').strip().lower().replace(' ', '_').replace('-', '_')
            if not slug:
                slug = f"rag_{index}"
            return slug
        generate_personalized_assets = self.generation_mode in (
            GenerationMode.FULL,
            GenerationMode.PERSONALIZED_ONLY,
        )
        generate_code_assets = self.generation_mode in (
            GenerationMode.FULL,
            GenerationMode.CODE_ONLY,
        )

        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(
            loader=FileSystemLoader(templates_path),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        env.globals['is_class'] = is_class
        env.globals['is_type'] = is_type
        env.globals['replace_bot_session_with_session_in_signature'] = replace_agent_session_with_session_in_signature
        agent_template = env.get_template('baf_agent_template.py.j2')
        agent_path = self.build_generation_path(file_name=f"{self.model.name}.py")
        personalized_agent_path = self.build_generation_path(file_name="personalized_agent_model.py")
        personalized_json_path = self.build_generation_path(file_name="personalized_agent_model.json")
        config_for_personalization = dict(self.config) if self.config else None
        if (
            generate_personalized_assets
            and _config_has_personalization_content(config_for_personalization)
            and self.generation_mode != GenerationMode.CODE_ONLY
        ):
            # Legacy ``personalizationrules`` (flat rule list) was the first
            # personalization format; it has been superseded by the structured
            # config consumed by ``configure_agent``. We no longer ship a
            # generator for the legacy format, so skip the pass entirely when
            # only legacy rules are present.
            if 'personalizationrules' not in config_for_personalization:
                configure_agent(
                    self.model,
                    config_for_personalization,
                    openai_api_key=self.openai_api_key,
                )

            # Persist personalized agent python for downstream conversion.
            agent_model_to_code(self.model, personalized_agent_path)

            # Emit JSON representation of the personalized agent for downstream
            # tooling (frontend preview, debugging). A conversion failure here
            # is non-fatal — the .py model is already on disk.
            try:
                with open(personalized_agent_path, "r", encoding="utf-8") as f:
                    personalized_code = f.read()
                personalized_json = agent_buml_to_json(personalized_code)
                with open(personalized_json_path, "w", encoding="utf-8") as jf:
                    json.dump(personalized_json, jf, indent=2)
                logger.info("Personalized agent JSON generated at %s", personalized_json_path)
            except Exception:
                logger.exception("Failed to convert personalized agent to JSON")
                # Remove any partial JSON written before the failure so a stale
                # file from a prior run is not mistaken for a successful export.
                if os.path.exists(personalized_json_path):
                    try:
                        os.remove(personalized_json_path)
                    except OSError:
                        logger.warning(
                            "Could not remove partial personalized JSON at %s",
                            personalized_json_path,
                        )

            if not generate_code_assets:
                return

        if config_for_personalization and 'personalizationMapping' in config_for_personalization:
            logger.info("Generating agent with personalization mappings")
            with open(agent_path, mode="w", encoding="utf-8") as f:
                generated_code = agent_template.render(
                    agent=self.model,
                    config=self.config,
                    personalization_mapping=config_for_personalization['personalizationMapping'],
                )
                f.write(generated_code)
        else:
            with open(agent_path, mode="w", encoding="utf-8") as f:
                # TODO: how to handle llm variable names that are used in bodies?
                generated_code = agent_template.render(agent=self.model, config=self.config)
                f.write(generated_code)
            logger.info("Agent script generated at %s", agent_path)
        if generate_code_assets:
            config_template = env.get_template('baf_config_template.py.j2')
            config_path = self.build_generation_path(file_name="config.yaml")
            with open(config_path, mode="w", encoding="utf-8") as f:
                properties = sorted(self.model.properties, key=lambda prop: prop.section)
                generated_code = config_template.render(properties=properties)
                f.write(generated_code)
            logger.info("Agent config file generated at %s", config_path)
            # Generate readme.txt using the Jinja2 template
            readme_template = env.get_template('readme.txt.j2')
            readme_path = self.build_generation_path(file_name="readme.txt")
            with open(readme_path, mode="w", encoding="utf-8") as f:
                generated_code = readme_template.render(agent=self.model)
                f.write(generated_code)
            logger.info("Agent readme file generated at %s", readme_path)

            rag_configs = getattr(self.model, 'rags', []) or []
            if rag_configs:
                rag_base_dir = self.build_generation_dir()
                for idx, rag in enumerate(rag_configs):
                    target_dir = os.path.join(rag_base_dir, rag_slug(getattr(rag, 'name', ''), idx))
                    os.makedirs(target_dir, exist_ok=True)
                    readme_path = os.path.join(target_dir, "README.txt")
                    if not os.path.exists(readme_path):
                        with open(readme_path, "w", encoding="utf-8") as readme_file:
                            readme_file.write(
                                "Place your PDF documents for this RAG database inside this "
                                "folder before running the agent.\n"
                            )
