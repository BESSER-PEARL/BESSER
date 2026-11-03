"""Run the generator a request names on the diagram(s) it needs.

``generate_from_diagram`` serves ``/generate-output`` and
``generate_from_project`` serves ``/generate-output-from-project``. Both
dispatch on the generator's registry entry (``category``,
``required_diagram_type``, ``output_type``); the generators that read their
own options from the request config have a builder in
``_CLASS_DIAGRAM_BUILDERS``. Bad input raises ``ValidationError`` /
``ConversionError`` (HTTP 400, see ``routers/error_handler.py``).
"""

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from typing import Optional

from besser.utilities.secret_redaction import sanitize_config
from besser.utilities.web_modeling_editor.backend.config import (
    SUPPORTED_GENERATORS,
    get_generator_info,
    get_nn_filename,
    is_generator_supported,
)
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    DEFAULT_DBMS,
    DEFAULT_DJANGO_APP_NAME,
    DEFAULT_DJANGO_PROJECT_NAME,
    DEFAULT_JAVA_VERSION,
    DEFAULT_JSONSCHEMA_MODE,
    DEFAULT_QISKIT_BACKEND,
    DEFAULT_QISKIT_SHOTS,
    DEFAULT_SPRING_APP_NAME,
    DEFAULT_SPRING_BOOT_VERSION,
    DEFAULT_SPRING_PACKAGE_NAME,
    DEFAULT_SPRING_PROJECT_NAME,
    DEFAULT_SQL_DIALECT,
    DEFAULT_SUPABASE_USER_ROOT,
    DEPLOYMENT_DIAGRAM_TYPE,
    TEMP_DIR_PREFIX,
)
from besser.utilities.web_modeling_editor.backend.constants.user_buml_model import (
    domain_model as user_reference_domain_model,
)
from besser.utilities.web_modeling_editor.backend.models import DiagramInput, ProjectInput
from besser.utilities.web_modeling_editor.backend.services.converters import (
    annotate_agent_with_a2a,
    link_method_neural_networks,
    process_agent_diagram,
    process_bpmn_diagram,
    process_class_diagram,
    process_deployment_diagram,
    process_gui_diagram,
    process_nn_diagram,
    process_object_diagram,
    process_quantum_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
    GenerationError,
    ValidationError,
)
from besser.utilities.web_modeling_editor.backend.services.generation.agents import (
    generate_agent_output,
    has_agent_components,
)
from besser.utilities.web_modeling_editor.backend.services.generation.downloads import (
    GeneratedOutput,
    file_output,
    with_version_comment,
    zip_output,
)
from besser.utilities.web_modeling_editor.backend.services.governance.swarm_bindings import (
    attach_entry_role_to_agents,
    attach_governance_to_agents,
)
from besser.utilities.web_modeling_editor.backend.services.utils.agent_generation_utils import (
    collect_agents_from_diagrams,
    normalize_personalization_mapping,
)
from besser.utilities.web_modeling_editor.backend.services.utils.user_profile_utils import (
    generate_user_profile_document as _generate_user_profile_document,
    normalize_user_model_output as _normalize_user_model_output,
    safe_path as _safe_path,
    user_criteria_comparators as _user_criteria_comparators,
)

logger = logging.getLogger(__name__)

async def generate_from_project(input_data: ProjectInput) -> GeneratedOutput:
    """Generate from a project: generators that need several diagrams (Web App,
    Docker Compose, the backend) or a non-class diagram, else the active diagram."""
    generator_type = input_data.settings.get("generator") if input_data.settings else None

    if not generator_type:
        raise ValidationError("Generator type is required in project settings")

    # Validate generator
    if not is_generator_supported(generator_type):
        raise ValidationError(
            f"Unsupported generator type: {generator_type}. Supported types: {list(SUPPORTED_GENERATORS.keys())}"
        )

    generator_info = get_generator_info(generator_type)

    # Get configuration from project settings
    config = input_data.settings.get("config", {}) if input_data.settings else {}

    # Handle Web App generator (requires both ClassDiagram and GUINoCodeDiagram)
    if generator_type == "web_app":
        return await _handle_web_app_project_generation(input_data, generator_info, config)

    # Deployment-diagram generators run at project scope so the AgentDiagrams
    # (and the BPMN diagram that wires them) are available for baking agents.
    if generator_info.required_diagram_type == DEPLOYMENT_DIAGRAM_TYPE:
        return await _handle_deployment_project_generation(
            input_data, generator_info, generator_type,
        )

    # The backend runs methods implemented by a neural network, so it needs the
    # project's NNDiagrams next to its ClassDiagram.
    if generator_type == "backend":
        return await _handle_backend_project_generation(input_data, generator_info)

    # Handle generators that consume a non-class diagram (Qiskit → quantum,
    # PyTorch/TensorFlow → neural network). The required diagram type comes
    # from the registry, so this branch covers every such generator without
    # name-literal switches.
    required_diagram_type = generator_info.required_diagram_type
    if required_diagram_type:
        diagram = input_data.get_active_diagram(required_diagram_type)
        if not diagram:
            raise ValidationError(
                f"{required_diagram_type} is required for "
                f"'{generator_type}' generator"
            )
        diagram_input = DiagramInput(
            id=diagram.id,
            title=diagram.title,
            model=diagram.model,
            lastUpdate=diagram.lastUpdate,
            generator=generator_type,
            config=config,
            configYaml=diagram.configYaml,
            referenceDiagramData=getattr(diagram, "referenceDiagramData", None),
        )
        return await generate_from_diagram(diagram_input)

    # For other generators, use the current diagram
    current_diagram = input_data.get_active_diagram(input_data.currentDiagramType)
    if not current_diagram:
        raise ValidationError(f"No diagram found for type: {input_data.currentDiagramType}")

    # Convert to DiagramInput and use existing logic
    diagram_input = DiagramInput(
        id=current_diagram.id,
        title=current_diagram.title,
        model=current_diagram.model,
        lastUpdate=current_diagram.lastUpdate,
        generator=generator_type,
        config=config,
        configYaml=current_diagram.configYaml,
        referenceDiagramData=current_diagram.referenceDiagramData
    )

    return await generate_from_diagram(diagram_input)


async def generate_from_diagram(input_data: DiagramInput) -> GeneratedOutput:
    """Generate from one diagram with the generator ``input_data.generator`` names."""
    with tempfile.TemporaryDirectory(prefix=f"{TEMP_DIR_PREFIX}{uuid.uuid4().hex}_") as temp_dir:
        json_data = input_data.model_dump()
        generator_type = input_data.generator

        # Validate generator type
        if not is_generator_supported(generator_type):
            raise ValidationError(
                f"Invalid generator type: {generator_type}. Supported types: {list(SUPPORTED_GENERATORS.keys())}"
            )

        generator_info = get_generator_info(generator_type)

        missing = generator_info.needs_beyond_one_diagram
        if missing:
            raise ValidationError(
                f"The '{generator_type}' generator cannot run on a single "
                f"{_get_diagram_type(json_data) or 'diagram'}: it needs {missing}."
            )

        # Handle agent generators (different diagram type)
        if generator_info.category == "ai_agent":
            return await generate_agent_output(json_data)

        # Handle quantum generators
        if generator_info.category == "quantum":
            return await _generate_qiskit(json_data, generator_info.generator_class, input_data.config, temp_dir)

        # Handle neural network generators
        if generator_info.category == "neural_network":
            return await _generate_nn(json_data, generator_type, generator_info.generator_class, input_data.config, temp_dir)

        # Handle BPMN generators (reads BPMNDiagram via process_bpmn_diagram)
        if generator_info.category == "business_process":
            return await _generate_bpmn(json_data, generator_type, generator_info.generator_class, temp_dir)

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

        if generator_info.required_diagram_type == DEPLOYMENT_DIAGRAM_TYPE:
            return await _handle_deployment_diagram_generation(
                json_data, generator_type, generator_info, temp_dir,
            )

        # Handle class diagram based generators
        return await _handle_class_diagram_generation(
            json_data, generator_type, generator_info, input_data.config, temp_dir
        )


async def _handle_backend_project_generation(input_data: ProjectInput, generator_info) -> GeneratedOutput:
    """Generate the FastAPI backend from the project's active ClassDiagram, with
    NN-implemented methods linked to the project's NNDiagrams."""
    class_diagram = input_data.get_active_diagram("ClassDiagram")
    if not class_diagram:
        raise ValidationError("ClassDiagram is required for Backend generator")

    with tempfile.TemporaryDirectory(prefix=f"{TEMP_DIR_PREFIX}{uuid.uuid4().hex}_") as temp_dir:
        buml_model = process_class_diagram(class_diagram.model_dump())
        link_method_neural_networks(buml_model, input_data.diagrams.get("NNDiagram", []))
        return await _generate_standard(
            buml_model, generator_info.generator_class, "backend", generator_info, temp_dir
        )


async def _handle_web_app_project_generation(input_data: ProjectInput, generator_info, config: dict) -> GeneratedOutput:
    """Handle Web App generation from a complete project with both ClassDiagram and GUINoCodeDiagram.
    Optionally includes AgentDiagram if agent components are present in the GUI."""
    # Extract active GUINoCodeDiagram first — it drives which other diagrams are used
    gui_diagram = input_data.get_active_diagram("GUINoCodeDiagram")
    if not gui_diagram:
        raise ValidationError("GUINoCodeDiagram is required for Web App generator")

    # Use the GUI diagram's per-diagram references to find the correct ClassDiagram
    class_diagram = input_data.get_referenced_diagram(gui_diagram, "ClassDiagram")
    if not class_diagram:
        raise ValidationError("ClassDiagram is required for Web App generator")

    # Personalization versions (optional). When present, the frontend has
    # pre-assembled one COMPLETE GUI model per version (base + each user
    # profile), each already resolved page-by-page. When absent, we generate a
    # single app from the GUI diagram's own model (unchanged behavior).
    web_app_versions = config.get("webAppVersions") if isinstance(config, dict) else None
    if web_app_versions:
        version_specs = [
            (str(v.get("slug") or f"version-{i + 1}"), v.get("guiModel"))
            for i, v in enumerate(web_app_versions)
        ]
    else:
        version_specs = [(None, gui_diagram.model)]

    multi = len(version_specs) > 1
    generator_class = generator_info.generator_class

    with tempfile.TemporaryDirectory(prefix=TEMP_DIR_PREFIX) as temp_dir:
        for slug, gui_json in version_specs:
            # Re-derive the class BUML per version so the generator can't leak
            # mutations from one version into the next.
            buml_model = process_class_diagram(class_diagram.model_dump())
            link_method_neural_networks(buml_model, input_data.diagrams.get("NNDiagram", []))
            gui_model = process_gui_diagram(gui_json, class_diagram.model, buml_model)

            # Collect every AgentDiagram in the project if this version's GUI uses
            # agent components. The frontend dropdown enumerates all agents, so the
            # generator must satisfy any binding — we don't filter to the active
            # reference here.
            agent_models = []
            agent_configs = {}
            agent_config_yamls: dict = {}
            if has_agent_components(gui_model):
                project_agent_config = config.get('agentConfig') if isinstance(config, dict) else None
                default_cfg = project_agent_config or config
                agent_models, agent_configs, agent_config_yamls = collect_agents_from_diagrams(
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

            # Single version → generate at the zip root (current layout).
            # Multiple versions → one profile-named subfolder each.
            out_dir = temp_dir
            if multi:
                out_dir = _safe_path(temp_dir, os.path.basename(slug))
                os.makedirs(out_dir, exist_ok=True)

            await _run_web_app_generator(
                buml_model, gui_model, generator_class, out_dir,
                agent_models=agent_models, agent_configs=agent_configs,
                agent_config_yamls=agent_config_yamls,
            )

        options = {"versions": ", ".join(slug for slug, _ in version_specs)} if multi else None
        return zip_output(temp_dir, "web_app", options)


async def _handle_deployment_project_generation(
    input_data: ProjectInput, generator_info, generator_type: str
) -> GeneratedOutput:
    """Generate Docker Compose output with the project's AgentDiagrams in scope.

    Resolves ``Artifact.agent_model_ref`` values against AgentDiagram ids, so each
    linked local artifact receives a baked BAF build context, and binds the
    project's BPMN diagrams onto those agents (governed merges, entry role; see
    ``services.governance.swarm_bindings``). Returns a ZIP with
    ``docker-compose.yml`` and the agent build contexts.
    """
    deployment_diagram = input_data.get_active_diagram(DEPLOYMENT_DIAGRAM_TYPE)
    if not deployment_diagram:
        raise ValidationError("DeploymentDiagram is required for the Docker Compose generator")

    with tempfile.TemporaryDirectory(prefix=f"{TEMP_DIR_PREFIX}{uuid.uuid4().hex}_") as temp_dir:
        deployment_model = _process_deployment_payload(deployment_diagram.model_dump())

        # AgentDiagram id -> BUML Agent. Keyed by diagram id (the value of
        # Artifact.agent_model_ref), not by agent name: agent names may repeat.
        agent_models_by_id: dict = {}
        agent_configs_by_id: dict = {}
        agent_config_yamls_by_id: dict = {}
        for entry in input_data.diagrams.get("AgentDiagram", []):
            entry_dict = entry.model_dump()
            model = entry_dict.get("model")
            if not entry_dict.get("id") or not (isinstance(model, dict) and model.get("elements")):
                continue
            agent_config = entry_dict.get("config")
            if isinstance(agent_config, dict) and isinstance(agent_config.get("personalizationMapping"), list):
                normalize_personalization_mapping(agent_config, entry_dict, _generate_user_profile_document)
            agent_model = process_agent_diagram(entry_dict)
            # The diagram's a2a:in / a2a:out tags become agent._a2a, which the
            # generator prefers over the to_/from_ state-name convention.
            annotate_agent_with_a2a(agent_model, entry_dict)
            agent_models_by_id[entry_dict["id"]] = agent_model
            agent_configs_by_id[entry_dict["id"]] = agent_config
            agent_config_yamls_by_id[entry_dict["id"]] = entry_dict.get("configYaml")

        for bpmn_diagram in input_data.diagrams.get("BPMN", []):
            bpmn_model = process_bpmn_diagram(bpmn_diagram.model_dump())
            attach_governance_to_agents(bpmn_model, agent_models_by_id)
            attach_entry_role_to_agents(bpmn_model, agent_models_by_id)

        generator_instance = generator_info.generator_class(
            deployment_model, output_dir=temp_dir,
            agent_models_by_id=agent_models_by_id,
            agent_configs_by_id=agent_configs_by_id,
            agent_config_yamls_by_id=agent_config_yamls_by_id,
        )
        await asyncio.to_thread(generator_instance.generate)
        return zip_output(temp_dir, generator_type)


def _process_deployment_payload(json_data: dict):
    """``process_deployment_diagram``, with a malformed payload reported as ``ConversionError``."""
    try:
        return process_deployment_diagram(json_data)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ConversionError(f"Malformed Deployment diagram payload: {exc}") from exc


async def _handle_class_diagram_generation(
    json_data: dict,
    generator_type: str,
    generator_info,
    config: dict,
    temp_dir: str
) -> GeneratedOutput:
    """Handle class diagram based generation."""
    # Process the class diagram JSON data
    buml_model = process_class_diagram(json_data)
    generator_class = generator_info.generator_class
    logger.info("Generator type %s", generator_type)

    builder = _CLASS_DIAGRAM_BUILDERS.get(generator_type)
    if builder:
        return await builder(buml_model, generator_class, config, temp_dir)

    return await _generate_standard(buml_model, generator_class, generator_type, generator_info, temp_dir)


async def _handle_object_diagram_generation(
    json_data: dict,
    generator_type: str,
    generator_info,
    config: dict,
    temp_dir: str,
) -> GeneratedOutput:
    """Handle generators that operate on object diagrams."""
    reference_payload = _extract_reference_class_diagram(json_data)
    if not reference_payload:
        raise ValidationError("Object diagram generation requires reference class diagram data.")

    domain_model = process_class_diagram(reference_payload)
    object_model = process_object_diagram(json_data, domain_model)

    generator_class = generator_info.generator_class
    generator_instance = generator_class(object_model, output_dir=temp_dir)
    await asyncio.to_thread(generator_instance.generate)

    return file_output(temp_dir, generator_type)


async def _handle_user_diagram_generation(
    json_data: dict,
    generator_type: str,
    generator_info,
    config: dict,
    temp_dir: str,
) -> GeneratedOutput:
    """Handle user diagram generation using the preset user reference domain."""
    try:
        object_model = process_object_diagram(json_data, user_reference_domain_model)
    except Exception as conversion_error:
        logger.exception("Failed to convert user diagram to BUML")
        raise ValidationError(
            "Failed to convert user diagram to BUML. Please check the diagram data."
        ) from conversion_error

    generator_class = generator_info.generator_class
    generator_instance = generator_class(object_model, output_dir=temp_dir)
    await asyncio.to_thread(generator_instance.generate)

    _normalize_user_model_output(object_model, temp_dir, _user_criteria_comparators(json_data))

    return file_output(temp_dir, generator_type)


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


async def _generate_django(buml_model, generator_class, config: dict, temp_dir: str) -> GeneratedOutput:
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

    # DjangoGenerator anchors all of its filesystem effects on output_dir
    # (its subprocesses run with an explicit cwd), so it can run directly on
    # a worker thread without touching the process-wide working directory.
    await asyncio.to_thread(generator_instance.generate)

    # Validate generation
    if not os.path.exists(project_dir) or not os.listdir(project_dir):
        raise ValueError("Django project generation failed: Output directory is empty")

    return zip_output(project_dir, "django", {
        "project_name": project_name, "app_name": app_name, "containerization": containerization,
    })


async def _generate_spring(buml_model, generator_class, config: dict, temp_dir: str) -> GeneratedOutput:
    """Generate a Spring Boot project and return it as a ZIP."""
    config = config or {}
    project_name = config.get("project_name") or DEFAULT_SPRING_PROJECT_NAME
    app_name = config.get("app_name") or DEFAULT_SPRING_APP_NAME
    spring_boot_version = config.get("spring_boot_version") or DEFAULT_SPRING_BOOT_VERSION
    java_version = config.get("java_version") or DEFAULT_JAVA_VERSION
    package_name = config.get("package_name") or DEFAULT_SPRING_PACKAGE_NAME

    # Sanitize project_name to prevent path traversal
    project_name = os.path.basename(project_name)
    if not project_name:
        project_name = DEFAULT_SPRING_PROJECT_NAME

    project_dir = _safe_path(temp_dir, project_name)
    os.makedirs(project_dir, exist_ok=True)

    # An invalid package name is a user input error, not a server fault.
    try:
        generator_instance = generator_class(
            buml_model,
            output_dir=project_dir,
            app_name=app_name,
            spring_boot_version=spring_boot_version,
            java_version=java_version,
            package_name=package_name,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    await asyncio.to_thread(generator_instance.generate)

    if not os.listdir(project_dir):
        raise ValueError("Spring Boot project generation failed: Output directory is empty")

    return zip_output(project_dir, "spring", {
        "project_name": project_name, "app_name": app_name, "spring_boot_version": spring_boot_version,
        "java_version": java_version, "package_name": package_name,
    })


async def _generate_sql(buml_model, generator_class, config: dict, temp_dir: str) -> GeneratedOutput:
    """Generate SQL files."""
    dialect = DEFAULT_SQL_DIALECT
    if config and "dialect" in config:
        dialect = config["dialect"]

    generator_instance = generator_class(
        buml_model, output_dir=temp_dir, sql_dialect=dialect
    )
    await asyncio.to_thread(generator_instance.generate)

    return file_output(temp_dir, "sql")


_SUPABASE_USER_ROOT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


async def _generate_supabase(buml_model, generator_class, config: dict, temp_dir: str) -> GeneratedOutput:
    """Generate Supabase Postgres DDL."""
    user_root = DEFAULT_SUPABASE_USER_ROOT
    if config and "user_root" in config:
        raw = config["user_root"]
        if raw == "" or raw is None:
            # Empty / None means "skip auth integration entirely"
            user_root = None
        else:
            if not isinstance(raw, str) or not _SUPABASE_USER_ROOT_RE.match(raw):
                raise ValidationError(
                    "Invalid user_root: must be a class name matching "
                    "[A-Za-z_][A-Za-z0-9_]{0,62} (or empty to skip auth integration)."
                )
            user_root = raw

    generator_instance = generator_class(
        buml_model, output_dir=temp_dir, user_root=user_root
    )
    await asyncio.to_thread(generator_instance.generate)

    return file_output(temp_dir, "supabase")


async def _generate_sqlalchemy(buml_model, generator_class, config: dict, temp_dir: str) -> GeneratedOutput:
    """Generate SQLAlchemy files."""
    dbms = DEFAULT_DBMS
    if config and "dbms" in config:
        dbms = config["dbms"]

    generator_instance = generator_class(buml_model, output_dir=temp_dir)
    await asyncio.to_thread(generator_instance.generate, dbms=dbms)

    return file_output(temp_dir, "sqlalchemy")


async def _generate_jsonschema(buml_model, generator_class, config: dict, temp_dir: str) -> GeneratedOutput:
    """Generate JSON Schema files."""
    mode = DEFAULT_JSONSCHEMA_MODE
    if config and "mode" in config:
        mode = config["mode"]

    generator_instance = generator_class(buml_model, output_dir=temp_dir, mode=mode)
    await asyncio.to_thread(generator_instance.generate)

    if mode == "smart_data":
        # For smart_data mode, we create a zip file with a specific name
        return zip_output(
            temp_dir, "jsonschema", {"mode": mode}, file_name="smart_data_schemas.zip",
        )
    else:
        return file_output(temp_dir, "jsonschema")


# Class-diagram generators that take options from the request config; every
# other one is built with ``(model, output_dir=...)`` by ``_generate_standard``.
_CLASS_DIAGRAM_BUILDERS = {
    "django": _generate_django,
    "spring": _generate_spring,
    "sql": _generate_sql,
    "supabase": _generate_supabase,
    "sqlalchemy": _generate_sqlalchemy,
    "jsonschema": _generate_jsonschema,
}


async def _handle_deployment_diagram_generation(
    json_data: dict,
    generator_type: str,
    generator_info,
    temp_dir: str,
) -> GeneratedOutput:
    """Generate Docker Compose from a single UML DeploymentDiagram.

    Single-diagram generation has no project-level AgentDiagram resolver, so the
    ZIP holds ``docker-compose.yml`` only. Project generation uses
    ``_handle_deployment_project_generation`` to also bake linked agent contexts.
    """
    deployment_model = _process_deployment_payload(json_data)
    generator_instance = generator_info.generator_class(deployment_model, output_dir=temp_dir)
    await asyncio.to_thread(generator_instance.generate)

    if generator_info.output_type == "zip":
        return zip_output(temp_dir, generator_type)
    return file_output(temp_dir, generator_type)


async def _generate_nn(json_data: dict, generator_type: str, generator_class, config: dict,
                       temp_dir: str) -> GeneratedOutput:
    """Generate neural network code (PyTorch or TensorFlow)."""
    if generator_class is None:
        raise ConversionError(
            f"Generator '{generator_type}' is not available. "
            f"Please install the required dependencies."
        )
    try:
        nn_model = process_nn_diagram(json_data)
    except (KeyError, TypeError, AttributeError) as exc:
        # Malformed NN payload (dangling element id, non-dict element,
        # non-string attribute value). Surface as a structured 400 via the
        # decorator instead of letting it leak as a 500.
        raise ConversionError(f"Malformed NN diagram payload: {exc}") from exc
    # ValueError from process_nn_diagram (missing mandatory attributes,
    # cycles, unresolved NNReferences, ...) is caught and mapped to 400 by
    # @handle_endpoint_errors — no explicit re-raise needed here.

    if generator_type == "forecaster":
        return await _generate_forecaster(nn_model, generator_class, config, temp_dir)

    generation_type = config.get("generation_type", "subclassing") if config else "subclassing"
    if generator_type == "tensorflow":
        generator_instance = generator_class(nn_model, output_dir=temp_dir, generation_type=generation_type)
    else:
        channel_last = config.get("channel_last", False) if config else False
        generator_instance = generator_class(
            nn_model, output_dir=temp_dir,
            generation_type=generation_type, channel_last=channel_last,
        )
    await asyncio.to_thread(generator_instance.generate)

    download_filename = get_nn_filename(generator_type, generation_type)

    # Read the known-named file emitted by the generator. Scanning the
    # directory and picking "the first file" is fragile — if the generator
    # drops helpers alongside the main artifact we'd ship the wrong one.
    output_file_path = _safe_path(temp_dir, download_filename)
    if not os.path.isfile(output_file_path):
        raise GenerationError(
            f"{generator_type} generation failed: expected output "
            f"{download_filename!r} was not produced."
        )
    with open(output_file_path, "rb") as f:
        file_content = f.read()
    file_content = with_version_comment(file_content, download_filename)

    return GeneratedOutput(file_content, download_filename, is_zip=False)


async def _generate_forecaster(nn_model, generator_class, config: dict, temp_dir: str) -> GeneratedOutput:
    """Generate a time-series forecaster from an NN model.

    ``config`` holds the ForecastConfig fields (window, horizon, season, log,
    difference, calendar, standardize, seed or seeds) plus ``channel_last``.
    Invalid settings, and a network the forecaster cannot take (its first layer
    must read the window, its last return the horizon), raise ValidationError,
    which @handle_endpoint_errors maps to a 400 with the generator's message.
    """
    # the generator class is only registered when this module imports
    from besser.generators.nn.forecaster import ForecastConfig

    settings = dict(config or {})
    settings.pop("generation_type", None)  # sent with every NN generation; not a forecaster setting
    channel_last = settings.pop("channel_last", False)
    if not isinstance(channel_last, bool):
        raise ValidationError(f"channel_last must be true or false, got {type(channel_last).__name__}")
    try:
        forecast_config = ForecastConfig.from_dict(settings)
        generator_instance = generator_class(
            nn_model, forecast_config, output_dir=temp_dir, channel_last=channel_last,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    await asyncio.to_thread(generator_instance.generate)
    # one file, forecaster_<model name>.py, shipped under that name (keep_generated_filename)
    return file_output(temp_dir, "forecaster")


async def _generate_bpmn(json_data: dict, generator_type: str, generator_class, temp_dir: str) -> GeneratedOutput:
    """Generate vendor-neutral BPMN 2.0 XML.

    Processes the BPMN diagram JSON via ``process_bpmn_diagram`` (re-wrapping
    malformed-payload exceptions as ``ConversionError`` so they surface as 400),
    then runs the BPMNGenerator and returns the emitted ``.bpmn`` file.
    """
    try:
        bpmn_model = process_bpmn_diagram(json_data)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ConversionError(f"Malformed BPMN diagram payload: {exc}") from exc

    generator_instance = generator_class(bpmn_model, output_dir=temp_dir)
    await asyncio.to_thread(generator_instance.generate)

    return file_output(temp_dir, generator_type)


async def _generate_qiskit(json_data: dict, generator_class, config: dict, temp_dir: str) -> GeneratedOutput:
    """Generate Qiskit code."""
    # Validate that this is a quantum diagram (has 'cols' in model)
    model_data = json_data.get('model', {})
    if not isinstance(model_data, dict) or 'cols' not in model_data:
        # Check if this looks like a ClassDiagram or other diagram type
        diagram_type = model_data.get('type', 'unknown') if isinstance(model_data, dict) else 'unknown'
        raise ValidationError(
            f"Invalid diagram type for Qiskit generator. Expected QuantumCircuitDiagram "
            f"but received '{diagram_type}'. Please use the 'generate-output-from-project' "
            f"endpoint or select the Quantum Circuit diagram."
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

    return file_output(temp_dir, "qiskit")


async def _run_web_app_generator(buml_model, gui_model, generator_class, out_dir: str,
                                 agent_models=None, agent_configs=None, agent_config_yamls=None):
    """Run the web app generator into ``out_dir`` WITHOUT zipping.

    Each personalization version is generated into its own subdirectory
    before a single zip is built. ``agent_models`` is a list; each is emitted
    under ``agents/<slug>/`` in the generated output.
    """
    generator_instance = generator_class(
        buml_model, gui_model, output_dir=out_dir,
        agent_models=agent_models, agent_configs=agent_configs,
        agent_config_yamls=agent_config_yamls,
    )
    await asyncio.to_thread(generator_instance.generate)


async def _generate_standard(buml_model, generator_class, generator_type: str, generator_info,
                             temp_dir: str) -> GeneratedOutput:
    """Generate standard files (non-Django, non-SQL)."""
    generator_instance = generator_class(buml_model, output_dir=temp_dir)
    await asyncio.to_thread(generator_instance.generate)

    # Return ZIP or single file based on generator info
    if generator_info.output_type == "zip":
        return zip_output(temp_dir, generator_type)
    else:
        return file_output(temp_dir, generator_type)
