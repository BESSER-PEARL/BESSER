"""
GitHub Deployment API - Deploy generated apps to user's GitHub account.

This endpoint generates a web app and pushes it to a GitHub repository
in the user's account, enabling one-click deployment to platforms like Render.
"""

import hashlib
import json
import logging
import os
import re
import tempfile
import uuid
from typing import Optional, List

import httpx
from fastapi import APIRouter, HTTPException, Request, Body, Header, Query
from pydantic import BaseModel, Field

from besser.utilities.web_modeling_editor.backend.services.deployment.github_service import (
    create_github_service
)
from besser.utilities.web_modeling_editor.backend.services.deployment.github_oauth import (
    get_user_token
)
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_class_diagram,
    process_gui_diagram,
    process_agent_diagram,
)
from besser.utilities.web_modeling_editor.backend.config import get_generator_info
from besser.utilities.buml_code_builder import (
    domain_model_to_code,
    gui_model_to_code,
    agent_model_to_code,
)
from besser.generators.web_app.web_app_generator import agent_slug
from besser.utilities.buml_code_builder.common import safe_var_name
from besser.utilities.web_modeling_editor.backend.routers.generation_router import (
    sanitize_config,
)
from besser.utilities.web_modeling_editor.backend.services.utils.agent_generation_utils import (
    collect_agents_from_diagrams,
    extract_openai_api_key,
    normalize_personalization_mapping,
)
from besser.utilities.web_modeling_editor.backend.services.utils.user_profile_utils import (
    generate_user_profile_document,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    GenerationError,
    ValidationError,
)

logger = logging.getLogger(__name__)

# Module-level constants for repeated literals used in render.yaml emission.
PYTHON_VERSION = "3.12.3"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
STREAMLIT_DB_HOST_ENV = "STREAMLIT_DB_HOST"
CHATBOT_DEPLOY_TARGETS = frozenset({
    "agent",
    "chatbot",
    "standalone-agent",
    "standalone_agent",
    "standalone-chatbot",
    "standalone_chatbot",
})


# Create router
router = APIRouter(prefix="/github", tags=["GitHub Deployment"])


class DeployToGitHubRequest(BaseModel):
    """Request model for GitHub deployment."""
    repo_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Repository name (alphanumeric, hyphens, underscores)"
    )
    description: Optional[str] = Field(
        None,
        max_length=350,
        description="Repository description"
    )
    is_private: bool = Field(
        default=False,
        description="Make repository private"
    )


class DeployToGitHubResponse(BaseModel):
    """Response model for GitHub deployment."""
    success: bool
    repo_url: str
    repo_name: str
    owner: str
    deployment_urls: dict
    files_uploaded: int
    message: str
    # True on the very first deploy to a repo (no previous ``render.yaml`` found),
    # False on every subsequent deploy. The frontend uses this to pick the right
    # "next action" button — create-blueprint vs. open-live-site.
    is_first_deploy: bool = True
    # Deployment flavor: ``webapp`` (Class+GUI) or ``chatbot`` (Agent-only).
    deployment_type: str = "webapp"


@router.post("/deploy-webapp", response_model=DeployToGitHubResponse)
async def deploy_webapp_to_github(
    request: Request,
    body: dict = Body(...),
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session")
):
    """
    Deploy generated web application to user's GitHub repository.

    Flow:
    1. Verify user has authenticated with GitHub
    2. Generate deployment project from diagrams (web app or standalone chatbot)
    3. Create repository in user's GitHub account
    4. Push generated code to repository
    5. Return repository URL and deployment links

    The user can then deploy to Render with one click.
    """
    try:
        # Verify GitHub authentication
        if not github_session:
            raise HTTPException(
                status_code=401,
                detail="GitHub authentication required. Please sign in with GitHub first."
            )

        access_token = get_user_token(github_session)
        if not access_token:
            raise HTTPException(
                status_code=401,
                detail="GitHub session expired. Please sign in again."
            )

        # Create GitHub service
        github = create_github_service(access_token)

        # Get authenticated user info
        user_info = await github.get_authenticated_user()
        username = user_info.get("login")

        # Extract deploy config
        deploy_config = body.get("deploy_config", {})
        deploy_target = _normalize_deploy_target(deploy_config.get("target"))
        repo_name = deploy_config.get("repo_name", "besser-webapp")
        default_description = (
            "Standalone chatbot generated by BESSER"
            if deploy_target == "agent"
            else "Web application generated"
        )
        description = deploy_config.get("description") or default_description
        is_private = deploy_config.get("is_private", False)
        use_existing = deploy_config.get("use_existing", False)
        custom_commit_message = deploy_config.get("commit_message", "")

        # Sanitize repo name
        repo_name = _sanitize_repo_name(repo_name)

        # Extract diagrams — supports both multi-diagram (list) and legacy (dict) format
        diagrams = body.get("diagrams", {})
        current_indices = body.get("currentDiagramIndices", {})

        def _get_active(diagram_type: str):
            """Return the active diagram dict for the given type, handling list or dict."""
            value = diagrams.get(diagram_type)
            if isinstance(value, list):
                idx = current_indices.get(diagram_type, 0)
                return value[idx] if 0 <= idx < len(value) else (value[0] if value else {})
            return value or {}

        def _get_all(diagram_type: str) -> list:
            """Return every diagram dict of the given type (normalized to a list)."""
            value = diagrams.get(diagram_type)
            if isinstance(value, list):
                return [v for v in value if v]
            if value:
                return [value]
            return []

        if deploy_target == "agent":
            agent_diagram_data = _get_active("AgentDiagram")
            if not agent_diagram_data or not agent_diagram_data.get("model"):
                raise HTTPException(
                    status_code=400,
                    detail="AgentDiagram is required for standalone chatbot deployment",
                )

            agent_model_payload = agent_diagram_data.get("model")
            if isinstance(agent_model_payload, dict) and not agent_model_payload.get("elements"):
                raise HTTPException(
                    status_code=400,
                    detail="AgentDiagram must contain at least one element for standalone chatbot deployment",
                )

            agent_model = process_agent_diagram(agent_diagram_data)
            settings = body.get("settings", {}) or {}
            settings_config = settings.get("config") or {}
            agent_config = agent_diagram_data.get("config") or settings_config

            logger.info(
                "[chatbot deploy] agent_config: %s; has personalizationMapping: %s (len=%s)",
                sanitize_config(agent_config) if isinstance(agent_config, dict) else type(agent_config).__name__,
                isinstance(agent_config, dict) and isinstance(agent_config.get("personalizationMapping"), list),
                len(agent_config.get("personalizationMapping", [])) if isinstance(agent_config, dict) and isinstance(agent_config.get("personalizationMapping"), list) else 0,
            )

            # If the agent config ships a personalization mapping, normalize it
            # the same way the generation router does. Without this step the
            # BAF generator receives raw UML JSON for ``user_profile`` /
            # ``agent_model`` entries and falls back to the default chatbot.
            if isinstance(agent_config, dict) and isinstance(agent_config.get("personalizationMapping"), list):
                normalize_personalization_mapping(
                    agent_config, agent_diagram_data, generate_user_profile_document,
                )
                logger.info(
                    "[chatbot deploy] normalized %d personalization mapping entries",
                    len(agent_config.get("personalizationMapping", [])),
                )

            existing_suffix = await _read_existing_chatbot_service_suffix(github, username, repo_name)

            with tempfile.TemporaryDirectory(prefix=f"besser_github_{uuid.uuid4().hex}_") as temp_dir:
                generator_info = get_generator_info("agent")
                if not generator_info:
                    raise GenerationError("Agent generator is not configured.")
                generator_class = generator_info.generator_class
                generator = generator_class(
                    agent_model,
                    output_dir=temp_dir,
                    config=agent_config,
                    openai_api_key=extract_openai_api_key(agent_config),
                )
                generator.generate()

                _write_personalization_user_profiles(temp_dir, agent_config)
                _add_chatbot_deployment_configs(
                    temp_dir,
                    repo_name,
                    agent_model=agent_model,
                    agent_config=agent_config,
                    service_suffix=existing_suffix,
                )

                reused_repo = False
                default_branch = "main"

                if use_existing:
                    try:
                        branches = await github.get_branches(username, repo_name)
                        reused_repo = True
                        if branches:
                            default_branch = branches[0]
                    except httpx.HTTPStatusError as e:
                        if e.response is not None and e.response.status_code == 404:
                            use_existing = False
                        else:
                            raise

                if not use_existing:
                    repo_info = await github.create_repository(
                        repo_name=repo_name,
                        description=description,
                        is_private=is_private,
                        auto_init=True,
                    )
                    default_branch = repo_info.get("default_branch", "main")

                    try:
                        await github.update_repository(
                            owner=username,
                            repo_name=repo_name,
                            description=f"{description} — Generated by BESSER",
                            homepage="https://editor.besser-pearl.org",
                            topics=[
                                "besser",
                                "low-code",
                                "model-driven",
                                "generated-agent",
                                "chatbot",
                                "streamlit",
                                "python",
                            ],
                        )
                    except Exception:
                        logger.debug("Non-critical: failed to update repository metadata", exc_info=True)

                repo_url = f"https://github.com/{username}/{repo_name}"

                buml_dir = os.path.join(temp_dir, "buml")
                os.makedirs(buml_dir, exist_ok=True)
                try:
                    agent_slug_name = agent_slug(agent_model.name)
                    agent_model_to_code(agent_model, os.path.join(buml_dir, f"agent_model_{agent_slug_name}.py"))
                except Exception:
                    logger.warning("Failed to export agent B-UML model — continuing without it", exc_info=True)

                try:
                    # Prefer the V2 project-export the frontend already builds via
                    # `exportProjectAsJson` — writing it as-is keeps `diagrams.json`
                    # re-importable through the editor's "Import Project" action.
                    project_export = body.get("projectExport") if isinstance(body, dict) else None
                    diagrams_payload = project_export if isinstance(project_export, dict) else diagrams
                    if not isinstance(project_export, dict):
                        logger.warning(
                            "Deploy body did not include 'projectExport'; falling back to "
                            "raw diagrams. The resulting diagrams.json will not be "
                            "re-importable via the editor."
                        )
                    json_path = os.path.join(buml_dir, "diagrams.json")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(diagrams_payload, f, indent=2, default=str)
                except Exception:
                    logger.warning("Failed to export diagram JSON — continuing without it", exc_info=True)

                readme_path = os.path.join(temp_dir, "README.md")
                with open(readme_path, "w", encoding="utf-8") as f:
                    f.write(github.generate_readme_content(
                        body.get("name", repo_name),
                        deploy_instructions=True,
                        owner=username,
                        repo_name=repo_name,
                        deployment_type="chatbot",
                    ))

                if custom_commit_message:
                    commit_message = custom_commit_message
                elif reused_repo:
                    commit_message = "Update chatbot - Generated by BESSER Web Editor"
                else:
                    commit_message = "Initial commit - Generated by BESSER Web Editor"

                push_results = await github.push_directory_to_repo(
                    owner=username,
                    repo_name=repo_name,
                    directory_path=temp_dir,
                    commit_message=commit_message,
                    branch=default_branch,
                    preserve_existing_files=reused_repo,
                )
                if not push_results.get("commit_sha"):
                    raise GenerationError("Failed to create GitHub commit for generated files.")

                deployment_urls = github.get_deployment_urls(username, repo_name)
                is_first_deploy = existing_suffix is None
                if existing_suffix:
                    chatbot_service_name = _fit_chatbot_service_name(repo_name, existing_suffix)
                    deployment_urls["live_chatbot"] = f"https://{chatbot_service_name}.onrender.com"
                    deployment_urls["render_dashboard"] = "https://dashboard.render.com/blueprints"

                action = "updated" if reused_repo else "created"
                return DeployToGitHubResponse(
                    success=True,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    owner=username,
                    deployment_urls=deployment_urls,
                    files_uploaded=push_results["total_files"],
                    message=f"Successfully {action} repository and pushed {push_results['total_files']} files",
                    is_first_deploy=is_first_deploy,
                    deployment_type="chatbot",
                )

        class_diagram_data = _get_active("ClassDiagram")
        gui_diagram_data = _get_active("GUINoCodeDiagram")

        if not class_diagram_data or not class_diagram_data.get("model"):
            raise HTTPException(
                status_code=400,
                detail="ClassDiagram is required for web app deployment"
            )

        if not gui_diagram_data or not gui_diagram_data.get("model"):
            raise HTTPException(
                status_code=400,
                detail="GUINoCodeDiagram is required for web app deployment"
            )

        # Process diagrams to BUML
        buml_model = process_class_diagram(class_diagram_data)
        gui_model = process_gui_diagram(
            gui_diagram_data.get("model"),
            class_diagram_data.get("model"),
            buml_model
        )

        # Collect every AgentDiagram in the project (multi-agent support).
        # Uniqueness enforcement and BUML conversion live in the shared helper.
        agent_models: list = []
        agent_configs: dict = {}
        has_agent_components = _has_agent_components(gui_model)
        if has_agent_components:
            settings = body.get("settings", {}) or {}
            settings_config = settings.get("config") or {}
            agent_models, agent_configs = collect_agents_from_diagrams(
                _get_all("AgentDiagram"),
                default_config=settings_config,
            )
            for name, cfg in agent_configs.items():
                logger.debug(
                    "Resolved agent_config for %s: %s",
                    name,
                    json.dumps(sanitize_config(cfg), indent=2, default=str) if isinstance(cfg, dict) else 'None',
                )

        # Try to reuse the service suffix from a previous deployment so
        # redeploys don't create a fresh set of zombie Render services.
        # On a brand-new repo (or if the existing render.yaml can't be read)
        # ``_add_deployment_configs`` will fall back to a random UUID suffix —
        # that's the right behavior for a first deploy since the hostname is
        # not yet taken on Render.
        existing_suffix = await _read_existing_service_suffix(github, username, repo_name)

        # Generate web app
        with tempfile.TemporaryDirectory(prefix=f"besser_github_{uuid.uuid4().hex}_") as temp_dir:
            generator_info = get_generator_info("web_app")
            generator_class = generator_info.generator_class
            generator = generator_class(
                buml_model,
                gui_model,
                output_dir=temp_dir,
                agent_models=agent_models,
                agent_configs=agent_configs,
            )
            generator.generate()

            # Add deployment configuration files
            _add_deployment_configs(
                temp_dir,
                repo_name,
                agent_models=agent_models,
                agent_configs=agent_configs,
                service_suffix=existing_suffix,
            )

            # Determine whether to reuse an existing repo or create a new one
            reused_repo = False
            default_branch = "main"

            if use_existing:
                try:
                    branches = await github.get_branches(username, repo_name)
                    # Repo exists and is accessible — reuse it
                    reused_repo = True
                    if branches:
                        default_branch = branches[0]
                except httpx.HTTPStatusError as e:
                    if e.response is not None and e.response.status_code == 404:
                        # Repo was deleted — fall back to creating a new one
                        use_existing = False
                    else:
                        raise

            if not use_existing:
                repo_info = await github.create_repository(
                    repo_name=repo_name,
                    description=description,
                    is_private=is_private,
                    auto_init=True
                )
                default_branch = repo_info.get("default_branch", "main")

                # Set repo metadata: homepage, topics, description
                try:
                    await github.update_repository(
                        owner=username,
                        repo_name=repo_name,
                        description=f"{description} — Generated by BESSER",
                        homepage="https://editor.besser-pearl.org",
                        topics=["besser", "low-code", "model-driven", "generated-app", "react", "fastapi", "web-app"]
                    )
                except Exception:
                    logger.debug("Non-critical: failed to update repository metadata", exc_info=True)

            repo_url = f"https://github.com/{username}/{repo_name}"

            # Export raw B-UML models so the project can be maintained without the web editor
            buml_dir = os.path.join(temp_dir, "buml")
            os.makedirs(buml_dir, exist_ok=True)
            try:
                domain_model_to_code(model=buml_model, file_path=os.path.join(buml_dir, "domain_model.py"))
                gui_model_to_code(model=gui_model, file_path=os.path.join(buml_dir, "gui_model.py"), domain_model=buml_model)
                for ag in agent_models:
                    slug = agent_slug(ag.name)
                    agent_model_to_code(ag, os.path.join(buml_dir, f"agent_model_{slug}.py"))
            except Exception:
                logger.warning("Failed to export B-UML models — continuing without them", exc_info=True)

            # Prefer the V2 project-export the frontend already builds via
            # `exportProjectAsJson` so the resulting diagrams.json can be
            # re-imported via the editor's "Import Project" action. Fall back
            # to the raw diagrams payload only when the frontend has not been
            # updated to send `projectExport` yet.
            try:
                project_export = body.get("projectExport") if isinstance(body, dict) else None
                diagrams_payload = project_export if isinstance(project_export, dict) else diagrams
                if not isinstance(project_export, dict):
                    logger.warning(
                        "Deploy body did not include 'projectExport'; falling back to "
                        "raw diagrams. The resulting diagrams.json will not be "
                        "re-importable via the editor."
                    )
                json_path = os.path.join(buml_dir, "diagrams.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(diagrams_payload, f, indent=2, default=str)
            except Exception:
                logger.warning("Failed to export diagram JSON — continuing without it", exc_info=True)

            # Add README before push so everything lands in one commit
            readme_path = os.path.join(temp_dir, "README.md")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(github.generate_readme_content(
                    body.get("name", repo_name),
                    deploy_instructions=True,
                    owner=username,
                    repo_name=repo_name,
                    deployment_type="webapp",
                ))

            # Push code to repository
            if custom_commit_message:
                commit_message = custom_commit_message
            elif reused_repo:
                commit_message = "Update app - Generated by BESSER Web Editor"
            else:
                commit_message = "Initial commit - Generated by BESSER Web Editor"

            push_results = await github.push_directory_to_repo(
                owner=username,
                repo_name=repo_name,
                directory_path=temp_dir,
                commit_message=commit_message,
                branch=default_branch,
                preserve_existing_files=reused_repo
            )
            if not push_results.get("commit_sha"):
                raise HTTPException(
                    status_code=500,
                    detail="Failed to create GitHub commit for generated files."
                )

            # Build deployment URLs. On a redeploy (existing_suffix was reused)
            # Render's blueprint webhook picks up the push automatically — the
            # user doesn't need to click "Create Blueprint" again. In that case
            # we also surface a direct link to the live frontend so the dialog
            # can jump straight there instead of sending them back through the
            # blueprint-creation flow.
            deployment_urls = github.get_deployment_urls(username, repo_name)
            is_first_deploy = existing_suffix is None
            app_host = repo_name.replace("_", "-")
            # Mirror the suffix fallback used inside ``_add_deployment_configs``:
            # if this was a first deploy we can't know the suffix the generator
            # rolled, so we only populate the live URL on redeploys where it's
            # stable. The frontend falls back to the ``render`` URL otherwise.
            if existing_suffix:
                deployment_urls["live_frontend"] = (
                    f"https://{app_host}-frontend-{existing_suffix}.onrender.com"
                )
                deployment_urls["live_backend"] = (
                    f"https://{app_host}-backend-{existing_suffix}.onrender.com"
                )
                # Point the user at the blueprint list so they can click into
                # their project's blueprint and hit "Manual Sync" — the single
                # reliable button that redeploys every service at once.
                deployment_urls["render_dashboard"] = "https://dashboard.render.com/blueprints"

            action = "updated" if reused_repo else "created"
            return DeployToGitHubResponse(
                success=True,
                repo_url=repo_url,
                repo_name=repo_name,
                owner=username,
                deployment_urls=deployment_urls,
                files_uploaded=push_results["total_files"],
                message=f"Successfully {action} repository and pushed {push_results['total_files']} files",
                is_first_deploy=is_first_deploy,
                deployment_type="webapp",
            )

    except HTTPException:
        raise
    except ValidationError as e:
        # Service-layer validation failures map to HTTP 400 with the original message.
        raise HTTPException(status_code=400, detail=str(e)) from e
    except GenerationError as e:
        # Service-layer generation/deployment failures map to HTTP 500 with the
        # original message so the frontend can surface the cause to the user.
        logger.exception("Generation error in deploy_webapp_to_github")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected error in deploy_webapp_to_github")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during GitHub deployment."
        )


def _sanitize_repo_name(name: str) -> str:
    """
    Sanitize repository name to meet GitHub requirements.

    - Lowercase
    - Replace spaces with hyphens
    - Remove special characters
    - Max 100 characters
    """
    # Convert to lowercase
    name = name.lower()

    # Replace spaces with hyphens
    name = name.replace(" ", "-")

    # Remove special characters (keep only alphanumeric, hyphens, underscores)
    name = "".join(c for c in name if c.isalnum() or c in "-_")

    # Remove leading/trailing hyphens
    name = name.strip("-_")

    # Limit length
    name = name[:100]

    # Ensure not empty
    if not name:
        name = f"besser-app-{uuid.uuid4().hex[:8]}"

    return name


def _normalize_deploy_target(raw_target: object) -> str:
    """Normalize deploy target values to ``webapp`` or ``agent``."""
    if not isinstance(raw_target, str):
        return "webapp"

    normalized = raw_target.strip().lower()
    if normalized in CHATBOT_DEPLOY_TARGETS:
        return "agent"
    return "webapp"


def _resolve_intent_recognition_technology(agent_config: object) -> str:
    """Read the intent-recognition technology from flat or structured config."""
    if not isinstance(agent_config, dict):
        return ""

    value = agent_config.get("intentRecognitionTechnology")
    if isinstance(value, str) and value.strip():
        return value.strip().lower()

    system_section = agent_config.get("system")
    if isinstance(system_section, dict):
        nested_value = system_section.get("intentRecognitionTechnology")
        if isinstance(nested_value, str) and nested_value.strip():
            return nested_value.strip().lower()

    return ""


def _write_personalization_user_profiles(directory: str, agent_config: object) -> None:
    """Emit user_profiles.json when personalization mappings are present."""
    if not isinstance(agent_config, dict):
        return

    personalization_mapping = agent_config.get("personalizationMapping")
    if not isinstance(personalization_mapping, list):
        return

    user_profiles = []
    for idx, entry in enumerate(personalization_mapping):
        if not isinstance(entry, dict):
            continue
        profile_data = entry.get("user_profile")
        profile_name = entry.get("name") or f"variant_{idx + 1}"
        if profile_data is not None:
            user_profiles.append(
                {
                    "name": profile_name,
                    "user_profile": profile_data,
                }
            )

    if not user_profiles:
        return

    profiles_path = os.path.join(directory, "user_profiles.json")
    with open(profiles_path, "w", encoding="utf-8") as profiles_file:
        json.dump(user_profiles, profiles_file, indent=2)


def _has_personalization(agent_config: object) -> bool:
    """Return True if the agent config carries a non-empty personalization mapping.

    Mirrors the condition the BAF generator uses to emit ``authenticate_users=True``
    on ``use_websocket_platform``, so the generated chatbot will try to connect to
    a Streamlit user DB whenever personalization is configured.
    """
    if not isinstance(agent_config, dict):
        return False
    mapping = agent_config.get("personalizationMapping")
    if not isinstance(mapping, list):
        return False
    return any(isinstance(entry, dict) for entry in mapping)


def _fit_chatbot_service_name(app_name: str, service_suffix: str) -> str:
    """Build a Render-safe service name for standalone chatbot deployments."""
    app_host = app_name.replace("_", "-")
    return _fit_service_name(
        f"{app_host}-chatbot-{service_suffix}",
        stable_key=f"{app_name}:chatbot",
    )


def _fit_chatbot_db_name(app_name: str, service_suffix: str) -> str:
    """Build a Render-safe database name for the chatbot's user/profile DB."""
    app_host = app_name.replace("_", "-")
    return _fit_service_name(
        f"{app_host}-userdb-{service_suffix}",
        stable_key=f"{app_name}:userdb",
    )


def _add_chatbot_deployment_configs(
    directory: str,
    app_name: str,
    agent_model,
    agent_config: object = None,
    service_suffix: str | None = None,
) -> None:
    """Add a Render blueprint for deploying one standalone chatbot service.

    When the agent is configured with a ``personalizationMapping``, the BAF
    generator emits ``authenticate_users=True`` on the websocket platform, which
    requires a PostgreSQL instance at runtime for login + profile persistence.
    In that case we also provision a Render-managed Postgres in the blueprint
    and wire its credentials into ``STREAMLIT_DB_*`` env vars on the web service
    so the deployed agent can connect without any manual setup.
    """
    if not service_suffix:
        service_suffix = uuid.uuid4().hex[:6]

    chatbot_service_name = _fit_chatbot_service_name(app_name, service_suffix)
    ic_tech = _resolve_intent_recognition_technology(agent_config)
    needs_user_db = _has_personalization(agent_config)
    db_name = _fit_chatbot_db_name(app_name, service_suffix) if needs_user_db else None

    if ic_tech == "classical":
        build_cmd = (
            "pip install torch==2.6.0+cpu scikit-learn==1.6.1 besser-agentic-framework[llms]==4.3.2 "
            "--extra-index-url https://download.pytorch.org/whl/cpu "
            '&& python -c "import nltk; nltk.download(\'punkt\', quiet=True); nltk.download(\'punkt_tab\', quiet=True)"'
        )
    else:
        build_cmd = (
            "pip install besser-agentic-framework[llms]==4.3.2 "
            '&& python -c "import nltk; nltk.download(\'punkt\', quiet=True); nltk.download(\'punkt_tab\', quiet=True)"'
        )

    # Bind Streamlit to Render's $PORT and populate db.streamlit.* from the
    # STREAMLIT_DB_* env vars Render injects from the managed Postgres. When no
    # DB env vars are present (e.g. no personalization), keep the DB disabled.
    yaml_patch = (
        "import yaml, os; "
        "c = yaml.safe_load(open('config.yaml')); "
        "c['platforms']['websocket']['host'] = '0.0.0.0'; "
        "c['platforms']['websocket']['port'] = 8765; "
        "ws = c['platforms']['websocket']; "
        "streamlit_cfg = ws.setdefault('streamlit', {}); "
        "streamlit_cfg['host'] = '0.0.0.0'; "
        "streamlit_cfg['port'] = int(os.environ['PORT']); "
        "c['db']['monitoring']['enabled'] = False; "
        "sdb = c['db'].setdefault('streamlit', {}); "
        f"db_host = os.environ.get('{STREAMLIT_DB_HOST_ENV}'); "
        "sdb['enabled'] = bool(db_host); "
        "sdb['dialect'] = os.environ.get('STREAMLIT_DB_DIALECT', 'postgresql'); "
        "sdb['host'] = db_host or 'localhost'; "
        "sdb['port'] = int(os.environ.get('STREAMLIT_DB_PORT') or 5432); "
        "sdb['database'] = os.environ.get('STREAMLIT_DB_NAME', ''); "
        "sdb['username'] = os.environ.get('STREAMLIT_DB_USER', ''); "
        "sdb['password'] = os.environ.get('STREAMLIT_DB_PASSWORD', ''); "
        "nlp_cfg = c.setdefault('nlp', {}); "
        "openai_cfg = nlp_cfg.setdefault('openai', {}); "
        f"api_key = os.environ.get('{OPENAI_API_KEY_ENV}'); "
        "openai_cfg['api_key'] = api_key if api_key else openai_cfg.get('api_key', ''); "
        "yaml.safe_dump(c, open('config.yaml','w'))"
    )
    # The BAFGenerator writes the agent file as f"{model.name}.py" with the
    # name normalized through safe_var_name(name, lowercase=False) (case
    # preserved, spaces and special chars → underscores). agent_slug() lowercases,
    # which mismatches the actual file on disk: e.g. "Greeting Agent" → file
    # "Greeting_Agent.py" but agent_slug → "greeting_agent" — Render then fails
    # to find the script. Mirror the BAFGenerator's transformation here.
    raw_agent_name = getattr(agent_model, "name", "agent")
    agent_script = f"{safe_var_name(raw_agent_name, lowercase=False)}.py"
    start_cmd = f'python -c "{yaml_patch}" && python -u "{agent_script}"'

    env_var_lines: list[str] = [
        "      - key: PYTHON_VERSION",
        f"        value: {PYTHON_VERSION}",
    ]
    if ic_tech != "classical":
        env_var_lines += [
            f"      - key: {OPENAI_API_KEY_ENV}",
            "        sync: false",
        ]
    if needs_user_db:
        for env_key, prop in (
            ("STREAMLIT_DB_HOST", "host"),
            ("STREAMLIT_DB_PORT", "port"),
            ("STREAMLIT_DB_NAME", "database"),
            ("STREAMLIT_DB_USER", "user"),
            ("STREAMLIT_DB_PASSWORD", "password"),
        ):
            env_var_lines += [
                f"      - key: {env_key}",
                "        fromDatabase:",
                f"          name: {db_name}",
                f"          property: {prop}",
            ]

    env_vars_block = "\n".join(env_var_lines)

    databases_block = ""
    if needs_user_db:
        databases_block = (
            "databases:\n"
            "  # Managed Postgres for Streamlit user auth + personalization profiles.\n"
            f"  - name: {db_name}\n"
            "    plan: free\n"
            "    postgresMajorVersion: \"16\"\n"
        )

    render_config = (
        f"{databases_block}"
        "services:\n"
        "  # Standalone chatbot service (WebSocket + Streamlit UI)\n"
        "  - type: web\n"
        f"    name: {chatbot_service_name}\n"
        "    runtime: python\n"
        "    plan: free\n"
        f"    buildCommand: {build_cmd}\n"
        f"    startCommand: {start_cmd}\n"
        "    envVars:\n"
        f"{env_vars_block}\n"
    )

    render_path = os.path.join(directory, "render.yaml")
    with open(render_path, "w", encoding="utf-8") as f:
        f.write(render_config)


async def _export_buml_files_to_repo(
    github, owner: str, repo: str, branch: str,
    json_file_path: str, project_data: dict, commit_message: str,
):
    """Export raw B-UML Python files alongside the project JSON in the repo.

    Non-blocking: logs warnings on failure but never raises.
    """
    import tempfile

    # Determine the buml directory next to the JSON file
    base_dir = json_file_path.rsplit("/", 1)[0] + "/" if "/" in json_file_path else ""
    buml_dir = f"{base_dir}buml"

    diagrams = project_data.get("diagrams", {})

    # Helper to get active diagram for a type
    current_indices = project_data.get("currentDiagramIndices", {})

    def _get_active(diagram_type: str):
        value = diagrams.get(diagram_type)
        if isinstance(value, list):
            idx = current_indices.get(diagram_type, 0)
            return value[idx] if 0 <= idx < len(value) else (value[0] if value else None)
        return value

    exports: list[tuple[str, str]] = []  # (repo_path, content)

    try:
        class_diagram_data = _get_active("ClassDiagram")
        if class_diagram_data and class_diagram_data.get("model"):
            buml_model = process_class_diagram(class_diagram_data)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
                domain_model_to_code(model=buml_model, file_path=f.name)
            with open(f.name, "r", encoding="utf-8") as fh:
                exports.append((f"{buml_dir}/domain_model.py", fh.read()))
            os.unlink(f.name)
    except Exception:
        logger.warning("Failed to export domain model B-UML", exc_info=True)

    try:
        # Walk every AgentDiagram (not just the active one) so a multi-agent project
        # exports one buml/agent_model_<slug>.py per agent.
        agent_diagrams_raw = diagrams.get("AgentDiagram")
        if isinstance(agent_diagrams_raw, list):
            agent_diagrams = [a for a in agent_diagrams_raw if a]
        elif agent_diagrams_raw:
            agent_diagrams = [agent_diagrams_raw]
        else:
            agent_diagrams = []

        for agent_data in agent_diagrams:
            if not (agent_data and agent_data.get("model", {}).get("elements")):
                continue
            agent_model = process_agent_diagram(agent_data)
            slug = agent_slug(agent_model.name)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
                agent_model_to_code(agent_model, f.name)
            with open(f.name, "r", encoding="utf-8") as fh:
                exports.append((f"{buml_dir}/agent_model_{slug}.py", fh.read()))
            os.unlink(f.name)
    except Exception:
        logger.warning("Failed to export agent model B-UML", exc_info=True)

    # Push each B-UML file to the repo
    for repo_path, content in exports:
        try:
            await github.create_or_update_file(
                owner=owner,
                repo_name=repo,
                file_path=repo_path,
                content=content,
                commit_message=f"{commit_message} [B-UML export]",
                branch=branch,
            )
        except Exception:
            logger.warning("Failed to push %s to GitHub", repo_path, exc_info=True)


# Render caps service hostnames at this many characters; longer names are
# silently truncated at dash boundaries, dropping the trailing suffix and
# breaking the URLs we bake into ``.env.production``.
_RENDER_HOSTNAME_MAX = 40

# Matches the ``name: <app>-backend-<suffix>`` line emitted by
# ``_add_deployment_configs`` so redeploys can recover the prior suffix and
# avoid minting a fresh set of Render services on every push. Also matches the
# ``besser-<hash>`` fallback form so overflow repos still reuse stable names.
_RENDER_BACKEND_NAME_RE = re.compile(
    r"^\s*name:\s*\S+-backend-([a-f0-9]{4,16})\s*$",
    re.MULTILINE,
)

_RENDER_CHATBOT_NAME_RE = re.compile(
    r"^\s*name:\s*\S+-chatbot-([a-f0-9]{4,16})\s*$",
    re.MULTILINE,
)


def _fit_service_name(readable: str, stable_key: str, max_length: int = _RENDER_HOSTNAME_MAX) -> str:
    """Return a Render-safe service name derived from ``readable``.

    Render caps service hostnames at 40 chars and truncates longer ones at
    dash boundaries — which silently eats the random suffix we rely on for
    uniqueness, so two agents in the same project end up at the same URL.

    When ``readable`` is over budget we fall back to ``besser-<12 hex chars>``
    where the hex is ``sha1(stable_key)``. ``stable_key`` must not include the
    per-deploy random suffix: it's a function of ``(app_name, role, slug)`` so
    the hashed name stays identical across redeploys of the same repo, which
    is what lets Render update the existing service in place.
    """
    if len(readable) <= max_length:
        return readable
    digest = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()[:12]
    return f"besser-{digest}"


async def _read_existing_service_suffix(github, owner: str, repo_name: str) -> Optional[str]:
    """Return the suffix used by the repo's current render.yaml, if any.

    Tries ``main`` then ``master`` as the default branch. Any failure
    (repo not found, render.yaml absent, parse miss, network error) returns
    ``None`` so the caller falls back to minting a fresh suffix.
    """
    for branch in ("main", "master"):
        try:
            result = await github.get_file_content(owner, repo_name, "render.yaml", branch=branch)
        except Exception:
            logger.debug(
                "[GitHub deploy] error reading render.yaml from %s/%s@%s",
                owner, repo_name, branch, exc_info=True,
            )
            continue
        if not result or not result.get("content"):
            continue
        match = _RENDER_BACKEND_NAME_RE.search(result["content"])
        if match:
            suffix = match.group(1)
            logger.info(
                "[GitHub deploy] reusing existing service suffix %s from %s/%s@%s",
                suffix, owner, repo_name, branch,
            )
            return suffix
    return None


async def _read_existing_chatbot_service_suffix(github, owner: str, repo_name: str) -> Optional[str]:
    """Return the suffix used by the repo's current chatbot service in render.yaml, if any."""
    for branch in ("main", "master"):
        try:
            result = await github.get_file_content(owner, repo_name, "render.yaml", branch=branch)
        except Exception:
            logger.debug(
                "[GitHub deploy] error reading render.yaml from %s/%s@%s",
                owner,
                repo_name,
                branch,
                exc_info=True,
            )
            continue
        if not result or not result.get("content"):
            continue
        match = _RENDER_CHATBOT_NAME_RE.search(result["content"])
        if match:
            suffix = match.group(1)
            logger.info(
                "[GitHub deploy] reusing existing chatbot suffix %s from %s/%s@%s",
                suffix,
                owner,
                repo_name,
                branch,
            )
            return suffix
    return None


def _has_agent_components(gui_model) -> bool:
    """Check if the GUI model contains any agent components."""
    from besser.BUML.metamodel.gui.dashboard import AgentComponent
    from besser.BUML.metamodel.gui import ViewContainer

    if not gui_model or not gui_model.modules:
        return False

    def _check_container(container) -> bool:
        for element in (container.view_elements or []):
            if isinstance(element, AgentComponent):
                return True
            if isinstance(element, ViewContainer) and _check_container(element):
                return True
        return False

    for module in gui_model.modules:
        for screen in (module.screens or []):
            if _check_container(screen):
                return True
    return False


def _add_deployment_configs(
    directory: str,
    app_name: str,
    agent_models: list | None = None,
    agent_configs: dict | None = None,
    service_suffix: str | None = None,
):
    """
    Add Render deployment configuration (only free platform for full-stack auto-deploy).

    Args:
        directory: Path to generated app directory
        app_name: Application name
        agent_models: List of BUML Agent models; one Render service block is emitted
            per entry, matching the ``agents/<slug>/`` directory layout produced by
            :class:`WebAppGenerator`.
        agent_configs: Mapping of ``agent.name`` -> config dict. Used to pick the
            intent-recognition technology per agent.
        service_suffix: Optional override for the 6-char unique suffix in service names.
    """
    agent_models = list(agent_models or [])
    agent_configs = dict(agent_configs or {})
    has_agent = bool(agent_models)

    # Render configuration - Full stack (backend + frontend + optional agents).
    # ``service_suffix`` should be a stable token per repo so redeploys update
    # the existing services instead of creating a fresh set every time.
    # Callers are expected to pass a reused suffix extracted from the repo's
    # previous render.yaml; if they don't, we fall back to a random UUID here
    # (first-deploy path — no preexisting Render hostname to collide with).
    if not service_suffix:
        service_suffix = uuid.uuid4().hex[:6]
    # Render rewrites underscores to dashes in service hostnames, so we do the
    # same conversion here — otherwise the ``VITE_API_URL`` / ``VITE_AGENT_URLS``
    # values baked into ``.env.production`` point at hostnames that don't resolve.
    app_host = app_name.replace("_", "-")
    backend_service_name = _fit_service_name(
        f"{app_host}-backend-{service_suffix}",
        stable_key=f"{app_name}:backend",
    )
    frontend_service_name = _fit_service_name(
        f"{app_host}-frontend-{service_suffix}",
        stable_key=f"{app_name}:frontend",
    )

    render_config = f"""services:
  # Backend API (Free tier - 750 hours/month, spins down after 15 min idle)
  - type: web
    name: {backend_service_name}
    runtime: python
    plan: free
    buildCommand: pip install -r backend/requirements.txt
    startCommand: cd backend && uvicorn main_api:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: {PYTHON_VERSION}

  # Frontend (Free static site)
  - type: web
    name: {frontend_service_name}
    runtime: static
    buildCommand: cd frontend && npm install && npm run build
    staticPublishPath: frontend/build
    envVars:
      - key: VITE_API_URL
        value: https://{backend_service_name}.onrender.com
    headers:
      - path: /assets/*.js
        name: Content-Type
        value: application/javascript
      - path: /assets/*.css
        name: Content-Type
        value: text/css
    routes:
      - type: rewrite
        source: /*
        destination: /index.html
"""

    # Build one Render service block per agent. Each agent lives under ``agents/<slug>/``
    # in the generated repo, so the startCommand cd's into that directory.
    #
    # Slug vs hostname: Render silently rewrites underscores to dashes in service
    # hostnames (e.g. ``my_agent`` → ``my-agent.onrender.com``). If we put an
    # underscore-slug straight into ``name:`` the DNS record ends up on a dashed
    # URL while our generated ``VITE_AGENT_URLS`` map still points at the
    # underscored one — every WebSocket connection then fails even though the
    # service is happily running. So we keep the underscore slug for the
    # filesystem layout (Python-friendly directory / script names) and build a
    # separate dash-slug for the Render service name and the URLs we hand to the
    # React frontend.
    agent_service_names: list[str] = []
    for agent in agent_models:
        slug = agent_slug(agent.name)           # underscores, filesystem-safe
        host_slug = slug.replace("_", "-")               # dashes, Render-hostname-safe
        agent_service_name = _fit_service_name(
            f"{app_host}-{host_slug}-agent-{service_suffix}",
            stable_key=f"{app_name}:agent:{slug}",
        )
        agent_service_names.append(agent_service_name)
        # The BAFGenerator writes f"{model.name}.py" with case preserved
        # (spaces → underscores, special chars → underscores). agent_slug
        # lowercases, which would mismatch the on-disk filename. Use
        # safe_var_name(..., lowercase=False) to match the BAFGenerator output.
        agent_script = f"{safe_var_name(agent.name, lowercase=False)}.py"

        cfg = agent_configs.get(agent.name) or {}
        ic_tech = cfg.get("intentRecognitionTechnology") if isinstance(cfg, dict) else None

        if ic_tech == "classical":
            # Classical IC needs PyTorch CPU + scikit-learn, plus [llms] for unconditional imports in generated code
            # Single pip call with --extra-index-url so PyTorch CPU resolves from the wheel index while the rest comes from PyPI.
            # BAFGenerator does NOT emit a requirements.txt, so everything has to be pinned inline here.
            build_cmd = (
                "pip install torch==2.6.0+cpu scikit-learn==1.6.1 besser-agentic-framework[llms]==4.3.2 "
                "--extra-index-url https://download.pytorch.org/whl/cpu "
                '&& python -c "import nltk; nltk.download(\'punkt\', quiet=True); nltk.download(\'punkt_tab\', quiet=True)"'
            )
        else:
            # LLM-based IC (default)
            build_cmd = (
                "pip install besser-agentic-framework[llms]==4.3.2 "
                '&& python -c "import nltk; nltk.download(\'punkt\', quiet=True); nltk.download(\'punkt_tab\', quiet=True)"'
            )

        # Patch config.yaml + agent script at container start.
        # - Bind websocket to 0.0.0.0:$PORT so Render's port scanner sees it
        # - Disable monitoring + streamlit DB (no external DB on free tier, template ships with YOUR-DB-HOST placeholder)
        # - Inject OPENAI_API_KEY into config.yaml only when present (classical IC path has no env var)
        # - Flip use_ui=True → use_ui=False in the agent script so the embedded Streamlit thread doesn't spin up
        #   (unreachable on free tier since only $PORT is exposed, and it wastes RAM on the 512 MB instance).
        yaml_patch = (
            "import yaml, os, re, pathlib; "
            "c = yaml.safe_load(open('config.yaml')); "
            "c['platforms']['websocket']['host'] = '0.0.0.0'; "
            "c['platforms']['websocket']['port'] = int(os.environ['PORT']); "
            "c['db']['monitoring']['enabled'] = False; "
            "c['db']['streamlit']['enabled'] = False; "
            f"c['nlp']['openai']['api_key'] = os.environ.get('{OPENAI_API_KEY_ENV}') or c['nlp']['openai'].get('api_key', ''); "
            "yaml.safe_dump(c, open('config.yaml','w')); "
            f"p = pathlib.Path('{agent_script}'); "
            "p.write_text(re.sub(r'use_ui=True', 'use_ui=False', p.read_text()))"
        )
        start_cmd = f'cd agents/{slug} && python -c "{yaml_patch}" && python -u "{agent_script}"'
        extra_env_vars = (
            ""
            if ic_tech == "classical"
            else f"\n      - key: {OPENAI_API_KEY_ENV}\n        sync: false"
        )

        render_config += f"""
  # Agent Service: {agent.name} (Free tier - WebSocket-based AI agent)
  - type: web
    name: {agent_service_name}
    runtime: python
    plan: free
    buildCommand: {build_cmd}
    startCommand: {start_cmd}
    envVars:
      - key: PYTHON_VERSION
        value: {PYTHON_VERSION}{extra_env_vars}
"""

    render_path = os.path.join(directory, "render.yaml")
    with open(render_path, "w") as f:
        f.write(render_config)

    # Build a per-agent URL map keyed by the BUML ``agent.name`` (Python-safe
    # identifier, e.g. "Library_Agent"). The generated React AgentComponent
    # normalizes its ``agent-name`` prop with a single space→underscore pass
    # before lookup, so one canonical key per agent is enough.
    # ``VITE_AGENT_URL`` stays as a back-compat pointer at the first agent.
    prod_agent_urls: dict[str, str] = {}
    local_agent_urls: dict[str, str] = {}
    for i, (agent, svc_name) in enumerate(zip(agent_models, agent_service_names)):
        prod_agent_urls[agent.name] = f"wss://{svc_name}.onrender.com"
        local_agent_urls[agent.name] = f"ws://localhost:{8765 + i}"

    env_production = f"""# Production environment variables
# Backend API URL - Update this with your deployed backend URL
VITE_API_URL=https://{backend_service_name}.onrender.com
"""

    if has_agent:
        first_agent = agent_models[0].name
        # Wrap the JSON map in single quotes so dotenv treats the whole thing as
        # a literal string regardless of the inner double quotes in the JSON.
        env_production += f"""# Agent WebSocket URLs
VITE_AGENT_URL={prod_agent_urls[first_agent]}
VITE_AGENT_URLS='{json.dumps(prod_agent_urls)}'
"""

    env_production += """
# For local development, use: http://localhost:8000 (backend) and ws://localhost:8765+ (agents)
"""

    env_prod_path = os.path.join(directory, "frontend", ".env.production")
    with open(env_prod_path, "w") as f:
        f.write(env_production)

    # Create .env for local development
    env_local = """# Local development environment variables
VITE_API_URL=http://localhost:8000
"""

    if has_agent:
        first_agent = agent_models[0].name
        env_local += f"""VITE_AGENT_URL={local_agent_urls[first_agent]}
VITE_AGENT_URLS='{json.dumps(local_agent_urls)}'
"""

    env_local_path = os.path.join(directory, "frontend", ".env")
    with open(env_local_path, "w") as f:
        f.write(env_local)


# ============================================================================
# GitHub Project Storage API
# ============================================================================

class GitHubRepoResponse(BaseModel):
    """Repository information response."""
    id: int
    name: str
    full_name: str
    description: Optional[str]
    html_url: str
    private: bool
    updated_at: str
    default_branch: str


class GitHubReposListResponse(BaseModel):
    """Response for list of repositories."""
    repositories: List[GitHubRepoResponse]


class GitHubCommitResponse(BaseModel):
    """Commit information response."""
    sha: str
    message: str
    author: str
    date: str
    html_url: str


class GitHubCommitsListResponse(BaseModel):
    """Response for list of commits."""
    commits: List[GitHubCommitResponse]


class SaveProjectRequest(BaseModel):
    """Request to save project to GitHub."""
    owner: str
    repo: str
    branch: str = "main"
    file_path: str = "besser-project.json"
    commit_message: str
    project_data: dict


class SaveProjectResponse(BaseModel):
    """Response for project save."""
    success: bool
    commit_sha: str
    message: str


class LoadProjectResponse(BaseModel):
    """Response for project load."""
    success: bool
    project: dict
    sha: str


class LoadProjectFromCommitResponse(BaseModel):
    """Response for loading project from specific commit."""
    success: bool
    project: dict
    commit_sha: str
    commit_message: str
    commit_date: str


class CreateRepoForProjectRequest(BaseModel):
    """Request to create a new repo and save project."""
    repo_name: str
    description: str = ""
    is_private: bool = False
    project_data: dict
    file_path: str = "besser-project.json"


class CreateRepoForProjectResponse(BaseModel):
    """Response for creating repo with project."""
    success: bool
    repo_url: str
    owner: str
    commit_sha: str
    message: str


@router.get("/repos", response_model=GitHubReposListResponse)
async def get_user_repositories(
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session")
):
    """
    Get list of repositories for the authenticated user.

    Returns only repositories owned by the user, sorted by last updated.
    """
    if not github_session:
        raise HTTPException(status_code=401, detail="GitHub authentication required")

    access_token = get_user_token(github_session)
    if not access_token:
        raise HTTPException(status_code=401, detail="GitHub session expired")

    try:
        github = create_github_service(access_token)
        repos = await github.get_user_repositories()

        return GitHubReposListResponse(
            repositories=[GitHubRepoResponse(**r) for r in repos]
        )
    except Exception:
        logger.exception("Unexpected error fetching GitHub repositories")
        raise HTTPException(status_code=500, detail="An internal error occurred while fetching repositories.")


class GitHubBranchesListResponse(BaseModel):
    """Response for list of branches."""
    branches: List[str]


@router.get("/branches", response_model=GitHubBranchesListResponse)
async def get_repository_branches(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session")
):
    """
    Get list of branches for a repository.
    """
    if not github_session:
        raise HTTPException(status_code=401, detail="GitHub authentication required")

    access_token = get_user_token(github_session)
    if not access_token:
        raise HTTPException(status_code=401, detail="GitHub session expired")

    try:
        github = create_github_service(access_token)
        branches = await github.get_branches(owner, repo)

        return GitHubBranchesListResponse(branches=branches)
    except Exception:
        logger.exception("Unexpected error fetching GitHub branches")
        raise HTTPException(status_code=500, detail="An internal error occurred while fetching branches.")


class FileExistsResponse(BaseModel):
    """Response for file existence check."""
    exists: bool
    path: str


@router.get("/file/exists", response_model=FileExistsResponse)
async def check_file_exists(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    branch: str = Query("main", description="Branch name"),
    file_path: str = Query(..., description="File path to check"),
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session")
):
    """
    Check if a file exists in a GitHub repository.
    """
    if not github_session:
        raise HTTPException(status_code=401, detail="GitHub authentication required")

    access_token = get_user_token(github_session)
    if not access_token:
        raise HTTPException(status_code=401, detail="GitHub session expired")

    try:
        github = create_github_service(access_token)
        exists = await github.file_exists(owner, repo, file_path, branch)

        return FileExistsResponse(exists=exists, path=file_path)
    except Exception:
        logger.exception("Unexpected error checking file existence on GitHub")
        raise HTTPException(status_code=500, detail="An internal error occurred while checking file existence.")



class GitHubContentItem(BaseModel):
    """Item in a repository directory."""
    name: str
    path: str
    type: str  # "file" or "dir"
    size: Optional[int] = 0
    sha: str


class GitHubContentsResponse(BaseModel):
    """Response for repository contents."""
    contents: List[GitHubContentItem]


@router.get("/contents", response_model=GitHubContentsResponse)
async def get_repository_contents(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    path: str = Query("", description="Path to directory"),
    branch: str = Query("main", description="Branch name"),
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session")
):
    """
    Get contents of a repository path.
    """
    if not github_session:
        raise HTTPException(status_code=401, detail="GitHub authentication required")

    access_token = get_user_token(github_session)
    if not access_token:
        raise HTTPException(status_code=401, detail="GitHub session expired")

    try:
        github = create_github_service(access_token)
        contents = await github.get_repository_contents(owner, repo, path, branch)

        return GitHubContentsResponse(
            contents=[
                GitHubContentItem(
                    name=item.get("name"),
                    path=item.get("path"),
                    type=item.get("type"),
                    size=item.get("size", 0),
                    sha=item.get("sha")
                )
                for item in contents
            ]
        )
    except Exception:
        logger.exception("Unexpected error fetching GitHub repository contents")
        raise HTTPException(status_code=500, detail="An internal error occurred while fetching repository contents.")


@router.get("/commits", response_model=GitHubCommitsListResponse)
async def get_repository_commits(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    path: Optional[str] = Query(None, description="File path to filter commits"),
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session")
):
    """
    Get commit history for a repository or specific file.
    """
    if not github_session:
        raise HTTPException(status_code=401, detail="GitHub authentication required")

    access_token = get_user_token(github_session)
    if not access_token:
        raise HTTPException(status_code=401, detail="GitHub session expired")

    try:
        github = create_github_service(access_token)
        commits = await github.get_commits(owner, repo, path)

        return GitHubCommitsListResponse(
            commits=[GitHubCommitResponse(**c) for c in commits]
        )
    except Exception:
        logger.exception("Unexpected error fetching GitHub commits")
        raise HTTPException(status_code=500, detail="An internal error occurred while fetching commits.")


@router.post("/project/save", response_model=SaveProjectResponse)
async def save_project_to_github(
    request: SaveProjectRequest,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session")
):
    """
    Save a BESSER project to a GitHub repository.

    Creates or updates the project JSON file in the specified repository.
    """
    if not github_session:
        raise HTTPException(status_code=401, detail="GitHub authentication required")

    access_token = get_user_token(github_session)
    if not access_token:
        raise HTTPException(status_code=401, detail="GitHub session expired")

    try:
        github = create_github_service(access_token)

        # Convert project data to JSON string
        project_json = json.dumps(request.project_data, indent=2)

        # Save to GitHub
        result = await github.create_or_update_file(
            owner=request.owner,
            repo_name=request.repo,
            file_path=request.file_path,
            content=project_json,
            commit_message=request.commit_message,
            branch=request.branch
        )

        # Also export raw B-UML models alongside the JSON
        await _export_buml_files_to_repo(
            github, request.owner, request.repo, request.branch,
            request.file_path, request.project_data, request.commit_message,
        )

        return SaveProjectResponse(
            success=True,
            commit_sha=result.get("commit_sha", ""),
            message="Project saved successfully"
        )
    except Exception:
        logger.exception("Unexpected error saving project to GitHub")
        raise HTTPException(status_code=500, detail="An internal error occurred while saving the project.")


@router.get("/project/load", response_model=LoadProjectResponse)
async def load_project_from_github(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    branch: str = Query("main", description="Branch name"),
    file_path: str = Query("besser-project.json", description="Project file path"),
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session")
):
    """
    Load a BESSER project from a GitHub repository.
    """
    if not github_session:
        raise HTTPException(status_code=401, detail="GitHub authentication required")

    access_token = get_user_token(github_session)
    if not access_token:
        raise HTTPException(status_code=401, detail="GitHub session expired")

    try:
        github = create_github_service(access_token)

        # Get file content
        file_data = await github.get_file_content(owner, repo, file_path, branch)

        if not file_data:
            raise HTTPException(
                status_code=404,
                detail=f"Project file not found: {file_path}"
            )

        # Parse JSON content
        project = json.loads(file_data.get("content", "{}"))

        return LoadProjectResponse(
            success=True,
            project=project,
            sha=file_data.get("sha", "")
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid project file format")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error loading project from GitHub")
        raise HTTPException(status_code=500, detail="An internal error occurred while loading the project.")


@router.get("/project/load-commit", response_model=LoadProjectFromCommitResponse)
async def load_project_from_commit(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    commit_sha: str = Query(..., description="Commit SHA to load from"),
    file_path: str = Query("besser-project.json", description="Project file path"),
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session")
):
    """
    Load a BESSER project from a specific commit in a GitHub repository.

    This allows restoring previous versions of the project.
    """
    if not github_session:
        raise HTTPException(status_code=401, detail="GitHub authentication required")

    access_token = get_user_token(github_session)
    if not access_token:
        raise HTTPException(status_code=401, detail="GitHub session expired")

    try:
        github = create_github_service(access_token)

        # Get file content at specific commit
        file_data = await github.get_file_content(owner, repo, file_path, ref=commit_sha)

        if not file_data:
            raise HTTPException(
                status_code=404,
                detail=f"Project file not found at commit {commit_sha[:7]}"
            )

        # Parse JSON content
        project = json.loads(file_data.get("content", "{}"))

        # Get commit info for display
        commits = await github.get_commits(owner, repo, file_path, per_page=100)
        commit_info = next((c for c in commits if c["sha"] == commit_sha), None)

        return LoadProjectFromCommitResponse(
            success=True,
            project=project,
            commit_sha=commit_sha,
            commit_message=commit_info.get("message", "") if commit_info else "",
            commit_date=commit_info.get("date", "") if commit_info else ""
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid project file format")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error loading project from GitHub commit")
        raise HTTPException(status_code=500, detail="An internal error occurred while loading the project from commit.")


@router.post("/project/create-repo", response_model=CreateRepoForProjectResponse)
async def create_repository_for_project(
    request: CreateRepoForProjectRequest,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session")
):
    """
    Create a new GitHub repository and save the project to it.

    This is a convenience endpoint that:
    1. Creates a new repository in the user's account
    2. Saves the project JSON as the initial commit
    3. Creates a README.md explaining the project
    """
    if not github_session:
        raise HTTPException(status_code=401, detail="GitHub authentication required")

    access_token = get_user_token(github_session)
    if not access_token:
        raise HTTPException(status_code=401, detail="GitHub session expired")

    try:
        github = create_github_service(access_token)

        # Get user info
        user_info = await github.get_authenticated_user()
        username = user_info.get("login")

        # Sanitize repo name
        repo_name = _sanitize_repo_name(request.repo_name)

        # Create repository
        repo_info = await github.create_repository(
            repo_name=repo_name,
            description=request.description or "BESSER project repository",
            is_private=request.is_private
        )

        # Save project to repo
        project_json = json.dumps(request.project_data, indent=2)
        file_path = request.file_path if hasattr(request, 'file_path') and request.file_path else "besser-project.json"
        save_result = await github.create_or_update_file(
            owner=username,
            repo_name=repo_name,
            file_path=file_path,
            content=project_json,
            commit_message=f"Initial commit: {request.project_data.get('name', 'BESSER Project')}",
            branch="main"
        )

        # Create README
        project_name = request.project_data.get("name", "BESSER Project")
        readme_content = f"""# {project_name}

BESSER project repository.

## About

This repository contains a [BESSER](https://github.com/BESSER-PEARL/BESSER) project.

## Files

- `{file_path}` - The project definition file

## Opening the Project

1. Go to [BESSER Web Editor](https://editor.besser-pearl.org/)
2. Connect your GitHub account
3. Link this repository to load the project

## Learn More

- [BESSER GitHub](https://github.com/BESSER-PEARL/BESSER)
- [Documentation](https://besser.readthedocs.io/)

---
*Generated by BESSER Web Modeling Editor*
"""
        await github.create_or_update_file(
            owner=username,
            repo_name=repo_name,
            file_path="README.md",
            content=readme_content,
            commit_message="Add README",
            branch="main"
        )

        return CreateRepoForProjectResponse(
            success=True,
            repo_url=repo_info.get("html_url", ""),
            owner=username,
            commit_sha=save_result.get("commit_sha", ""),
            message=f"Repository '{repo_name}' created with project"
        )
    except Exception:
        logger.exception("Unexpected error creating GitHub repository for project")
        raise HTTPException(status_code=500, detail="An internal error occurred while creating the repository.")


# ============================================================================
# GitHub Gist API
# ============================================================================

class CreateGistRequest(BaseModel):
    """Request to create a GitHub Gist."""
    project_data: dict
    description: str = ""
    is_public: bool = False


class CreateGistResponse(BaseModel):
    """Response for Gist creation."""
    success: bool
    gist_url: str
    gist_id: str
    message: str


@router.post("/gist/create", response_model=CreateGistResponse)
async def create_gist_from_project(
    request: CreateGistRequest,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session")
):
    """
    Create a GitHub Gist from project data.

    This creates a shareable Gist containing the project JSON.
    Useful for quick sharing without creating a full repository.
    """
    if not github_session:
        raise HTTPException(status_code=401, detail="GitHub authentication required")

    access_token = get_user_token(github_session)
    if not access_token:
        raise HTTPException(status_code=401, detail="GitHub session expired")

    try:
        github = create_github_service(access_token)

        # Convert project to JSON
        project_json = json.dumps(request.project_data, indent=2)

        # Get project name for filename
        project_name = request.project_data.get("name", "besser-project")
        filename = f"{project_name.lower().replace(' ', '-')}.json"

        # Create gist
        result = await github.create_gist(
            content=project_json,
            filename=filename,
            description=request.description or f"BESSER Project: {project_name}",
            is_public=request.is_public
        )

        return CreateGistResponse(
            success=True,
            gist_url=result.get("gist_url", ""),
            gist_id=result.get("gist_id", ""),
            message="Gist created successfully"
        )
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code if e.response is not None else 500
        # GitHub returns 404/403 when the token lacks the "gist" scope or gists are disabled
        if status_code in (403, 404):
            detail = "GitHub token is missing access to gists. Please reconnect GitHub and grant the 'gist' scope."
        else:
            detail = f"GitHub API error while creating Gist: {str(e)}"
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception:
        logger.exception("Unexpected error creating GitHub Gist")
        raise HTTPException(status_code=500, detail="An internal error occurred while creating the Gist.")

