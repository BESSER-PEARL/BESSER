"""
Deployment Router

Handles deployment and feedback endpoints for the BESSER web modeling editor backend.
"""

import logging
import os
import re
import shutil
import tempfile
import uuid
import asyncio
import ipaddress

from fastapi import APIRouter, HTTPException, Request

# Backend models
from besser.utilities.web_modeling_editor.backend.models import (
    DiagramInput,
    FeedbackResponse,
    FeedbackSubmission,
)

# Backend services - Converters
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_class_diagram,
)

# Backend services - Other services
from besser.utilities.web_modeling_editor.backend.services.feedback_service import (
    submit_feedback,
)
from besser.utilities.web_modeling_editor.backend.services.deployment import (
    run_docker_compose,
)
from besser.utilities.web_modeling_editor.backend.services.utils import (
    safe_path as _safe_path,
    validate_generator,
)

# Backend configuration
from besser.utilities.web_modeling_editor.backend.config import (
    SUPPORTED_GENERATORS,
)

# Backend constants
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    TEMP_DIR_PREFIX,
    DEFAULT_DJANGO_PROJECT_NAME,
    DEFAULT_DJANGO_APP_NAME,
)

from besser.utilities.web_modeling_editor.backend.dependencies import SettingsDep

# Centralized error handling
from besser.utilities.web_modeling_editor.backend.routers.error_handler import (
    handle_endpoint_errors,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/besser_api", tags=["deployment"])

# A reverse proxy in front of the backend sets one of these; the request then
# comes from someone else even when the proxy connects over loopback.
_FORWARDING_HEADERS = ("forwarded", "x-forwarded-for", "x-real-ip")


def _is_loopback(address) -> bool:
    try:
        return ipaddress.ip_address(address[0]).is_loopback
    except (TypeError, ValueError, IndexError):
        return False


def _is_local_request(request: Request) -> bool:
    """True when the browser runs on the backend's own machine: a loopback
    connection on both ends, not relayed by a proxy."""
    return (_is_loopback(request.client)
            and _is_loopback(request.scope.get("server"))
            and not any(name in request.headers for name in _FORWARDING_HEADERS))


@router.post("/deploy-app")
@handle_endpoint_errors("deploy_app")
async def deploy_app(input_data: DiagramInput, request: Request, settings: SettingsDep):
    """
    Deploy application using Docker Compose.

    Args:
        input_data: DiagramInput containing diagram data and generator type

    Returns:
        dict: Deployment status and information

    Raises:
        HTTPException: If deployment fails
    """
    # This builds and starts containers on the backend's own Docker daemon,
    # unauthenticated, so only a single-user local install may use it. An explicit
    # BESSER_ENABLE_DOCKER_DEPLOY wins; unset, only local requests are served.
    # (The docstring is in the editor's OpenAPI schema.)
    enabled = settings.docker_deploy_enabled
    if enabled is None:
        enabled = _is_local_request(request)
    if not enabled:
        raise HTTPException(
            status_code=403,
            detail="Docker deployment is disabled on this server (set BESSER_ENABLE_DOCKER_DEPLOY to enable it).",
        )
    with tempfile.TemporaryDirectory(prefix=f'{TEMP_DIR_PREFIX}{uuid.uuid4().hex}_') as temp_dir:
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
            deploy_config = input_data.config if input_data.config else {}
            project_name = deploy_config.get('project_name', DEFAULT_DJANGO_PROJECT_NAME)
            app_name = deploy_config.get('app_name', DEFAULT_DJANGO_APP_NAME)

            # Validate project_name to prevent path traversal
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_-]*$', project_name):
                raise HTTPException(status_code=400, detail="Invalid project name")
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', app_name):
                raise HTTPException(status_code=400, detail="Invalid app name")

            # Sanitize and resolve project directory safely within temp_dir
            project_dir = _safe_path(temp_dir, project_name)
            if os.path.exists(project_dir):
                shutil.rmtree(project_dir)

            # Create a fresh project directory
            os.makedirs(temp_dir, exist_ok=True)

            generator_instance = generator_class(
                model=buml_model,
                project_name=project_name,
                app_name=app_name,
                containerization=True,
                output_dir=temp_dir
            )

            # Generate the Django project
            await asyncio.to_thread(generator_instance.generate)

            # Check if the project directory exists and has content
            if not os.path.exists(project_dir) or not os.listdir(project_dir):
                raise ValueError("Django project generation failed: Output directory is empty")

            await asyncio.to_thread(run_docker_compose, directory=project_dir, project_name=project_name)
            return {"status": "deployed", "project_name": project_name}

        else:
            raise ValueError("Deployment only possible for Django projects")


@router.post("/feedback", response_model=FeedbackResponse)
@handle_endpoint_errors("feedback_endpoint")
async def feedback_endpoint(feedback: FeedbackSubmission):
    """
    Feedback submission endpoint.

    Receives user feedback and processes it via the feedback service.
    See feedback_service.py for implementation details.

    Args:
        feedback: FeedbackSubmission model containing user feedback data

    Returns:
        dict: Status message confirming feedback receipt

    Raises:
        HTTPException: If feedback processing fails
    """
    return submit_feedback(feedback)
