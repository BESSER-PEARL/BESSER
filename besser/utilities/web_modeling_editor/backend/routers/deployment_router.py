"""
Deployment Router

Handles deployment and feedback endpoints for the BESSER web modeling editor backend.
"""

import logging
import os
import shutil
import tempfile
import uuid
import asyncio

from fastapi import APIRouter, HTTPException

# Backend models
from besser.utilities.web_modeling_editor.backend.models import (
    DiagramInput,
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
    cleanup_temp_resources,
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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/besser_api", tags=["deployment"])


@router.post("/deploy-app")
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
    temp_dir = tempfile.mkdtemp(prefix=f'{TEMP_DIR_PREFIX}{uuid.uuid4().hex}_')
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
            deploy_config = input_data.config if input_data.config else {}
            project_name = deploy_config.get('project_name', DEFAULT_DJANGO_PROJECT_NAME)
            app_name = deploy_config.get('app_name', DEFAULT_DJANGO_APP_NAME)

            # Clean up any existing project directory first
            project_dir = os.path.join(temp_dir, project_name)
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

            full_path = os.path.join(temp_dir, project_name)
            await asyncio.to_thread(run_docker_compose, directory=full_path, project_name=project_name)

            cleanup_temp_resources(temp_dir)

        else:
            raise ValueError("Deployment only possible for Django projects")
    except HTTPException as e:
        # Handle known exceptions with specific status codes
        cleanup_temp_resources(temp_dir)
        raise e
    except Exception as e:
        cleanup_temp_resources(temp_dir)
        logger.error("Error during file generation or django deployment: %s", str(e))
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@router.post("/feedback")
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
    try:
        return submit_feedback(feedback)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
