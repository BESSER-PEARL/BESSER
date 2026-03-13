"""
BESSER Backend API

This module provides FastAPI endpoints for the BESSER web modeling editor backend.
It handles code generation from UML diagrams using various generators.

The editor is available at: https://editor.besser-pearl.org
"""

# Standard library imports
import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse

from besser.utilities.web_modeling_editor.backend.middleware import setup_middleware

# Backend constants (no circular dependency risk)
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    API_VERSION,
    DEFAULT_CORS_ORIGINS,
)

# Pre-load services.converters before config to avoid circular import:
# config -> baf_generator -> services.converters -> services/__init__ -> deployment -> config
import besser.utilities.web_modeling_editor.backend.services.converters  # noqa: F401

# Backend configuration (safe now that converters are loaded)
from besser.utilities.web_modeling_editor.backend.config import (
    SUPPORTED_GENERATORS,
)

# Backend response models
from besser.utilities.web_modeling_editor.backend.models.responses import (
    ApiInfoResponse,
)

# Backend exceptions
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
    ValidationError,
    GenerationError,
)

# Temporary file cleanup
from besser.utilities.web_modeling_editor.backend.services.cleanup import (
    cleanup_old_temp_files,
    schedule_cleanup,
)

# Deployment routers (already separate files)
from besser.utilities.web_modeling_editor.backend.services.deployment import (
    github_oauth_router,
    github_deploy_router,
)

# Application routers
from besser.utilities.web_modeling_editor.backend.routers import (
    generation_router,
    conversion_router,
    validation_router,
    deployment_router,
)

logger = logging.getLogger(__name__)

MAX_REQUEST_SIZE = 50 * 1024 * 1024  # 50MB


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: blob:; "
            "connect-src 'self' https://api.github.com https://github.com; "
            "frame-ancestors 'none'"
        )
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests that exceed MAX_REQUEST_SIZE."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_SIZE:
            return StarletteResponse("Request too large", status_code=413)

        # For requests without content-length or to prevent spoofing,
        # wrap the body stream to enforce the limit
        if request.method in ("POST", "PUT", "PATCH"):
            body = await request.body()
            if len(body) > MAX_REQUEST_SIZE:
                return StarletteResponse("Request too large", status_code=413)

        return await call_next(request)


# ---------------------------------------------------------------------------
# Startup: environment variable validation
# ---------------------------------------------------------------------------

_REQUIRED_ENV_VARS = [
    "GITHUB_CLIENT_ID",
    "GITHUB_CLIENT_SECRET",
]

_OPTIONAL_ENV_VARS = [
    "SMTP_PASSWORD",
    "OPENAI_API_KEY",
]


def _validate_env_vars() -> None:
    """Check that expected environment variables are set and log diagnostics.

    Required variables trigger ``logger.error`` when absent; optional ones
    trigger ``logger.warning``.  The application is **not** terminated so
    that services which do not depend on the missing variables can still
    operate (graceful degradation).
    """
    for var in _REQUIRED_ENV_VARS:
        if not os.environ.get(var):
            logger.error(
                "Required environment variable %s is not set. "
                "Features depending on it will not work.", var,
            )

    for var in _OPTIONAL_ENV_VARS:
        if not os.environ.get(var):
            logger.warning(
                "Optional environment variable %s is not set. "
                "Related functionality will be unavailable.", var,
            )


@asynccontextmanager
async def lifespan(_: FastAPI):
    _validate_env_vars()

    # Run an immediate cleanup of stale temp files left from previous runs,
    # then schedule a background task that repeats every hour.
    cleanup_old_temp_files()
    cleanup_task = schedule_cleanup()

    yield

    # Cancel the periodic cleanup task on shutdown.
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


# Initialize FastAPI application
app = FastAPI(
    title="BESSER Backend API",
    description="Backend services for web modeling editor",
    version=API_VERSION,
    lifespan=lifespan,
)

# Security middlewares (applied before CORS so headers are set on all responses)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestSizeLimitMiddleware)

# Configure CORS middleware
_cors_env = os.environ.get("CORS_ORIGINS", "")
ALLOWED_ORIGINS = [origin.strip() for origin in _cors_env.split(",") if origin.strip()] if _cors_env else DEFAULT_CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-GitHub-Session", "Content-Disposition", "Authorization"],
    expose_headers=["Content-Disposition"],
)

# Request logging middleware (outermost – added last so it wraps everything)
setup_middleware(app)


# Include GitHub OAuth and deployment routers
app.include_router(github_oauth_router, prefix="/besser_api")
app.include_router(github_deploy_router, prefix="/besser_api")

# Include application routers
app.include_router(generation_router.router)
app.include_router(conversion_router.router)
app.include_router(validation_router.router)
app.include_router(deployment_router.router)


# Exception handlers
@app.exception_handler(ConversionError)
async def conversion_error_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ValidationError)
async def validation_error_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(GenerationError)
async def generation_error_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check for load balancers and monitoring."""
    logger.debug("Health check requested")
    return {"status": "ok"}


# API Root Endpoint
@app.get("/besser_api/", response_model=ApiInfoResponse)
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
            "generate_from_project": "/besser_api/generate-output-from-project",
            "deploy": "/besser_api/deploy-app",
            "export_buml": "/besser_api/export-buml",
            "export_project": "/besser_api/export-project-as-buml",
            "get_project_json_model": "/besser_api/get-project-json-model",
            "get_json_model": "/besser_api/get-json-model",
            "validate_diagram": "/besser_api/validate-diagram",
            "csv_to_domain_model": "/besser_api/csv-to-domain-model",
            "get_json_model_from_image": "/besser_api/get-json-model-from-image",
            "get_json_model_from_kg": "/besser_api/get-json-model-from-kg",
            "transform_agent_model": "/besser_api/transform-agent-model-json",
            "feedback": "/besser_api/feedback",
            "check_ocl": "/besser_api/check-ocl (deprecated, use validate-diagram)"
        }
    }


# Main application entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9000,
        log_level="info"
    )
#The editor is available at: https://editor.besser-pearl.org
