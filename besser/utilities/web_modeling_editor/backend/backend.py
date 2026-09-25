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
    spec_driven_router,
    telemetry_router,
    agent_simulator_router,
)

# Spec-driven download registry — started/cancelled in the lifespan below
from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    DURABLE_RUN_MANAGER,
    SMART_RUN_REGISTRY,
)
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    LLM_DOWNLOAD_TTL_SECONDS,
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


class RequestSizeLimitMiddleware:
    """Reject requests that exceed MAX_REQUEST_SIZE.

    Pure ASGI on purpose, and not ``BaseHTTPMiddleware``. The previous
    version read the body to enforce the cap, which consumes the receive
    channel, and then every attempt to put it back was wrong on one
    Starlette or the other:

    * consuming and NOT replaying deadlocks every POST on 0.27.0 - the
      endpoint waits forever for a body that is already gone;
    * replaying a constant ``http.request`` fixes 0.27.0 and breaks 0.36.3,
      where ``BaseHTTPMiddleware`` polls receive again to await the client
      disconnect, sees a second ``http.request`` and raises "Unexpected
      message received" - a 500 on every POST;
    * replaying then returning ``http.disconnect`` still breaks 0.36.3,
      because that version already caches and replays the body itself, so
      any manual ``_receive`` collides with its own machinery.

    `fastapi` is unpinned in this package's requirements, so a fresh build
    takes whichever Starlette pip resolves. The size check therefore must
    not depend on that at all - so it never reads the body. It inspects
    ``content-length`` and otherwise counts bytes as they stream past,
    leaving the body untouched for the application to read normally.
    """

    def __init__(self, app, max_size: int = None):
        self.app = app
        self.max_size = MAX_REQUEST_SIZE if max_size is None else max_size

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared:
            try:
                if int(declared) > self.max_size:
                    return await StarletteResponse(
                        "Request too large", status_code=413)(scope, receive, send)
            except ValueError:
                pass  # malformed header; the byte counter below still applies

        seen = 0

        async def counting_receive():
            """Count what actually arrives, without holding on to it.

            Covers an absent or understated ``content-length``. Once the cap
            is passed the body is cut off with a disconnect rather than
            streamed on: we are already past the point where a clean 413 can
            be sent, and continuing would defeat the limit entirely.
            """
            nonlocal seen
            message = await receive()
            if message.get("type") == "http.request":
                seen += len(message.get("body", b"") or b"")
                if seen > self.max_size:
                    logger.warning(
                        "Request body exceeded %d bytes with content-length %r; "
                        "cutting the stream", self.max_size, declared)
                    return {"type": "http.disconnect"}
            return message

        return await self.app(scope, counting_receive, send)


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
    "AGENT_SIMULATOR_API_TOKEN",  # live agent simulation returns 503 without it
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

    # Download metadata lives beside artifacts on the persistent run volume,
    # so a backend/container restart does not invalidate a completed result.
    await SMART_RUN_REGISTRY.restore_persisted()

    # Sweep expired spec-driven generation download entries every minute.
    smart_gen_sweeper = asyncio.create_task(
        SMART_RUN_REGISTRY.periodic_sweep(),
        name="spec-driven-registry-sweeper",
    )
    durable_run_sweeper = asyncio.create_task(
        DURABLE_RUN_MANAGER.periodic_sweep(LLM_DOWNLOAD_TTL_SECONDS),
        name="spec-driven-durable-run-sweeper",
    )

    yield

    # Cancel background tasks on shutdown.
    cleanup_task.cancel()
    smart_gen_sweeper.cancel()
    durable_run_sweeper.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    try:
        await smart_gen_sweeper
    except asyncio.CancelledError:
        pass
    try:
        await durable_run_sweeper
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
    # Idempotency-Key lets the spec-driven client safely retry a run start
    # that failed at the transport layer (see spec_driven_router).
    allow_headers=["Content-Type", "X-GitHub-Session", "Content-Disposition",
                   "Authorization", "Idempotency-Key"],
    expose_headers=["Content-Disposition", "X-BESSER-Run-Id"],
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
app.include_router(spec_driven_router.router)
app.include_router(telemetry_router.router)
app.include_router(agent_simulator_router.router)


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
            "smart_generate": "/besser_api/spec-driven/generate",
            "download_smart": "/besser_api/spec-driven/download/{run_id}",
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
            "recommend_agent_config_llm": "/besser_api/recommend-agent-config-llm",
            "get_agent_config_manual_mapping": "/besser_api/agent-config-manual-mapping",
            "recommend_agent_config_mapping": "/besser_api/recommend-agent-config-mapping",
            "feedback": "/besser_api/feedback",
            "check_ocl": "/besser_api/check-ocl (deprecated, use validate-diagram)"
        }
    }


# Main application entry point
if __name__ == "__main__":
    import uvicorn
    # Single process on purpose: the agent simulator router keeps its rate
    # limiter, per-actor session cap and session-ownership map in memory.
    # Adding workers would give each worker its own copy of that state.
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9000,
        log_level="info"
    )
#The editor is available at: https://editor.besser-pearl.org
