"""
BESSER Backend API

This module provides FastAPI endpoints for the BESSER web modeling editor backend.
It handles code generation from UML diagrams using various generators.

The editor is available at: https://editor.besser-pearl.org
"""

# Standard library imports
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse

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
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Content-Length exceeds MAX_REQUEST_SIZE."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_SIZE:
            return StarletteResponse("Request too large", status_code=413)
        return await call_next(request)


# Initialize FastAPI application
app = FastAPI(
    title="BESSER Backend API",
    description="Backend services for web modeling editor",
    version=API_VERSION,
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
            "deploy": "/besser_api/deploy-app",
            "export_buml": "/besser_api/export-buml",
            "export_project": "/besser_api/export-project_as_buml",
            "get_project_json_model": "/besser_api/get-project-json-model",
            "get_single_json_model": "/besser_api/get-single-json-model",
            "validate_diagram": "/besser_api/validate-diagram",
            "check_ocl": "/besser_api/check-ocl (deprecated, use validate_diagram)"
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
