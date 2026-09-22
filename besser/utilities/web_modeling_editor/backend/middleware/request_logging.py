"""
Structured request logging and performance timing middleware.

Provides observability for the BESSER backend by logging every request with
structured data (method, path, status_code, duration_ms, request_id) and
injecting an X-Request-ID header into each response.

Slow requests (>1 second) are logged at WARNING level; all others at INFO.

Registration
------------
To register this middleware in ``backend.py``, import and call
``setup_middleware`` **before** adding CORS or other middlewares so it wraps
the entire request lifecycle::

    from besser.utilities.web_modeling_editor.backend.middleware import setup_middleware
    setup_middleware(app)

References: GitHub issues #15 (structured logging) and #16 (performance timing).
"""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("besser.backend.requests")

# Threshold in seconds above which a request is considered slow.
SLOW_REQUEST_THRESHOLD_S = 1.0


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with structured context and performance timing.

    For each request the middleware:
    1. Generates a unique ``request_id`` (UUID4).
    2. Measures wall-clock duration.
    3. Attaches ``X-Request-ID`` to the response headers.
    4. Emits a structured log entry containing *method*, *path*,
       *status_code*, *duration_ms*, and *request_id*.
    5. Uses WARNING level when *duration_ms* exceeds 1 000 ms, INFO otherwise.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            # If an unhandled exception propagates, still log the failed
            # request before re-raising so it appears in observability data.
            duration_ms = (time.perf_counter() - start_time) * 1000
            log_data = {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
                "duration_ms": round(duration_ms, 2),
            }
            logger.warning(
                "request failed with unhandled exception | %s",
                log_data,
            )
            raise

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Attach the request ID so callers and upstream proxies can correlate.
        response.headers["X-Request-ID"] = request_id

        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        }

        if duration_ms > SLOW_REQUEST_THRESHOLD_S * 1000:
            logger.warning("slow request | %s", log_data)
        else:
            logger.info("request completed | %s", log_data)

        return response


def setup_middleware(app):
    """Register observability middleware on a FastAPI/Starlette application.

    Call this function in ``backend.py`` to enable structured request logging
    and performance timing::

        from besser.utilities.web_modeling_editor.backend.middleware import setup_middleware
        setup_middleware(app)

    The middleware should be added **before** CORS and other middlewares so
    that the timing measurement covers the full request lifecycle.

    Args:
        app: The FastAPI (or Starlette) application instance.
    """
    app.add_middleware(RequestLoggingMiddleware)
