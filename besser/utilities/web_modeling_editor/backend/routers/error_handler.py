"""
Centralized error handling decorator for all router endpoints.

Provides a consistent pattern for catching and mapping exceptions to
appropriate HTTP responses, replacing duplicated try/except blocks
across router modules.
"""

import functools
import logging
import asyncio

from fastapi import HTTPException

from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
    ValidationError,
    GenerationError,
)

logger = logging.getLogger(__name__)


def handle_endpoint_errors(endpoint_name: str):
    """Decorator that wraps an endpoint function with standard error handling.

    Args:
        endpoint_name: A human-readable label used in log messages when an
            unexpected error occurs (e.g. ``"generate_code_output"``).

    Returns:
        A decorator that can be applied to both sync and async endpoint
        functions.

    Exception mapping:
        * ``ConversionError``  -> HTTP 400
        * ``ValidationError``  -> HTTP 400
        * ``GenerationError``  -> HTTP 500
        * ``HTTPException``    -> re-raised as-is
        * ``Exception``        -> HTTP 500 with generic message

    Example::

        @router.post("/my-endpoint")
        @handle_endpoint_errors("my_endpoint")
        async def my_endpoint(input_data: DiagramInput):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                raise
            except ConversionError as exc:
                logger.warning("Conversion error in %s: %s", endpoint_name, exc)
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except ValidationError as exc:
                logger.warning("Validation error in %s: %s", endpoint_name, exc)
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except GenerationError as exc:
                logger.error("Generation error in %s: %s", endpoint_name, exc)
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            except ValueError as exc:
                logger.warning("Value error in %s: %s", endpoint_name, exc)
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception:
                logger.exception("Unexpected error in %s", endpoint_name)
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except HTTPException:
                raise
            except ConversionError as exc:
                logger.warning("Conversion error in %s: %s", endpoint_name, exc)
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except ValidationError as exc:
                logger.warning("Validation error in %s: %s", endpoint_name, exc)
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except GenerationError as exc:
                logger.error("Generation error in %s: %s", endpoint_name, exc)
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            except ValueError as exc:
                logger.warning("Value error in %s: %s", endpoint_name, exc)
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception:
                logger.exception("Unexpected error in %s", endpoint_name)
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
