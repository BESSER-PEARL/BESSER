"""
Middleware package for the BESSER web modeling editor backend.

Provides observability middleware including structured request logging
and performance timing.
"""

from besser.utilities.web_modeling_editor.backend.middleware.request_logging import (
    setup_middleware,
)

__all__ = ["setup_middleware"]
