"""Traceability service layer behind ``routers/traceability_router.py``."""

from .traced_generation import generate_with_trace

__all__ = ["generate_with_trace"]
