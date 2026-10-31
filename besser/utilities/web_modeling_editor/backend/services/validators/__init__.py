"""
Validators module for OCL checking and model validation.
"""

from .ocl_checker import check_ocl_constraint
from .legacy_format import (
    ensure_diagram_not_legacy,
    ensure_not_legacy_model,
    ensure_project_not_legacy,
    is_legacy_v3_model,
)

__all__ = [
    "check_ocl_constraint",
    "ensure_diagram_not_legacy",
    "ensure_not_legacy_model",
    "ensure_project_not_legacy",
    "is_legacy_v3_model",
]
