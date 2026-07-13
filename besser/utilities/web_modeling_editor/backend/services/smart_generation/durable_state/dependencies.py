"""Lazy optional-dependency loading for production adapters."""

from __future__ import annotations

import importlib
from types import ModuleType

from .errors import MissingOptionalDependencyError


def require_optional_dependency(module_name: str, install_hint: str) -> ModuleType:
    """Import an adapter dependency or fail without selecting a weaker backend."""

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            f"Production SmartGen adapter requires {module_name!r}. Install {install_hint}; "
            "the service will not fall back to local/in-memory storage."
        ) from exc
