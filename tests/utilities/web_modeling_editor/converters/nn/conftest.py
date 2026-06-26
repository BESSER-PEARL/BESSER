"""
Mock optional heavy dependencies only when they aren't actually installed.

The services/__init__.py eagerly imports several backend-only packages (bocl,
yaml, docker, etc.) that aren't relevant to the NN diagram processor. When
running this test suite in a minimal environment those imports would fail, so
we fall back to a MagicMock.

IMPORTANT: we must try to import the real module first. Using a blind
``sys.modules.setdefault(name, MagicMock())`` pollutes the global import cache
at pytest collection time — before any other test has had a chance to import
the real package — and causes unrelated tests (e.g. the OCL constraint tests)
to receive MagicMock objects instead of the real ``bocl.OCLWrapper``.
"""

import importlib
import sys
from unittest.mock import MagicMock

_OPTIONAL = [
    "bocl",
    "bocl.OCLWrapper",
    "yaml",
    "docker",
    "docker.errors",
    "github",
    "github.Github",
    "openai",
]

for _mod in _OPTIONAL:
    if _mod in sys.modules:
        continue
    try:
        importlib.import_module(_mod)
    except ImportError:
        sys.modules[_mod] = MagicMock()
