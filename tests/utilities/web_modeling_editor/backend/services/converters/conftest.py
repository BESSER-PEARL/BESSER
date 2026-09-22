"""Mock optional heavy backend dependencies only when they aren't installed.

The backend's ``services/__init__.py`` eagerly imports several backend-only packages
(``bocl``, ``yaml``, ``docker``, ``github``, …) that aren't relevant to the converter
unit tests in this directory. In a minimal local environment those imports fail and
pytest aborts collection before any test runs.

Pattern copied from ``tests/utilities/web_modeling_editor/converters/nn/conftest.py``:
try the real import first, only register a ``MagicMock`` if it isn't installed. This
avoids polluting the import cache for environments that *do* have the real package.
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
