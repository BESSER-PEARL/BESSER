"""
Mock optional heavy dependencies that are not needed for NN diagram processor tests.
The services/__init__.py eagerly imports several backend-only packages (bocl, yaml, docker, etc.)
that are not relevant to the NN diagram processor. We stub them out here.
"""

import sys
from unittest.mock import MagicMock

_MOCKED = [
    "bocl",
    "bocl.OCLWrapper",
    "yaml",
    "docker",
    "docker.errors",
    "github",
    "github.Github",
    "openai",
]

for _mod in _MOCKED:
    sys.modules.setdefault(_mod, MagicMock())