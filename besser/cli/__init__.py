"""The ``besser`` command-line interface.

``besser new``, ``validate``, ``info``, ``generators``, ``generate``,
``regenerate`` and ``convert``, for people and for coding agents (``--json``).
Loading a B-UML ``.py`` model executes it; see ``besser --help``.
"""

from besser.cli.main import main

__all__ = ["main"]
