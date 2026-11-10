"""The ``besser`` command-line interface.

``besser validate``, ``besser generators``, ``besser generate`` and
``besser convert``, for people and for coding agents (``--json``). Loading a
B-UML ``.py`` model executes it; see ``besser --help``.
"""

from besser.cli.main import main

__all__ = ["main"]
