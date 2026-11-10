"""What a ``besser`` command reports, and how it is printed.

Standard output carries the command's result and nothing else: one JSON
object with ``--json``, a short report otherwise. Whatever a model file or a
generator prints while it runs goes to standard error.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Iterator

EXIT_OK = 0
# The command ran and reports a failure: validation errors, a generator that raised.
EXIT_FAILED = 1
# The command could not run: bad arguments, a model file that does not load.
EXIT_USAGE = 2


@dataclass
class Result:
    """The outcome of one command."""

    command: str
    exit_code: int = EXIT_OK
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Files the command wrote, as paths the caller can open from its working directory.
    files: list[str] = field(default_factory=list)
    # The command's own keys in the JSON output.
    data: dict[str, Any] = field(default_factory=dict)
    # The report printed on standard output without ``--json``.
    text: str = ""
    # True when ``text`` already shows the errors and warnings (``validate``).
    text_has_diagnostics: bool = False

    def as_json(self) -> dict[str, Any]:
        """The JSON output: ``ok``, ``errors``, ``warnings`` and ``files`` on every command."""
        return {
            "ok": self.exit_code == EXIT_OK,
            "command": self.command,
            "errors": self.errors,
            "warnings": self.warnings,
            "files": self.files,
            **self.data,
        }


def failure(command: str, message: str, exit_code: int = EXIT_USAGE, data: dict[str, Any] | None = None) -> Result:
    """A result that reports one error."""
    return Result(command=command, exit_code=exit_code, errors=[message], data=data or {})


def print_result(result: Result, as_json: bool) -> None:
    """Print ``result``: JSON on standard output, or the report with its diagnostics on standard error."""
    if as_json:
        print(json.dumps(result.as_json(), indent=2))
        return
    if result.text:
        print(result.text)
    if not result.text_has_diagnostics:
        for warning in result.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        for error in result.errors:
            print(f"error: {error}", file=sys.stderr)


@contextlib.contextmanager
def stdout_to_stderr() -> Iterator[None]:
    """Send everything written to standard output to standard error.

    Model files and generators print progress, and a generator's child process
    (``django-admin``) writes to the inherited descriptor, so both ``sys.stdout``
    and file descriptor 1 are redirected.
    """
    sys.stdout.flush()
    try:
        saved_fd = os.dup(1)
        os.dup2(2, 1)
    except OSError:  # a standard stream is closed: only sys.stdout can be redirected
        saved_fd = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        if saved_fd is not None:
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
