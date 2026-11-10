"""What a ``besser`` command reports, and how it is printed.

Standard output carries the command's result and nothing else: one JSON
object with ``--json``, a short report otherwise. Whatever a model file or a
generator prints while it runs goes to standard error, or nowhere with ``--quiet``.

``--quiet`` drops the diagnostics (notes, warnings, what models and generators print)
and the report of a command that writes or checks files; it keeps the output of a
command whose output is what was asked for (``Result.text_is_result``: a completion
script, the version, the generators, a model's summary) and every error.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Iterator

EXIT_OK = 0
# The command ran and reports a failure: validation errors, a generator that raised.
EXIT_FAILED = 1
# The command could not run: bad arguments, a model file that does not load.
EXIT_USAGE = 2

# How much is printed besides the result: set once per run by ``main``.
QUIET, NORMAL, VERBOSE = -1, 0, 1
_verbosity = NORMAL


def set_verbosity(verbosity: int) -> None:
    """Set what this run prints: ``QUIET``, ``NORMAL`` or ``VERBOSE``."""
    global _verbosity
    _verbosity = verbosity


@dataclass
class Result:
    """The outcome of one command."""

    command: str
    exit_code: int = EXIT_OK
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # What the command chose on its own and the user may want to know (e.g. which model it used):
    # printed as "note: ..." on standard error, not with --quiet.
    notes: list[str] = field(default_factory=list)
    # Files the command wrote, as paths the caller can open from its working directory.
    files: list[str] = field(default_factory=list)
    # The command's own keys in the JSON output.
    data: dict[str, Any] = field(default_factory=dict)
    # The report printed on standard output without ``--json``.
    text: str = ""
    # The errors and warnings ``text`` already shows (``validate`` lists them under each model).
    shown_in_text: frozenset[str] = frozenset()
    # The traceback behind a model or generator failure, printed with ``--verbose``.
    traceback: str = ""
    # ``text`` is what the command was run for (a completion script, the version), not a report
    # of what it did: printed with ``--quiet`` too.
    text_is_result: bool = False

    def as_json(self) -> dict[str, Any]:
        """The JSON output: the keys every command has, then the command's own."""
        return {
            "ok": self.exit_code == EXIT_OK,
            "command": self.command,
            "exit_code": self.exit_code,
            "errors": self.errors,
            "warnings": self.warnings,
            "notes": self.notes,
            "files": self.files,
            **self.data,
        }


def failure(command: str, message: str, exit_code: int = EXIT_USAGE, data: dict[str, Any] | None = None,
            cause: BaseException | None = None) -> Result:
    """A result that reports one error; ``cause`` is the exception behind it, for ``--verbose``."""
    return Result(command=command, exit_code=exit_code, errors=[message], data=data or {},
                  traceback=format_traceback(cause) if cause else "")


def format_traceback(exc: BaseException) -> str:
    """The traceback of ``exc`` with the exceptions that caused it."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def print_result(result: Result, as_json: bool) -> None:
    """Print ``result``: JSON on standard output, or the report with its diagnostics on standard error.

    ``--quiet`` prints the errors, and ``text`` only when it is the result (``text_is_result``);
    a run that fails on its warnings alone (``validate --strict``) prints them too, since they
    are why it failed. The JSON object is printed whole. ``--verbose`` adds the traceback of a
    failure on standard error.
    """
    if _verbosity == VERBOSE and result.traceback:
        print(result.traceback, file=sys.stderr, end="")
    if as_json:
        print(json.dumps(result.as_json(), indent=2))
        return
    if _verbosity == QUIET:
        _print_quiet(result)
        return
    if result.text:
        print(result.text)
    for note in result.notes:
        print(f"note: {note}", file=sys.stderr)
    for warning in result.warnings:
        if warning not in result.shown_in_text:
            print(f"warning: {warning}", file=sys.stderr)
    for error in result.errors:
        if error not in result.shown_in_text:
            print(f"error: {error}", file=sys.stderr)


def plural(count: int, noun: str) -> str:
    """``count`` and ``noun``, with an "s" unless ``count`` is 1: "1 model", "2 models"."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _print_quiet(result: Result) -> None:
    if result.text and result.text_is_result:
        print(result.text)
    for error in result.errors:
        print(f"error: {error}", file=sys.stderr)
    if result.exit_code == EXIT_FAILED and not result.errors and result.warnings:
        print(f"error: {plural(len(result.warnings), 'warning')} (--strict)", file=sys.stderr)
        for warning in result.warnings:
            print(f"warning: {warning}", file=sys.stderr)


@contextlib.contextmanager
def stdout_to_stderr() -> Iterator[None]:
    """Send everything written to standard output to standard error, or nowhere with ``--quiet``.

    Model files and generators print progress, and a generator's child process
    (``django-admin``) writes to the inherited descriptor, so both ``sys.stdout``
    and file descriptor 1 are redirected.
    """
    sys.stdout.flush()
    with contextlib.ExitStack() as stack:
        if _verbosity == QUIET:
            target = stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
            target_fd = target.fileno()
        else:
            target, target_fd = sys.stderr, 2
        try:
            saved_fd = os.dup(1)
            os.dup2(target_fd, 1)
        except OSError:  # a standard stream is closed: only sys.stdout can be redirected
            saved_fd = None
        try:
            with contextlib.redirect_stdout(target):
                yield
        finally:
            if saved_fd is not None:
                os.dup2(saved_fd, 1)
                os.close(saved_fd)
