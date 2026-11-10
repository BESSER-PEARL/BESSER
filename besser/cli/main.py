"""``besser``: validate and inspect B-UML models, run generators and convert model files.

This module is the console-script entry point registered in ``setup.cfg``;
``python -m besser`` runs it too.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from typing import List, Optional

from besser.cli import commands
from besser.cli.output import (
    EXIT_FAILED,
    NORMAL,
    QUIET,
    VERBOSE,
    Result,
    failure,
    print_result,
    set_verbosity,
)
from besser.utilities.provenance import get_besser_version

PROG = "besser"

_EXECUTES_PYTHON = (
    "A B-UML .py model is Python code and is EXECUTED, with your permissions, to load it: "
    "only pass files you trust. Nothing but the file you name is executed, and its "
    "'if __name__ == \"__main__\":' block does not run. A .json model is parsed as data."
)
_MODEL_HELP = "a B-UML .py file or an editor .json export; - reads either from standard input"

_DESCRIPTION = f"""\
Validate and inspect B-UML models, run BESSER's code generators and convert
model files.

A model file is a B-UML Python script (.py) or a Web Modeling Editor export
(.json: a project or a single diagram). Where a command takes a model file,
- reads it from standard input.

{_EXECUTES_PYTHON}
"""

_EPILOG = """\
exit status:
  0  success
  1  the command ran and failed: validation errors (or warnings with --strict),
     a generator that raised, conflicts left by regenerate
  2  the command could not run: bad arguments, an unknown generator or option,
     a model file that does not load, an output folder generate or regenerate
     refuses

With --json, standard output is one JSON object with the keys ok, errors,
warnings and files (plus the command's own); everything a model or a
generator prints goes to standard error.

examples:
  besser new model.py
  besser validate model.py --json
  besser info model.py
  besser generators
  besser generate sqlalchemy model.py -o out --option dbms=postgresql
  besser regenerate backend model.py -o app
  besser convert model.py -o project.json
"""


class _UsageError(Exception):
    """The arguments do not parse; carries argparse's message."""


class _Parser(argparse.ArgumentParser):
    """An ``ArgumentParser`` that raises instead of exiting, so ``--json`` can report a usage error."""

    def error(self, message: str):
        raise _UsageError(f"{message} (see '{self.prog} --help')")


def build_parser() -> argparse.ArgumentParser:
    """Return the ``besser`` argument parser."""
    # These are accepted before and after the command. SUPPRESS keeps a
    # subparser's default from overwriting the value parsed before the command.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS,
        help="print one JSON object on standard output (keys: ok, errors, warnings, files, ...)",
    )
    common.add_argument(
        "-q", "--quiet", action="store_true", default=argparse.SUPPRESS,
        help="print errors only: no report, no warnings, nothing of what a model or a generator prints",
    )
    common.add_argument(
        "-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
        help="print the traceback of a model file or generator that fails, on standard error",
    )

    parser = _Parser(
        prog=PROG, description=_DESCRIPTION, epilog=_EPILOG, parents=[common],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="store_true", help="print the BESSER version and exit")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>", parser_class=_Parser)

    def add_command(name: str, summary: str, description: str) -> argparse.ArgumentParser:
        return subparsers.add_parser(
            name, help=summary, description=description, parents=[common],
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

    def add_generator_arguments(command: argparse.ArgumentParser, output_help: str) -> None:
        command.add_argument("generator", help="a generator id (see 'besser generators')")
        command.add_argument("model", help=_MODEL_HELP)
        command.add_argument("-o", "--output", required=True, metavar="DIR", help=output_help)
        command.add_argument(
            "--option", action="append", default=[], metavar="NAME=VALUE",
            help="a generator option; repeat for several (a list is comma-separated: http_methods=GET,POST)",
        )
        command.add_argument(
            "--use", action="append", default=[], metavar="NAME",
            help="the model to generate from when the file holds several of a kind the generator reads "
                 "(see 'besser info'); repeat to name one of each kind",
        )

    new = add_command(
        "new", "write a starter model file",
        "Write a small, commented B-UML model (a library, its books and their authors) to start from.",
    )
    new.add_argument("path", help="the .py file to write")
    new.add_argument("--force", action="store_true", help="overwrite the file if it exists")

    validate = add_command(
        "validate", "check models with their validate() rules",
        "Load each model file and run validate() on every model in it.\n\n"
        "Exit status: 0 no errors, 1 errors (or warnings with --strict), 2 a file does not load.\n\n"
        + _EXECUTES_PYTHON,
    )
    validate.add_argument("model", nargs="+", help=_MODEL_HELP + "; several files are checked in one run")
    validate.add_argument("--strict", action="store_true", help="fail (exit 1) on warnings too")

    info = add_command(
        "info", "summarise the models of a file",
        "Print what a model file holds: its models, the classes, attributes and associations of a\n"
        "domain model, and the generators that can run on the file.\n\n" + _EXECUTES_PYTHON,
    )
    info.add_argument("model", help=_MODEL_HELP)

    add_command(
        "generators", "list the generators and their options",
        "List the registered generators: the models each reads, the options it takes and whether\n"
        "'besser regenerate' works with it.",
    )

    generate = add_command(
        "generate", "run a generator on a model",
        "Validate the models the generator reads, run it and print the files it wrote.\n"
        "A model with validation errors is not generated from.\n\n"
        "generate overwrites, so it refuses an output directory that is not empty unless --force\n"
        "is given. To bring generated code up to a changed model and keep the edits made in it,\n"
        "use 'besser regenerate'.\n\n" + _EXECUTES_PYTHON,
    )
    add_generator_arguments(generate, "directory to write to (created if missing); it must be empty")
    generate.add_argument("--force", action="store_true",
                          help="write into a directory that is not empty, overwriting files in it")

    regenerate = add_command(
        "regenerate", "update generated code, keeping hand edits",
        "Bring the output directory up to the model and keep the edits made in it.\n\n"
        "The generator runs in a scratch folder and its output is merged, three ways, with what it\n"
        "generated last time and with the files as they are now. What it generated last time is\n"
        "recorded in .besser_generated.json in the output directory: keep that file with the code.\n"
        "The first run, into an empty directory, generates and starts the record.\n\n"
        "An edited file takes the model's changes (merged). Where an edit and the model changed the\n"
        "same lines, conflict markers are written and the exit status is 1; resolve them, then run\n"
        "the command again.\n\n"
        "A directory written by 'besser generate' has no record: run regenerate once with --adopt\n"
        "and the model it was generated from, before changing the model.\n\n"
        "Needs pygit2 (pip install \"besser[regen]\"); 'besser generators' says which generators\n"
        "support it.\n\n" + _EXECUTES_PYTHON,
    )
    add_generator_arguments(regenerate, "the directory holding the generated code (created if missing)")
    regenerate.add_argument(
        "--adopt", action="store_true",
        help="record a directory generated without a record as this model's output, changing no file",
    )

    convert = add_command(
        "convert", "convert between B-UML .py and editor JSON",
        "Convert a B-UML .py file to an editor project (.json), or an editor export (.json)\n"
        "to a B-UML .py file. The direction follows the two file extensions.\n\n" + _EXECUTES_PYTHON,
    )
    convert.add_argument("input", help="the file to convert (.py or .json); - reads it from standard input")
    convert.add_argument("-o", "--output", required=True, metavar="FILE",
                         help="the file to write (.json or .py); overwritten if it exists")
    return parser


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Result:
    if args.version:
        version = get_besser_version()
        return Result(command="version", data={"version": version}, text=f"{PROG} {version}")
    if args.command == "new":
        return commands.new(args.path, force=args.force)
    if args.command == "validate":
        return commands.validate(args.model, strict=args.strict)
    if args.command == "info":
        return commands.info(args.model)
    if args.command == "generators":
        return commands.generators()
    if args.command == "generate":
        return commands.generate(args.generator, args.model, args.output, args.option, args.use, args.force)
    if args.command == "regenerate":
        return commands.regenerate(args.generator, args.model, args.output, args.option, args.use, args.adopt)
    if args.command == "convert":
        return commands.convert(args.input, args.output)
    return failure("", f"a command is required\n\n{parser.format_usage().strip()} (see '{PROG} --help')")


def _verbosity(args: argparse.Namespace) -> int:
    quiet, verbose = getattr(args, "quiet", False), getattr(args, "verbose", False)
    if quiet and verbose:
        raise _UsageError(f"--quiet and --verbose cannot be combined (see '{PROG} --help')")
    return QUIET if quiet else VERBOSE if verbose else NORMAL


def main(argv: Optional[List[str]] = None) -> int:
    """Run ``besser`` with ``argv`` (default: the process arguments) and return its exit status."""
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    as_json = "--json" in argv
    set_verbosity(NORMAL)
    try:
        args = parser.parse_args(argv)
        as_json = getattr(args, "json", False)
        set_verbosity(_verbosity(args))
        # --quiet also mutes the warnings BESSER logs, which would otherwise reach standard error.
        logging.disable(logging.WARNING if getattr(args, "quiet", False) else logging.NOTSET)
        result = _run(args, parser)
    except _UsageError as exc:
        result = failure("", str(exc))
    except Exception as exc:
        # A defect in BESSER, not in the input: the traceback is what a bug report needs.
        traceback.print_exc()
        result = failure("", f"unexpected {type(exc).__name__}: {exc}", EXIT_FAILED)
    finally:
        logging.disable(logging.NOTSET)
    print_result(result, as_json)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
