"""``besser``: validate B-UML models, run generators and convert model files.

This module is the console-script entry point registered in ``setup.cfg``;
``python -m besser`` runs it too.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import List, Optional

from besser.cli import commands
from besser.cli.output import EXIT_FAILED, EXIT_OK, EXIT_USAGE, Result, failure, print_result
from besser.utilities.provenance import get_besser_version

PROG = "besser"

_EXECUTES_PYTHON = (
    "A B-UML .py model is Python code and is EXECUTED, with your permissions, to load it: "
    "only pass files you trust. Nothing but the file you name is executed, and its "
    "'if __name__ == \"__main__\":' block does not run. A .json model is parsed as data."
)

_DESCRIPTION = f"""\
Validate B-UML models, run BESSER's code generators and convert model files.

A model file is a B-UML Python script (.py) or a Web Modeling Editor export
(.json: a project or a single diagram).

{_EXECUTES_PYTHON}
"""

_EPILOG = """\
exit status:
  0  success
  1  the command ran and failed: validation errors (or warnings with --strict),
     a generator that raised
  2  the command could not run: bad arguments, an unknown generator or option,
     a model file that does not load

With --json, standard output is one JSON object with the keys ok, errors,
warnings and files (plus the command's own); everything a model or a
generator prints goes to standard error.

examples:
  besser validate model.py --json
  besser generators
  besser generate sqlalchemy model.py -o out --option dbms=postgresql
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
    # ``--json`` is accepted before and after the command. SUPPRESS keeps a
    # subparser's default from overwriting the value parsed before the command.
    json_option = argparse.ArgumentParser(add_help=False)
    json_option.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS,
        help="print one JSON object on standard output (keys: ok, errors, warnings, files, ...)",
    )

    parser = _Parser(
        prog=PROG, description=_DESCRIPTION, epilog=_EPILOG, parents=[json_option],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="store_true", help="print the BESSER version and exit")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>", parser_class=_Parser)

    def add_command(name: str, summary: str, description: str) -> argparse.ArgumentParser:
        return subparsers.add_parser(
            name, help=summary, description=description, parents=[json_option],
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

    validate = add_command(
        "validate", "check a model with its validate() rules",
        "Load a model file and run validate() on every model in it.\n\n"
        "Exit status: 0 no errors, 1 errors (or warnings with --strict), 2 the file does not load.\n\n"
        + _EXECUTES_PYTHON,
    )
    validate.add_argument("model", help="a B-UML .py file or an editor .json export")
    validate.add_argument("--strict", action="store_true", help="fail (exit 1) on warnings too")

    add_command(
        "generators", "list the generators and their options",
        "List the registered generators: the models each reads and the options it takes.",
    )

    generate = add_command(
        "generate", "run a generator on a model",
        "Validate the models the generator reads, run it and print the files it wrote.\n"
        "A model with validation errors is not generated from.\n\n" + _EXECUTES_PYTHON,
    )
    generate.add_argument("generator", help="a generator id (see 'besser generators')")
    generate.add_argument("model", help="a B-UML .py file or an editor .json export")
    generate.add_argument("-o", "--output", required=True, metavar="DIR",
                          help="directory to write to (created if missing)")
    generate.add_argument(
        "--option", action="append", default=[], metavar="NAME=VALUE",
        help="a generator option; repeat for several (a list is comma-separated: http_methods=GET,POST)",
    )

    convert = add_command(
        "convert", "convert between B-UML .py and editor JSON",
        "Convert a B-UML .py file to an editor project (.json), or an editor export (.json)\n"
        "to a B-UML .py file. The direction follows the two file extensions.\n\n" + _EXECUTES_PYTHON,
    )
    convert.add_argument("input", help="the file to convert (.py or .json)")
    convert.add_argument("-o", "--output", required=True, metavar="FILE",
                         help="the file to write (.json or .py); overwritten if it exists")
    return parser


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Result:
    if args.version:
        version = get_besser_version()
        return Result(command="version", data={"version": version}, text=f"{PROG} {version}")
    if args.command == "validate":
        return commands.validate(args.model, strict=args.strict)
    if args.command == "generators":
        return commands.generators()
    if args.command == "generate":
        return commands.generate(args.generator, args.model, args.output, args.option)
    if args.command == "convert":
        return commands.convert(args.input, args.output)
    return failure("", f"a command is required\n\n{parser.format_usage().strip()} (see '{PROG} --help')")


def main(argv: Optional[List[str]] = None) -> int:
    """Run ``besser`` with ``argv`` (default: the process arguments) and return its exit status."""
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    as_json = "--json" in argv
    try:
        args = parser.parse_args(argv)
        as_json = getattr(args, "json", False)
        result = _run(args, parser)
    except _UsageError as exc:
        result = failure("", str(exc))
    except Exception as exc:
        # A defect in BESSER, not in the input: the traceback is what a bug report needs.
        traceback.print_exc()
        result = failure("", f"unexpected {type(exc).__name__}: {exc}", EXIT_FAILED)
    print_result(result, as_json)
    return result.exit_code


__all__ = ["EXIT_FAILED", "EXIT_OK", "EXIT_USAGE", "PROG", "build_parser", "main"]


if __name__ == "__main__":
    sys.exit(main())
