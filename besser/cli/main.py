"""``besser``: start, describe, validate and inspect B-UML models, run generators and convert model files.

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
from besser.cli.completion import COMPLETION_SHELLS, completion_script
from besser.cli.config import CONFIG_FILENAME
from besser.cli.importing import IMPORT_FORMATS
from besser.cli.starters import DEFAULT_KIND, STARTERS
from besser.cli.output import (
    EXIT_FAILED,
    EXIT_OK,
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
    "only pass files you trust. Nothing but the files you name is executed (and those besser.toml "
    "names for 'build', and, with 'validate --exec-py', the .py files found in a directory you name), and a file's "
    "'if __name__ == \"__main__\":' block does not run. A .json model is parsed as data."
)
_MODEL_HELP = "a B-UML .py file or an editor .json export; - reads either from standard input"

_DESCRIPTION = f"""\
Start, validate and inspect B-UML models, run BESSER's code generators and
convert model files. Every command takes --json, asks nothing and exits with
0, 1 or 2: it is made to be driven by a coding agent or a script as well as
by hand.

A model file is a B-UML Python script (.py), the format to write and edit, or
a Web Modeling Editor export (.json: a project or a single diagram). Where a
command takes a model file, - reads it from standard input.

{_EXECUTES_PYTHON}
"""

_EPILOG = """\
exit status:
  0  success
  1  the command ran and failed: validation errors (or warnings with --strict),
     a generator that raised or wrote nothing, conflicts left by regenerate
  2  the command could not run: bad arguments, an unknown generator or option,
     a model file or besser.toml that does not load, an output folder generate
     or regenerate refuses

With --json, standard output is one JSON object with the keys ok, command,
exit_code, errors, warnings, notes and files (plus the command's own;
'besser schema result' prints its JSON Schema); everything a model or a
generator prints goes to standard error.

examples, in the order of a session:
  besser new model.py                          a starter model to edit (--list: the other kinds)
  besser schema domain                         the classes and arguments to write it with
  besser validate model.py --json              errors and warnings, with file and line, as JSON
  besser validate models/ --strict             every editor .json export under models/
  besser validate models/ --exec-py --ocl      ... and the .py models, with their OCL constraints
  besser info project.json                     the models, classes and generators of a file
  besser generators                            the generators and their options
  besser generate sqlalchemy model.py -o out --option dbms=postgresql
  besser generate backend model.py -o app --dry-run    what would be written
  besser regenerate backend model.py -o app    after a model change: keeps your edits
  besser build                                 every output listed in besser.toml, edits kept
  besser convert model.py -o project.json      B-UML .py <-> editor JSON
  besser convert diagram.plantuml -o model.py  import another notation (see convert --help)
  eval "$(besser completion bash)"             tab completion (also zsh, fish)
  besser mcp --root .                          the commands as MCP tools for an AI agent
"""


class _UsageError(Exception):
    """The arguments do not parse; carries the message and the command it is about ("" for none)."""

    def __init__(self, message: str, command: str = ""):
        super().__init__(message)
        self.command = command


class _Parser(argparse.ArgumentParser):
    """An ``ArgumentParser`` that raises instead of exiting, so ``--json`` can report a usage error.

    A command's parser is named ``besser <command>``: its error names the command and points to its help.
    """

    def error(self, message: str):
        raise _UsageError(f"{message} (see '{self.prog} --help')", self.prog.partition(" ")[2])


def build_parser() -> argparse.ArgumentParser:
    """Return the ``besser`` argument parser."""
    # These are accepted before and after the command. SUPPRESS keeps a
    # subparser's default from overwriting the value parsed before the command.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS,
        help="print one JSON object on standard output (keys: ok, command, exit_code, errors, warnings, notes, files, ...)",
    )
    common.add_argument(
        "-q", "--quiet", action="store_true", default=argparse.SUPPRESS,
        help="print only errors and the output of completion, generators, info, schema and --version "
             "(and, with --strict, the warnings that fail the run)",
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

    def add_dry_run(command: argparse.ArgumentParser) -> None:
        command.add_argument("--dry-run", action="store_true",
                             help="print the files that would be written or changed, and write nothing")

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
        "Write a small, commented model to start from: a class diagram (a library, its books\n"
        "and their authors) unless --kind names another kind of model. A .py file is a B-UML\n"
        "script, the format to write and edit; a .json file is the same model as an editor project.\n\n"
        "kinds:\n" + "".join(f"  {kind:<15} {starter.summary}\n" for kind, starter in STARTERS.items()),
    )
    new.add_argument("path", nargs="?", help="the .py or .json file to write")
    new.add_argument("--kind", choices=list(STARTERS), default=DEFAULT_KIND,
                     help=f"the kind of model to start from (default: {DEFAULT_KIND})")
    new.add_argument("--list", action="store_true", dest="list_kinds", help="list the kinds and write nothing")
    new.add_argument("--force", action="store_true", help="overwrite the file if it exists")

    schema = add_command(
        "schema", "describe the model formats, the JSON output and besser.toml",
        "Print what to write a model with: the B-UML classes of a model kind and their constructor\n"
        "arguments, read from the metamodel; the JSON Schema of an editor project; the JSON Schema\n"
        "of the object every command prints with --json; the besser.toml file 'besser build' reads.\n"
        "Without a topic, list the topics.",
    )
    schema.add_argument("topic", nargs="?", choices=commands.schema_topics(),
                        help="a model kind, project, result or config")

    validate = add_command(
        "validate", "check models with their validate() rules",
        "Load each model file and run validate() on every model in it.\n\n"
        "A directory is searched, recursively, for model files. Only its .json files are loaded\n"
        "unless --exec-py is given, since loading a .py model executes it; a .py file named on\n"
        "the command line is always loaded.\n\n"
        "--ocl evaluates invariants with Python built from the object models' values, so it\n"
        "evaluates those of editor JSON only with --exec-py; a .py model's are evaluated, since\n"
        "it was executed to load it.\n\n"
        "Exit status, that of the worst file: 0 no errors, 1 errors (or warnings with --strict),\n"
        "2 a file does not load or there is nothing to check.\n\n"
        + _EXECUTES_PYTHON,
    )
    validate.add_argument(
        "model", nargs="+", metavar="model",
        help=_MODEL_HELP + ", or a directory of them; several are checked in one run",
    )
    validate.add_argument("--strict", action="store_true", help="fail (exit 1) on warnings too")
    validate.add_argument(
        "--exec-py", action="store_true",
        help="also load (execute) the .py files found in a directory, which are skipped without it; "
             "with --ocl, also evaluate the invariants of editor JSON models",
    )
    validate.add_argument(
        "--ocl", action="store_true",
        help="also check the models' OCL constraints: parse each one and evaluate the invariants on the "
             "object models, those of editor JSON only with --exec-py (the evaluator runs Python built from "
             'their values); needs "besser[ocl]"',
    )

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
    generate.add_argument(
        "--keep-edits", action="store_true",
        help="keep the edits made in the generated code: the same as 'besser regenerate'",
    )
    add_dry_run(generate)

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
        "'besser generate' starts the record too. A directory generated without one (by an older\n"
        "BESSER, or by other means) needs one run with --adopt and the model it was generated from,\n"
        "before changing the model.\n\n"
        "Needs pygit2 (pip install \"besser[regen]\"); 'besser generators' says which generators\n"
        "support it.\n\n" + _EXECUTES_PYTHON,
    )
    add_generator_arguments(regenerate, "the directory holding the generated code (created if missing)")
    regenerate.add_argument(
        "--adopt", action="store_true",
        help="record a directory generated without a record as this model's output, changing no file",
    )
    add_dry_run(regenerate)

    build = add_command(
        "build", "generate every output listed in besser.toml",
        "Generate the outputs a configuration file lists, so each generator's model, directory and\n"
        "options are written once ('besser schema config' prints an example). An output is generated\n"
        "as 'besser generate' does the first time, and brought up to the model as 'besser regenerate'\n"
        "does afterwards, keeping the edits made in it. A generator that does not support regenerate\n"
        "needs 'overwrite = true' to write over its output.\n\n"
        "Exit status, that of the worst output: an output that fails does not stop the others.\n\n"
        + _EXECUTES_PYTHON,
    )
    build.add_argument("output_name", nargs="*", metavar="output",
                       help="the outputs to build, by the NAME of their [output.NAME] table (default: all)")
    build.add_argument("-c", "--config", default=CONFIG_FILENAME, metavar="FILE",
                       help=f"the configuration file (default: {CONFIG_FILENAME} in the current directory)")
    add_dry_run(build)

    convert = add_command(
        "convert", "convert a model file, or import another notation",
        "Convert a B-UML .py file to an editor project (.json), or an editor export (.json)\n"
        "to a B-UML .py file, or import a file of another notation as a model. What is read\n"
        "follows the input's extension (or --from), what is written the output's.\n\n"
        "imports (read as data: nothing in them is executed):\n"
        + "".join(
            f"  {fmt.name:<9} {', '.join(fmt.suffixes) or '(--from ' + fmt.name + ')'}\n"
            f"            {fmt.summary}\n" for fmt in IMPORT_FORMATS
        )
        + "\nA knowledge graph is also written as RDF: -o graph.ttl (Turtle), .owl or .rdf (RDF/XML).\n\n"
        + _EXECUTES_PYTHON,
    )
    convert.add_argument(
        "input", nargs="+",
        help="the file to convert (.py or .json; - reads either from standard input) or to import; "
             "several files for --from csv",
    )
    convert.add_argument("-o", "--output", required=True, metavar="FILE",
                         help="the file to write (.py, .json, or .ttl/.owl/.rdf for a knowledge graph); "
                              "overwritten if it exists")
    convert.add_argument("--from", dest="source_format", choices=[fmt.name for fmt in IMPORT_FORMATS],
                         help="the notation of the input, when its extension does not say")
    convert.add_argument("--class-diagram", action="store_true",
                         help="write the class diagram derived from each knowledge graph, instead of the graph")
    add_dry_run(convert)

    completion = add_command(
        "completion", "print a shell completion script",
        "Print a completion script for bash, zsh or fish on standard output. To load it:\n\n"
        "  bash  eval \"$(besser completion bash)\"         (in ~/.bashrc)\n"
        "  zsh   eval \"$(besser completion zsh)\"          (in ~/.zshrc)\n"
        "  fish  besser completion fish > ~/.config/fish/completions/besser.fish",
    )
    completion.add_argument("shell", choices=COMPLETION_SHELLS, help="the shell to complete for")

    mcp = add_command(
        "mcp", "serve the commands to an AI agent over MCP (stdio)",
        "Run a Model Context Protocol server on standard input and output, exposing validate,\n"
        "info, generators, generate, regenerate and convert as tools to an MCP client (an AI\n"
        "coding agent).\n\n"
        "Tools only read and write files under --root. A .py model is Python code: the tools\n"
        "load .json models only, unless --exec-py is given.\n\n"
        'Needs the MCP SDK (pip install "besser[mcp]").',
    )
    mcp.add_argument("--root", default=".", metavar="DIR",
                     help="the directory the tools may read and write (default: the current directory)")
    mcp.add_argument("--exec-py", action="store_true",
                     help="let the tools load (execute) B-UML .py model files under --root")
    return parser


def _parse(parser: argparse.ArgumentParser, argv: List[str]) -> argparse.Namespace:
    """``argv`` parsed; an unknown option is reported by the parser of the command it was given to."""
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        command_parser = _subparsers(parser).choices.get(args.command, parser)
        command_parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    return args


def _subparsers(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    return next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))


def _command_name(args: argparse.Namespace) -> str:
    return "version" if args.version else args.command or ""


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Optional[Result]:
    """Run the command ``args`` names; ``None`` when there is nothing to print (``mcp`` served)."""
    if args.version:
        version = get_besser_version()
        return Result(command="version", data={"version": version}, text=f"{PROG} {version}", text_is_result=True)
    if args.command == "new":
        return commands.new(args.path, kind=args.kind, force=args.force, list_kinds=args.list_kinds)
    if args.command == "schema":
        return commands.schema(args.topic)
    if args.command == "build":
        return commands.build(args.output_name, config_path=args.config, dry_run=args.dry_run)
    if args.command == "validate":
        return commands.validate(args.model, strict=args.strict, exec_py=args.exec_py, ocl=args.ocl)
    if args.command == "info":
        return commands.info(args.model)
    if args.command == "generators":
        return commands.generators()
    if args.command == "generate":
        if args.keep_edits:
            if args.force:
                raise _UsageError(f"--keep-edits and --force cannot be combined (see '{PROG} generate --help')",
                                  "generate")
            return commands.regenerate(args.generator, args.model, args.output, args.option, args.use,
                                       dry_run=args.dry_run)
        return commands.generate(args.generator, args.model, args.output, args.option, args.use, args.force,
                                 dry_run=args.dry_run)
    if args.command == "regenerate":
        return commands.regenerate(args.generator, args.model, args.output, args.option, args.use, args.adopt,
                                   dry_run=args.dry_run)
    if args.command == "convert":
        return commands.convert(args.input, args.output, dry_run=args.dry_run, source_format=args.source_format,
                                class_diagram=args.class_diagram)
    if args.command == "completion":
        return completion_script(args.shell, parser)
    if args.command == "mcp":
        # Imported here: only this command needs the server (and the MCP SDK it imports).
        from besser.cli import mcp_server

        return mcp_server.serve(args.root, exec_py=args.exec_py)
    return failure("", f"a command is required\n\n{parser.format_usage().strip()} (see '{PROG} --help')")


def _verbosity(args: argparse.Namespace) -> int:
    quiet, verbose = getattr(args, "quiet", False), getattr(args, "verbose", False)
    if quiet and verbose:
        raise _UsageError(f"--quiet and --verbose cannot be combined (see '{PROG} --help')", _command_name(args))
    return QUIET if quiet else VERBOSE if verbose else NORMAL


def main(argv: Optional[List[str]] = None) -> int:
    """Run ``besser`` with ``argv`` (default: the process arguments) and return its exit status."""
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    as_json = "--json" in argv
    set_verbosity(NORMAL)
    try:
        args = _parse(parser, argv)
        as_json = getattr(args, "json", False)
        set_verbosity(_verbosity(args))
        # --quiet also mutes the warnings BESSER logs, which would otherwise reach standard error.
        logging.disable(logging.WARNING if getattr(args, "quiet", False) else logging.NOTSET)
        result = _run(args, parser)
    except _UsageError as exc:
        result = failure(exc.command, str(exc))
    except Exception as exc:
        # A defect in BESSER, not in the input: the traceback is what a bug report needs.
        traceback.print_exc()
        result = failure("", f"unexpected {type(exc).__name__}: {exc}", EXIT_FAILED)
    finally:
        logging.disable(logging.NOTSET)
    if result is None:  # mcp served: standard output was the protocol's, and the session is over
        return EXIT_OK
    print_result(result, as_json)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
