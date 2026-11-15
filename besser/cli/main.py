"""``besser``: start, describe, validate and inspect B-UML models, run generators and convert model files.

This module is the console-script entry point registered in ``setup.cfg``;
``python -m besser`` runs it too.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shlex
import sys
import traceback
from typing import List, Optional

from besser.cli import commands
from besser.cli.completion import COMPLETION_SHELLS, completion_script
from besser.cli.config import CONFIG_FILENAME
from besser.cli.importing import IMPORT_FORMATS
from besser.cli.model_edit import OPERATIONS, Operation
from besser.cli.starters import DEFAULT_KIND, STARTERS
from besser.cli.output import (
    EXIT_FAILED,
    EXIT_OK,
    NORMAL,
    QUIET,
    VERBOSE,
    Result,
    default_verbosity,
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

When standard output is not a terminal (a coding agent, a script), a command
that writes or checks files prints a one-line summary, the warnings and the
errors, and no notes. BESSER_OUTPUT=full prints the whole report anyway;
BESSER_OUTPUT=terse prints the summary on a terminal too.

a coding agent's loop ('besser guide' prints its rules, 'init' writes them to AGENTS.md):
  besser init                                  model.py, besser.toml, AGENTS.md, CLAUDE.md, .ignore
  besser info model.py                         the model and the API's routes, in place of reading files
  besser info model.py --class Deal            one class in full, with the generated names to import
  besser model add-attribute Deal score:int=0  change the model without reading or patching model.py
  besser apply                                 after each model change: validate, build, run the tests

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


_MODEL_EDIT_EXAMPLES = """\
attributes are name:type, name:type? (optional) or name:type=default; types as 'besser info'
prints them (str, int, float, bool, date, datetime, time, timedelta, any) or an enumeration of
the model (a link to a class is an association); an association end is role:Class[multiplicity]
([1], [0..1], [*], [1..*]).

examples:
  besser model add-class Invoice number:str total:float paid:bool=false
  besser model add-attribute Contact source:str? phone:str
  besser model add-association invoice_customer invoices:Invoice[*] customer:Contact[1]
  besser model set-enum Stage LEAD WON LOST
  besser model add-method Deal "discount(rate:float) -> float" --code "def discount(self, rate): ..."
  besser model add-constraint Deal "self.amount >= 0" --name amount_positive
  besser model add-parent Customer Party
  besser model rename Customer Account          (also Class.attribute, Enum.LITERAL, association,
                                                 association.end)
  besser model remove Contact.phone
  besser model add-screen Invoice --title Bills (a page with a table of the class, in the GUI model)
  besser model add-class Tag label:str --screen (a class and its page)
  besser model add-menu                         (a menu with a link to every page, on every page)
  besser model add-column Invoice customer.name --label Customer   (a column of a related object)
  besser model order-columns Invoice number customer.name          (these columns first in its tables)
  besser model add-page Overview --route /home --main              (a page for cards and charts)
  besser model add-card Overview "Paid" sum:Invoice.total --where "paid == true"
  besser model add-chart Overview "Invoices by status" count:Invoice --by status
  besser model batch - <<'EOF'                  (several changes, one write)
  besser model set-attribute Book.pages :int?=0 (type, ? and default in place; = drops the default)
  besser model set-multiplicity book_authors.authors "[*]"   (also Class.end)
  besser model set-abstract Party true
  besser model remove-parent Customer Party     (the class stays)
"""

# What rename and remove take.
_MODEL_PATH_HELP = "Class, Enumeration, association, association.end, constraint, Class.member or Enumeration.LITERAL"

# Operation -> its arguments, named as the function of besser.cli.model_edit takes them.
_MODEL_OPERATION_ARGUMENTS: dict[str, list[tuple[tuple[str, ...], dict]]] = {
    "add-class": [
        (("name",), {"help": "the class name"}),
        (("attributes",), {"nargs": "*", "metavar": "ATTRIBUTE", "help": "name:type, name:type? or name:type=default"}),
        (("--parent",), {"metavar": "CLASS", "help": "the class it specializes"}),
        (("--abstract",), {"action": "store_true", "help": "an abstract class"}),
        (("--screen",), {"action": "store_true", "help": "and a screen with a table of it (see add-screen)"}),
        (("--title",), {"help": "the screen's title (asks for the screen)"}),
        (("--route",), {"metavar": "/PATH", "help": "the screen's address (asks for the screen)"}),
        (("--rows",), {"metavar": "N", "type": int, "help": "rows its table shows a page (asks for the screen)"}),
    ],
    "add-attribute": [
        (("class_name",), {"metavar": "CLASS", "help": "the class"}),
        (("attributes",), {"nargs": "+", "metavar": "ATTRIBUTE", "help": "name:type, name:type? or name:type=default"}),
        (("--derived",), {"action": "store_true", "help": "computed by the server: clients never send them (your "
                                                          "code sets them)"}),
    ],
    "add-parent": [
        (("class_name",), {"metavar": "CLASS", "help": "the class"}),
        (("parent",), {"metavar": "PARENT", "help": "the class it specializes"}),
    ],
    "add-association": [
        (("name",), {"help": "the association name"}),
        (("first",), {"metavar": "END", "help": "role:Class[multiplicity], e.g. invoices:Invoice[*]"}),
        (("second",), {"metavar": "END", "help": "role:Class[multiplicity], e.g. customer:Contact[1]"}),
        (("--composite",), {"action": "store_true", "help": "the first end is the whole, the second its parts"}),
    ],
    "add-method": [
        (("class_name",), {"metavar": "CLASS", "help": "the class"}),
        (("signature",), {"help": "name(param:type, ...) -> type"}),
        (("--code",), {"help": "the method's Python code (a def), which the backend runs"}),
    ],
    "add-constraint": [
        (("class_name",), {"metavar": "CLASS", "help": "the context class"}),
        (("expression",), {"help": "the OCL invariant, e.g. 'self.amount >= 0'"}),
        (("--name",), {"help": "the constraint name (default: <class>_rule_<n>)"}),
    ],
    "add-screen": [
        (("class_name",), {"metavar": "CLASS", "help": "the class the screen's table shows"}),
        (("--title",), {"help": "the screen's title (default: the plural of the class name)"}),
        (("--route",), {"metavar": "/PATH", "help": "the page's address (default: the title's words, in lower case)"}),
        (("--rows",), {"metavar": "N", "type": int, "help": "rows the table shows a page (default: 5)"}),
    ],
    "add-page": [
        (("title",), {"help": "the page's title, also its heading"}),
        (("--route",), {"metavar": "/PATH", "help": "the page's address (default: the title's words, in lower case)"}),
        (("--main",), {"action": "store_true", "help": "make it the main page, the one the application opens on"}),
    ],
    "add-menu": [],
    "add-column": [
        (("class_name",), {"metavar": "CLASS", "help": "the class whose tables get the column"}),
        (("path",), {"metavar": "END.ATTRIBUTE", "help": "an association end to one object, then an attribute of "
                                                         "its class: account.name (or an attribute of the class "
                                                         "itself)"}),
        (("--label",), {"help": "the column's label (default: the end's name)"}),
    ],
    "order-columns": [
        (("class_name",), {"metavar": "CLASS", "help": "the class whose tables are reordered"}),
        (("columns",), {"metavar": "COLUMN", "nargs": "+",
                        "help": "the columns that come first, in this order: an attribute, or end.attribute for a "
                                "related object's column; the other columns follow as they were"}),
    ],
    "add-card": [
        (("screen",), {"help": "the screen: its name, title or route"}),
        (("title",), {"help": "the card's title"}),
        (("value",), {"help": "sum:Class.attribute (also avg, min, max, median) or count:Class"}),
        (("--where",), {"metavar": "CONDITION", "help": "only the objects where attribute == value (also != > >= < "
                                                        "<=), several joined by 'and'"}),
    ],
    "add-link": [
        (("screen",), {"help": "the screen: its name, title or route"}),
        (("label",), {"help": "the link's text"}),
        (("url",), {"metavar": "URL", "help": "/route of the application, or http(s)://... (a route of the backend "
                                              "needs its whole address)"}),
    ],
    "add-chart": [
        (("screen",), {"help": "the screen: its name, title or route"}),
        (("title",), {"help": "the chart's title"}),
        (("value",), {"help": "sum:Class.attribute (also avg, min, max, median) or count:Class"}),
        (("--by",), {"required": True, "metavar": "ATTRIBUTE", "help": "the attribute of the class with one bar, "
                                                                      "point or slice per value"}),
        (("--kind",), {"choices": ["bar", "line", "pie"], "default": "bar", "help": "the chart (default: bar)"}),
        (("--where",), {"metavar": "CONDITION", "help": "only the objects where attribute == value (also != > >= < "
                                                        "<=), several joined by 'and'"}),
    ],
    "set-enum": [
        (("name",), {"help": "the enumeration (created if missing)"}),
        (("literals",), {"nargs": "+", "metavar": "LITERAL", "help": "all its literals, in order"}),
    ],
    "rename": [
        (("path",), {"help": _MODEL_PATH_HELP}),
        (("new_name",), {"metavar": "NEW_NAME", "help": "the new name"}),
        (("--keep-pages",), {"action": "store_true", "help": "the pages keep their titles, column labels and "
                                                             "routes (default: those derived from the old name "
                                                             "follow)"}),
    ],
    "remove": [
        (("path",), {"help": _MODEL_PATH_HELP}),
    ],
    "set-attribute": [
        (("path",), {"metavar": "CLASS.ATTRIBUTE", "help": "the attribute"}),
        (("spec",), {"metavar": "CHANGE", "help": "what changes, as add-attribute writes it without the name: "
                                                  ":type, :type? (optional), =default, = (no default)"}),
    ],
    "set-multiplicity": [
        (("path",), {"metavar": "END", "help": "the association end: association.end or Class.end"}),
        (("multiplicity",), {"metavar": "MULTIPLICITY", "help": "[1], [0..1], [*], [1..*] or [n..m]"}),
    ],
    "set-abstract": [
        (("class_name",), {"metavar": "CLASS", "help": "the class"}),
        (("abstract",), {"choices": ["true", "false"], "help": "true: abstract; false: concrete"}),
    ],
    "remove-parent": [
        (("class_name",), {"metavar": "CLASS", "help": "the class"}),
        (("parent",), {"metavar": "PARENT", "help": "the parent it no longer specializes"}),
    ],
}


def _add_model_file_options(parser: argparse.ArgumentParser) -> None:
    """``--file`` and ``--use``, accepted before and after the operation."""
    parser.add_argument("-f", "--file", default=argparse.SUPPRESS, metavar="MODEL",
                        help="the .py model file (default: the model of besser.toml, else model.py)")
    parser.add_argument("--use", action="append", default=argparse.SUPPRESS, metavar="NAME",
                        help="the domain model to change, or the GUI model add-screen adds to, when the file "
                             "holds several; repeat to name one of each")


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
        help="print only errors and the output of completion, generators, guide, info, new --list, schema and --version "
             "(and, with --strict, the warnings that fail the run)",
    )
    common.add_argument(
        "-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
        help="print the whole report, even when standard output is not a terminal, and the traceback of a "
             "model file or generator that fails, on standard error",
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
    new.add_argument("--no-agent-files", dest="agent_files", action="store_false",
                     help="write only the model, not the agent guide (AGENTS.md, CLAUDE.md) and .ignore beside it "
                          "(written only in a folder with a besser.toml)")

    init = add_command(
        "init", "set up a project for a coding agent",
        "Write a starter model (model.py), a besser.toml that builds it (a 'backend' API for a\n"
        "class diagram), the agent guide in AGENTS.md (imported by CLAUDE.md) and the .ignore\n"
        "rules that keep the generated folders out of an agent's searches. A model or besser.toml\n"
        "that exists is kept; the guide and the rules are blocks of their files, updated in place.\n"
        "A kind no generator reads gets no besser.toml, and so no guide.",
    )
    init.add_argument("folder", nargs="?", default=".", help="the project folder (default: the current one)")
    init.add_argument("--kind", choices=list(STARTERS), default=DEFAULT_KIND,
                      help=f"the kind of model to start from (default: {DEFAULT_KIND})")

    add_command(
        "guide", "print the rules for a coding agent",
        "Print what a coding agent needs to work in a BESSER project, in about 3 KB: what not to read,\n"
        "the commands that change the model, the one command after a change, and where hand-written\n"
        "code goes; with the lines on screens when an output of besser.toml in the current folder reads\n"
        "a GUI model (web_app). 'besser new' and 'besser init' write it to AGENTS.md.",
    )

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
    info.add_argument("--class", dest="class_name", metavar="CLASS",
                      help="describe one class in full: members, inherited attributes, association ends, "
                           "constraints, the backend's routes and code names for it, and its page")
    info.add_argument("--all", dest="everything", action="store_true",
                      help="print a large model in full (by default one with more than 20 classes is "
                           "summarised by its names)")

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
        "Merging an edited file with a changed one needs pygit2 (pip install \"besser[regen]\");\n"
        "'besser generators' says which generators support it.\n\n" + _EXECUTES_PYTHON,
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

    apply = add_command(
        "apply", "build every output, then run the tests",
        "The one command to run after a model change: 'besser build' (each model is validated first),\n"
        "then, when the build succeeds, the 'test' command line of besser.toml, run from its folder\n"
        "without a shell. Prints one line when everything passes, the errors or the end of the\n"
        "test output otherwise. Exit status 1 when the build or the tests fail, 2 when besser.toml, a model\n"
        "or an output cannot be used (as for 'besser build').\n\n" + _EXECUTES_PYTHON,
    )
    apply.add_argument("-c", "--config", default=CONFIG_FILENAME, metavar="FILE",
                       help=f"the configuration file (default: {CONFIG_FILENAME} in the current directory)")

    model = add_command(
        "model", "change the domain model of a .py file: add, rename, remove; add screens",
        "Change the domain model of a B-UML .py file with one command per change, instead of\n"
        "reading the file and editing it. Only the characters of the change are written: the\n"
        "rest of the file, comments and layout, stays as it is. The file is written only when\n"
        "the result loads, holds the change and is as valid as before; one line says what changed.\n"
        "With a GUI model in the file, a class's tables follow its attributes; the line says so.\n\n"
        "operations:\n" + "".join(f"  {name:<16} {summary}\n" for name, (summary, _) in OPERATIONS.items())
        + "  batch            several operations, one per line of a file (- reads standard input), "
        "written together\n\n" + _MODEL_EDIT_EXAMPLES + "\n" + _EXECUTES_PYTHON,
    )
    _add_model_file_options(model)
    model_operations = model.add_subparsers(dest="operation", metavar="<operation>", parser_class=_Parser)
    for name, (summary, _) in OPERATIONS.items():
        operation = model_operations.add_parser(name, help=summary, description=summary, parents=[common])
        _add_model_file_options(operation)
        for argument, options in _MODEL_OPERATION_ARGUMENTS[name]:
            operation.add_argument(*argument, **options)
    batch = model_operations.add_parser(
        "batch", parents=[common], help="several operations, one per line, written together",
        description="Make the operations listed one per line in FILE (- reads standard input), each\n"
                    "written as after 'besser model' (blank lines and # comments are skipped); the\n"
                    "file is written once, when all of them apply.",
    )
    _add_model_file_options(batch)
    batch.add_argument("source", nargs="?", default="-", metavar="FILE", help="the operations (default: -)")

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
        "Run a Model Context Protocol server on standard input and output, exposing new,\n"
        "schema, validate, info, generators, generate, regenerate and convert as tools to an\n"
        "MCP client (an AI coding agent).\n\n"
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
        return commands.new(args.path, kind=args.kind, force=args.force, list_kinds=args.list_kinds,
                            agent_files=args.agent_files)
    if args.command == "init":
        return commands.init(args.folder, kind=args.kind)
    if args.command == "guide":
        return commands.guide()
    if args.command == "apply":
        return commands.apply(config_path=args.config)
    if args.command == "schema":
        return commands.schema(args.topic)
    if args.command == "build":
        return commands.build(args.output_name, config_path=args.config, dry_run=args.dry_run)
    if args.command == "validate":
        return commands.validate(args.model, strict=args.strict, exec_py=args.exec_py, ocl=args.ocl)
    if args.command == "info":
        return commands.info(args.model, args.class_name, args.everything)
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
    if args.command == "model":
        return _model(args, parser)
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


def _route_argument(value: object) -> object:
    """A route as it was typed, when an MSYS shell (Git Bash on Windows) rewrote it on the way: it turns an
    argument that starts with ``/`` into a Windows path (``/dashboard`` -> ``C:/Program Files/Git/dashboard``,
    ``/a/home`` -> ``A:/home``). No route is a Windows path, so the rewrite is undone."""
    if os.name != "nt" or not os.environ.get("MSYSTEM") or not isinstance(value, str):
        return value
    text = value.replace("\\", "/")
    root = os.path.dirname(os.environ.get("EXEPATH", "").rstrip("\\/")).replace("\\", "/")
    for prefix in filter(None, (root, re.sub(r"/(usr|mingw64|mingw32)$", "", root))):
        if text.lower().startswith(prefix.lower() + "/"):
            return text[len(prefix):]
    drive = re.fullmatch(r"([A-Za-z]):(/.*)", text)
    return f"/{drive[1].lower()}{drive[2]}" if drive else value


def _model_operation(args: argparse.Namespace) -> Operation:
    """The operation ``args`` (parsed by an operation's parser) names."""
    names = [argument[0].lstrip("-").replace("-", "_") for argument, _ in _MODEL_OPERATION_ARGUMENTS[args.operation]]
    values = {name: getattr(args, name) for name in names}
    for name in ("route", "screen"):  # a screen is named by its route too
        if name in values:
            values[name] = _route_argument(values[name])
    return Operation(args.operation, values)


# The options of an operation's parser that are the whole batch's (an operation line's would be ignored).
_BATCH_OPTIONS = ("file", "use", "json", "quiet", "verbose")


def _model(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Result:
    if not args.operation:
        raise _UsageError(f"an operation is required (see '{PROG} model --help')", "model")
    if args.operation != "batch":
        operations = [_model_operation(args)]
    else:
        operation_parsers = _subparsers(_subparsers(parser).choices["model"]).choices
        try:
            text = sys.stdin.read() if args.source == "-" else open(args.source, encoding="utf-8").read()
        except (OSError, UnicodeDecodeError) as exc:
            raise _UsageError(f"cannot read {args.source}: {exc}", "model") from None
        operations = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                words = shlex.split(line)
            except ValueError as exc:
                raise _UsageError(f"{args.source}: line {number}: {exc}", "model") from None
            where = f"{args.source}: line {number}"
            if words[0] not in OPERATIONS:
                raise _UsageError(f"{where}: '{words[0]}' is not an operation ({', '.join(OPERATIONS)})", "model")
            if "-h" in words or "--help" in words:  # argparse would print the help and apply nothing
                raise _UsageError(f"{where}: a batch line takes no -h ('{PROG} model {words[0]} --help')", "model")
            try:
                parsed = operation_parsers[words[0]].parse_args(words[1:])
            except _UsageError as exc:
                raise _UsageError(f"{where}: {exc}", "model") from None
            whole_batch = sorted(option for option in _BATCH_OPTIONS if option in vars(parsed))
            if whole_batch:
                raise _UsageError(f"{where}: {', '.join('--' + option for option in whole_batch)} applies to the "
                                  f"whole batch: give it before 'batch'", "model")
            parsed.operation = words[0]
            operations.append(_model_operation(parsed))
        if not operations:
            raise _UsageError(f"{args.source}: no operation", "model")
    return commands.model(operations, getattr(args, "file", None), getattr(args, "use", []))


def _verbosity(args: argparse.Namespace) -> int:
    quiet, verbose = getattr(args, "quiet", False), getattr(args, "verbose", False)
    if quiet and verbose:
        raise _UsageError(f"--quiet and --verbose cannot be combined (see '{PROG} --help')", _command_name(args))
    if quiet:
        return QUIET
    if verbose:
        return VERBOSE
    if args.command == "mcp":  # standard output is the protocol's: what models print goes to standard error
        return NORMAL
    try:
        return default_verbosity()
    except ValueError as exc:
        raise _UsageError(str(exc), _command_name(args)) from None


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
    try:
        print_result(result, as_json)
        sys.stdout.flush()
    except BrokenPipeError:
        # The reader closed the pipe early ('besser schema domain | head -1'): what it read is the
        # output, and the status stays the command's. Standard output goes to devnull so that the
        # interpreter's flush at exit does not raise again.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
