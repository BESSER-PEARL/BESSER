"""``besser completion``: shell completion scripts generated from the argument parser.

The parser is read once into a shell-neutral tree (:func:`describe`): the global options, then
each command with its options and positional arguments, and what completes each value. Three
small renderers turn that tree into a bash, zsh or fish script. Since the tree is read at run
time, an option added to the parser is completed without a change here. The generator ids are
written into the script, so it completes the generators installed when it was printed.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

from besser.cli.output import Result

COMPLETION_SHELLS = ("bash", "zsh", "fish")

# What completes a value: a fixed list of words, or one of these kinds.
GENERATOR = "generator"  # a registered generator id
MODEL = "model"  # a file, .py and .json first, or a directory
DIRECTORY = "directory"
FILE = "file"
TEXT = "text"  # free text: nothing is offered

# The metavars whose value is a path, and what completes it.
_METAVAR_KINDS = {"DIR": DIRECTORY, "FILE": FILE}


@dataclass(frozen=True)
class Value:
    """What completes an option's value or a positional argument."""

    kind: str
    words: tuple[str, ...] = ()  # the choices, or the generator ids


@dataclass(frozen=True)
class Option:
    """An option: its spellings, its help, and its value (``None`` for a flag)."""

    names: tuple[str, ...]
    help: str
    value: Value | None
    repeatable: bool

    @property
    def short(self) -> list[str]:
        return [name[1:] for name in self.names if not name.startswith("--")]

    @property
    def long(self) -> list[str]:
        return [name[2:] for name in self.names if name.startswith("--")]


@dataclass(frozen=True)
class Positional:
    """A positional argument; ``many`` when it takes several words (``nargs`` ``+`` or ``*``)."""

    name: str
    help: str
    value: Value
    many: bool


@dataclass(frozen=True)
class Command:
    """A command (or, with an empty name, the program itself): its options and positionals."""

    name: str
    help: str
    options: tuple[Option, ...]
    positionals: tuple[Positional, ...]


@dataclass(frozen=True)
class Tree:
    """The program's global options and its commands."""

    prog: str
    root: Command
    commands: tuple[Command, ...]


def _help(action: argparse.Action) -> str:
    """The help of ``action`` on one line, as ``--help`` shows it (``%%`` is a percent sign)."""
    text = " ".join((action.help or "").split())
    return text.replace("%%", "%")


def _value(action: argparse.Action, generator_ids: tuple[str, ...]) -> Value:
    if action.choices:
        return Value("words", tuple(str(choice) for choice in action.choices))
    if not action.option_strings:
        if action.dest == "generator":
            return Value(GENERATOR, generator_ids)
        return Value(MODEL)
    return Value(_METAVAR_KINDS.get(str(action.metavar), TEXT))


def _command(name: str, help_text: str, parser: argparse.ArgumentParser,
             generator_ids: tuple[str, ...]) -> Command:
    options, positionals = [], []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction) or action.help == argparse.SUPPRESS:
            continue
        if action.option_strings:
            takes_value = action.nargs != 0
            options.append(Option(
                names=tuple(action.option_strings), help=_help(action),
                value=_value(action, generator_ids) if takes_value else None,
                repeatable=isinstance(action, (argparse._AppendAction, argparse._CountAction)),
            ))
        else:
            positionals.append(Positional(
                name=action.dest, help=_help(action), value=_value(action, generator_ids),
                many=action.nargs in ("+", "*", argparse.REMAINDER),
            ))
    return Command(name, help_text, tuple(options), tuple(positionals))


def _subparsers(parser: argparse.ArgumentParser) -> argparse._SubParsersAction | None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def describe(parser: argparse.ArgumentParser, generator_ids: tuple[str, ...]) -> Tree:
    """Read ``parser`` into a :class:`Tree`; ``generator_ids`` complete the ``generator`` arguments."""
    root = _command("", parser.description or "", parser, generator_ids)
    commands = []
    subparsers = _subparsers(parser)
    if subparsers is not None:
        helps = {choice.dest: _help(choice) for choice in subparsers._choices_actions}
        for name, subparser in subparsers.choices.items():
            commands.append(_command(name, helps.get(name, ""), subparser, generator_ids))
    return Tree(parser.prog, root, tuple(commands))


def _function_name(prog: str) -> str:
    return "_" + re.sub(r"\W", "_", prog)


# -- bash ---------------------------------------------------------------------------------------

def _bash_words(words) -> str:
    """``words`` as one single-quoted bash word."""
    return "'" + " ".join(words).replace("'", "'\\''") + "'"


def _bash_value_case(value: Value) -> str:
    """What ``_<prog>_complete`` takes for ``value``: its kind, then the words for a list."""
    if value.kind in ("words", GENERATOR):
        return _bash_words(("words", *value.words))
    return value.kind


def render_bash(tree: Tree) -> str:
    """A bash script defining the completion function and registering it with ``complete``."""
    fn = _function_name(tree.prog)
    commands = [tree.root, *tree.commands]
    options_cases, value_cases, positional_cases = [], [], []
    for command in commands:
        names = [name for option in command.options for name in option.names]
        options_cases.append(f'        "{command.name}") echo {_bash_words(names)} ;;')
        for option in command.options:
            if option.value is not None:
                pattern = "|".join(f'"{command.name} {name}"' for name in option.names)
                value_cases.append(f"        {pattern}) echo {_bash_value_case(option.value)} ;;")
        for index, positional in enumerate(command.positionals):
            pattern = f'"{command.name} "*' if positional.many else f'"{command.name} {index}"'
            positional_cases.append(f"        {pattern}) echo {_bash_value_case(positional.value)} ;;")
    command_names = [command.name for command in tree.commands]
    lines = [
        f"# bash completion for {tree.prog}, generated by '{tree.prog} completion bash'.",
        f'# Load it with: eval "$({tree.prog} completion bash)"',
        "",
        f"# The options of a command ({fn}_options generate), or of {tree.prog} itself (\"\").",
        f"{fn}_options() {{",
        '    case "$1" in',
        *options_cases,
        "    esac",
        "}",
        "",
        "# What completes the value of an option (\"generate -o\"); nothing for a flag.",
        f"{fn}_option_value() {{",
        '    case "$1 $2" in',
        *value_cases,
        "    esac",
        "}",
        "",
        "# What completes a command's positional argument number N, from 0 (\"generate 0\").",
        f"{fn}_positional() {{",
        '    case "$1 $2" in',
        *positional_cases,
        "    esac",
        "}",
        "",
        "# Fill COMPREPLY with the completions of \"$cur\": words W1 W2..., model, directory, file or text.",
        f"{fn}_complete() {{",
        '    case "$1" in',
        '        words) mapfile -t COMPREPLY < <(compgen -W "${*:2}" -- "$cur") ;;',
        "        model)",
        "            compopt -o filenames 2>/dev/null",
        '            mapfile -t COMPREPLY < <(compgen -d -- "$cur"; compgen -f -X \'!*.py\' -- "$cur";'
        ' compgen -f -X \'!*.json\' -- "$cur") ;;',
        '        directory) compopt -o filenames 2>/dev/null; mapfile -t COMPREPLY < <(compgen -d -- "$cur") ;;',
        '        file) compopt -o filenames 2>/dev/null; mapfile -t COMPREPLY < <(compgen -f -- "$cur") ;;',
        "        *) COMPREPLY=() ;;",
        "    esac",
        "}",
        "",
        f"{fn}() {{",
        '    local cur="${COMP_WORDS[COMP_CWORD]}" command="" word value index=0 i',
        "    COMPREPLY=()",
        "    # The command is the first word that is not an option; count the positionals after it,",
        "    # skipping the value of an option that takes one.",
        "    for ((i = 1; i < COMP_CWORD; i++)); do",
        '        word="${COMP_WORDS[i]}"',
        '        if [[ -z "$command" ]]; then',
        '            [[ "$word" != -* ]] && command="$word"',
        '        elif [[ "$word" == -* ]]; then',
        f'            [[ -n "$({fn}_option_value "$command" "$word")" ]] && ((i++))',
        "        else",
        "            ((index++))",
        "        fi",
        "    done",
        f'    value="$({fn}_option_value "$command" "${{COMP_WORDS[COMP_CWORD-1]}}")"',
        '    if [[ -n "$command" && -n "$value" ]]; then',
        f"        {fn}_complete $value",
        '    elif [[ "$cur" == -* ]]; then',
        f'        mapfile -t COMPREPLY < <(compgen -W "$({fn}_options "$command")" -- "$cur")',
        '    elif [[ -z "$command" ]]; then',
        f"        mapfile -t COMPREPLY < <(compgen -W {_bash_words(command_names)} -- \"$cur\")",
        "    else",
        f'        {fn}_complete $({fn}_positional "$command" "$index")',
        "    fi",
        "}",
        "",
        f"complete -F {fn} {tree.prog}",
    ]
    return "\n".join(lines)


# -- zsh ----------------------------------------------------------------------------------------

def _zsh_quote(text: str) -> str:
    """``text`` as one single-quoted zsh word."""
    return "'" + text.replace("'", "'\\''") + "'"


def _zsh_escape(text: str) -> str:
    """``text`` inside an ``_arguments`` spec: brackets and colons escaped."""
    return re.sub(r"([\\\[\]:])", r"\\\1", text)


def _zsh_action(value: Value) -> str:
    if value.kind in ("words", GENERATOR):
        return "(" + " ".join(value.words) + ")"
    return {MODEL: '_files -g "*.(py|json)"', DIRECTORY: "_files -/", FILE: "_files"}.get(value.kind, " ")


def _zsh_specs(command: Command) -> list[str]:
    """The ``_arguments`` specs of ``command``'s options and positionals."""
    specs = []
    for option in command.options:
        exclusive = "" if option.repeatable or len(option.names) == 1 else "(" + " ".join(option.names) + ")"
        for name in option.names:
            spec = f"{'*' if option.repeatable else ''}{exclusive}{name}[{_zsh_escape(option.help)}]"
            if option.value is not None:
                metavar = option.value.kind if option.value.kind != "words" else "value"
                spec += f":{metavar}:{_zsh_action(option.value)}"
            specs.append(spec)
    for index, positional in enumerate(command.positionals, start=1):
        position = "*" if positional.many else str(index)
        specs.append(f"{position}:{_zsh_escape(positional.name)}:{_zsh_action(positional.value)}")
    return specs


def render_zsh(tree: Tree) -> str:
    """A zsh script for ``compsys``: an autoloadable ``#compdef`` file, or one to ``eval``."""
    fn = _function_name(tree.prog)
    lines = [
        f"#compdef {tree.prog}",
        f"# zsh completion for {tree.prog}, generated by '{tree.prog} completion zsh'.",
        f'# Load it with: eval "$({tree.prog} completion zsh)"',
        "",
        f"{fn}() {{",
        "    local curcontext=\"$curcontext\" state line",
        "    typeset -A opt_args",
        "    _arguments -C \\",
        *(f"        {_zsh_quote(spec)} \\" for spec in _zsh_specs(tree.root)),
        "        '1: :->command' '*:: :->argument'",
        '    case "$state" in',
        "        command)",
        "            local -a commands=(",
        *(f"                {_zsh_quote(_zsh_escape(c.name) + ':' + c.help)}" for c in tree.commands),
        "            )",
        "            _describe command commands ;;",
        "        argument)",
        '            case "$line[1]" in',
    ]
    for command in tree.commands:
        specs = _zsh_specs(command)
        lines.append(f"                {command.name})")
        if specs:
            lines.append("                    _arguments \\")
            lines.extend(f"                        {_zsh_quote(spec)} \\" for spec in specs[:-1])
            lines.append(f"                        {_zsh_quote(specs[-1])} ;;")
        else:
            lines.append("                    ;;")
    lines += [
        "            esac ;;",
        "    esac",
        "}",
        "",
        "if [[ \"${zsh_eval_context[-1]}\" == loadautofunc ]]; then",
        f'    {fn} "$@"',
        "else",
        f"    compdef {fn} {tree.prog}",
        "fi",
    ]
    return "\n".join(lines)


# -- fish ---------------------------------------------------------------------------------------

def _fish_quote(text: str) -> str:
    """``text`` as one single-quoted fish word."""
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _fish_arguments(value: Value) -> str:
    """The ``complete`` arguments that offer ``value``."""
    if value.kind in ("words", GENERATOR):
        return "-a " + _fish_quote(" ".join(value.words))
    return {
        MODEL: "-a '(__fish_complete_suffix .py) (__fish_complete_suffix .json)'",
        DIRECTORY: "-a '(__fish_complete_directories)'",
        FILE: "-F",
    }.get(value.kind, "")


def _fish_option(prog: str, condition: str, option: Option) -> str:
    parts = [f"complete -c {prog} -n {_fish_quote(condition)}"]
    parts += [f"-s {name}" for name in option.short] + [f"-l {name}" for name in option.long]
    if option.value is not None:
        parts.append("-r")
        arguments = _fish_arguments(option.value)
        if arguments:
            parts.append(arguments)
    parts.append(f"-d {_fish_quote(option.help)}")
    return " ".join(parts)


def render_fish(tree: Tree) -> str:
    """A fish script of ``complete`` commands, with two helper functions."""
    prog, fn = tree.prog, "_" + _function_name(tree.prog)
    value_cases = []
    for command in tree.commands:
        names = [name for option in command.options if option.value is not None for name in option.names]
        if names:
            value_cases.append(f"        case {command.name}")
            value_cases.append(f"            contains -- $argv[2] {' '.join(names)}")
    lines = [
        f"# fish completion for {prog}, generated by '{prog} completion fish'.",
        f"# Install it with: {prog} completion fish > ~/.config/fish/completions/{prog}.fish",
        "",
        "# Whether option $argv[2] of command $argv[1] takes a value.",
        f"function {fn}_takes_value",
        "    switch $argv[1]",
        *value_cases,
        "        case '*'",
        "            return 1",
        "    end",
        "end",
        "",
        "# Whether the word being typed is positional argument $argv[2] (from 0) of command $argv[1];",
        "# with a third argument, positional $argv[2] or any after it.",
        f"function {fn}_positional",
        "    set -l words (commandline -opc)",
        "    set -l command",
        "    set -l index 0",
        "    set -l skip 0",
        "    for word in $words[2..-1]",
        "        if test $skip = 1",
        "            set skip 0",
        "        else if test -z \"$command\"",
        "            string match -q -- '-*' $word; or set command $word",
        "        else if string match -q -- '-*' $word",
        f"            {fn}_takes_value $command $word; and set skip 1",
        "        else",
        "            set index (math $index + 1)",
        "        end",
        "    end",
        "    test \"$command\" = $argv[1]; or return 1",
        "    test $skip = 0; or return 1",
        "    if set -q argv[3]",
        "        test $index -ge $argv[2]",
        "    else",
        "        test $index -eq $argv[2]",
        "    end",
        "end",
        "",
        f"complete -c {prog} -f",
    ]
    for command in tree.commands:
        lines.append(f"complete -c {prog} -n __fish_use_subcommand -a {command.name} -d {_fish_quote(command.help)}")
    lines.extend(_fish_option(prog, "__fish_use_subcommand", option) for option in tree.root.options)
    for command in tree.commands:
        lines.append("")
        condition = f"__fish_seen_subcommand_from {command.name}"
        lines.extend(_fish_option(prog, condition, option) for option in command.options)
        for index, positional in enumerate(command.positionals):
            arguments = _fish_arguments(positional.value)
            if arguments:
                test = f"{fn}_positional {command.name} {index}" + (" +" if positional.many else "")
                lines.append(f"complete -c {prog} -n {_fish_quote(test)} {arguments}")
    return "\n".join(lines)


_RENDERERS = {"bash": render_bash, "zsh": render_zsh, "fish": render_fish}


def _generator_ids() -> tuple[str, ...]:
    from besser.generators.registry import GENERATORS

    return tuple(spec.id for spec in GENERATORS)


def completion_script(shell: str, parser: argparse.ArgumentParser) -> Result:
    """The completion script for ``shell``, built from ``parser``.

    The script is the text printed on standard output; with ``--json`` it is the ``script`` key.
    """
    script = _RENDERERS[shell](describe(parser, _generator_ids()))
    return Result(command="completion", data={"shell": shell, "script": script}, text=script)
