"""``besser completion``: the scripts follow the parser, parse in their shell and complete in bash."""

import argparse
import re
import shutil
import subprocess

import pytest

from besser.cli.completion import COMPLETION_SHELLS
from besser.cli.main import build_parser
from besser.generators.registry import GENERATORS

GENERATOR_IDS = [spec.id for spec in GENERATORS]

# The command that checks a script's syntax without running it, per shell.
SYNTAX_CHECK = {"bash": ["bash", "-n"], "zsh": ["zsh", "-n"], "fish": ["fish", "--no-execute"]}


def _commands() -> dict[str, argparse.ArgumentParser]:
    parser = build_parser()
    subparsers = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return dict(subparsers.choices)


def _option_names(parser: argparse.ArgumentParser) -> list[str]:
    """The option spellings of ``parser``, in its order, as the scripts list them."""
    return [name for action in parser._actions if action.help != argparse.SUPPRESS for name in action.option_strings]


def test_bash_lists_exactly_the_options_of_each_command(besser):
    script = besser("completion", "bash").stdout.splitlines()

    for name, parser in {"": build_parser(), **_commands()}.items():
        assert f"        \"{name}\") echo '{' '.join(_option_names(parser))}' ;;" in script, name
    assert f"        \"generate 0\") echo 'words {' '.join(GENERATOR_IDS)}' ;;" in script
    assert f"        \"regenerate 0\") echo 'words {' '.join(GENERATOR_IDS)}' ;;" in script


def _zsh_blocks(script: str) -> dict[str, list[str]]:
    """The option names in each command's ``_arguments`` block of the zsh script."""
    blocks = re.findall(r"^ {16}([a-z]+)\)\n(.*?);;$", script, flags=re.MULTILINE | re.DOTALL)
    return {name: re.findall(r"^ +'\*?(?:\([^)]*\))?(-[\w-]+)\[", body, flags=re.MULTILINE) for name, body in blocks}


def test_zsh_lists_exactly_the_options_of_each_command(besser):
    blocks = _zsh_blocks(besser("completion", "zsh").stdout)

    assert blocks == {name: _option_names(parser) for name, parser in _commands().items()}


def test_fish_lists_exactly_the_options_of_each_command(besser):
    script = besser("completion", "fish").stdout.splitlines()

    for name, parser in _commands().items():
        prefix = f"complete -c besser -n '__fish_seen_subcommand_from {name}' "
        spelled = [" ".join(re.findall(r"-[sl] [\w-]+", line[len(prefix):].split(" -d ")[0]))
                   for line in script if line.startswith(prefix) and " -d " in line]
        expected = [" ".join([f"-s {n[1:]}" for n in action.option_strings if not n.startswith("--")]
                             + [f"-l {n[2:]}" for n in action.option_strings if n.startswith("--")])
                    for action in parser._actions if action.option_strings and action.help != argparse.SUPPRESS]
        assert spelled == expected, name
    commands = [line.split(" -a ")[1].split(" ")[0] for line in script
                if line.startswith("complete -c besser -n __fish_use_subcommand -a ")]
    assert commands == list(_commands())


def test_quiet_still_prints_the_script(besser):
    assert besser("completion", "bash", "-q").stdout == besser("completion", "bash").stdout


def test_the_script_is_the_same_on_every_run(besser):
    assert besser("completion", "zsh").stdout == besser("completion", "zsh").stdout


def test_json_carries_the_shell_and_the_script(besser):
    text = besser("completion", "fish").stdout
    run = besser("completion", "fish", "--json")
    assert run.returncode == 0
    assert run.json["ok"] is True
    assert run.json["shell"] == "fish"
    assert run.json["script"] + "\n" == text


def test_an_unknown_shell_is_a_usage_error(besser):
    run = besser("completion", "powershell")
    assert run.returncode == 2
    assert "invalid choice" in run.stderr


@pytest.mark.parametrize("shell", COMPLETION_SHELLS)
def test_the_script_parses_in_its_shell(besser, tmp_path, shell):
    if shutil.which(shell) is None:
        pytest.skip(f"{shell} is not installed")
    script = tmp_path / f"besser.{shell}"
    script.write_text(besser("completion", shell).stdout, encoding="utf-8")
    check = subprocess.run([*SYNTAX_CHECK[shell], str(script)], capture_output=True, text=True, check=False)
    assert check.returncode == 0, check.stderr


# Each line typed so far (the last word is the one being completed), and the completions bash offers.
BASH_CASES = [
    (["besser", "gen"], ["generators", "generate"]),
    (["besser", "generate", "sql"], ["sqlalchemy", "sql"]),
    (["besser", "validate", "--"], ["--help", "--json", "--quiet", "--verbose", "--strict", "--exec-py", "--ocl"]),
    (["besser", "completion", ""], ["bash", "zsh", "fish"]),
    (["besser", "--js"], ["--json"]),
    (["besser", "generate", "sqlalchemy", ""], ["models", "library.py", "project.json"]),
    (["besser", "generate", "sqlalchemy", "-o", ""], ["models"]),
    (["besser", "regenerate", "--option", "dbms=sqlite", "back"], ["backend"]),
    (["besser", "convert", "library.py", "-o", "n"], ["notes.txt"]),
    (["besser", "model", "-f", ""], ["models", "library.py", "project.json"]),
    (["besser", "model", "add-class", "Shelf", "--file", "l"], ["library.py"]),
]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")
@pytest.mark.parametrize(("words", "expected"), BASH_CASES, ids=[" ".join(w) for w, _ in BASH_CASES])
def test_bash_completes(besser, tmp_path, words, expected):
    script = tmp_path / "besser.bash"
    script.write_text(besser("completion", "bash").stdout, encoding="utf-8")
    for name in ("library.py", "project.json", "notes.txt"):
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "models").mkdir()
    quoted = " ".join(f"'{word}'" for word in words)
    program = (
        f"source '{script}'\n"
        f"COMP_WORDS=({quoted}); COMP_CWORD={len(words) - 1}\n"
        "_besser\n"
        'printf "%s\\n" "${COMPREPLY[@]}"\n'
    )
    run = subprocess.run(["bash", "-c", program], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    assert run.stdout.splitlines() == expected


# zsh completes in its line editor only: a pseudo-terminal (zpty) runs an interactive zsh with compinit and
# the script, the line is typed with a tab, and compadd, wrapped, prints each match it adds.
ZSH_INIT = r"""
PROMPT=
autoload -U compinit; compinit -u -D
bindkey '^I' complete-word
null-line () { echo -E - $'\0' }
compprefuncs=( null-line )
comppostfuncs=( null-line exit )
zstyle ':completion:*' list-grouped false
zstyle ':completion:*' insert-tab false
compadd () {
    if [[ ${@[1,(i)(-|--)]} == *-(O|A|D)\ * ]]; then
        builtin compadd "$@"
        return $?
    fi
    typeset -a __hits
    builtin compadd -A __hits "$@"
    builtin compadd "$@"  # added too, so that _files sees the matches and tries no other tag
    (( $#__hits )) || return 1
    local hit
    for hit in $__hits; do
        echo -E - $hit
    done
}
"""
ZSH_CAPTURE = r"""
zmodload zsh/zpty
zpty z zsh -f -i
zpty -w z "source ${(q)1}; source ${(q)2}"
zpty -w z "$3"$'\t'
integer seen=0
while zpty -r z; do :; done | while IFS= read -r line; do
    if [[ $line == *$'\0\r' ]]; then
        (( seen++ )) && return 0 || continue
    fi
    (( seen )) && echo -E - ${line%$'\r'}
done
"""

# The line typed so far and what each shell offers for its last word (a directory without its "/").
SHELL_CASES = [
    ("besser gen", ["generate", "generators"]),
    ("besser generate sql", ["sql", "sqlalchemy"]),
    ("besser completion ", ["bash", "fish", "zsh"]),
    ("besser --js", ["--json"]),
    ("besser validate --s", ["--strict"]),
    ("besser generate sqlalchemy ", ["library.py", "models", "project.json"]),  # model files, not notes.txt
    ("besser info l", ["library.py"]),
    ("besser generate sqlalchemy -o ", ["models"]),  # a directory
    ("besser generate sqlalchemy library.py -o out --option ", []),  # free text: no file
    ("besser regenerate --option dbms=sqlite back", ["backend"]),
    ("besser convert library.py -o n", ["notes.txt"]),  # any file
    ("besser model -f ", ["library.py", "models", "project.json"]),
    ("besser model add-class Shelf --file l", ["library.py"]),
]


def _completion_files(tmp_path):
    for name in ("library.py", "project.json", "notes.txt"):
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "models").mkdir()


def _offered(lines: str) -> list[str]:
    return sorted({line.split("\t")[0].rstrip("/") for line in lines.splitlines() if line})


@pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh is not installed")
@pytest.mark.parametrize(("line", "expected"), SHELL_CASES, ids=[line for line, _ in SHELL_CASES])
def test_zsh_completes(besser, tmp_path, line, expected):
    (tmp_path / "_besser").write_text(besser("completion", "zsh").stdout, encoding="utf-8")
    (tmp_path / "init.zsh").write_text(ZSH_INIT, encoding="utf-8")
    (tmp_path / "capture.zsh").write_text(ZSH_CAPTURE, encoding="utf-8")
    _completion_files(tmp_path)
    run = subprocess.run(["zsh", "-f", "capture.zsh", "init.zsh", "_besser", line], cwd=tmp_path,
                         capture_output=True, text=True, check=False, timeout=60)
    assert run.returncode == 0, run.stderr
    assert _offered(run.stdout) == expected


@pytest.mark.skipif(shutil.which("fish") is None, reason="fish is not installed")
@pytest.mark.parametrize(("line", "expected"), SHELL_CASES, ids=[line for line, _ in SHELL_CASES])
def test_fish_completes(besser, tmp_path, line, expected):
    script = tmp_path / "besser.fish"
    script.write_text(besser("completion", "fish").stdout, encoding="utf-8")
    _completion_files(tmp_path)
    run = subprocess.run(["fish", "--no-config", "-c", "source $argv[1]; complete -C $argv[2]", str(script), line],
                         cwd=tmp_path, capture_output=True, text=True, check=False, timeout=60)
    assert run.returncode == 0, run.stderr
    # fish 4 adds its own fuzzy matches to the list (--json and --verbose for --s): keep what the word starts.
    word = line.rsplit(" ", 1)[-1]
    assert [offer for offer in _offered(run.stdout) if offer.startswith(word)] == expected
