"""``besser completion``: the scripts follow the parser, parse in their shell and complete in bash."""

import argparse
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


def _long_options(parser: argparse.ArgumentParser) -> set[str]:
    return {name for action in parser._actions for name in action.option_strings if name.startswith("--")}


@pytest.mark.parametrize("shell", COMPLETION_SHELLS)
def test_the_script_names_every_command_option_and_generator(besser, shell):
    run = besser("completion", shell)
    assert run.returncode == 0, run.stderr
    script = run.stdout
    for name, subparser in _commands().items():
        assert name in script
        for option in _long_options(subparser):
            # fish spells a long option "-l name"; bash and zsh spell it "--name".
            assert (f"-l {option[2:]}" if shell == "fish" else option) in script, (name, option)
    for generator_id in GENERATOR_IDS:
        assert generator_id in script


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
