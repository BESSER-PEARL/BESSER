"""The ``besser`` entry points, ``--version``, ``--help`` and usage errors."""

import configparser
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from besser.cli import commands
from besser.cli.main import main
from besser.cli.output import stdout_to_stderr
from besser.utilities.provenance import get_besser_version
from tests.cli.conftest import PROJECT_EXPORT

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_besser_console_script_is_registered():
    setup_cfg = configparser.ConfigParser()
    setup_cfg.read(REPO_ROOT / "setup.cfg", encoding="utf-8")
    scripts = dict(
        line.split(" = ") for line in setup_cfg["options.entry_points"]["console_scripts"].strip().splitlines()
    )
    module_name, _, function_name = scripts["besser"].partition(":")

    assert getattr(importlib.import_module(module_name), function_name) is main


def test_version(besser):
    run = besser("--version")

    assert run.returncode == 0
    assert run.stdout == f"besser {get_besser_version()}\n"


def test_version_as_json(besser):
    result = besser("--version", "--json").json

    assert result["ok"] is True and result["version"] == get_besser_version()


def test_help_says_that_a_python_model_is_executed(besser):
    run = besser("--help")

    assert run.returncode == 0
    assert "is Python code and is EXECUTED" in " ".join(run.stdout.split())
    for command in ("validate", "generators", "generate", "convert"):
        assert command in run.stdout


@pytest.mark.parametrize("command", ["validate", "generate", "convert"])
def test_the_help_of_a_command_that_loads_a_model_says_it_too(besser, command):
    assert "is Python code and is EXECUTED" in " ".join(besser(command, "--help").stdout.split())


def test_no_command_is_a_usage_error(besser):
    run = besser()

    assert run.returncode == 2
    assert run.stdout == "" and "error: a command is required" in run.stderr


def test_an_unknown_command_is_a_usage_error(besser):
    run = besser("frobnicate")

    assert run.returncode == 2
    assert "invalid choice: 'frobnicate'" in run.stderr


def test_a_usage_error_is_json_with_the_json_option(besser):
    run = besser("--json", "validate")

    assert run.returncode == 2
    assert run.json["ok"] is False and run.json["files"] == [] and run.json["warnings"] == []
    assert "the following arguments are required: model" in run.json["errors"][0]


def test_the_json_option_goes_before_or_after_the_command(besser, model):
    assert besser("--json", "validate", model).json == besser("validate", model, "--json").json


def test_the_json_option_is_not_taken_from_an_argument_value(besser):
    run = besser("validate", "--", "--json")

    assert run.returncode == 2
    assert "error: --json: no such file" in run.stderr


def test_an_unexpected_error_exits_1_with_valid_json_and_a_traceback(monkeypatch, capsys):
    def broken():
        raise RuntimeError("a defect")
    monkeypatch.setattr(commands, "generators", broken)

    assert main(["generators", "--json"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["errors"] == ["unexpected RuntimeError: a defect"]
    assert "Traceback" in captured.err


def test_editor_json_without_the_backend_packages_says_how_to_install_them(monkeypatch, tmp_path, capsys):
    """The core install lacks what the editor's converters import: a JSON model needs the backend requirements."""
    monkeypatch.setitem(sys.modules, "pydantic", None)  # importing it now raises ModuleNotFoundError
    project = tmp_path / "project.json"
    project.write_text(json.dumps(PROJECT_EXPORT), encoding="utf-8")

    assert main(["validate", str(project), "--json"]) == 2
    [error] = json.loads(capsys.readouterr().out)["errors"]
    assert "editor JSON needs the Web Modeling Editor backend's packages (missing module: 'pydantic')" in error
    assert "pip install -r " in error and error.endswith("requirements.txt")


def test_output_of_a_model_or_generator_and_of_its_child_processes_goes_to_standard_error(capfd):
    with stdout_to_stderr():
        print("printed by a generator")
        subprocess.run([sys.executable, "-c", "print('printed by its child process')"], check=True)
    print("the result")

    captured = capfd.readouterr()
    assert captured.out == "the result\n"
    assert "printed by a generator" in captured.err and "printed by its child process" in captured.err
