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
from tests.cli.conftest import INVALID_MODEL, LIBRARY_MODEL, PROJECT_EXPORT, WARNING_MODEL

NOISY_MODEL = 'print("loading the model")\n' + WARNING_MODEL

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
    for command in ("new", "validate", "info", "generators", "generate", "regenerate", "convert"):
        assert f"\n    {command} " in run.stdout


@pytest.mark.parametrize("command", ["validate", "info", "generate", "regenerate", "convert"])
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


def test_quiet_prints_no_report_no_warning_and_nothing_the_model_prints(besser, write):
    write("noisy.py", NOISY_MODEL)
    plain = besser("generate", "python", "noisy.py", "-o", "plain")
    quiet = besser("generate", "python", "noisy.py", "-o", "quiet", "--quiet")

    assert plain.returncode == 0 and "Generated 1 file" in plain.stdout
    assert "loading the model" in plain.stderr and "warning: " in plain.stderr
    assert quiet.returncode == 0
    assert quiet.stdout == "" and quiet.stderr == ""


def test_quiet_still_prints_the_errors(besser, write):
    run = besser("-q", "validate", write("noisy.py", NOISY_MODEL), "missing.py", write("broken.py", INVALID_MODEL))

    assert run.returncode == 2 and run.stdout == ""
    errors = run.stderr.splitlines()
    assert errors[0] == "error: missing.py: no such file"
    assert errors[1].startswith("error: broken.py: broken_model: Attribute 'title' of class 'Book' has type 'Text'")
    assert len(errors) == 2


def test_quiet_leaves_the_json_object_whole(besser, write):
    run = besser("validate", write("noisy.py", NOISY_MODEL), "--json", "--quiet")

    assert run.returncode == 0 and run.stderr == ""
    assert len(run.json["warnings"]) == 1


def test_verbose_prints_the_traceback_of_a_model_file_that_raises(besser, write):
    write("crash.py", "def build():\n    raise RuntimeError('no')\n\nbuild()\n")
    plain = besser("validate", "crash.py")
    verbose = besser("validate", "crash.py", "--verbose")

    assert plain.stderr == "error: crash.py: line 2: RuntimeError: no\n"
    assert verbose.returncode == 2
    assert "Traceback (most recent call last):" in verbose.stderr and "in build" in verbose.stderr
    assert verbose.stderr.endswith("error: crash.py: line 2: RuntimeError: no\n")


def test_verbose_prints_the_traceback_of_a_generator_that_raises(monkeypatch, tmp_path, capsys):
    def broken_generator(*args, **kwargs):
        raise RuntimeError("template missing")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "library.py").write_text(LIBRARY_MODEL, encoding="utf-8")
    monkeypatch.setattr(commands, "run_generator", broken_generator)

    status = main(["-v", "generate", "python", "library.py", "-o", "out", "--json"])
    captured = capsys.readouterr()

    assert status == 1
    assert json.loads(captured.out)["errors"] == ["the 'python' generator failed: RuntimeError: template missing"]
    assert "Traceback (most recent call last):" in captured.err and "in broken_generator" in captured.err


def test_quiet_and_verbose_together_is_a_usage_error(besser, model):
    run = besser("validate", model, "--quiet", "--verbose")

    assert run.returncode == 2
    assert run.stderr == "error: --quiet and --verbose cannot be combined (see 'besser --help')\n"


@pytest.mark.parametrize("args", [["--version"], ["completion", "bash"], ["generators"], ["info", "library.py"]],
                         ids=["version", "completion", "generators", "info"])
def test_quiet_keeps_the_output_that_is_the_result(besser, model, args):
    plain = besser(*args)
    quiet = besser(*args, "-q")

    assert quiet.returncode == plain.returncode == 0
    assert quiet.stdout == plain.stdout != ""
    assert quiet.stderr == ""


def test_quiet_eval_of_the_completion_script_defines_the_function(besser):
    script = besser("completion", "bash", "-q").stdout

    assert "complete -F _besser besser" in script


def test_quiet_with_strict_prints_the_warnings_that_fail_the_run(besser, write):
    run = besser("validate", write("warn.py", WARNING_MODEL), "--strict", "-q")

    assert run.returncode == 1 and run.stdout == ""
    first, *rest = run.stderr.splitlines()
    assert first == "error: 1 warning (--strict)"
    assert len(rest) == 1 and rest[0].startswith("warning: library_model: ") and "Shelf" in rest[0]


@pytest.mark.parametrize("flag", ["-q", "-v", "--json"])
def test_version_takes_the_common_options(besser, flag):
    run = besser("--version", flag)

    assert run.returncode == 0 and run.stderr == ""
    if flag == "--json":
        assert run.json["command"] == "version" and run.json["version"] == get_besser_version()
    else:
        assert run.stdout == f"besser {get_besser_version()}\n"


@pytest.mark.parametrize(("args", "command"), [
    (["generate", "python", "m.py", "-o", "out", "--keep-edits", "--force"], "generate"),
    (["validate", "m.py", "-q", "-v"], "validate"),
    (["--version", "-q", "-v"], "version"),
    (["validate"], "validate"),
    (["generate", "python", "m.py", "-o", "out", "--adopt"], "generate"),
    (["--bogus"], ""),
])
def test_a_usage_error_names_the_command_it_is_about(besser, args, command):
    run = besser(*args, "--json")

    assert run.returncode == 2
    assert run.json["ok"] is False and run.json["command"] == command


def test_an_unknown_option_of_a_command_points_to_the_command_help(besser):
    run = besser("generate", "python", "m.py", "-o", "out", "--adopt")

    assert run.returncode == 2
    assert run.stderr == "error: unrecognized arguments: --adopt (see 'besser generate --help')\n"


def test_a_run_does_not_import_the_mcp_server():
    """It builds the tools from every generator class: loading it on each run made every command slow."""
    probe = ("import sys; from besser.cli.main import main; main(['--version']); "
             "print('besser.cli.mcp_server' in sys.modules, 'besser.generators.django' in sys.modules)")
    run = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=120)

    assert run.stdout.splitlines()[-1] == "False False"
