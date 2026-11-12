"""``besser new``: a starter model file, of every kind, that the other commands accept as it is."""

import ast
import importlib.util
import shlex
import sys

import pytest

from besser.cli.schema import editor_diagram_type
from besser.cli.starters import STARTERS


def test_it_writes_a_model_that_validates_without_warnings(besser, tmp_path):
    run = besser("new", "model.py")

    assert run.returncode == 0
    assert run.stdout.startswith("Wrote a starter domain model to model.py. Next:\n  besser validate model.py\n")
    assert "besser validate model.py" in (tmp_path / "model.py").read_text()
    assert besser("validate", "model.py", "--strict").returncode == 0


def test_the_starter_model_is_one_the_generators_run_on(besser, tmp_path):
    besser("new", "model.py")
    run = besser("generate", "backend", "model.py", "-o", "app", "--json")

    assert run.returncode == 0 and run.json["warnings"] == []
    assert "app/main_api.py" in run.json["files"]


def test_json_lists_the_file(besser):
    result = besser("new", "models/shop.py", "--json").json

    assert result == {
        "ok": True, "command": "new", "exit_code": 0, "errors": [], "warnings": [],
        "notes": ["no agent guide: models has no besser.toml for it to describe ('besser init' sets one up)"],
        "files": ["models/shop.py"],
        "model": "models/shop.py", "kind": "domain",
    }


def test_an_existing_file_is_kept(besser, write, tmp_path):
    write("model.py", "mine = 1\n")
    run = besser("new", "model.py")

    assert run.returncode == 2
    assert run.stderr == "error: model.py exists: pass --force to overwrite it\n"
    assert (tmp_path / "model.py").read_text() == "mine = 1\n"


def test_force_overwrites_an_existing_file(besser, write, tmp_path):
    write("model.py", "mine = 1\n")

    assert besser("new", "model.py", "--force").returncode == 0
    assert "DomainModel(" in (tmp_path / "model.py").read_text()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows file names hold no quote nor backslash")
@pytest.mark.parametrize("name", ['say """hi""".py', "back\\slash.py", "my model.py"])
def test_any_file_name_gives_a_starter_that_loads_and_a_command_to_copy(besser, name, tmp_path):
    assert besser("new", name).returncode == 0
    run = besser("validate", name, "--strict")

    assert run.returncode == 0 and run.stderr == ""  # no SyntaxError, no SyntaxWarning on an escape
    shown = ast.get_docstring(ast.parse((tmp_path / name).read_text()))
    assert f"besser validate {shlex.quote(name)} " in shown


def test_a_name_that_is_not_a_model_file_is_a_usage_error(besser, tmp_path):
    run = besser("new", "model.txt")

    assert run.returncode == 2
    assert run.stderr == "error: a model file is a B-UML .py file or an editor .json project; got model.txt\n"
    assert not (tmp_path / "model.txt").exists()


def test_without_a_file_it_says_what_to_give(besser):
    run = besser("new")

    assert run.returncode == 2
    assert run.stderr == "error: name the file to write (besser new model.py), or pass --list for the kinds\n"


def test_there_is_a_starter_for_every_model_kind():
    from besser.cli.loading import MODEL_KINDS

    assert set(STARTERS) == {kind for kind, _ in MODEL_KINDS} | {"project"}


def test_list_names_every_kind_and_writes_nothing(besser, tmp_path):
    run = besser("new", "--list", "--json")

    assert run.returncode == 0 and run.json["files"] == []
    assert [entry["kind"] for entry in run.json["kinds"]] == list(STARTERS)
    domain = run.json["kinds"][0]
    assert (domain["summary"], domain["editor_diagram"]) == (STARTERS["domain"].summary, "ClassDiagram")
    assert "sqlalchemy" in domain["generators"]
    assert list(tmp_path.iterdir()) == []


def test_an_unknown_kind_is_a_usage_error_that_lists_the_kinds(besser):
    run = besser("new", "model.py", "--kind", "petri_net")

    assert run.returncode == 2
    assert "invalid choice: 'petri_net'" in run.stderr and "state_machine" in run.stderr


@pytest.mark.parametrize("kind", list(STARTERS))
def test_every_starter_validates_and_says_what_it_holds(besser, kind, tmp_path):
    assert besser("new", "model.py", "--kind", kind).returncode == 0
    validated = besser("validate", "model.py", "--strict", "--json")
    info = besser("info", "model.py", "--json").json

    assert validated.returncode == 0, validated.stdout
    assert validated.json["warnings"] == []
    assert "MODEL_FILE" not in (tmp_path / "model.py").read_text()
    if kind != "project":
        assert kind in [model["kind"] for model in info["models"]]
    else:
        assert len(info["models"]) > 1


@pytest.mark.parametrize("kind", [kind for kind, starter in STARTERS.items() if starter.generator])
def test_the_generator_a_starter_suggests_runs_on_it(besser, kind):
    generator = STARTERS[kind].generator
    created = besser("new", "model.py", "--kind", kind)
    run = besser("generate", generator, "model.py", "-o", "out", "--json")

    assert f"besser generate {generator} model.py -o out" in created.stdout
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.json["files"]


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="editor JSON needs the backend's packages")
@pytest.mark.parametrize("kind", [kind for kind in STARTERS if kind == "project" or editor_diagram_type(kind)])
def test_every_starter_is_also_an_editor_project(besser, kind, tmp_path):
    assert besser("new", "model.json", "--kind", kind).returncode == 0
    validated = besser("validate", "model.json", "--strict", "--json")
    info = besser("info", "model.json", "--json").json
    first = (tmp_path / "model.json").read_bytes()

    assert validated.returncode == 0, validated.stdout
    if kind != "project":
        assert kind in [model["kind"] for model in info["models"]]
    # The same starter writes the same file: ids derived from the model, a fixed time.
    assert besser("new", "model.json", "--kind", kind, "--force").returncode == 0
    assert (tmp_path / "model.json").read_bytes() == first


@pytest.mark.parametrize("kind", ["deployment", "feature"])
def test_a_kind_the_editor_has_no_diagram_for_is_a_py_file_only(besser, kind, tmp_path):
    run = besser("new", "model.json", "--kind", kind)

    assert run.returncode == 2
    assert run.stderr == f"error: the editor has no diagram for a {kind} model: write a .py file\n"
    assert not (tmp_path / "model.json").exists()


def test_the_deployment_starter_generates_once_its_cluster_file_exists(besser, write, tmp_path):
    besser("new", "model.py", "--kind", "deployment")
    without = besser("generate", "terraform", "model.py", "-o", "out", "--json")
    left_out = (tmp_path / "out").exists()
    dry_run = besser("generate", "terraform", "model.py", "-o", "out", "--json", "--dry-run")
    write("cluster.conf", 'project = "my-gcp-project"\n')
    with_file = besser("generate", "terraform", "model.py", "-o", "out", "--json")

    # The reason is in the result, which a --json caller or an MCP client reads, not only on standard error.
    reason = "Error reading configuration file cluster.conf: [Errno 2] No such file or directory: 'cluster.conf'"
    assert without.returncode == 1
    assert without.json["errors"] == [f"the 'terraform' generator wrote no file; the last it printed and logged: "
                                      f"{reason}"]
    assert reason in without.stderr  # still logged on standard error, as it was
    assert dry_run.returncode == 1 and dry_run.json["would_write"] == {}
    assert dry_run.json["errors"] == [f"the 'terraform' generator would write no file; the last it printed and "
                                      f"logged: {reason}"]
    assert not left_out  # the folder the run created and wrote nothing in is removed
    assert with_file.returncode == 0 and with_file.json["files"]


def test_quiet_and_verbose_write_the_model_and_print_nothing_on_standard_error(besser, tmp_path):
    for flag, name in (("-q", "quiet.py"), ("-v", "verbose.py")):
        run = besser("new", name, flag, "--no-agent-files")

        assert run.returncode == 0
        assert run.stderr == ""
        assert "DomainModel(" in (tmp_path / name).read_text()


def test_verbose_on_a_file_it_cannot_write_prints_the_traceback(besser, write):
    # A file where the model's directory should be.
    write("blocker", "")

    run = besser("new", "blocker/model.py", "-v")

    assert run.returncode == 1
    assert "Traceback (most recent call last):" in run.stderr
    assert run.stderr.rstrip().splitlines()[-1].startswith("error: cannot write blocker/model.py: ")
