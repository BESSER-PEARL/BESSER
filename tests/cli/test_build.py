"""``besser build``: the outputs of ``besser.toml``, generated first and regenerated afterwards."""

import importlib.util

import pytest

from tests.cli.conftest import LIBRARY_MODEL_WITH_ISBN

needs_pygit2 = pytest.mark.skipif(importlib.util.find_spec("pygit2") is None,
                                  reason='regenerate merges with pygit2: pip install "besser[regen]"')

CONFIG = """\
model = "library.py"

[output.classes]
generator = "python"
dir = "generated/classes"

[output.db]
generator = "sqlalchemy"
dir = "generated/db"
options = { dbms = "postgresql" }
"""


@pytest.fixture
def config(write, model):
    return write("besser.toml", CONFIG)


def test_it_generates_every_output_with_its_options(besser, config, tmp_path):
    run = besser("build", "--json")

    assert run.returncode == 0, run.stdout
    assert [(output["name"], output["command"], output["ok"]) for output in run.json["outputs"]] == [
        ("classes", "generate", True), ("db", "generate", True),
    ]
    assert run.json["files"] == ["generated/classes/classes.py", "generated/db/sql_alchemy.py"]
    assert run.json["config"] == "besser.toml"
    assert "postgresql" in (tmp_path / "generated" / "db" / "sql_alchemy.py").read_text()


def test_the_report_names_each_output(besser, config):
    run = besser("build")

    assert run.returncode == 0
    assert "[classes] python: library.py -> generated/classes: ok\n" in run.stdout
    assert run.stdout.endswith("OK: 2 outputs, 0 failed\n")


def test_an_output_is_built_alone_by_name(besser, config, tmp_path):
    run = besser("build", "db", "--json")
    unknown = besser("build", "api")

    assert [output["name"] for output in run.json["outputs"]] == ["db"]
    assert not (tmp_path / "generated" / "classes").exists()
    assert unknown.returncode == 2
    assert unknown.stderr == "error: besser.toml has no output named api (it has: classes, db)\n"


@needs_pygit2
def test_a_second_build_keeps_the_edits_made_in_the_output(besser, config, write, tmp_path):
    besser("build")
    generated = tmp_path / "generated" / "classes" / "classes.py"
    generated.write_text("# mine\n" + generated.read_text())
    write("library.py", LIBRARY_MODEL_WITH_ISBN)

    run = besser("build", "classes", "--json")

    assert run.returncode == 0, run.stdout
    assert run.json["outputs"][0]["command"] == "regenerate"
    assert run.json["outputs"][0]["merged"] == ["generated/classes/classes.py"]
    assert generated.read_text().startswith("# mine\n") and "isbn" in generated.read_text()


def test_without_pygit2_init_then_build_twice_succeeds(besser, write, tmp_path):
    # 'python -m' puts the working directory first on sys.path, so this module stands in for pygit2:
    # a base install, without besser[regen].
    write("pygit2.py", "raise ImportError('hidden by the test')\n")
    assert besser("init").returncode == 0

    first, second = besser("build"), besser("build")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "up to date" in second.stdout


def test_without_pygit2_a_model_change_rebuilds_when_nothing_was_edited(besser, config, write, tmp_path):
    write("pygit2.py", "raise ImportError('hidden by the test')\n")
    assert besser("build").returncode == 0
    write("library.py", LIBRARY_MODEL_WITH_ISBN)

    run = besser("build", "--json")

    assert run.returncode == 0, run.stdout
    assert "isbn" in (tmp_path / "generated" / "classes" / "classes.py").read_text()


def test_overwrite_discards_the_edits(besser, config, write, tmp_path):
    write("besser.toml", CONFIG.replace('dir = "generated/classes"', 'dir = "generated/classes"\noverwrite = true'))
    besser("build")
    generated = tmp_path / "generated" / "classes" / "classes.py"
    generated.write_text("# mine\n" + generated.read_text())

    run = besser("build", "classes", "--json")

    assert run.returncode == 0 and run.json["outputs"][0]["command"] == "generate"
    assert not generated.read_text().startswith("# mine")


def test_a_dry_run_writes_nothing(besser, config, tmp_path):
    run = besser("build", "--dry-run", "--json")

    assert run.returncode == 0 and run.json["files"] == [] and run.json["dry_run"] is True
    assert run.json["outputs"][0]["would_write"] == {"generated/classes/classes.py": "new"}
    assert not (tmp_path / "generated").exists()


def test_an_output_that_fails_does_not_stop_the_others(besser, write, model):
    write("besser.toml", CONFIG.replace('generator = "python"', 'generator = "cobol"'))

    run = besser("build", "--json")

    assert run.returncode == 2
    assert [output["ok"] for output in run.json["outputs"]] == [False, True]
    assert run.json["errors"][0].startswith("classes: unknown generator 'cobol'")
    assert run.json["files"] == ["generated/db/sql_alchemy.py"]


def test_a_generator_without_regenerate_needs_overwrite_for_a_second_build(besser, write):
    besser("new", "agent.py", "--kind", "agent")
    write("besser.toml", '[output.bot]\ngenerator = "agent"\nmodel = "agent.py"\ndir = "bot"\n')

    first = besser("build")
    second = besser("build", "--json")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 2
    assert second.json["errors"][-1] == "bot: to overwrite it on every build, set 'overwrite = true' in [output.bot]"


def test_paths_are_relative_to_the_configuration_file(besser, write, model, tmp_path):
    write("conf/besser.toml", '[output.classes]\ngenerator = "python"\nmodel = "../library.py"\ndir = "out"\n')

    run = besser("build", "-c", "conf/besser.toml", "--json")

    assert run.returncode == 0, run.stdout
    assert (tmp_path / "conf" / "out" / "classes.py").is_file()


@pytest.mark.parametrize("content, message", [
    (None, "besser.toml: no such file; 'besser schema config' prints an example to start from"),
    ("model = ", "besser.toml: not a TOML file: "),
    ('modle = "library.py"\n', "besser.toml: unknown key modle; the file takes 'model', 'test' and "
                                   "[output.NAME] tables"),
    ('model = "library.py"\n', "besser.toml: no output to build: add an [output.NAME] table with 'generator' and 'dir'"),
    ('[output.a]\ngenerator = "python"\n', "besser.toml: [output.a]: missing dir, model (or a top-level model)"),
    ('[output.a]\ngenerator = "python"\ndir = "o"\nmodel = "m.py"\noption = {}\n',
     "besser.toml: [output.a]: unknown key option; an output takes generator, dir, model, use, options, overwrite"),
    ('[output.a]\ngenerator = "python"\ndir = "o"\nmodel = "m.py"\nuse = "x"\n',
     "besser.toml: [output.a]: use must be a list of strings"),
])
def test_a_configuration_it_cannot_read_is_a_usage_error(besser, write, content, message):
    if content is not None:
        write("besser.toml", content)

    run = besser("build", "--json")

    assert run.returncode == 2 and run.json["outputs"] == []
    assert run.json["errors"][0].startswith(message)
