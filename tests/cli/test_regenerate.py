"""``besser regenerate``: bring generated code up to a changed model and keep the edits made in it."""

import importlib.util
import json

import pytest

from besser.cli import commands
from besser.cli.main import main
from tests.cli.conftest import INVALID_MODEL, LIBRARY_MODEL, LIBRARY_MODEL_WITH_ISBN

RECORD = ".besser_generated.json"
HAND_EDIT = "# Reviewed by hand: keep this line.\n"
BOOK_CONSTRUCTOR = "def __init__(self, title: str, pages: int,"

# The merge is pygit2's, which the besser[regen] extra installs (CI does); everything
# the command does before the merge is tested without it.
needs_pygit2 = pytest.mark.skipif(
    importlib.util.find_spec("pygit2") is None,
    reason='regenerate merges with pygit2, which is not installed: pip install "besser[regen]"',
)


@pytest.fixture
def classes(tmp_path):
    """The file the python generator writes into ``out``."""
    return tmp_path / "out" / "classes.py"


@needs_pygit2
def test_the_first_run_generates_and_starts_the_record(besser, model, classes, tmp_path):
    run = besser("regenerate", "python", model, "-o", "out")

    assert run.returncode == 0
    assert run.stdout == "Regenerated out with the 'python' generator (1 written):\n  written  out/classes.py\n"
    assert "class Book" in classes.read_text()
    record = json.loads((tmp_path / "out" / RECORD).read_text())
    assert record["generator"] == "PythonGenerator" and list(record["files"]) == ["classes.py"]


@needs_pygit2
def test_a_hand_edit_is_kept_and_the_model_change_is_applied(besser, model, write, classes):
    assert besser("regenerate", "python", model, "-o", "out").returncode == 0
    assert "isbn" not in classes.read_text()
    classes.write_text(HAND_EDIT + classes.read_text())
    write("library.py", LIBRARY_MODEL_WITH_ISBN)

    run = besser("regenerate", "python", model, "-o", "out")

    regenerated = classes.read_text()
    assert run.returncode == 0
    assert run.stdout == "Regenerated out with the 'python' generator (1 merged):\n  merged  out/classes.py\n"
    assert regenerated.startswith(HAND_EDIT)
    assert "def isbn(self) -> str:" in regenerated and "<<<<<<<" not in regenerated
    assert run.stderr.startswith("warning: the model's changes were merged into your edits;")
    assert run.stderr.rstrip().endswith("so run your tests: out/classes.py")


@needs_pygit2
def test_json_reports_what_happened_to_each_file(besser, model, write, classes):
    besser("regenerate", "python", model, "-o", "out")
    classes.write_text(HAND_EDIT + classes.read_text())
    write("library.py", LIBRARY_MODEL_WITH_ISBN)

    result = besser("regenerate", "python", model, "-o", "out", "--json").json

    assert result == {
        "ok": True, "command": "regenerate", "exit_code": 0, "errors": [], "notes": [],
        "warnings": ["the model's changes were merged into your edits; a clean merge can still break, "
                     "so run your tests: out/classes.py"],
        "files": ["out/classes.py"],
        "generator": "python", "model": model, "output_dir": "out", "adopted": False,
        "written": [], "merged": ["out/classes.py"], "conflicted": [], "deleted": [],
        "kept_removed_from_model": [], "not_restored": [], "not_merged": [], "differs_from_model": [],
    }


@needs_pygit2
def test_an_edit_on_the_lines_the_model_changes_is_a_conflict(besser, model, write, classes):
    besser("regenerate", "python", model, "-o", "out")
    generated = classes.read_text()
    assert BOOK_CONSTRUCTOR in generated
    classes.write_text(generated.replace(BOOK_CONSTRUCTOR, "def __init__(self, title: str, pages: int = 0,"))
    write("library.py", LIBRARY_MODEL_WITH_ISBN)

    run = besser("regenerate", "python", model, "-o", "out", "--json")

    conflicted = classes.read_text()
    assert run.returncode == 1
    assert run.json["ok"] is False
    assert run.json["conflicted"] == ["out/classes.py"] and run.json["files"] == ["out/classes.py"]
    [error] = run.json["errors"]
    assert error.startswith("out/classes.py: your edit and the model changed the same lines. Resolve the conflict")
    assert "<<<<<<< your edit\n    def __init__(self, title: str, pages: int = 0," in conflicted
    assert "    def __init__(self, title: str, pages: int, isbn: str," in conflicted
    assert ">>>>>>> generated from your model\n" in conflicted


@needs_pygit2
def test_the_conflict_files_are_listed_and_the_next_run_waits_for_them(besser, model, write, classes):
    besser("regenerate", "python", model, "-o", "out")
    classes.write_text(classes.read_text().replace(BOOK_CONSTRUCTOR, "def __init__(self, title: str, pages: int = 0,"))
    write("library.py", LIBRARY_MODEL_WITH_ISBN)

    run = besser("regenerate", "python", model, "-o", "out")
    again = besser("regenerate", "python", model, "-o", "out")

    assert run.returncode == 1
    assert run.stdout == "Regenerated out with the 'python' generator (1 conflict):\n  conflict  out/classes.py\n"
    assert "error: out/classes.py: your edit and the model changed the same lines." in run.stderr
    assert again.returncode == 2
    assert again.stderr == "error: Resolve the conflict markers in classes.py first.\n"

    resolved = [line for line in classes.read_text().splitlines(keepends=True)
                if not line.startswith(("<<<<<<<", "=======", ">>>>>>>")) and "pages: int = 0" not in line]
    classes.write_text("".join(resolved))
    assert besser("regenerate", "python", model, "-o", "out").returncode == 0


@needs_pygit2
def test_a_second_run_with_the_same_model_changes_nothing(besser, model, classes):
    besser("regenerate", "python", model, "-o", "out")
    classes.write_text(HAND_EDIT + classes.read_text())

    run = besser("regenerate", "python", model, "-o", "out", "--json")

    assert run.returncode == 0 and run.json["files"] == [] and run.json["warnings"] == []
    assert classes.read_text().startswith(HAND_EDIT)
    assert besser("regenerate", "python", model, "-o", "out").stdout == (
        "out is up to date with library.py: no file changed.\n"
    )


@needs_pygit2
def test_a_file_the_model_no_longer_produces_is_removed(besser, model, write, tmp_path):
    besser("regenerate", "java", model, "-o", "out")
    library = next((tmp_path / "out").rglob("Library.java"))
    write("library.py", LIBRARY_MODEL.replace("types={library, book}, associations={holds}", "types={book}"))

    run = besser("regenerate", "java", model, "-o", "out", "--json")

    assert run.returncode == 0
    assert run.json["deleted"] == [library.relative_to(tmp_path).as_posix()]
    assert not library.exists()


@needs_pygit2
def test_an_option_reaches_the_generator(besser, model, tmp_path):
    run = besser("regenerate", "sqlalchemy", model, "-o", "out", "--option", "dbms=postgresql")

    assert run.returncode == 0
    assert "postgresql" in (tmp_path / "out" / "sql_alchemy.py").read_text()


@needs_pygit2
def test_a_directory_written_by_generate_is_refused_until_it_is_adopted(besser, model, write, classes):
    besser("generate", "python", model, "-o", "out")

    refused = besser("regenerate", "python", model, "-o", "out")

    assert refused.returncode == 2
    assert "out holds generated files (classes.py) but no .besser_generated.json" in refused.stderr
    assert "'besser regenerate python <that model> -o out --adopt'" in refused.stderr

    adopted = besser("regenerate", "python", model, "-o", "out", "--adopt", "--json")
    assert adopted.returncode == 0
    assert adopted.json["adopted"] is True and adopted.json["files"] == []
    assert adopted.json["differs_from_model"] == []

    classes.write_text(HAND_EDIT + classes.read_text())
    write("library.py", LIBRARY_MODEL_WITH_ISBN)
    run = besser("regenerate", "python", model, "-o", "out", "--json")
    assert run.returncode == 0 and run.json["merged"] == ["out/classes.py"]
    assert classes.read_text().startswith(HAND_EDIT) and "def isbn(self)" in classes.read_text()


@needs_pygit2
def test_adopting_says_which_files_differ_from_the_model(besser, model, write, classes):
    besser("generate", "python", model, "-o", "out")
    write("library.py", LIBRARY_MODEL_WITH_ISBN)

    run = besser("regenerate", "python", model, "-o", "out", "--adopt")

    assert run.returncode == 0
    assert run.stdout.startswith("Recorded out as the 'python' generator's output for library.py; no file was changed.")
    assert "warning: these files differ from this model's output and count as your edits from now on" in run.stderr
    assert "isbn" not in classes.read_text()


@needs_pygit2
def test_generate_refuses_a_regenerated_directory_and_points_to_regenerate(besser, model, classes):
    besser("regenerate", "python", model, "-o", "out")
    classes.write_text(HAND_EDIT + classes.read_text())

    run = besser("generate", "python", model, "-o", "out")

    assert run.returncode == 2
    assert ("error: out holds generated code and the record of it (.besser_generated.json): "
            "'besser regenerate python library.py -o out' brings it up to the model and keeps hand edits; "
            "--force overwrites them\n") == run.stderr
    assert classes.read_text().startswith(HAND_EDIT)


@needs_pygit2
def test_generate_force_overwrites_a_regenerated_directory_and_drops_its_record(besser, model, classes, tmp_path):
    besser("regenerate", "python", model, "-o", "out")
    classes.write_text(HAND_EDIT + classes.read_text())

    run = besser("generate", "python", model, "-o", "out", "--force", "--json")

    assert run.returncode == 0
    assert not classes.read_text().startswith(HAND_EDIT)
    assert not (tmp_path / "out" / RECORD).exists()
    [warning] = run.json["warnings"]
    assert warning.startswith("removed out/.besser_generated.json: it recorded the output this run overwrote.")
    assert "'besser regenerate python library.py -o out --adopt'" in warning


def test_a_generator_that_does_not_support_it_is_a_usage_error(besser, model, tmp_path):
    run = besser("regenerate", "supabase", model, "-o", "out", "--json")

    assert run.returncode == 2
    [error] = run.json["errors"]
    assert error.startswith("the 'supabase' generator does not support regenerate: its output differs from run to run")
    assert "'besser generate supabase ... --force' overwrites the folder instead" in error
    assert "Generators that support it: python, java, " in error
    assert not (tmp_path / "out").exists()


def test_without_pygit2_it_says_how_to_install_it(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "library.py").write_text(LIBRARY_MODEL, encoding="utf-8")
    monkeypatch.setattr(commands, "merge_available", lambda: False)

    status = main(["regenerate", "python", "library.py", "-o", "out"])

    assert status == 2
    assert capsys.readouterr().err == (
        'error: regenerate needs the pygit2 package, which is not installed: pip install "besser[regen]"\n'
    )
    assert not (tmp_path / "out").exists()


def test_an_unknown_generator_is_a_usage_error(besser, model):
    run = besser("regenerate", "cobol", model, "-o", "out")

    assert run.returncode == 2
    assert "error: unknown generator 'cobol'; available generators: python, java, " in run.stderr


@needs_pygit2
def test_a_model_with_validation_errors_is_not_regenerated_from(besser, model, write, classes):
    besser("regenerate", "python", model, "-o", "out")
    before = classes.read_text()

    run = besser("regenerate", "python", write("library.py", INVALID_MODEL), "-o", "out", "--json")

    assert run.returncode == 1
    assert run.json["errors"][-1] == "nothing was generated: the model has validation errors"
    assert classes.read_text() == before
