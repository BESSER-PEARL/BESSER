"""``besser regenerate``: bring generated code up to a changed model and keep the edits made in it."""

import importlib.util
import json
import shutil

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
        "generator": "python", "model": model, "output_dir": "out", "dry_run": False,
        "used": {"domain": "library_model"}, "adopted": False,
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
    assert again.stderr == (
        "error: out/classes.py still holds the conflict markers of the last regenerate: resolve them "
        "('<<<<<<< your edit' ... '>>>>>>> generated from your model'), then run regenerate again.\n"
    )

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
def test_a_directory_written_by_generate_merges_a_hand_edit(besser, model, write, classes):
    besser("generate", "python", model, "-o", "out")
    classes.write_text(HAND_EDIT + classes.read_text())
    write("library.py", LIBRARY_MODEL_WITH_ISBN)

    run = besser("regenerate", "python", model, "-o", "out", "--json")

    assert run.returncode == 0 and run.json["merged"] == ["out/classes.py"]
    assert classes.read_text().startswith(HAND_EDIT) and "def isbn(self)" in classes.read_text()


def _generate_without_record(besser, model, tmp_path):
    """A folder as an older BESSER's 'besser generate' left it: generated code and no record."""
    assert besser("generate", "python", model, "-o", "out").returncode == 0
    (tmp_path / "out" / RECORD).unlink()


@needs_pygit2
def test_a_directory_generated_without_a_record_is_refused_until_it_is_adopted(besser, model, write, classes,
                                                                              tmp_path):
    _generate_without_record(besser, model, tmp_path)

    refused = besser("regenerate", "python", model, "-o", "out")

    assert refused.returncode == 2
    assert "out holds generated files (classes.py) but no .besser_generated.json" in refused.stderr
    assert ("Record it once with the model those files came from: "
            "'besser regenerate python <that model> -o out --adopt'") in refused.stderr

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
def test_adopting_says_which_files_differ_from_the_model(besser, model, write, classes, tmp_path):
    _generate_without_record(besser, model, tmp_path)
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
def test_generate_force_overwrites_a_regenerated_directory_and_records_what_it_wrote(besser, model, classes,
                                                                                    tmp_path):
    besser("regenerate", "python", model, "-o", "out")
    first_record = (tmp_path / "out" / RECORD).read_bytes()
    classes.write_text(HAND_EDIT + classes.read_text())

    run = besser("generate", "python", model, "-o", "out", "--force", "--json")

    assert run.returncode == 0 and run.json["recorded"] is True and run.json["warnings"] == []
    assert not classes.read_text().startswith(HAND_EDIT)
    # The same record a first regenerate run writes.
    assert (tmp_path / "out" / RECORD).read_bytes() == first_record


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


# ── --dry-run ───────────────────────────────────────────────────────────────


def _tree(folder) -> dict:
    """Every file under ``folder``, by path, with its bytes: what a dry run must leave as it was."""
    return {path.relative_to(folder).as_posix(): path.read_bytes() for path in folder.rglob("*") if path.is_file()}


@pytest.fixture
def edited(besser, model, write, classes, tmp_path):
    """``out`` generated, then hand-edited, and the model changed since: the next run merges."""
    besser("generate", "python", model, "-o", "out")
    classes.write_text(HAND_EDIT + classes.read_text())
    write("library.py", LIBRARY_MODEL_WITH_ISBN)
    return tmp_path / "out"


@needs_pygit2
def test_dry_run_reports_the_merge_and_changes_nothing(besser, model, edited):
    before = _tree(edited)

    run = besser("regenerate", "python", model, "-o", "out", "--dry-run", "--json")

    assert run.returncode == 0
    assert run.json["dry_run"] is True and run.json["files"] == []
    assert run.json["merged"] == ["out/classes.py"]
    assert run.json["would_write"] == {"out/classes.py": "merged"}
    assert run.json["warnings"] == ["dry run: the model's changes would be merged into your edits; a clean merge "
                                    "can still break, so run your tests: out/classes.py"]
    assert _tree(edited) == before


@needs_pygit2
def test_dry_run_text_says_nothing_was_written(besser, model, edited):
    run = besser("regenerate", "python", model, "-o", "out", "--dry-run")

    assert run.stdout == (
        "Dry run: nothing was written.\n"
        "Regenerating out with the 'python' generator would change 1 file (1 merged):\n"
        "  merged  out/classes.py\n"
    )


@needs_pygit2
def test_dry_run_of_a_conflict_fails_and_writes_no_markers(besser, model, edited, classes):
    classes.write_text(classes.read_text().replace(BOOK_CONSTRUCTOR, "def __init__(self, title: str, pages: int = 0,"))
    before = _tree(edited)

    run = besser("regenerate", "python", model, "-o", "out", "--dry-run", "--json")

    assert run.returncode == 1 and run.json["conflicted"] == ["out/classes.py"]
    assert run.json["errors"] == ["out/classes.py: your edit and the model change the same lines: regenerate "
                                  "would write conflict markers into it"]
    assert _tree(edited) == before
    assert besser("regenerate", "python", model, "-o", "out").returncode == 1


@needs_pygit2
def test_dry_run_of_a_first_run_creates_nothing(besser, model, tmp_path):
    run = besser("regenerate", "python", model, "-o", "out", "--dry-run", "--json")

    assert run.returncode == 0 and run.json["written"] == ["out/classes.py"]
    assert not (tmp_path / "out").exists()


@needs_pygit2
def test_dry_run_of_an_adoption_writes_no_record(besser, model, tmp_path):
    _generate_without_record(besser, model, tmp_path)

    run = besser("regenerate", "python", model, "-o", "out", "--adopt", "--dry-run")

    assert run.returncode == 0
    assert run.stdout.startswith("Dry run: nothing was written.\nAdopting would record out as the 'python' generator")
    assert not (tmp_path / "out" / RECORD).exists()


@needs_pygit2
def test_dry_run_refusals_name_the_output_directory(besser, model, tmp_path):
    _generate_without_record(besser, model, tmp_path)

    run = besser("regenerate", "python", model, "-o", "out", "--dry-run", "--json")

    assert run.returncode == 2
    assert run.json["errors"][0].startswith("out holds generated files (classes.py) but no .besser_generated.json")


# ── generate --keep-edits ───────────────────────────────────────────────────


@needs_pygit2
def test_generate_keep_edits_is_regenerate(besser, model, edited, tmp_path):
    before = _tree(edited)
    regenerated = besser("regenerate", "python", model, "-o", "out", "--json")
    after_regenerate = _tree(edited)
    shutil.rmtree(edited)
    for path, content in before.items():
        (edited / path).parent.mkdir(parents=True, exist_ok=True)
        (edited / path).write_bytes(content)

    kept = besser("generate", "python", model, "-o", "out", "--keep-edits", "--json")

    assert kept.returncode == regenerated.returncode == 0
    assert kept.json == regenerated.json and kept.json["command"] == "regenerate"
    assert _tree(edited) == after_regenerate


@needs_pygit2
def test_generate_keep_edits_dry_run_is_regenerate_dry_run(besser, model, edited):
    before = _tree(edited)

    kept = besser("generate", "python", model, "-o", "out", "--keep-edits", "--dry-run", "--json")

    assert kept.json == besser("regenerate", "python", model, "-o", "out", "--dry-run", "--json").json
    assert kept.json["dry_run"] is True and kept.json["merged"] == ["out/classes.py"]
    assert _tree(edited) == before


def test_generate_keep_edits_with_force_is_a_usage_error(besser, model, tmp_path):
    run = besser("generate", "python", model, "-o", "out", "--keep-edits", "--force", "--json")

    assert run.returncode == 2
    assert run.json["errors"] == ["--keep-edits and --force cannot be combined (see 'besser generate --help')"]
    assert not (tmp_path / "out").exists()


# ── refusals and the next step they name ────────────────────────────────────


@needs_pygit2
def test_the_unrecorded_refusal_gives_the_cli_command_once(besser, model, tmp_path):
    _generate_without_record(besser, model, tmp_path)

    run = besser("regenerate", "python", model, "-o", "out", "--json")

    assert run.json["errors"] == [
        "out holds generated files (classes.py) but no .besser_generated.json, so your edits cannot be told from "
        "generated code. Record it once with the model those files came from: "
        "'besser regenerate python <that model> -o out --adopt', then regenerate with the changed model."
    ]


@needs_pygit2
def test_the_adopt_command_it_names_keeps_the_options(besser, model, tmp_path):
    besser("generate", "sqlalchemy", model, "-o", "out", "--option", "dbms=postgresql")
    (tmp_path / "out" / RECORD).unlink()

    run = besser("regenerate", "sqlalchemy", model, "-o", "out", "--option", "dbms=postgresql")

    assert run.returncode == 2
    assert "'besser regenerate sqlalchemy <that model> -o out --option dbms=postgresql --adopt'" in run.stderr


@needs_pygit2
def test_adopting_a_recorded_directory_says_to_run_it_without_adopt(besser, model):
    besser("generate", "python", model, "-o", "out")

    run = besser("regenerate", "python", model, "-o", "out", "--adopt")

    assert run.returncode == 2
    assert run.stderr == ("error: out already has .besser_generated.json: there is nothing to adopt; "
                          "run it without --adopt\n")


@needs_pygit2
def test_an_output_that_is_a_file_is_refused_before_the_run(besser, model, write, tmp_path):
    write("out", "mine\n")

    for dry_run in ([], ["--dry-run"]):
        run = besser("regenerate", "python", model, "-o", "out", *dry_run, "--json")

        assert run.returncode == 2
        assert run.json["errors"] == ["out is a file: -o takes a directory"]
    assert (tmp_path / "out").read_text() == "mine\n"


# ── what the dry run predicts ───────────────────────────────────────────────

_OUTCOMES = ("written", "merged", "conflicted", "deleted")


def _predicts(dry: dict, real: dict) -> None:
    assert dry["exit_code"] == real["exit_code"]
    assert {name: dry[name] for name in _OUTCOMES} == {name: real[name] for name in _OUTCOMES}
    assert dry["would_write"] == {path: name for name in _OUTCOMES for path in real[name]}
    assert set(real["files"]) <= set(dry["would_write"])


@needs_pygit2
def test_a_dry_run_predicts_a_merge(besser, model, edited):
    dry = besser("regenerate", "python", model, "-o", "out", "--dry-run", "--json").json
    real = besser("regenerate", "python", model, "-o", "out", "--json").json

    _predicts(dry, real)
    assert dry["would_write"] == {"out/classes.py": "merged"}


@needs_pygit2
def test_a_dry_run_predicts_a_conflict(besser, model, edited, classes):
    classes.write_text(classes.read_text().replace(BOOK_CONSTRUCTOR, "def __init__(self, title: str, pages: int = 0,"))

    dry = besser("regenerate", "python", model, "-o", "out", "--dry-run", "--json").json
    real = besser("regenerate", "python", model, "-o", "out", "--json").json

    _predicts(dry, real)
    assert dry["exit_code"] == 1 and dry["would_write"] == {"out/classes.py": "conflicted"}


@needs_pygit2
def test_a_dry_run_predicts_a_first_run_and_a_deletion(besser, model, write):
    dry = besser("regenerate", "java", model, "-o", "out", "--dry-run", "--json").json
    real = besser("regenerate", "java", model, "-o", "out", "--json").json
    _predicts(dry, real)
    assert sorted(dry["would_write"]) == real["files"]

    write("library.py", LIBRARY_MODEL.replace("types={library, book}, associations={holds}", "types={book}"))
    dry = besser("regenerate", "java", model, "-o", "out", "--dry-run", "--json").json
    real = besser("regenerate", "java", model, "-o", "out", "--json").json
    _predicts(dry, real)
    assert "deleted" in dry["would_write"].values()


@needs_pygit2
def test_dry_run_warnings_say_what_would_happen(besser, model, edited, classes):
    classes.unlink()

    run = besser("regenerate", "python", model, "-o", "out", "--dry-run", "--json")

    assert run.json["warnings"] == ["dry run: you deleted these files and the model changes them; they would "
                                    "stay deleted: out/classes.py"]


# ── --quiet and --verbose ───────────────────────────────────────────────────


@needs_pygit2
def test_quiet_prints_nothing_but_errors(besser, model, tmp_path):
    first = besser("regenerate", "python", model, "-o", "out", "--quiet")
    (tmp_path / "out" / RECORD).unlink()
    refused = besser("regenerate", "python", model, "-o", "out", "--quiet")

    assert first.returncode == 0 and first.stdout == "" and first.stderr == ""
    assert refused.returncode == 2 and refused.stdout == ""
    assert refused.stderr.startswith("error: out holds generated files") and "\nwarning" not in refused.stderr


@needs_pygit2
def test_verbose_adds_the_traceback_of_a_refusal(besser, model, tmp_path):
    _generate_without_record(besser, model, tmp_path)

    run = besser("regenerate", "python", model, "-o", "out", "--verbose")

    assert run.returncode == 2 and run.stdout == ""
    assert "Traceback (most recent call last):" in run.stderr
    assert "UnrecordedOutputError" in run.stderr
