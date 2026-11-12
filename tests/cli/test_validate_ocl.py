"""``besser validate --ocl``: the models' OCL constraints, parsed and evaluated on object models."""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

from besser.cli.ocl_check import OCL_PREFIX
from tests.cli.conftest import LIBRARY_MODEL, Run

# On the tests that check constraints; the ones about a missing evaluator run without it.
needs_bocl = pytest.mark.skipif(importlib.util.find_spec("bocl") is None, reason="the OCL check needs besser[ocl]")
# The note's end when invariants of editor JSON were not evaluated.
NOT_EVALUATED = ("of editor JSON: the OCL evaluator runs Python built from the models' values; "
                 "pass --exec-py to evaluate them")

BOOK_MODEL = '''
from besser.BUML.metamodel.object import AttributeLink, DataValue, Object, ObjectModel
from besser.BUML.metamodel.structural import Class, Constraint, DomainModel, IntegerType, Property, StringType

title = Property(name="title", type=StringType)
pages = Property(name="pages", type=IntegerType)
book = Class(name="Book", attributes={title, pages})
positive = Constraint(name="positive_pages", context=book, language="OCL",
                      expression="context Book inv positive_pages: self.pages > 0")
BROKEN
book_model = DomainModel(name="Books", types={book}, constraints={CONSTRAINTS})
'''

OBJECTS = '''
novel = Object(name="novel", classifier=book, slots=[
    AttributeLink(attribute=title, value=DataValue(classifier=StringType, value="Novel")),
    AttributeLink(attribute=pages, value=DataValue(classifier=IntegerType, value=PAGES)),
])
shelf = ObjectModel(name="Shelf", objects={novel})
'''

BROKEN_CONSTRAINT = '''
broken = Constraint(name="broken", context=book, language="OCL", expression="context Book inv broken: self.pages >")
'''


def book_model(pages: int | None = None, broken: bool = False) -> str:
    source = BOOK_MODEL.replace("BROKEN", BROKEN_CONSTRAINT if broken else "")
    source = source.replace("{CONSTRAINTS}", "{positive, broken}" if broken else "{positive}")
    return source + (OBJECTS.replace("PAGES", str(pages)) if pages is not None else "")


@needs_bocl
def test_a_valid_constraint_passes(besser, write):
    run = besser("validate", write("books.py", book_model()), "--ocl")

    assert run.returncode == 0
    assert "book_model (DomainModel): ok" in run.stdout
    assert "note: OCL: 1 constraint checked, invariants evaluated on 0 object models\n" in run.stderr


@needs_bocl
def test_an_object_model_that_satisfies_the_constraints_passes(besser, write):
    run = besser("validate", write("books.py", book_model(pages=120)), "--ocl", "--json")

    assert run.returncode == 0, run.stdout
    assert run.json["notes"] == ["OCL: 1 constraint checked, invariants evaluated on 1 object model"]


@needs_bocl
def test_a_constraint_that_does_not_parse_fails_naming_it(besser, write):
    run = besser("validate", write("books.py", book_model(broken=True)), "--ocl", "--json")

    assert run.returncode == 1
    [error] = run.json["errors"]
    assert error == (f"book_model: {OCL_PREFIX} 'broken' on Book does not parse: line 1, column 37: Expression ends "
                     "unexpectedly — the operand on the right is missing.")
    assert run.json["models"][0]["errors"] == [error.removeprefix("book_model: ")]


@needs_bocl
def test_an_object_model_that_violates_a_constraint_fails(besser, write):
    run = besser("validate", write("books.py", book_model(pages=-3)), "--ocl", "--json")

    assert run.returncode == 1
    [error] = run.json["errors"]
    assert error.startswith(f"shelf: {OCL_PREFIX} [Book inv positive_pages]: ")
    assert "Constraint violation" in error
    assert [(report["name"], len(report["errors"])) for report in run.json["models"]] == [
        ("book_model", 0), ("shelf", 1),
    ]


@needs_bocl
def test_a_broken_constraint_is_reported_once_with_an_object_model(besser, write):
    run = besser("validate", write("books.py", book_model(pages=-3, broken=True)), "--ocl", "--json")

    assert run.returncode == 1
    assert sorted(error.split(":")[1].strip() for error in run.json["errors"]) == [
        f"{OCL_PREFIX} 'broken' on Book does not parse", f"{OCL_PREFIX} [Book inv positive_pages]",
    ]


def test_without_ocl_the_constraints_are_parsed_and_not_evaluated(besser, write):
    violated = besser("validate", write("books.py", book_model(pages=-3)), "--json")
    broken = besser("validate", write("broken.py", book_model(pages=-3, broken=True)), "--json")

    assert violated.returncode == 0 and violated.json["notes"] == []
    assert broken.returncode == 1
    assert broken.json["errors"] == [f"book_model: {OCL_PREFIX} 'broken' on Book does not parse: line 1, column 37: "
                                     "Expression ends unexpectedly — the operand on the right is missing."]


@needs_bocl
def test_a_model_without_constraints_says_so(besser, model):
    run = besser("validate", model, "--ocl", "--json")

    assert run.returncode == 0
    assert run.json["notes"] == ["OCL: the models have no OCL constraints"]


@needs_bocl
def test_ocl_runs_on_every_file_of_a_directory(besser, write):
    write("models/good.py", book_model(pages=10))
    write("models/bad.py", book_model(pages=0))
    run = besser("validate", "models", "--exec-py", "--ocl", "--json")

    assert run.returncode == 1
    assert [(entry["file"], entry["exit_code"]) for entry in run.json["results"]] == [
        ("models/bad.py", 1), ("models/good.py", 0),
    ]
    assert run.json["errors"][0].startswith(f"models/bad.py: shelf: {OCL_PREFIX} [Book inv positive_pages]: ")


@pytest.fixture
def besser_without_bocl(besser_without):
    """Run ``python -m besser`` where importing ``bocl`` fails, as when ``besser[ocl]`` is not installed."""
    return lambda *args: besser_without(("bocl",), *args)


def test_without_bocl_ocl_is_skipped_with_a_warning_saying_what_to_install(besser_without_bocl, write):
    write("books.py", book_model(pages=-3))
    run = besser_without_bocl("validate", "books.py", "--ocl")

    assert run.returncode == 0
    assert "Traceback" not in run.stderr
    assert ("warning: OCL constraints not checked: the OCL evaluator is not installed (missing module: 'bocl'); "
            'install it with: pip install "besser[ocl]"') in run.stderr
    assert run.stdout.rstrip().endswith("OK: 0 errors, 1 warning")

    result = besser_without_bocl("validate", "books.py", "--ocl", "--json").json
    assert result["ok"] is True and result["warnings"][0].startswith(f"{OCL_PREFIX}s not checked: ")
    # It is a warning: --strict fails on it, so a CI job cannot pass without the check it asked for.
    assert besser_without_bocl("validate", "books.py", "--ocl", "--strict").returncode == 1


def test_without_bocl_validate_still_runs_when_ocl_is_not_asked_for(besser_without_bocl, write):
    write("library.py", LIBRARY_MODEL)

    assert besser_without_bocl("validate", "library.py", "--strict").returncode == 0
    # Parsing the constraints needs no optional package.
    assert besser_without_bocl("validate", write("broken.py", book_model(broken=True))).returncode == 1


# An object value that breaks out of the string literal bocl builds for eval(): it writes MARKER when evaluated.
HOSTILE_VALUE_MODEL = '''
from besser.BUML.metamodel.object import AttributeLink, DataValue, Object, ObjectModel
from besser.BUML.metamodel.structural import Class, Constraint, DomainModel, Property, StringType

title = Property(name="title", type=StringType)
book = Class(name="Book", attributes={title})
named = Constraint(name="named", context=book, language="OCL", expression="context Book inv named: self.title <> 'x'")
book_model = DomainModel(name="Books", types={book}, constraints={named})
hostile = '" or open(MARKER, "w").write("ran") or "'
novel = Object(name="novel", classifier=book, slots=[
    AttributeLink(attribute=title, value=DataValue(classifier=StringType, value=hostile)),
])
shelf = ObjectModel(name="Shelf", objects={novel})
'''


@pytest.fixture
def hostile_json(besser, write, tmp_path):
    """The hostile model as an editor export, and the file its value writes when it runs."""
    marker = tmp_path / "ran.txt"
    source = write("hostile.py", HOSTILE_VALUE_MODEL.replace("MARKER", json.dumps(str(marker))))
    assert besser("convert", source, "-o", "hostile.json").returncode == 0
    assert not marker.exists()
    return "hostile.json", marker


@needs_bocl
def test_invariants_of_editor_json_are_not_evaluated_without_exec_py(besser, hostile_json):
    model, marker = hostile_json

    run = besser("validate", model, "--ocl", "--json")

    assert not marker.exists()
    assert run.returncode == 0, run.stdout
    # One note on the check, which does not claim it evaluated anything.
    assert run.json["notes"] == [
        f"OCL: 1 constraint checked; invariants not evaluated on 1 object model {NOT_EVALUATED}",
    ]


@needs_bocl
def test_exec_py_evaluates_the_invariants_of_editor_json(besser, write):
    source = write("books.py", book_model(pages=0))
    assert besser("convert", source, "-o", "books.json").returncode == 0

    run = besser("validate", "books.json", "--ocl", "--exec-py", "--json")

    assert run.returncode == 1
    assert any("positive_pages" in error and "Evaluates to False" in error for error in run.json["errors"])


@needs_bocl
def test_a_constraint_dropped_from_editor_json_is_an_error(besser, write):
    source = write("books.py", book_model(broken=True))
    assert besser("convert", source, "-o", "books.json").returncode == 0

    run = besser("validate", "books.json", "--ocl", "--json")

    assert run.returncode == 1
    assert any(f"{OCL_PREFIX} dropped:" in error and "broken" in error for error in run.json["errors"]), run.json


@needs_bocl
def test_a_run_of_json_and_python_models_gives_one_note(besser, write):
    source = write("books.py", book_model(pages=5))
    assert besser("convert", source, "-o", "books.json").returncode == 0

    run = besser("validate", "books.py", "books.json", "--ocl", "--json")

    assert run.returncode == 0, run.json
    assert run.json["notes"] == [
        f"OCL: 2 constraints checked, invariants evaluated on 1 object model; invariants not evaluated on "
        f"1 object model {NOT_EVALUATED}"
    ]


@needs_bocl
def test_a_constraint_that_does_not_parse_counts_as_checked(besser, write):
    run = besser("validate", write("books.py", book_model(broken=True)), "--ocl", "--json")

    assert run.json["notes"] == ["OCL: 2 constraints checked, invariants evaluated on 0 object models"]


# A second domain model and an object model of it: each object model is checked against its own classes.
SECOND_DOMAIN = '''
pen = Class(name="Pen", attributes={Property(name="ink", type=IntegerType)})
ink = list(pen.attributes)[0]
full = Constraint(name="full", context=pen, language="OCL", expression="context Pen inv full: self.ink > 50")
pen_model = DomainModel(name="Pens", types={pen}, constraints={full})
blue = Object(name="blue", classifier=pen, slots=[
    AttributeLink(attribute=ink, value=DataValue(classifier=IntegerType, value=INK)),
])
desk = ObjectModel(name="Desk", objects={blue})
'''


@needs_bocl
@pytest.mark.parametrize(("pages", "ink", "failing"), [(5, 80, []), (-1, 80, ["shelf"]), (5, 10, ["desk"])])
def test_each_object_model_is_evaluated_against_the_domain_model_it_instantiates(besser, write, pages, ink, failing):
    run = besser("validate", write("two.py", book_model(pages=pages) + SECOND_DOMAIN.replace("INK", str(ink))),
                 "--ocl", "--json")

    assert run.returncode == (1 if failing else 0), run.json
    assert [error.split(":")[0] for error in run.json["errors"]] == failing
    assert run.json["notes"] == ["OCL: 2 constraints checked, invariants evaluated on 2 object models"]


@pytest.fixture
def besser_without(tmp_path):
    """Run ``python -m besser`` where importing the named top-level modules fails, as when they are not installed."""
    def run(missing: tuple[str, ...], *args) -> Run:
        shim = tmp_path / "shim"
        shim.mkdir(exist_ok=True)
        (shim / "sitecustomize.py").write_text(
            "import sys\n" + "".join(f"sys.modules[{name!r}] = None\n" for name in missing)
        )
        env = {**os.environ,
               "PYTHONPATH": os.pathsep.join(filter(None, [str(shim), os.environ.get("PYTHONPATH")]))}
        process = subprocess.run(
            [sys.executable, "-m", "besser", *map(str, args)],
            cwd=tmp_path, capture_output=True, text=True, timeout=120, check=False, env=env,
        )
        return Run(process.returncode, process.stdout, process.stderr)
    return run


def test_without_the_backend_packages_ocl_is_skipped_with_a_warning_saying_what_to_install(besser_without, write):
    write("books.py", book_model(pages=-3))
    run = besser_without(("deep_translator",), "validate", "books.py", "--ocl", "--json")

    assert run.returncode == 0, run.stdout
    [warning] = run.json["warnings"]
    assert warning.startswith(f"{OCL_PREFIX}s not checked: editor JSON needs the Web Modeling Editor backend's "
                              "packages (missing module: 'deep_translator'); install them with: pip install -r ")
    assert warning.endswith("requirements.txt")
    assert besser_without(("deep_translator",), "validate", "books.py", "--ocl", "--strict").returncode == 1
