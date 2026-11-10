"""``besser validate --ocl``: the models' OCL constraints, parsed and evaluated on object models."""

import os
import subprocess
import sys

import pytest

from besser.cli.ocl_check import OCL_PREFIX
from tests.cli.conftest import LIBRARY_MODEL, Run

pytest.importorskip("bocl", reason="the OCL check needs besser[ocl]")

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


def test_a_valid_constraint_passes(besser, write):
    run = besser("validate", write("books.py", book_model()), "--ocl")

    assert run.returncode == 0
    assert "book_model (DomainModel): ok" in run.stdout
    assert "note: OCL: 1 constraint parsed, invariants evaluated on 0 object models" in run.stderr


def test_an_object_model_that_satisfies_the_constraints_passes(besser, write):
    run = besser("validate", write("books.py", book_model(pages=120)), "--ocl", "--json")

    assert run.returncode == 0, run.stdout
    assert run.json["notes"] == ["OCL: 1 constraint parsed, invariants evaluated on 1 object model"]


def test_a_constraint_that_does_not_parse_fails_naming_it(besser, write):
    run = besser("validate", write("books.py", book_model(broken=True)), "--ocl", "--json")

    assert run.returncode == 1
    [error] = run.json["errors"]
    assert error.startswith(f"book_model: {OCL_PREFIX} [Book inv broken]: 'context Book inv broken: self.pages >'")
    assert run.json["models"][0]["errors"] == [error.removeprefix("book_model: ")]


def test_an_object_model_that_violates_a_constraint_fails(besser, write):
    run = besser("validate", write("books.py", book_model(pages=-3)), "--ocl", "--json")

    assert run.returncode == 1
    [error] = run.json["errors"]
    assert error.startswith(f"shelf: {OCL_PREFIX} [Book inv positive_pages]: ")
    assert "Constraint violation" in error
    assert [(report["name"], len(report["errors"])) for report in run.json["models"]] == [
        ("book_model", 0), ("shelf", 1),
    ]


def test_a_broken_constraint_is_reported_once_with_an_object_model(besser, write):
    run = besser("validate", write("books.py", book_model(pages=-3, broken=True)), "--ocl", "--json")

    assert run.returncode == 1
    assert sorted(error.split(":")[1].strip() for error in run.json["errors"]) == [
        f"{OCL_PREFIX} [Book inv broken]", f"{OCL_PREFIX} [Book inv positive_pages]",
    ]


def test_without_ocl_the_constraints_are_not_checked(besser, write):
    run = besser("validate", write("books.py", book_model(pages=-3, broken=True)), "--json")

    assert run.returncode == 0 and run.json["notes"] == []


def test_a_model_without_constraints_says_so(besser, model):
    run = besser("validate", model, "--ocl", "--json")

    assert run.returncode == 0
    assert run.json["notes"] == ["OCL: the models have no OCL constraints"]


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
def besser_without_bocl(tmp_path):
    """Run ``python -m besser`` where importing ``bocl`` fails, as when ``besser[ocl]`` is not installed."""
    shim = tmp_path / "shim" / "bocl"
    shim.mkdir(parents=True)
    (shim / "__init__.py").write_text("raise ModuleNotFoundError(\"No module named 'bocl'\", name='bocl')\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(shim.parent), os.environ.get("PYTHONPATH")]))}

    def run(*args) -> Run:
        process = subprocess.run(
            [sys.executable, "-m", "besser", *map(str, args)],
            cwd=tmp_path, capture_output=True, text=True, timeout=120, check=False, env=env,
        )
        return Run(process.returncode, process.stdout, process.stderr)
    return run


def test_without_bocl_ocl_is_skipped_with_a_warning_saying_what_to_install(besser_without_bocl, write):
    write("books.py", book_model(pages=-3, broken=True))
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
