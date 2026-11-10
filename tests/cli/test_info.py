"""``besser info``: what a model file holds and the generators that read it."""

import json

from besser.generators.registry import GENERATORS
from tests.cli.conftest import INVALID_MODEL, LIBRARY_MODEL, PROJECT_EXPORT, TWO_DOMAIN_MODELS

GUI_AND_ENUM_MODEL = LIBRARY_MODEL + '''
from besser.BUML.metamodel.gui import GUIModel, Module, Screen
from besser.BUML.metamodel.structural import Enumeration, EnumerationLiteral, Generalization, Method

genre = Enumeration(name="Genre", literals={EnumerationLiteral(name="novel")})
novel = Class(name="Novel", is_abstract=True, methods={Method(name="summary", type=StringType)})
library_model.add_type(genre)
library_model.add_type(novel)
library_model.generalizations = {Generalization(general=book, specific=novel)}
home = Screen(name="Home", description="", x_dpi="", y_dpi="", screen_size="Small", view_elements=set(),
              is_main_page=True)
gui = GUIModel(name="Shelf", package="p", versionCode="1", versionName="1", description="",
               screenCompatibility=True, modules={Module(name="Main", screens={home})})
'''


def test_it_prints_the_classes_attributes_and_associations(besser, model):
    run = besser("info", model)

    assert run.returncode == 0
    assert run.stdout.splitlines()[:5] == [
        "library.py: 1 model",
        "  library_model (DomainModel 'LibraryModel'): 2 classes, 1 association",
        "    class Library: name: str",
        "    class Book: title: str, pages: int",
        "    association holds: library: Library [1] -- books: Book [0..*]",
    ]


def test_it_names_the_generators_that_read_the_file(besser, model):
    last_line = besser("info", model).stdout.splitlines()[-1]
    domain_only = [spec.id for spec in GENERATORS if spec.models == ("domain",)]

    assert last_line == f"Generators that read this model: {', '.join(domain_only)}"
    assert "python" in domain_only and "react" not in domain_only


def test_json_describes_each_model(besser, model):
    result = besser("info", model, "--json").json

    assert result["ok"] is True and result["command"] == "info" and result["model"] == model
    assert result["models"] == [{
        "name": "library_model", "kind": "domain", "type": "DomainModel", "model_name": "LibraryModel", "current": False,
        "counts": {"classes": 2, "enumerations": 0, "associations": 1, "generalizations": 0, "constraints": 0},
        "classes": [
            {"name": "Library", "is_abstract": False, "parents": [], "methods": [],
             "attributes": [{"name": "name", "type": "str"}]},
            {"name": "Book", "is_abstract": False, "parents": [], "methods": [],
             "attributes": [{"name": "title", "type": "str"}, {"name": "pages", "type": "int"}]},
        ],
        "enumerations": [],
        "associations": [{"name": "holds", "ends": [
            {"name": "library", "type": "Library", "multiplicity": "1"},
            {"name": "books", "type": "Book", "multiplicity": "0..*"},
        ]}],
    }]
    assert "python" in result["generators"] and "react" not in result["generators"]


def test_enumerations_inheritance_methods_and_other_model_kinds(besser, write):
    run = besser("info", write("app.py", GUI_AND_ENUM_MODEL))

    assert run.returncode == 0
    assert "(DomainModel 'LibraryModel'): 3 classes, 1 enumeration, 1 association, 1 generalization" in run.stdout
    assert "    abstract class Novel(Book): summary()\n" in run.stdout
    assert "    enumeration Genre: novel\n" in run.stdout
    assert "  gui (GUIModel 'Shelf'): 1 module, 1 screen\n" in run.stdout
    assert "react" in run.stdout.splitlines()[-1]


def test_every_model_of_a_file_is_listed_by_the_name_use_takes(besser, write):
    result = besser("info", write("two.py", TWO_DOMAIN_MODELS), "--json").json

    assert [(entry["name"], entry["model_name"]) for entry in result["models"]] == [
        ("library_model", "LibraryModel"), ("draft", "Draft"),
    ]


def test_an_editor_project_is_summarised(besser, write):
    run = besser("info", write("project.json", PROJECT_EXPORT))

    assert run.returncode == 0
    assert "    class Book: title: str\n" in run.stdout


def test_a_model_from_standard_input_is_summarised(besser):
    run = besser("info", "-", stdin=json.dumps(PROJECT_EXPORT))

    assert run.returncode == 0
    assert run.stdout.startswith("<stdin>: 1 model\n")


def test_a_model_with_validation_errors_is_still_summarised(besser, write):
    run = besser("info", write("broken.py", INVALID_MODEL))

    assert run.returncode == 0
    assert "    class Book: title: Text\n" in run.stdout


def test_a_file_that_does_not_load_is_a_load_error(besser):
    run = besser("info", "missing.py")

    assert run.returncode == 2
    assert run.stderr == "error: missing.py: no such file\n"
