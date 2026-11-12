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
        "    class Library: name:str",
        "    class Book: title:str, pages:int",
        "    association holds: library:Library[1] books:Book[*]",
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
             "attributes": [{"name": "name", "type": "str", "optional": False, "default": None}]},
            {"name": "Book", "is_abstract": False, "parents": [], "methods": [],
             "attributes": [{"name": "title", "type": "str", "optional": False, "default": None},
                            {"name": "pages", "type": "int", "optional": False, "default": None}]},
        ],
        "enumerations": [],
        "associations": [{"name": "holds", "ends": [
            {"name": "library", "type": "Library", "multiplicity": "1"},
            {"name": "books", "type": "Book", "multiplicity": "*"},
        ]}],
        "constraints": [],
        "backend_endpoints": {
            "Library": ["POST /library/", "GET /library/", "GET /library/{library_id}/", "PUT /library/{library_id}/",
                        "DELETE /library/{library_id}/", "POST /library/bulk/", "DELETE /library/bulk/",
                        "GET /library/count/", "GET /library/paginated/", "GET /library/search/"],
            "Book": ["POST /book/", "GET /book/", "GET /book/{book_id}/", "PUT /book/{book_id}/",
                     "DELETE /book/{book_id}/", "POST /book/bulk/", "DELETE /book/bulk/", "GET /book/count/",
                     "GET /book/paginated/", "GET /book/search/"],
        },
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
    assert "    class Book: title:str\n" in run.stdout


def test_a_model_from_standard_input_is_summarised(besser):
    run = besser("info", "-", stdin=json.dumps(PROJECT_EXPORT))

    assert run.returncode == 0
    assert run.stdout.startswith("<stdin>: 1 model\n")


def test_a_model_with_validation_errors_is_still_summarised(besser, write):
    run = besser("info", write("broken.py", INVALID_MODEL))

    assert run.returncode == 0
    assert "    class Book: title:Text\n" in run.stdout


def test_a_file_that_does_not_load_is_a_load_error(besser):
    run = besser("info", "missing.py")

    assert run.returncode == 2
    assert run.stderr == "error: missing.py: no such file\n"


def test_quiet_prints_no_notes_or_warnings(besser, write):
    # Two domain models and no current one: info notes that generators need --use.
    project = dict(PROJECT_EXPORT["project"])
    project["diagrams"] = {"ClassDiagram": [
        {**PROJECT_EXPORT["project"]["diagrams"]["ClassDiagram"][0], "id": "d1", "title": "One"},
        {**PROJECT_EXPORT["project"]["diagrams"]["ClassDiagram"][0], "id": "d2", "title": "Two"},
    ]}
    model = write("two.json", {"project": project})
    assert "note: two.json holds 2 domain models" in besser("info", model).stderr

    run = besser("info", model, "-q")

    assert run.returncode == 0
    assert run.stderr == ""


def test_verbose_prints_the_traceback_of_a_model_file_that_raises(besser, write):
    model = write("crash.py", "def build():\n    return 1 / 0\n\nbuild()\n")

    run = besser("info", model, "-v")

    assert run.returncode == 2
    assert "Traceback (most recent call last):" in run.stderr
    assert 'crash.py", line 2, in build' in run.stderr
    assert run.stderr.endswith("error: crash.py: line 2: ZeroDivisionError: division by zero\n")
    assert "Traceback" not in besser("info", model).stderr


def test_a_model_that_is_not_a_domain_model_lists_what_it_holds_by_name(besser):
    besser("new", "machine.py", "--kind", "state_machine")

    run = besser("info", "machine.py", "--json")
    states = run.json["models"][0]["elements"]["states"]

    assert run.json["models"][0]["counts"] == {"states": len(states)}
    assert "created" in states
    assert f"    states: {', '.join(states)}\n" in besser("info", "machine.py").stdout


SHAPES_MODEL = '''
from besser.BUML.metamodel.structural import (
    AssociationClass, BinaryAssociation, Class, DomainModel, FloatType, Generalization, Multiplicity, Property,
    StringType,
)

cat = Class(name="Cat", attributes={Property(name="name", type=StringType)})
item = Class(name="Item")
stock = BinaryAssociation(name="stock", ends={
    Property(name="cats", type=cat, multiplicity=Multiplicity(0, "*")),
    Property(name="items", type=item, multiplicity=Multiplicity(0, "*")),
})
listing = AssociationClass(name="Listing", attributes={Property(name="price", type=FloatType)}, association=stock)
prod = Class(name="Prod", is_read_only=True, attributes={Property(name="n", type=StringType)})
kitten = Class(name="Kitten")
shapes = DomainModel(name="Shapes", types={cat, item, listing, prod, kitten}, associations={stock},
                     generalizations={Generalization(general=cat, specific=kitten)})
'''


def test_class_routes_follow_association_classes_and_read_only_classes(besser, write):
    model = write("m.py", SHAPES_MODEL)

    listing = besser("info", model, "--class", "Listing").stdout
    prod = besser("info", model, "--class", "Prod").stdout
    item = besser("info", model, "--class", "Item").stdout
    kitten = besser("info", model, "--class", "Kitten").stdout
    summary = besser("info", model).stdout

    assert ("  backend routes: POST|GET /listing/, GET|PUT|DELETE /listing/{cats_id:int}/{items_id:int}/, "
            "GET /listing/count/, GET /listing/paginated/, GET /listing/search/\n") in listing
    assert ("  backend routes: POST|GET /prod/, GET /prod/{prod_id}/, POST|DELETE /prod/bulk/, GET /prod/count/, "
            "GET /prod/paginated/, GET /prod/search/\n") in prod
    assert "/item/search/" not in item and "GET /kitten/search/" in kitten
    assert "  inherited ends: items:Item[*] (stock, from Cat)\n" in kitten
    assert "    Listing routes: POST|GET /listing/, GET|PUT|DELETE /listing/{cats_id:int}/{items_id:int}/," in summary
    assert "    Cat routes:" not in summary
