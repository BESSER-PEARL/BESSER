"""``besser info``: what a model file holds and the generators that read it."""

import json

import pytest

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
             "attributes": [{"name": "name", "type": "str", "optional": False, "default": None, "derived": False}]},
            {"name": "Book", "is_abstract": False, "parents": [], "methods": [],
             "attributes": [{"name": "title", "type": "str", "optional": False, "default": None, "derived": False},
                            {"name": "pages", "type": "int", "optional": False, "default": None, "derived": False}]},
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


def test_each_screen_says_the_class_and_the_columns_of_its_tables(besser):
    assert besser("init", "--kind", "gui").returncode == 0
    run = besser("info", "model.py")

    assert run.stdout.splitlines()[-4:-1] == [
        "    modules: library_module",
        "    screen Authors (main page), /Authors: Author table author_table (name, email); buttons to Books",
        "    screen Books, /Books: Book table book_table (title, pages)",
    ]
    screens = besser("info", "model.py", "--json").json["models"][1]["screens"]
    assert screens[1] == {"name": "Books", "route": "/Books", "main": False, "opens": [],
                          "cards": [], "charts": [], "menu": False, "links": [],
                          "tables": [{"name": "book_table", "class": "Book", "columns": ["title", "pages"]}]}


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


DIRECTORY_GIVEN = ("models is a directory: name a model file (a B-UML .py file or an editor .json export); "
                   "'besser validate' takes a directory and checks the model files in it")


@pytest.mark.parametrize("args", [
    ("info", "models"),
    ("generate", "python", "models", "-o", "out"),
    ("regenerate", "python", "models", "-o", "out"),
])
def test_a_directory_where_a_model_file_goes_is_named_a_directory(besser, tmp_path, args):
    (tmp_path / "models").mkdir()
    run = besser(*args, "--json")

    assert run.returncode == 2
    assert run.json["errors"] == [DIRECTORY_GIVEN]


def test_a_directory_named_like_a_model_file_is_named_a_directory(besser, tmp_path):
    (tmp_path / "models.py").mkdir()
    (tmp_path / "graph.ttl").mkdir()
    converted = besser("convert", "models.py", "-o", "x.json", "--json")
    imported = besser("convert", "graph.ttl", "-o", "x.py", "--from", "rdf", "--json")
    edited = besser("model", "-f", "models.py", "add-class", "Shelf", "--json")

    assert converted.json["errors"] == [DIRECTORY_GIVEN.replace("models is", "models.py is")]
    assert imported.json["errors"] == ["graph.ttl is a directory: --from rdf reads a file"]
    assert edited.json["errors"] == ["models.py is a directory: name the model file with -f FILE"]


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


def test_a_large_model_is_summarised_by_its_names(besser, tmp_path):
    # Why: at 48 classes the full report is 17,000 characters, read at the start of every session of a coding
    # agent that then changes one class; --class gives that class, with the page that shows it.
    assert besser("init", "--kind", "gui").returncode == 0
    batch = "".join(f"add-class Thing{n} code:str label:str --screen\n" for n in range(1, 21))
    assert besser("model", "batch", "-", stdin=batch).returncode == 0
    full = besser("info", "model.py", "--all").stdout
    assert "class Thing7: code:str, label:str" in full and "screen Thing7s, /thing7s: Thing7 table" in full

    overview = besser("info", "model.py").stdout
    assert len(overview) < len(full) / 2
    assert "    classes: Book, Author, Thing1, Thing2," in overview and "class Thing7:" not in overview
    assert "    screens: Authors /Authors (main page), Books /Books, Thing10s /thing10s," in overview
    assert ("A large model: names only. 'besser info model.py --class X' shows a class in full with its page; "
            "--all shows everything.") in overview
    one = besser("info", "model.py", "--class", "Thing7").stdout
    assert "  attributes: code:str, label:str\n" in one
    assert "  page: screen Thing7s, /thing7s: Thing7 table thing7_table (code, label); buttons to" not in one
    assert "  page: screen Thing7s, /thing7s: Thing7 table thing7_table (code, label)\n" in one

    # A model of ordinary size is printed in full, as before.
    assert besser("model", "batch", "-", stdin="remove Thing20\nremove Thing19\n").returncode == 0
    assert "class Thing7: code:str, label:str" in besser("info", "model.py").stdout
