"""``besser model``: change a domain model's ``.py`` file in place, one command per change.

Each test checks the file's text after the change: only the lines of the change differ, comments and
layout stay, and a change that cannot be made or makes the model invalid writes nothing.
"""

import difflib
import subprocess
import sys

import pytest

from tests.cli.conftest import LIBRARY_MODEL

# The three ways a model file is written: inline sets (besser new), a variable per element with the
# attributes set afterwards (besser convert), and helper functions with wrapped sets (an agent's own).
INLINE = '''\
"""A library."""

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    IntegerType,
    Multiplicity,
    Property,
    StringType,
)

# Classes and their attributes
library = Class(name="Library", attributes={
    Property(name="name", type=StringType),  # shown on the front page
})
book = Class(name="Book", attributes={
    Property(name="title", type=StringType),
    Property(name="pages", type=IntegerType),
})

# Associations
library_books = BinaryAssociation(name="library_books", ends={
    Property(name="library", type=library, multiplicity=Multiplicity(1, 1)),
    Property(name="books", type=book, multiplicity=Multiplicity(0, "*")),
})

# The model
library_model = DomainModel(
    name="Library",
    types={library, book},
    associations={library_books},
)
'''

VARIABLES = '''\
from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType, Constraint

# Classes
Book = Class(name="Book")

# Book class attributes and methods
Book_title: Property = Property(name="title", type=StringType)
Book.attributes={Book_title}

rule = Constraint(name="titled", context=Book, language="OCL", expression="context Book inv titled: self.title <> ''")
domain_model = DomainModel(name="Shop", types={Book}, constraints={rule})
'''

HELPERS = '''\
from besser.BUML.metamodel.structural import Class, DomainModel, Enumeration, EnumerationLiteral, Property, StringType


def _enum(name, *literals):
    return Enumeration(name=name, literals={EnumerationLiteral(name=l) for l in literals})


Stage = _enum("Stage", "LEAD", "WON")
deal_title = Property(name="title", type=StringType)
deal_stage = Property(name="stage", type=Stage)
Deal = Class(name="Deal", attributes={deal_title, deal_stage})
crm = DomainModel(name="CRM", types={Stage, Deal})

COLUMNS = [("Title", deal_title)]  # what a GUI model would refer to
'''


def _changed_lines(before: str, after: str) -> tuple[list[str], list[str]]:
    """The lines removed from ``before`` and those added in ``after``."""
    diff = list(difflib.ndiff(before.splitlines(), after.splitlines()))
    return [line[2:] for line in diff if line.startswith("- ")], [line[2:] for line in diff if line.startswith("+ ")]


@pytest.fixture
def edit(besser, write, tmp_path):
    """Write ``text`` to model.py, run ``besser model ARGS`` on it, and return the run and the new text."""
    def run(text: str, *args, **kwargs):
        (tmp_path / "model.py").write_text(text, encoding="utf-8")
        result = besser("model", *args, **kwargs)
        return result, (tmp_path / "model.py").read_text(encoding="utf-8")
    return run


def test_an_attribute_goes_into_its_class_set_and_nothing_else_changes(edit):
    run, text = edit(INLINE, "add-attribute", "Book", "isbn:str", "subtitle:str?", output=None)

    assert run.returncode == 0
    assert run.stdout == "model.py: added Book.isbn:str, subtitle:str?\n"
    removed, added = _changed_lines(INLINE, text)
    assert removed == []
    assert added == ['    Property(name="isbn", type=StringType),',
                     '    Property(name="subtitle", type=StringType, is_optional=True),']
    assert text.index("isbn") > text.index('"pages"') and text.index("isbn") < text.index("# Associations")


def test_a_class_with_a_parent_and_an_association_follow_the_inline_style(edit, besser):
    _, text = edit(INLINE, "add-class", "Member", "joined:date", "--parent", "Book")
    run, text = edit(text, "add-association", "loans", "loans:Book[*]", "borrower:Member[0..1]")

    assert run.returncode == 0
    assert 'member = Class(name="Member", attributes={\n    Property(name="joined", type=DateType),\n})' in text
    assert "    types={library, book, member},\n" in text
    assert "    associations={library_books, loans},\n" in text
    assert "    generalizations={Generalization(general=book, specific=member)},\n" in text
    assert "    DateType,\n" in text and "    Generalization,\n" in text  # imported where the file imports
    info = besser("info", "model.py").stdout
    assert "class Member(Book): joined:date" in info
    assert "association loans: loans:Book[*] borrower:Member[0..1]" in info


def test_a_new_enumeration_is_defined_before_the_classes_that_use_it(edit, besser):
    _, text = edit(INLINE, "set-enum", "Genre", "NOVEL", "ESSAY")
    run, text = edit(text, "add-attribute", "Book", "genre:Genre=NOVEL")

    assert run.returncode == 0
    assert text.index("genre = Enumeration(") < text.index("# Classes and their attributes")
    assert '    Property(name="genre", type=genre, default_value="NOVEL"),\n' in text
    assert besser("validate", "model.py").returncode == 0


def test_set_enum_replaces_the_literals_in_place(edit):
    _, text = edit(INLINE, "set-enum", "Genre", "NOVEL", "ESSAY")
    run, text = edit(text, "set-enum", "Genre", "NOVEL", "POEM", output=None)

    assert run.stdout == "model.py: set Genre: NOVEL, POEM\n"
    assert 'literals={EnumerationLiteral(name="NOVEL"), EnumerationLiteral(name="POEM")}' in text


def test_a_literal_used_as_a_default_cannot_be_removed(edit):
    _, text = edit(INLINE, "set-enum", "Genre", "NOVEL", "ESSAY")
    _, text = edit(text, "add-attribute", "Book", "genre:Genre=NOVEL")
    run, after = edit(text, "set-enum", "Genre", "ESSAY")

    assert run.returncode == 2 and after == text
    assert "the default of Book.genre is a literal this removes" in run.stderr


def test_rename_a_class_renames_its_variable_and_the_name_only(edit):
    run, text = edit(INLINE, "rename", "Book", "Volume", output=None)

    assert run.stdout == "model.py: renamed Book to Volume (variable book -> volume)\n"
    removed, added = _changed_lines(INLINE, text)
    assert removed == ['book = Class(name="Book", attributes={',
                       '    Property(name="books", type=book, multiplicity=Multiplicity(0, "*")),',
                       '    types={library, book},']
    assert added == ['volume = Class(name="Volume", attributes={',
                     '    Property(name="books", type=volume, multiplicity=Multiplicity(0, "*")),',
                     '    types={library, volume},']


def test_rename_an_attribute_updates_the_constraints_that_name_it(edit):
    run, text = edit(VARIABLES, "rename", "Book.title", "heading")

    assert run.returncode == 0
    assert 'Book_title: Property = Property(name="heading", type=StringType)' in text
    assert 'expression="context Book inv titled: self.heading <> \'\'")' in text


def test_the_variable_style_gets_a_variable_per_attribute(edit):
    run, text = edit(VARIABLES, "add-attribute", "Book", "price:float=9.5")

    assert run.returncode == 0
    _, added = _changed_lines(VARIABLES, text)
    assert added[1:] == ['Book_price = Property(name="price", type=FloatType, default_value=9.5)',
                         "Book.attributes={Book_title, Book_price}"]
    assert text.startswith("from besser.BUML.metamodel.structural import Class, DomainModel, Property, "
                           "StringType, Constraint, FloatType\n")


def test_an_element_a_helper_builds_is_changed_by_a_statement_after_it(edit, besser):
    run, text = edit(HELPERS, "set-enum", "Stage", "LEAD", "WON", "LOST")

    assert run.returncode == 0
    assert ('Stage = _enum("Stage", "LEAD", "WON")\nStage.literals = {EnumerationLiteral(name="LEAD"), '
            'EnumerationLiteral(name="WON"), EnumerationLiteral(name="LOST")}\n') in text
    assert "enumeration Stage: LEAD, WON, LOST" in besser("info", "model.py").stdout


def test_removing_what_other_code_uses_is_refused_with_the_lines(edit):
    run, text = edit(HELPERS, "remove", "Deal.title")

    assert run.returncode == 2 and text == HELPERS
    assert run.stderr == ("error: remove Deal.title: Deal.title is still used on line 14 of model.py: "
                          "change those lines first (model.py not changed)\n")


def test_remove_a_class_takes_its_associations_along(edit, besser):
    run, text = edit(INLINE, "remove", "Library", output=None)

    assert run.stdout == "model.py: removed Library and association library_books\n"
    assert "library = Class(" not in text and "library_books" not in text and "type=library," not in text
    assert "    types={book},\n    associations=set(),\n" in text
    assert "# Classes and their attributes" in text  # comments stay
    assert besser("validate", "model.py").returncode == 0


def test_remove_an_attribute(edit):
    run, text = edit(INLINE, "remove", "Book.pages")

    assert run.returncode == 0
    removed, added = _changed_lines(INLINE, text)
    assert removed == ['    Property(name="pages", type=IntegerType),'] and added == []


def test_methods_and_constraints(edit, besser):
    _, text = edit(INLINE, "add-method", "Book", "discount(rate:float) -> float")
    run, text = edit(text, "add-constraint", "Book", "self.pages > 0", "--name", "has_pages")

    assert run.returncode == 0
    info = besser("info", "model.py", "--class", "Book").stdout
    assert "  methods: discount(rate:float) -> float\n" in info
    assert "  constraints: has_pages: self.pages > 0\n" in info


def test_a_change_that_makes_the_model_invalid_writes_nothing(edit):
    run, text = edit(INLINE, "add-class", "Novel", "title:str", "--parent", "Book")

    assert run.returncode == 1 and text == INLINE
    assert "defines attribute 'title' which already exists in a parent class" in run.stderr
    assert "model.py not changed: the change makes the model invalid" in run.stderr


@pytest.mark.parametrize("args, message", [
    (("add-attribute", "Book", "isbn"), "'isbn' is not an attribute: write name:type"),
    (("add-attribute", "Book", "title:str"), "'title' exists already"),
    (("add-attribute", "Book", "year:Date"), "has no class or enumeration named 'Date'"),
    (("add-attribute", "Shelf", "year:int"), "has no class named 'Shelf' (it has: Book, Library)"),
    (("add-association", "x", "a:Book", "b:Library[1]"), "'a:Book' is not an association end"),
    (("rename", "Book.nope", "x"), "Book has no member named 'nope'"),
    (("rename", "Book", "Library"), "'Library' is taken"),
])
def test_a_change_that_does_not_apply_says_why_and_writes_nothing(edit, args, message):
    run, text = edit(INLINE, *args)

    assert run.returncode == 2 and text == INLINE
    assert message in run.stderr


def test_batch_makes_every_change_in_one_write(edit, besser):
    changes = "# the new feature\nadd-class Tag label:str\n\nadd-association tagged tags:Tag[*] books:Book[*]\n"
    run, text = edit(INLINE, "batch", "-", stdin=changes, output=None)

    assert run.returncode == 0
    assert run.stdout == ("model.py: added class Tag with 1 attribute; added association tagged: tags:Tag[*] "
                          "books:Book[*]\n")
    assert "association tagged" in besser("info", "model.py").stdout


def test_batch_writes_nothing_when_one_change_fails(edit):
    run, text = edit(INLINE, "batch", "-", stdin="add-class Tag label:str\nrename Nope Other\n")

    assert run.returncode == 2 and text == INLINE
    assert "rename Nope Other: Library has no class" in run.stderr


def test_the_model_file_comes_from_besser_toml(besser, write, tmp_path):
    write("models/shop.py", LIBRARY_MODEL)
    write("besser.toml", 'model = "models/shop.py"\n[output.api]\ngenerator = "backend"\ndir = "api"\n')

    run = besser("model", "add-attribute", "Book", "isbn:str", output=None)

    assert run.stdout == "models/shop.py: added Book.isbn:str\n"


def test_json_reports_the_changes(edit):
    run, _ = edit(INLINE, "add-attribute", "Book", "isbn:str", "--json")

    assert run.json["ok"] is True and run.json["command"] == "model"
    assert run.json["changes"] == ["added Book.isbn:str"] and run.json["files"] == ["model.py"]


def test_windows_line_endings_are_kept(besser, tmp_path):
    (tmp_path / "model.py").write_bytes(INLINE.replace("\n", "\r\n").encode())

    run = besser("model", "add-attribute", "Book", "isbn:str")

    data = (tmp_path / "model.py").read_bytes()
    assert run.returncode == 0 and b'    Property(name="isbn", type=StringType),\r\n' in data
    assert b"\n" not in data.replace(b"\r\n", b"")


def test_an_editor_json_file_is_refused(besser, write):
    write("project.json", {"project": {}})

    run = besser("model", "-f", "project.json", "add-class", "X")

    assert run.returncode == 2 and "'besser convert project.json -o model.py' first" in run.stderr


def test_adding_then_removing_a_class_gives_the_file_back(edit, besser):
    _, text = edit(INLINE, "add-class", "Member", "name:str")
    _, text = edit(text, "add-association", "loans", "loans:Book[*]", "borrower:Member[0..1]")
    run, text = edit(text, "remove", "Member")

    assert run.returncode == 0
    assert text == INLINE


# ── OCL: the constraints name members of the model ──────────────────────────


@pytest.fixture
def project(besser, tmp_path):
    """A project of 'besser init' (the starter's Library, Book and Author); returns a reader of model.py."""
    assert besser("init").returncode == 0
    return lambda: (tmp_path / "model.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("expression, reason", [
    ("self.weight > 0", "Property 'weight' not found in context 'Book'"),
    ("self.pages >>> 0", "mismatched input '>'"),
])
def test_a_constraint_that_does_not_parse_against_the_model_is_refused(besser, project, expression, reason):
    before = project()

    run = besser("model", "add-constraint", "Book", expression)

    assert run.returncode == 1 and project() == before
    assert "error: OCL constraint 'book_rule_1' on Book does not parse: " in run.stderr and reason in run.stderr
    assert "model.py not changed: the change makes the model invalid" in run.stderr


def test_validate_and_apply_refuse_a_constraint_that_names_no_member(besser, write, tmp_path):
    write("model.py", LIBRARY_MODEL.replace("library_model = DomainModel(", '''
from besser.BUML.metamodel.structural import Constraint
rule = Constraint(name="heavy", context=book, language="OCL", expression="context Book inv heavy: self.weight > 0")
library_model = DomainModel(constraints={rule}, '''))
    write("besser.toml", 'model = "model.py"\n[output.backend]\ngenerator = "backend"\ndir = "backend"\n')

    validated = besser("validate", "model.py")
    applied = besser("apply")

    assert validated.returncode == 1
    assert ("error: OCL constraint 'heavy' on Book does not parse: Property 'weight' not found in context 'Book'"
            in validated.stdout)
    assert applied.returncode == 1 and "nothing was generated: the model has validation errors" in applied.stderr
    assert not (tmp_path / "backend").exists()


def test_validate_ocl_reports_a_constraint_that_does_not_parse_once(besser, write):
    write("model.py", LIBRARY_MODEL.replace("library_model = DomainModel(", '''
from besser.BUML.metamodel.structural import Constraint
rule = Constraint(name="odd", context=book, language="OCL", expression="context Book inv odd: self.pages >>> 0")
library_model = DomainModel(constraints={rule}, '''))

    run = besser("validate", "model.py", "--ocl", "--json")

    assert run.json["exit_code"] == 1
    assert [error for error in run.json["errors"] if "odd" in error] == [
        error for error in run.json["errors"] if error.startswith("library_model: OCL constraint 'odd' on Book")]
    assert len(run.json["errors"]) == 1


def test_remove_refuses_a_member_a_constraint_uses(besser, project):
    besser("model", "add-constraint", "Book", "self.pages > 0")
    besser("model", "add-constraint", "Library", "self.books->forAll(b | b.title <> '')", "--name", "titled")
    before = project()

    attribute = besser("model", "remove", "Book.pages")
    navigated = besser("model", "remove", "Book.title")
    end = besser("model", "remove", "library_books")
    unused = besser("model", "remove", "Author.email")

    assert attribute.returncode == 2 and navigated.returncode == 2 and end.returncode == 2
    assert "constraint book_rule_1 uses Book.pages: remove or change it first (model.py not changed)" \
        in attribute.stderr
    assert "constraint titled uses Book.title: remove or change it first" in navigated.stderr
    assert "constraint titled uses Library.books: remove or change it first" in end.stderr
    assert unused.returncode == 0 and project() == before.replace(
        '    Property(name="email", type=StringType),\n', "")


def test_rename_rewrites_the_names_a_constraint_navigates_and_no_quoted_text(besser, project, tmp_path):
    besser("model", "add-constraint", "Author", "self.name <> 'Book'")
    besser("model", "add-constraint", "Library", "self.name <> ''")
    besser("model", "add-constraint", "Library", "self.books->forAll(b | b.title <> self.name)", "--name", "titles")

    runs = [besser("model", "rename", "Library.name", "label"), besser("model", "rename", "Book", "Volume"),
            besser("model", "rename", "Volume.title", "heading"), besser("model", "rename", "titles", "unique")]
    applied = besser("apply")
    imported = subprocess.run([sys.executable, "-c", "import main_api"], cwd=tmp_path / "backend",
                              capture_output=True, text=True, check=False)

    assert [run.returncode for run in runs] == [0, 0, 0, 0]
    text = project()
    assert "expression=\"context Author inv author_rule_1: self.name <> 'Book'\"" in text
    assert "expression=\"context Library inv library_rule_1: self.label <> ''\"" in text
    assert ("expression=\"context Library inv unique: self.books->forAll(b | b.heading <> self.label)\""
            in text)
    assert applied.returncode == 0, applied.stderr
    assert imported.returncode == 0, imported.stderr


def test_rename_a_literal_renames_the_defaults_that_use_it(besser, project):
    besser("model", "set-enum", "Genre", "NOVEL", "POEM")
    besser("model", "add-attribute", "Book", "genre:Genre=NOVEL")

    renamed = besser("model", "rename", "Genre.NOVEL", "FICTION")
    removed = besser("model", "remove", "Genre.FICTION")

    assert renamed.returncode == 0
    assert 'Property(name="genre", type=genre, default_value="FICTION")' in project()
    assert "class Book: title:str, pages:int, release:date, genre:Genre='FICTION'" in besser("info", "model.py").stdout
    assert removed.returncode == 2
    assert "the default of Book.genre is FICTION: change it first" in removed.stderr
    assert besser("apply").returncode == 0


def test_validate_reports_a_default_that_is_no_literal_of_its_enumeration(besser, project, tmp_path):
    besser("model", "set-enum", "Genre", "NOVEL", "POEM")
    besser("model", "add-attribute", "Book", "genre:Genre=NOVEL")
    (tmp_path / "model.py").write_text(project().replace('default_value="NOVEL"', 'default_value="ESSAY"'),
                                       encoding="utf-8")

    run = besser("validate", "model.py")

    assert run.returncode == 1
    assert ("error: Attribute 'genre' of class 'Book' has the default 'ESSAY', which is not a literal of its "
            "enumeration 'Genre' (NOVEL, POEM).") in run.stdout
