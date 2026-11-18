"""``besser model``: change an element in place (an attribute, an association end, a class's parents),
and the checks that keep a change from being accepted wrong (defaults, enumeration literals in OCL).

Each in-place change is checked on the file's text (only the changed element's lines differ) and where
it shows: ``besser validate --ocl`` passes and the generated backend imports and applies it.
"""

import difflib
import json
import os
import subprocess
import sys
import textwrap

import pytest

STARTER_BOOK = (
    'book = Class(name="Book", attributes={\n'
    '    Property(name="title", type=StringType),\n'
    '    Property(name="pages", type=IntegerType),\n'
    '    Property(name="release", type=DateType),\n'
    '})\n'
)


def _changed_lines(before: str, after: str) -> tuple[list[str], list[str]]:
    """The lines removed from ``before`` and those added in ``after``."""
    diff = list(difflib.ndiff(before.splitlines(), after.splitlines()))
    return [line[2:] for line in diff if line.startswith("- ")], [line[2:] for line in diff if line.startswith("+ ")]


@pytest.fixture
def project(besser, tmp_path):
    """A project of 'besser init' (the starter's Library, Book and Author); returns a reader of model.py."""
    assert besser("init").returncode == 0
    return lambda: (tmp_path / "model.py").read_text(encoding="utf-8")


@pytest.fixture
def change(besser):
    """Run ``besser model ARGS``; check it succeeded with one line and return that line."""
    def run(*args, stdin=""):
        result = besser("model", *args, stdin=stdin, output=None)
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(result.stdout.splitlines()) == 1, result.stdout
        return result.stdout.strip()
    return run


@pytest.fixture
def refused(besser, tmp_path):
    """Run ``besser model ARGS``; check it was refused (exit 2) and wrote nothing; return its message."""
    def run(*args, stdin=""):
        before = (tmp_path / "model.py").read_bytes()
        result = besser("model", *args, stdin=stdin)
        assert result.returncode == 2, result.stdout + result.stderr
        assert (tmp_path / "model.py").read_bytes() == before
        return result.stderr
    return run


# ── defaults are literals of their attribute's type ─────────────────────────


@pytest.mark.parametrize("attribute, message", [
    # Accepted before, and the backend then wrote default='now' on a Date column.
    ("created:date=now", "default 'now': a date attribute takes no default. The generated backend writes a "
                         "default as text, which its date column refuses on insert, and has no 'now'. Leave it "
                         "without one, make it optional (date?) or name it created_at or updated_at (no default)"),
    ("created:date=2024-01-31", "a date attribute takes no default"),
    ("opened:datetime=2024-01-31T10:00", "a datetime attribute takes no default"),
    ("at:time=10:00", "a time attribute takes no default. "),
    ("copies:int=1.5", "default '1.5' is not an int (a whole number, e.g. 0)"),  # accepted before
    ("copies:int=true", "default 'true' is not an int"),
    ("price:float=cheap", "default 'cheap' is not a number (e.g. 0.5)"),
    ("lent:bool=1", "default '1' is not true or false"),
])
def test_a_default_that_is_no_literal_of_its_type_is_refused(project, refused, attribute, message):
    assert message in refused("add-attribute", "Book", attribute)


def test_a_default_of_an_enumeration_names_its_literals(project, change, refused):
    change("set-enum", "Genre", "NOVEL", "POEM")

    assert "default 'ESSAY' is not a literal of Genre (NOVEL, POEM)" in refused("add-attribute", "Book",
                                                                                 "genre:Genre=ESSAY")


def test_the_defaults_the_backend_applies_are_accepted(project, change):
    line = change("add-attribute", "Book", "copies:int=1", "price:float=2", "lent:bool=false", "note:str=now",
                  "created_at:datetime")

    assert line == ("model.py: added Book.copies:int=1, price:float=2, lent:bool=false, note:str=now, "
                    "created_at:datetime")


# ── the generated backend, run on the changed model ─────────────────────────

# Creates a library and an author, then POSTs the book of argv[1] (JSON); prints the status and the body.
_POST_BOOK = '''
import json, sys
from fastapi.testclient import TestClient
import main_api


def ident(body):
    return body.get("id") or next(v["id"] for v in body.values() if isinstance(v, dict) and "id" in v)


with TestClient(main_api.app) as client:
    library = ident(client.post("/library/", json={"name": "L", "address": "A"}).json())
    author = ident(client.post("/author/", json={"name": "N", "email": "e"}).json())
    book = json.loads(sys.argv[1])
    book.setdefault("library", library)
    book.setdefault("authors", [author])
    response = client.post("/book/", json=book)
    body = response.json()
    print(json.dumps({"status": response.status_code, "book": body.get("book", body)}))
'''


@pytest.fixture
def post_book(besser, tmp_path):
    """Check model.py with 'validate --ocl', generate the backend from it, run its API on a new SQLite
    database, POST a book (dict) and return the status and the book the API answers."""
    count = iter(range(100))

    def run(book: dict) -> tuple[int, dict]:
        validated = besser("validate", "model.py", "--ocl")
        assert validated.returncode == 0, validated.stdout + validated.stderr
        out = tmp_path / f"api{next(count)}"
        generated = besser("generate", "backend", "model.py", "-o", out.name)
        assert generated.returncode == 0, generated.stdout + generated.stderr
        (out / "post_book.py").write_text(_POST_BOOK, encoding="utf-8")
        env = {**os.environ, "DATABASE_URL": f"sqlite:///{(out / 'test.db').as_posix()}"}
        process = subprocess.run([sys.executable, "post_book.py", json.dumps(book)], cwd=out, env=env,
                                 capture_output=True, text=True, timeout=120, check=False)
        assert process.returncode == 0, process.stderr[-3000:]
        answer = json.loads(process.stdout.strip().splitlines()[-1])
        return answer["status"], answer["book"]
    return run


BOOK = {"title": "Dune", "pages": 412, "release": "1965-08-01"}


# ── set-attribute ───────────────────────────────────────────────────────────


def test_set_attribute_changes_the_line_of_the_attribute_only(project, change):
    before = project()

    line = change("set-attribute", "Book.pages", ":int?=0")

    assert line == "model.py: changed Book.pages from int to int?=0"
    assert _changed_lines(before, project()) == (
        ['    Property(name="pages", type=IntegerType),'],
        ['    Property(name="pages", type=IntegerType, is_optional=True, default_value=0),'])
    assert change("set-attribute", "Book.pages", "=") == "model.py: changed Book.pages from int?=0 to int?"
    # The add-attribute notation with the attribute's own name is taken too.
    assert change("set-attribute", "Book.pages", "pages:int") == "model.py: changed Book.pages from int? to int"
    assert project() == before


def test_the_backend_applies_a_default_and_an_optional_attribute(project, change, post_book):
    change("batch", stdin="set-attribute Book.pages :int=7\nset-attribute Book.release :date?\n")

    status, book = post_book({"title": "Dune"})

    assert status == 200, book
    assert book["pages"] == 7 and book["release"] is None


def test_the_backend_enforces_a_new_type(project, change, post_book):
    assert post_book({**BOOK, "pages": "many"})[0] == 422
    change("set-attribute", "Book.pages", ":str")

    status, book = post_book({**BOOK, "pages": "many"})

    assert status == 200 and book["pages"] == "many"
    change("set-attribute", "Book.title", ":int")
    assert post_book(BOOK)[0] == 422


# Every way a hand-written file writes an attribute; a constraint navigates to one, an object model holds them,
# a method's code reads one, and a __main__ block uses an import.
HAND_WRITTEN = '''\
"""A shop, written by hand."""
import datetime

from besser.BUML.metamodel.object import ObjectModel
from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, Constraint, DomainModel, FloatType, IntegerType, Method, Multiplicity, Property,
    StringType,
)

# pages: a constructor over several lines
book_pages = Property(
    name="pages",
    type=IntegerType,  # how long it is
)
book_title = Property("title", StringType)  # positional
book_price = Property(name="price", type=FloatType)
book_price.is_optional = True  # set after the constructor
book = Class(name="Book", attributes={book_title, book_pages, book_price}, methods={
    Method(name="thick", code="def thick(self):\\n    return self.pages > 300\\n"),
})
shelf = Class(name="Shelf", attributes={Property(name="label", type=StringType)})
shelf_books = BinaryAssociation(
    name="shelf_books",
    ends={
        Property(name="shelf", type=shelf, multiplicity=Multiplicity(1, 1)),
        Property(name="books", type=book, multiplicity=Multiplicity(0, "*")),  # the books on it
    },
)
long_enough = Constraint(name="long_enough", context=book, language="OCL",
                         expression="context Book inv long_enough: self.pages > 10")
shop = DomainModel(name="Shop", types={book, shelf}, associations={shelf_books}, constraints={long_enough})

# Objects
dune = book("dune").attributes(title="Dune", pages=412, price=9.5).build()
top = shelf("top").attributes(label="Top").build()
top.books = {dune}
shop_objects = ObjectModel(name="ShopObjects", objects={dune, top})

if __name__ == "__main__":
    print(datetime.date.today(), shop.name)
'''


@pytest.fixture
def hand_written(write, tmp_path):
    write("model.py", HAND_WRITTEN)
    return lambda: (tmp_path / "model.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("path, change_, summary, removed, added", [
    # The constraint holds and stays as written; the comment stays on its line.
    ("Book.pages", ":int?=20", "changed Book.pages from int to int?=20",
     [], ["    is_optional=True,", "    default_value=20,"]),
    ("Book.title", ":str?", "changed Book.title from str to str?",
     ['book_title = Property("title", StringType)  # positional'],
     ['book_title = Property("title", StringType, is_optional=True)  # positional']),
    ("Book.price", ":float", "changed Book.price from float? to float",
     ["book_price.is_optional = True  # set after the constructor"],
     ["book_price.is_optional = False  # set after the constructor"]),
])
def test_set_attribute_edits_a_hand_written_file_where_it_writes_the_attribute(
        besser, hand_written, change, path, change_, summary, removed, added):
    before = hand_written()

    assert change("set-attribute", path, change_) == f"model.py: {summary}"

    assert _changed_lines(before, hand_written()) == (removed, added)
    checked = besser("validate", "model.py", "--ocl")
    assert checked.returncode == 0, checked.stdout + checked.stderr


@pytest.mark.parametrize("args, message", [
    (("Book.pages", ":str"), "constraint long_enough uses Book.pages as int and would not hold for str: change it "
                             "first, or remove it (besser model remove long_enough)"),
    (("Book.price", ":int"), "object dune (price=9.5) holds no int: change the value first"),
    # The object model takes no int for a float either.
    (("Book.pages", ":float"), "object dune (pages=412) holds no float: change the value first"),
    (("Book.pages", ":Shelf"), "'Shelf' is a class: an attribute cannot hold a class, an association links them: "
                               "besser model remove Book.pages, then besser model add-association book_pages "
                               "book:Book[*] pages:Shelf[1]"),
    (("Shelf.books", ":int"), "Shelf.books is an association end: besser model set-multiplicity Shelf.books [*]"),
    (("Book.pages", "count:int"), "'count:int' names another attribute: write :type, :type? or =default, the name "
                                  "stays (besser model rename Book.pages count renames it)"),
    (("Book.pages", "int"), "'int' changes nothing: write the parts to change as add-attribute writes them, "
                            "without the name: :type, :type? (optional), =default, = (no default); e.g. besser "
                            "model set-attribute Book.pages :int?=0"),
    (("Book.pages", ":int"), "Book.pages is int already"),
    (("Book.pages", "="), "Book.pages has no default to drop"),
    (("Book.pages", "=x"), "default 'x' is not an int"),
])
def test_set_attribute_refuses_what_it_would_break(hand_written, refused, args, message):
    assert message in refused("set-attribute", *args)


def test_a_kept_default_that_does_not_fit_the_new_type_is_refused_with_both_commands(project, change, refused):
    change("set-attribute", "Book.pages", "=12")
    assert change("set-attribute", "Book.pages", ":str") == "model.py: changed Book.pages from int=12 to str=12"
    change("set-attribute", "Book.pages", "=many")

    assert ("the default many of Book.pages does not fit int (default 'many' is not an int (a whole number, e.g. 0)): "
            "give one (besser model set-attribute Book.pages :int=VALUE) or drop it "
            "(besser model set-attribute Book.pages :int=)") in refused("set-attribute", "Book.pages", ":int")
    assert 'Property(name="pages", type=StringType, default_value="many")' in project()


def test_an_enumeration_type_and_its_literal_as_default(project, change, besser):
    change("set-enum", "Format", "PAPER", "EBOOK")

    assert change("set-attribute", "Book.title", ":Format=EBOOK") == ("model.py: changed Book.title from str to "
                                                                     "Format=EBOOK")
    assert 'Property(name="title", type=format, default_value="EBOOK")' in project()
    assert besser("validate", "model.py").returncode == 0


def test_an_attribute_defined_on_its_own_takes_a_new_enumeration_as_type(besser, change, tmp_path):
    """The GUI starter defines each attribute before its class: a new enumeration goes before them."""
    assert besser("init", "--kind", "gui").returncode == 0
    change("set-enum", "Genre", "NOVEL", "POEM")

    assert change("set-attribute", "Book.title", ":Genre=POEM") == "model.py: changed Book.title from str to Genre=POEM"
    text = (tmp_path / "model.py").read_text(encoding="utf-8")
    assert text.index("genre = Enumeration(") < text.index('book_title = Property(name="title", type=genre')
    assert besser("validate", "model.py").returncode == 0


def test_set_attribute_keeps_windows_line_endings(project, change, tmp_path):
    path = tmp_path / "model.py"
    path.write_bytes(project().replace("\n", "\r\n").encode())

    change("set-attribute", "Book.pages", ":float?")

    data = path.read_bytes()
    assert b'    Property(name="pages", type=FloatType, is_optional=True),\r\n' in data
    assert b"\n" not in data.replace(b"\r\n", b"")


# ── set-multiplicity ────────────────────────────────────────────────────────


def test_set_multiplicity_changes_the_bounds_only(project, change):
    before = project()

    by_association = change("set-multiplicity", "book_authors.authors", "[*]")
    by_class = change("set-multiplicity", "Book.library", "0..1")

    assert by_association == "model.py: changed Book.authors (book_authors) from [1..*] to [*]"
    assert by_class == "model.py: changed Book.library (library_books) from [1] to [0..1]"
    assert _changed_lines(before, project()) == (
        ['    Property(name="library", type=library, multiplicity=Multiplicity(1, 1)),',
         '    Property(name="authors", type=author, multiplicity=Multiplicity(1, "*")),'],
        ['    Property(name="library", type=library, multiplicity=Multiplicity(0, 1)),',
         '    Property(name="authors", type=author, multiplicity=Multiplicity(0, "*")),'])
    # The association keeps its place in the file: the lines around it are as they were.
    assert project().index("library_books = BinaryAssociation(") == before.index("library_books = BinaryAssociation(")


def test_the_backend_follows_a_required_or_optional_end(project, change, post_book):
    assert post_book({**BOOK, "library": None})[0] == 422  # [1]: a book needs its library

    change("batch", stdin="set-multiplicity Book.library [0..1]\nset-multiplicity book_authors.authors [*]\n")

    status, book = post_book({**BOOK, "library": None, "authors": []})
    assert status == 200, book
    assert book["library_id"] is None
    change("set-multiplicity", "Book.library", "[1]")
    assert post_book({**BOOK, "library": None})[0] == 422


def test_set_multiplicity_in_a_hand_written_file(hand_written, change, besser):
    before = hand_written()

    assert change("set-multiplicity", "Shelf.books", "[1..*]") == ("model.py: changed Shelf.books (shelf_books) "
                                                                  "from [*] to [1..*]")

    assert _changed_lines(before, hand_written()) == (
        ['        Property(name="books", type=book, multiplicity=Multiplicity(0, "*")),  # the books on it'],
        ['        Property(name="books", type=book, multiplicity=Multiplicity(1, "*")),  # the books on it'])
    assert besser("validate", "model.py", "--ocl").returncode == 0


@pytest.mark.parametrize("args, message", [
    # The object model links the shelf to one book: it would have too few.
    (("Shelf.books", "[2..*]"), "the object model would be invalid once Shelf.books is [2..*]: Object 'top' "
                                "violates multiplicity 2..* for association end 'books'"),
    (("Shelf.books", "[*]"), "Shelf.books is [*] already"),
    (("Book.pages", "[1]"), "'Book.pages' is not an association end: name one as Class.end or association.end"),
    (("Shelf.books", "[2..1]"), "'[2..1]': not a multiplicity"),
    (("Shelf.books", "many"), "'many' is not a multiplicity: write [1], [0..1], [*], [1..*] or [n..m]"),
])
def test_set_multiplicity_refuses_what_it_would_break(hand_written, refused, args, message):
    assert message in refused("set-multiplicity", *args)


def test_an_end_without_a_multiplicity_gets_one(write, change, tmp_path):
    write("model.py", '''\
        from besser.BUML.metamodel.structural import BinaryAssociation, Class, DomainModel, Property, StringType

        a = Class(name="A", attributes={Property(name="x", type=StringType)})
        b = Class(name="B", attributes={Property(name="y", type=StringType)})
        ab = BinaryAssociation(name="ab", ends={Property(name="a", type=a), Property(name="bs", type=b)})
        m = DomainModel(name="M", types={a, b}, associations={ab})
        ''')

    assert change("set-multiplicity", "ab.bs", "[*]") == "model.py: changed A.bs (ab) from [1] to [*]"
    text = (tmp_path / "model.py").read_text()
    # The line would grow past the limit: the new argument goes on a line of its own.
    assert 'Property(name="bs", type=b,\n' in text and ' multiplicity=Multiplicity(0, "*"))})\n' in text


# ── set-abstract, remove-parent ─────────────────────────────────────────────


def test_set_abstract_and_back(project, change, besser, post_book):
    before = project()

    run = besser("model", "batch", "-", stdin="add-class Work\nset-abstract Work true\n", output=None)

    assert run.returncode == 0, run.stderr
    assert run.stdout == "model.py: added class Work; made Work abstract\n"
    # No class specializes it: what validate warns of is said.
    assert ("warning: no concrete class specializes Work, so no Work object can exist: add one "
            "(besser model add-class NAME --parent Work)") in run.stderr
    assert 'work = Class(name="Work", is_abstract=True)' in project()
    change("add-parent", "Book", "Work")
    post_book(BOOK)
    assert change("set-abstract", "Work", "false") == "model.py: made Work concrete"
    assert 'work = Class(name="Work")' in project()
    change("remove-parent", "Book", "Work")
    change("remove", "Work")
    # What add-parent left: the empty set of generalizations (its import went with the last one).
    assert _changed_lines(before, project()) == ([], ["    generalizations=set(),"])


@pytest.mark.parametrize("args, message", [
    (("Book", "true"), "object dune (shop_objects) is a Book: an abstract class has no objects; make each an object "
                       "of a subclass first"),
    (("Book", "false"), "Book is concrete already"),
])
def test_set_abstract_refusals(hand_written, refused, args, message):
    assert message in refused("set-abstract", *args)


# A hand-written file with a parent: Book specializes Work by a variable, Author specializes Person in the set.
WITH_PARENTS = '''\
from besser.BUML.metamodel.structural import (
    Class, Constraint, DomainModel, Generalization, Property, StringType,
)

work = Class(name="Work", attributes={Property(name="code", type=StringType)})
book = Class(name="Book", attributes={Property(name="title", type=StringType)})
person = Class(name="Person", attributes={Property(name="name", type=StringType)})
author = Class(name="Author", attributes={Property(name="email", type=StringType)})
book_is_work = Generalization(general=work, specific=book)  # a book is a work
coded = Constraint(name="coded", context=book, language="OCL", expression="context Book inv coded: self.code <> ''")
model = DomainModel(
    name="Library",
    types={work, book, person, author},
    generalizations={book_is_work, Generalization(general=person, specific=author)},
    constraints={coded},
)
'''


def test_remove_parent_keeps_the_class(write, change, refused, besser, tmp_path):
    write("model.py", WITH_PARENTS)

    assert change("remove-parent", "Author", "Person") == "model.py: Author is no longer a Person"
    after = (tmp_path / "model.py").read_text()
    assert _changed_lines(WITH_PARENTS, after) == (
        ["    generalizations={book_is_work, Generalization(general=person, specific=author)},"],
        ["    generalizations={book_is_work},"])
    assert ("constraint coded would no longer hold against the model once Book is no Work: change it first, or "
            "remove it (besser model remove coded)") in refused("remove-parent", "Book", "Work")
    assert "Author is no direct Person (its parents: none)" in refused("remove-parent", "Author", "Person")
    change("remove", "coded")
    assert change("remove-parent", "Book", "Work") == "model.py: Book is no longer a Work"
    text = (tmp_path / "model.py").read_text()
    assert "book_is_work" not in text and "    generalizations=set(),\n" in text
    assert besser("info", "model.py", "--class", "Book", "--json").json["classes"][0]["parents"] == []


def test_remove_parent_refuses_an_object_that_holds_what_was_inherited(write, refused):
    write("model.py", WITH_PARENTS + 'from besser.BUML.metamodel.object import ObjectModel\n'
                                     'emma = book("emma").attributes(title="Emma", code="E1").build()\n'
                                     'objects = ObjectModel(name="Objects", objects={emma})\n')

    assert ("object emma (code) holds a value of an attribute Book has from Work: change the objects first"
            in refused("remove-parent", "Book", "Work"))


# ── an enumeration literal in OCL text ──────────────────────────────────────

STATUS = """\
set-enum Status Open Done
add-attribute Book status:Status=Open
add-attribute Author status:str?
add-constraint Book "self.status <> 'Done'" --name not_done
add-constraint Author "self.status <> 'Done'" --name author_text
add-constraint Library "self.books->forAll(b | b.status <> 'Done')" --name all_open
"""


def test_rename_a_literal_renames_its_quoted_name_where_it_is_compared(project, change, besser):
    change("batch", stdin=STATUS)
    before = project()

    run = besser("model", "rename", "Status.Done", "Finished", output=None)

    # Before: the constraint kept 'Done', which no status is any longer: it held for every book.
    assert run.returncode == 0, run.stderr
    assert run.stdout == "model.py: renamed Status.Done to Status.Finished; constraint not_done: 'Done' -> 'Finished'\n"
    removed, added = _changed_lines(before, project())
    assert [line for line in added if "not_done" in line] == [line.replace("<> 'Done'", "<> 'Finished'")
                                                             for line in removed if "not_done" in line]
    assert "self.status <> 'Done'\", language=\"OCL\")" in project()  # Author's status is text
    # b.status: Book's or Author's status, which the text does not tell: the file keeps it, a warning names it.
    assert ("warning: constraint all_open holds 'Done' where it may not mean Status.Done: check it says 'Finished' "
            "where it should") in run.stderr
    assert besser("validate", "model.py", "--ocl").returncode == 0


@pytest.mark.parametrize("args", [("remove", "Status.Done"), ("set-enum", "Status", "Open")])
def test_a_literal_a_constraint_compares_to_is_not_removed(project, change, refused, args):
    change("batch", stdin=STATUS)

    assert ("constraint not_done compares to Status.Done: change it first, or remove it "
            "(besser model remove not_done)") in refused(*args)


# The GUI starter, with Book a Work whose code its table shows.
def _gui_with_parent(text: str) -> str:
    text = text.replace('book_title = Property(name="title", type=StringType)\n',
                        'work_code = Property(name="code", type=StringType)\n'
                        'work = Class(name="Work", attributes={work_code})\n'
                        'book_title = Property(name="title", type=StringType)\n')
    text = text.replace('types={book, author}, associations={book_authors})',
                        'types={work, book, author}, associations={book_authors},\n'
                        '                             generalizations={Generalization(general=work, specific=book)})')
    text = text.replace('    FieldColumn(label="Pages", field=book_pages),\n',
                        '    FieldColumn(label="Pages", field=book_pages),\n'
                        '    FieldColumn(label="Code", field=work_code),\n')
    return text.replace("    Class,\n", "    Class,\n    Generalization,\n")


def test_the_gui_tables_follow_each_in_place_change(besser, change, tmp_path):
    assert besser("init", "--kind", "gui").returncode == 0
    path = tmp_path / "model.py"
    path.write_text(_gui_with_parent(path.read_text()), encoding="utf-8")
    assert besser("validate", "model.py").returncode == 0

    line = change("batch", stdin=(
        "set-attribute Book.pages :float?=0\n"
        "set-multiplicity book_authors.authors [*]\n"
        "set-abstract Work true\n"
        "remove-parent Book Work\n"))

    assert line == ("model.py: changed Book.pages from int to float?=0; changed Book.authors (book_authors) from "
                    "[1..*] to [*]; made Work abstract; Book is no longer a Work; its columns of Work from table "
                    "book_table")
    text = path.read_text()
    # The column of the retyped attribute stays; the one of what Book no longer inherits goes.
    assert '    FieldColumn(label="Pages", field=book_pages),\n])' in text and "field=work_code" not in text
    assert besser("validate", "model.py", "--ocl").returncode == 0
    generated = besser("generate", "web_app", "model.py", "-o", "out")
    assert generated.returncode == 0, generated.stdout + generated.stderr
    page = (tmp_path / "out" / "frontend" / "src" / "pages" / "Books.tsx").read_text(encoding="utf-8")
    assert '"label": "Pages"' in page and '"label": "Code"' not in page


def test_a_batch_mixes_in_place_changes_and_renames(project, change, besser):
    line = change("batch", stdin=(
        "set-attribute Book.pages :int?=1\n"
        "rename Book.pages length\n"
        "set-multiplicity Library.books [1..*]\n"
        "rename library_books.books volumes\n"
        "set-attribute Book.length =\n"))

    assert line == ("model.py: changed Book.pages from int to int?=1; renamed Book.pages to Book.length; "
                    "changed Library.books (library_books) from [*] to [1..*]; renamed library_books.books to "
                    "library_books.volumes; changed Book.length from int?=1 to int?")
    info = besser("info", "model.py").stdout
    assert "length:int?" in info and "volumes:Book[1..*]" in info
    assert besser("validate", "model.py", "--ocl").returncode == 0


# ── the imports a change adds or leaves unused ──────────────────────────────


def _imports(text: str) -> str:
    return text[text.index("from besser.BUML.metamodel.structural import"):text.index("\n)\n") + 3]


def test_a_new_name_goes_in_order_into_a_sorted_import(project, change):
    before = project()

    change("add-attribute", "Book", "due:datetime", "opened:time")

    # Before: DateTimeType and TimeType were appended after StringType.
    assert _imports(project()) == (
        "from besser.BUML.metamodel.structural import (\n    BinaryAssociation,\n    Class,\n    DateTimeType,\n"
        "    DateType,\n    DomainModel,\n    IntegerType,\n    Multiplicity,\n    Property,\n    StringType,\n"
        "    TimeType,\n)\n")
    assert _changed_lines(before, project())[0] == []


ONE_LINE_IMPORT = '''\
from besser.BUML.metamodel.structural import BinaryAssociation, Class, DomainModel, Property, StringType

a = Class(name="A", attributes={Property(name="x", type=StringType)})
b = Class(name="B", attributes={Property(name="y", type=StringType)})
ab = BinaryAssociation(name="ab", ends={Property(name="a", type=a), Property(name="bs", type=b)})
m = DomainModel(name="M", types={a, b}, associations={ab})
'''


def test_an_import_on_one_line_stays_in_order_and_is_wrapped_past_the_line_limit(write, change, tmp_path):
    write("model.py", ONE_LINE_IMPORT)

    change("set-multiplicity", "ab.bs", "[*]")
    one_line = (tmp_path / "model.py").read_text()
    change("add-attribute", "A", "n:int")

    assert one_line.startswith("from besser.BUML.metamodel.structural import BinaryAssociation, Class, DomainModel, "
                               "Multiplicity, Property, StringType\n\n")
    # Before: 'IntegerType' was appended, a line of 130 characters.
    assert (tmp_path / "model.py").read_text().startswith(
        "from besser.BUML.metamodel.structural import (\n    BinaryAssociation,\n    Class,\n    DomainModel,\n"
        "    IntegerType,\n    Multiplicity,\n    Property,\n    StringType,\n)\n\n")


def test_an_import_a_change_leaves_unused_goes(project, change):
    before = project()

    change("set-attribute", "Book.pages", ":float")
    change("set-attribute", "Book.pages", ":int")
    assert project() == before  # FloatType came and went
    change("remove", "Book.release")

    # Before: DateType stayed imported, used by nothing.
    assert _changed_lines(before, project()) == (["    DateType,", '    Property(name="release", type=DateType),'], [])


def test_an_import_used_elsewhere_stays(write, change, tmp_path):
    # IntegerType: in a __main__ block; DateType: by another model of the file; Method: by no one before.
    write("model.py", '''\
        from besser.BUML.metamodel.structural import Class, DateType, DomainModel, IntegerType, Method, Property

        book = Class(name="Book", attributes={Property(name="pages", type=IntegerType),
                                              Property(name="due", type=DateType)})
        shelf = Class(name="Shelf", attributes={Property(name="since", type=DateType)})
        library = DomainModel(name="Library", types={book})
        storage = DomainModel(name="Storage", types={shelf})

        if __name__ == "__main__":
            print(IntegerType)
        ''')
    before = (tmp_path / "model.py").read_text()

    change("--use", "library", "batch", stdin="remove Book.pages\nremove Book.due\n")

    assert _changed_lines(before, (tmp_path / "model.py").read_text()) == (
        ['book = Class(name="Book", attributes={Property(name="pages", type=IntegerType),',
         '                                      Property(name="due", type=DateType)})'],
        ['book = Class(name="Book", attributes=set())'])


def test_a_parent_class_the_cli_wrote_is_removed_with_its_generalizations(project, change, besser):
    # The generalizations are written as add-class --parent and add-parent write them: inline, unnamed.
    change("batch", stdin="add-class Person nickname:str\nadd-class Student --parent Person\nadd-parent Book Person\n")
    assert "generalizations={Generalization(general=person, specific=student)" in project()

    # Before: refused, "cannot find where 'Generalization(...)' is written ...; edit it by hand".
    assert change("remove", "Person") == "model.py: removed Person"

    text = project()
    assert "person" not in text and "Generalization" not in text and "    generalizations=set(),\n" in text
    assert besser("info", "model.py", "--class", "Book", "--json").json["classes"][0]["parents"] == []


def test_remove_parent_drops_the_inherited_columns_add_screen_and_add_parent_wrote(besser, change, tmp_path):
    assert besser("init", "--kind", "gui").returncode == 0
    change("batch", stdin="add-class Work code:str\nadd-parent Book Work\nadd-class Novel genre:str --parent Book "
                          "--screen\n")
    assert (tmp_path / "model.py").read_text().count("field=work_code") == 2  # Books and Novels show the code

    line = change("remove-parent", "Book", "Work")

    assert line == "model.py: Book is no longer a Work; its columns of Work from table book_table, novel_table"
    assert "field=work_code" not in (tmp_path / "model.py").read_text()
    assert besser("validate", "model.py", "--ocl").returncode == 0
    assert besser("generate", "web_app", "model.py", "-o", "out").returncode == 0
