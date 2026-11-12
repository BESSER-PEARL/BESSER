"""``besser model`` on a file that holds a GUI model too: ``add-screen``, and tables that follow their class.

Each change is checked where it shows: the file loads and validates, and the React pages the ``web_app``
generator writes from it hold the new page or column, and no longer the removed one.
"""

import re

import pytest

# A GUI model written by hand, unlike the starter: the domain model after the GUI model, screens written
# inside two modules (no variable), two tables bound to one class, one of them with a title of its own.
HAND_WRITTEN = '''\
from besser.BUML.metamodel.gui import DataBinding, FieldColumn, GUIModel, Module, Screen, Table
from besser.BUML.metamodel.structural import Class, DomainModel, IntegerType, Property, StringType

customer_name = Property(name="name", type=StringType)
customer_age = Property(name="age", type=IntegerType)
customer = Class(name="Customer", attributes={customer_name, customer_age})

recent = Table(name="recent", title="Recent customers", data_binding=DataBinding(domain_concept=customer),
               columns=[FieldColumn(label="Who", field=customer_name)])
everyone = Table(name="everyone", title="Customers", action_buttons=True, data_binding=DataBinding(
    domain_concept=customer), columns=[
    FieldColumn(label="Name", field=customer_name),
    FieldColumn(label="Age", field=customer_age),
])

admin = Module(name="admin", screens={Screen(name="Customers", description="Customers", view_elements={everyone})})
shop_gui = GUIModel(name="ShopApp", package="shop", versionCode="1", versionName="1.0", description="A shop", modules={
    Module(name="home", screens={
        Screen(name="Home", description="Home", is_main_page=True, view_elements={recent}),
    }),
    admin,
})
shop = DomainModel(name="Shop", types={customer})
'''

DOMAIN_ONLY = '''\
from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType

book = Class(name="Book", attributes={Property(name="title", type=StringType)})
library = DomainModel(name="Library", types={book})
'''


@pytest.fixture
def starter(besser, tmp_path):
    """A project made with 'besser init --kind gui': model.py holds the Library and its two screens."""
    assert besser("init", "--kind", "gui").returncode == 0
    return tmp_path / "model.py"


@pytest.fixture
def change(besser):
    """Run ``besser model ARGS`` and check it succeeded with one line; returns that line."""
    def run(*args, stdin=""):
        result = besser("model", *args, stdin=stdin, output=None)
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(result.stdout.splitlines()) == 1, result.stdout
        return result.stdout.strip()
    return run


@pytest.fixture
def app(besser, tmp_path):
    """Validate model.py, generate the web_app from it into a new folder, and return its pages (name -> text)
    and its App.tsx."""
    count = iter(range(100))

    def run():
        assert besser("validate", "model.py").returncode == 0
        out = f"out{next(count)}"
        generated = besser("generate", "web_app", "model.py", "-o", out)
        assert generated.returncode == 0, generated.stdout + generated.stderr
        pages = {path.stem: path.read_text(encoding="utf-8")
                 for path in (tmp_path / out / "frontend" / "src" / "pages").glob("*.tsx")}
        return pages, (tmp_path / out / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8"), tmp_path / out
    return run


def _columns(page: str) -> list[str]:
    """The labels of the table columns of a page (not those of its create/edit form)."""
    return re.findall(r'"label": "([^"]*)",\s*"column_type"', page)


def test_add_class_with_screen_makes_a_page_the_main_page_opens(starter, change, app):
    line = change("add-class", "Publisher", "name:str", "country:str", "--screen")
    assert line == ("model.py: added class Publisher with 2 attributes; added screen Publishers "
                    "(table of Publisher: name, country; button on Authors)")
    pages, routes, _ = app()
    assert _columns(pages["Publishers"]) == ["Name", "Country"]
    assert '"actionButtons": true' in pages["Publishers"]  # add, edit and delete
    assert '<Route path="/Publishers"' in routes
    assert 'navigate("/Publishers")' in pages["Authors"]
    # The starter's style: a variable per attribute, which the columns refer to.
    assert 'publisher = Class(name="Publisher", attributes={publisher_name, publisher_country})' in starter.read_text()


def test_add_attribute_adds_its_column(starter, change, app):
    line = change("add-attribute", "Author", "bio:str?", "active:bool=true")
    assert line == "model.py: added Author.bio:str?, active:bool=true; columns added to table author_table"
    pages, _, _ = app()
    assert _columns(pages["Authors"]) == ["Name", "Email", "Bio", "Active"]


def test_an_enumeration_attribute_is_a_column_too(starter, change, app):
    line = change("batch", stdin="set-enum Genre NOVEL POETRY ESSAY\nadd-attribute Book genre:Genre?\n")
    assert line.endswith("added Book.genre:Genre?; column added to table book_table")
    pages, _, _ = app()
    assert _columns(pages["Books"]) == ["Title", "Pages", "Genre"]


def test_rename_a_class_renames_the_page_titled_after_it(starter, change, app):
    line = change("rename", "Author", "Writer")
    assert "GUI titles Authors -> Writers" in line
    pages, routes, _ = app()
    assert "Authors" not in pages and _columns(pages["Writers"]) == ["Name", "Email"]
    assert '<Route path="/Writers"' in routes and '"Authors"' not in pages["Writers"]


def test_remove_an_attribute_removes_its_column(starter, change, app):
    # Before: refused, "Author.email is still used on line 52" (the column referred to it).
    assert change("remove", "Author.email") == "model.py: removed Author.email; its column from table author_table"
    pages, _, out = app()
    assert _columns(pages["Authors"]) == ["Name"]
    assert "email" not in pages["Authors"] and "email" not in (out / "backend" / "pydantic_classes.py").read_text()


def test_rename_an_attribute_relabels_a_column_labelled_after_it(starter, change, app):
    assert change("rename", "Author.email", "mail").endswith("; column Email -> Mail in table author_table")
    pages, _, _ = app()
    assert _columns(pages["Authors"]) == ["Name", "Mail"]


def test_a_loan_of_one_book_and_one_member_with_its_page(starter, change, app):
    line = change("batch", stdin=(
        "add-class Member name:str --screen\n"
        "add-class Loan due:date --screen\n"
        "add-association loan_book loans:Loan[*] book:Book[1]\n"
        "add-association loan_member loans:Loan[*] member:Member[1]\n"))
    assert "added screen Loans" in line and "added screen Members" in line
    pages, routes, _ = app()
    assert _columns(pages["Loans"]) == ["Due"]
    assert '"entity": "Book"' in pages["Loans"] and '"entity": "Member"' in pages["Loans"]  # the form picks them
    assert 'navigate("/Loans")' in pages["Authors"] and '<Route path="/Members"' in routes


def test_batch_adds_a_class_its_association_and_its_screen(starter, change, app):
    line = change("batch", stdin=(
        "add-class Review stars:int\n"
        "add-association book_reviews reviews:Review[*] book:Book[1]\n"
        "add-screen Review --title 'Book reviews'\n"))
    assert line.endswith("added screen BookReviews (table of Review: stars; button on Authors)")
    pages, _, _ = app()
    assert _columns(pages["BookReviews"]) == ["Stars"]


def test_remove_a_class_removes_its_screen_and_the_buttons_to_it(starter, change, app):
    line = change("remove", "Book")
    assert line == ("model.py: removed Book and association book_authors; and table book_table, screen Books, "
                    "button to_books")
    pages, routes, _ = app()
    assert set(pages) == {"Authors"} and "/Books" not in routes + pages["Authors"]


@pytest.mark.parametrize("class_name, title", [
    ("Person", "People"), ("Shelf", "Shelves"), ("News", "News"), ("Categories", "Categories"),
    ("Child", "Children"), ("SalesPerson", "SalesPeople"), ("Knife", "Knives"), ("Category", "Categories"),
    ("Box", "Boxes"), ("Status", "Statuses"),
])
def test_a_page_is_titled_with_the_plural_of_its_class(starter, change, class_name, title):
    # Before: Persons, Shelfs, Newses, Categorieses.
    line = change("add-class", class_name, "code:str", "--screen")
    assert f"added screen {title} (table of {class_name}: code;" in line
    assert f'description="Go to the {title.lower()}", label="{title}"' in starter.read_text()


def test_a_class_named_as_its_own_plural_keeps_its_page_titled_after_it(starter, change):
    change("add-class", "News", "headline:str", "--screen")
    assert "GUI titles News -> Articles" in change("rename", "News", "Article")


def test_a_class_whose_variable_is_title_can_be_renamed_and_removed(starter, change, tmp_path):
    # Before: the rename turned every title= of the tables into label=, and remove was refused: "'Title' is
    # still used on lines 54, 55, ..." (each title= counted as a use of the variable title).
    change("add-class", "Tag", "code:str")
    assert change("rename", "Tag", "Title").endswith("(variable tag -> title)")
    assert change("rename", "Title", "Label").endswith("(variable title -> label)")
    assert 'title="Authors"' in starter.read_text()
    assert change("remove", "Label") == "model.py: removed Label"


def test_add_screen_is_refused_when_it_cannot_be_made(besser, write, starter):
    def refused(*args):
        result = besser("model", *args)
        assert result.returncode == 2
        return result.stderr

    assert "screen Authors shows Author already" in refused("add-screen", "Author")
    assert besser("model", "add-class", "Tag").returncode == 0
    assert "Tag has no attribute for its table to show: add one first" in refused("add-screen", "Tag")
    write("model.py", DOMAIN_ONLY)
    assert "model.py has no GUI model to add a screen to" in refused("add-screen", "Book")
    assert "has no GUI model" in refused("add-class", "Shelf", "name:str", "--screen")


def test_a_hand_written_gui_model_follows_its_class(write, change, app, tmp_path):
    write("model.py", HAND_WRITTEN)
    line = change("add-attribute", "Customer", "email:str?")
    assert line == "model.py: added Customer.email:str?; column added to table everyone, recent"
    pages, _, _ = app()
    assert _columns(pages["Customers"]) == ["Name", "Age", "Email"]
    assert _columns(pages["Home"]) == ["Who", "Email"]

    assert change("remove", "Customer.age").endswith("; its column from table everyone")
    line = change("rename", "Customer", "Client")
    assert "GUI titles Customers -> Clients" in line
    text = (tmp_path / "model.py").read_text()
    assert 'title="Recent customers"' in text and 'Screen(name="Clients", description="Clients"' in text
    pages, _, _ = app()
    assert _columns(pages["Clients"]) == ["Name", "Email"] and "Customers" not in pages


def test_a_column_relabelled_in_several_tables_is_named_once(write, change):
    # Before: "...; column Name -> Full name; column Name -> Full name", once per table.
    write("model.py", HAND_WRITTEN.replace('label="Who"', 'label="Name"'))
    assert change("rename", "Customer.name", "full_name") == (
        "model.py: renamed Customer.name to Customer.full_name; column Name -> Full name in table everyone, recent")


def test_a_column_appended_or_added_to_a_list_is_relabelled_too(write, besser, change, app, tmp_path):
    # Before: "renamed Customer.name to Customer.full_name" alone: no column relabelled, and nothing said.
    write("model.py", HAND_WRITTEN.replace(
        'columns=[FieldColumn(label="Who", field=customer_name)])',
        'columns=[FieldColumn(label="Name", field=customer_name)] + [FieldColumn(label="Age", field=customer_age)])',
    ) + 'everyone.columns.append(FieldColumn("Name", customer_name))\n')
    assert change("rename", "Customer.name", "full_name") == (
        "model.py: renamed Customer.name to Customer.full_name; column Name -> Full name in table everyone, recent")
    text = (tmp_path / "model.py").read_text()
    assert 'columns=[FieldColumn(label="Full name", field=customer_name)] + [' in text
    assert 'everyone.columns.append(FieldColumn("Full name", customer_name))' in text
    pages, _, _ = app()
    assert _columns(pages["Home"]) == ["Full name", "Age"]
    assert _columns(pages["Customers"]) == ["Full name", "Age", "Full name"]
    # A column the file builds with an expression keeps its label, and the warning says so.
    with (tmp_path / "model.py").open("a", encoding="utf-8") as file:
        file.write('recent.columns.append(FieldColumn(label="Age", field=globals()["customer_age"]))\n')
    renamed = besser("model", "rename", "Customer.age", "years")
    assert renamed.returncode == 0
    assert renamed.stdout.endswith("; column Age -> Years in table everyone, recent\n")
    assert ("warning: the column Age of table recent keeps its label: it is not written as "
            "FieldColumn(label=..., field=<variable>); relabel it by hand") in renamed.stderr


def test_a_screen_goes_in_the_first_module_and_the_main_page_inline_gets_its_button(write, change, app, tmp_path):
    write("model.py", HAND_WRITTEN)
    line = change("add-class", "Order", "number:str", "--screen")
    assert line.endswith("added screen Orders (table of Order: number; button on Home)")
    text = (tmp_path / "model.py").read_text()
    assert "admin = Module(name=\"admin\", screens={Screen(" in text and "order_screen})" in text
    pages, routes, _ = app()
    assert _columns(pages["Orders"]) == ["Number"] and 'navigate("/Orders")' in pages["Home"]


def test_a_table_whose_columns_are_not_one_list_gets_its_column_appended(write, change, app, tmp_path):
    # Before: refused, "the change would break model.py (... keyword argument repeated: columns)".
    write("model.py", HAND_WRITTEN.replace(
        'columns=[FieldColumn(label="Who", field=customer_name)])',
        'columns=[FieldColumn(label="Who", field=customer_name)] + [])'))
    line = change("add-attribute", "Customer", "email:str?")
    assert line == "model.py: added Customer.email:str?; column added to table everyone, recent"
    assert 'recent.columns.append(FieldColumn(label="Email", field=customer_email))' in (
        tmp_path / "model.py").read_text()
    pages, _, _ = app()
    assert _columns(pages["Home"]) == ["Who", "Email"]


def test_use_names_the_gui_model_a_screen_goes_in(starter, besser, change, app):
    with starter.open("a", encoding="utf-8") as file:
        file.write('admin_gui = GUIModel(name="AdminApp", package="admin", versionCode="1", versionName="1.0", '
                   'description="Admin", modules={Module(name="admin_module", screens={Screen(\n'
                   '    name="Start", description="Start", is_main_page=True, view_elements=set())})})\n')
    refused = besser("model", "add-screen", "Book", "--title", "Catalog")
    assert refused.returncode == 2 and "name one with --use (library_gui, admin_gui)" in refused.stderr
    # What the message says to do: before, "model.py has no domain model named admin_gui".
    line = change("--use", "admin_gui", "add-screen", "Book", "--title", "Catalog")
    assert line == "model.py: added screen Catalog (table of Book: title, pages; button on Start)"
    assert besser("validate", "model.py").returncode == 0


def test_a_screen_whose_route_differs_only_in_case_is_refused(besser, starter, change, app):
    # The generated app's routes ignore case: /BOOKS would open the Books page, and the new one never.
    assert change("add-class", "Shelf", "code:str").startswith("model.py: added class Shelf")
    refused = besser("model", "add-screen", "Shelf", "--title", "BOOKS")
    assert refused.returncode == 2
    assert "LibraryApp has a screen named 'Books' already: give another --title" in refused.stderr
    # Nor does a renamed class take such a name for its screen.
    assert change("add-class", "Box", "code:str", "--screen").endswith("button on Authors)")
    renamed = besser("model", "rename", "Box", "BOOK")
    assert renamed.returncode == 0 and renamed.stdout == "model.py: renamed Box to BOOK\n"
    # Before: the table was titled BOOKs and the rest kept Boxes: one page, two names.
    assert renamed.stderr == ("warning: screen Boxes keeps its titles, as do its heading, tables and the buttons "
                              "that open it: BOOKs would be the route or page file of screen Books (they ignore case)\n")
    pages, routes, _ = app()
    assert '<Route path="/Boxes"' in routes and "/BOOKS" not in routes
    assert 'title="Boxes"' in pages["Boxes"] and '{"Boxes"}' in pages["Boxes"] and "BOOKs" not in pages["Boxes"]
    assert '"Boxes"' in pages["Authors"] and "BOOKs" not in pages["Authors"]  # the button that opens it
    assert "GUI titles Authors -> AUTHORs" in change("rename", "Author", "AUTHOR")  # its own route, but for case


def test_a_screen_whose_page_file_differs_only_in_case_is_refused(besser, starter, change, app):
    # The React generator names a page file after its title in PascalCase with inner words lowercased:
    # OrderLines -> Orderlines.tsx, Order_lines -> OrderLines.tsx, one file on macOS or Windows.
    assert change("add-class", "OrderLine", "x:str", "--screen").endswith("added screen OrderLines "
                                                                         "(table of OrderLine: x; button on Authors)")
    assert change("add-class", "Item", "y:str").startswith("model.py: added class Item")
    refused = besser("model", "add-screen", "Item", "--title", "Order_lines")
    # Before: added, and the app had pages/Orderlines.tsx and pages/OrderLines.tsx.
    assert refused.returncode == 2
    assert "LibraryApp has a screen named 'OrderLines' already: give another --title" in refused.stderr
    # Nor does a renamed class's screen take such a name.
    renamed = besser("model", "rename", "Book", "Order_line")
    assert renamed.returncode == 0 and ("screen Books keeps its titles, as do its heading, tables and the buttons "
                                        "that open it: Order_lines would be the route or page file of screen "
                                        "OrderLines") in renamed.stderr
    pages, _, _ = app()
    assert sorted(pages) == ["Authors", "Books", "Orderlines"]
