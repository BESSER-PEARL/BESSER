"""``besser model`` on a file that holds a GUI model too: ``add-screen``, and tables that follow their class.

Each change is checked where it shows: the file loads and validates, and the React pages the ``web_app``
generator writes from it hold the new page or column, and no longer the removed one.
"""

import os
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
                    "at /publishers (table of Publisher: name, country; button on Authors)")
    pages, routes, _ = app()
    assert _columns(pages["Publishers"]) == ["Name", "Country"]
    assert '"actionButtons": true' in pages["Publishers"]  # add, edit and delete
    assert '<Route path="/publishers"' in routes
    assert 'navigate("/publishers")' in pages["Authors"]
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
    assert 'navigate("/loans")' in pages["Authors"] and '<Route path="/members"' in routes


def test_batch_adds_a_class_its_association_and_its_screen(starter, change, app):
    line = change("batch", stdin=(
        "add-class Review stars:int\n"
        "add-association book_reviews reviews:Review[*] book:Book[1]\n"
        "add-screen Review --title 'Book reviews'\n"))
    assert line.endswith("added screen BookReviews titled 'Book reviews' at /bookreviews "
                         "(table of Review: stars; button on Authors)")
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
    assert f"added screen {title} at /{title.lower()} (table of {class_name}: code;" in line
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


def _form_fields(page: str) -> list[str]:
    """The fields of a page's create/edit form."""
    return re.findall(r'"column_type": "field",\s*"field": "([^"]*)",\s*"label"', page)


def test_a_subclass_page_shows_and_follows_its_inherited_attributes(starter, change, app):
    change("add-class", "Person", "name:str")
    # Before: the table of Student: school, without the name it inherits.
    assert change("add-class", "Student", "school:str", "--parent", "Person", "--screen").endswith(
        "added screen Students at /students (table of Student: name, school; button on Authors)")
    # Before: refused, "Teacher has no attribute for its table to show".
    assert change("add-class", "Teacher", "--parent", "Person", "--screen").endswith(
        "added screen Teachers at /teachers (table of Teacher: name; button on Authors)")
    assert change("add-attribute", "Person", "email:str?") == (
        "model.py: added Person.email:str?; column added to table student_table, teacher_table")
    assert change("rename", "Person.email", "mail").endswith("; column Email -> Mail in table student_table, "
                                                             "teacher_table")
    change("add-class", "Worker", "salary:int", "--screen")
    assert change("add-parent", "Worker", "Person") == (
        "model.py: made Worker a Person; columns name, mail added to table worker_table")
    pages, _, _ = app()
    assert _columns(pages["Students"]) == ["Name", "School", "Mail"]
    assert _columns(pages["Teachers"]) == ["Name", "Mail"]
    assert _columns(pages["Workers"]) == ["Salary", "Name", "Mail"]
    assert sorted(_form_fields(pages["Teachers"])) == ["mail", "name"]  # the form submits the inherited fields
    assert sorted(_form_fields(pages["Workers"])) == ["mail", "name", "salary"]
    assert change("remove", "Person.mail") == (
        "model.py: removed Person.mail; its column from table student_table, teacher_table, worker_table")
    pages, _, _ = app()
    assert _columns(pages["Students"]) == ["Name", "School"] and _form_fields(pages["Teachers"]) == ["name"]


def test_removing_the_main_page_makes_another_the_main_page_with_its_buttons(starter, change, app):
    change("add-class", "Publisher", "name:str", "--screen")
    # Before: "warning: screen Authors was the main page: the app opens on Books now, and the buttons it held
    # are gone"; the Publishers page could not be reached.
    assert change("remove", "Author") == (
        "model.py: removed Author and association book_authors; and table author_table, screen Authors, "
        "button to_books, button to_publishers; Books is the main page now, with a button to Publishers")
    pages, routes, _ = app()
    assert sorted(pages) == ["Books", "Publishers"]
    assert '<Route path="/" element={<Navigate to="/Books" replace />} />' in routes
    assert 'navigate("/publishers")' in pages["Books"]


# The main page shows a class's table; the other screens are written inside their modules.
INLINE_SCREENS = HAND_WRITTEN.replace(
    '        Screen(name="Home", description="Home", is_main_page=True, view_elements={recent}),\n',
    '        Screen(name="Home", description="Home", is_main_page=True, view_elements={recent}),\n'
    '        Screen(name="About", description="About", view_elements=set()),\n',
).replace('admin = Module(name="admin", screens={Screen(name="Customers", description="Customers", '
          'view_elements={everyone})})',
          'admin = Module(name="admin", screens={Screen(name="Archive", description="Archive", view_elements=set())})')


def test_the_new_main_page_written_inline_gets_its_flag_and_buttons(write, besser, change, app, tmp_path):
    write("model.py", INLINE_SCREENS)
    change("add-class", "Order", "number:str", "--screen")  # in the first module, admin, with a variable
    removed = besser("model", "remove", "Customer")
    assert removed.returncode == 0, removed.stderr
    assert removed.stdout == ("model.py: removed Customer; and table everyone, table recent, screen Home, "
                              "button to_orders; Archive is the main page now, with a button to Orders\n")
    assert removed.stderr == ("warning: no button on Archive opens screen About: it has no variable to refer to "
                              "it by; give it one and add the button\n")
    assert re.search(r'Screen\(name="Archive", description="Archive", view_elements=\{to_orders\},\s+is_main_page=True\)',
                     (tmp_path / "model.py").read_text())
    pages, routes, _ = app()
    assert '<Navigate to="/Archive" replace />' in routes and 'navigate("/orders")' in pages["Archive"]


def test_add_screen_is_refused_when_it_cannot_be_made(besser, write, starter):
    def refused(*args):
        result = besser("model", *args)
        assert result.returncode == 2
        return result.stderr

    assert "screen Authors shows Author already" in refused("add-screen", "Author")
    assert besser("model", "add-class", "Tag").returncode == 0
    assert "Tag has no attribute, own or inherited, for its table to show: add one first" in refused(
        "add-screen", "Tag")
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
    assert line.endswith("added screen Orders at /orders (table of Order: number; button on Home)")
    text = (tmp_path / "model.py").read_text()
    assert "admin = Module(name=\"admin\", screens={Screen(" in text and "order_screen})" in text
    pages, routes, _ = app()
    assert _columns(pages["Orders"]) == ["Number"] and 'navigate("/orders")' in pages["Home"]


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
    assert line == "model.py: added screen Catalog at /catalog (table of Book: title, pages; button on Start)"
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
    assert '<Route path="/boxes"' in routes and "/BOOKS" not in routes
    assert 'title="Boxes"' in pages["Boxes"] and '{"Boxes"}' in pages["Boxes"] and "BOOKs" not in pages["Boxes"]
    assert '"Boxes"' in pages["Authors"] and "BOOKs" not in pages["Authors"]  # the button that opens it
    assert "GUI titles Authors -> AUTHORs" in change("rename", "Author", "AUTHOR")  # its own route, but for case


def test_a_screen_whose_page_file_differs_only_in_case_is_refused(besser, starter, change, app):
    # The React generator names a page file after its title in PascalCase with inner words lowercased:
    # OrderLines -> Orderlines.tsx, Order_lines -> OrderLines.tsx, one file on macOS or Windows.
    assert change("add-class", "OrderLine", "x:str", "--screen").endswith(
        "added screen OrderLines at /orderlines (table of OrderLine: x; button on Authors)")
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


# ── routes, a menu, columns of related objects, cards and charts ────────────
#
# Why: a coding agent given an ordinary specification (pages at fixed routes, a menu on every page, a
# dashboard, the name of a related object in a table) had only add-screen. In two measured runs it left
# the model: once it wrote the GUI part of model.py by hand, after which add-attribute was refused
# ("cannot find where 'deal_table' is written in model.py; edit it by hand"); once it wrote a whole
# React frontend, and the model no longer drove the pages.


def _links(page: str) -> list[tuple[str, str]]:
    """The links of a page's menu: (address, label)."""
    return re.findall(r'<a href="([^"]*)">\{"([^"]*)"\}</a>', page)


def test_a_new_screen_is_at_its_title_in_lower_case_or_at_the_route_asked_for(starter, change, besser, app):
    assert change("add-class", "Publisher", "name:str", "--screen").endswith(
        "added screen Publishers at /publishers (table of Publisher: name; button on Authors)")
    assert change("add-class", "Shelf", "code:str").startswith("model.py: added class Shelf")
    assert "added screen Shelves at /library/shelves" in change("add-screen", "Shelf", "--route", "/library/shelves")
    _, routes, _ = app()
    assert '<Route path="/publishers"' in routes and '<Route path="/library/shelves"' in routes

    assert change("add-class", "Tag", "label:str").startswith("model.py: added class Tag")
    taken = besser("model", "add-screen", "Tag", "--route", "/PUBLISHERS")
    assert taken.returncode == 2
    assert "screen Publishers of LibraryApp is at /publishers already: give another --route" in taken.stderr
    bad = besser("model", "add-screen", "Tag", "--route", "tags")
    assert bad.returncode == 2 and "'tags' is not a route: write /name or /group/name" in bad.stderr


def test_a_menu_names_once_each_page_its_buttons_opened(starter, change):
    # Before: with a button to Books on two pages, "it replaces the buttons to Authors, Books, Books".
    change("add-page", "Overview", "--route", "/overview", "--main")
    assert change("add-menu").endswith("; it replaces the buttons to Authors, Books")


def test_a_menu_links_every_page_from_every_page_and_follows_the_screens(starter, change, besser, app):
    # The menu replaces the starter's button to Books: beside it, the button was a second navigation.
    assert change("add-menu") == "model.py: added menu menu to Authors, Books; it replaces the button to Books"
    pages, _, _ = app()
    assert "to_books" not in pages["Authors"]
    for page in ("Authors", "Books"):
        assert _links(pages[page]) == [("/Authors", "Authors"), ("/Books", "Books")]
        # Before: the generator compared the menu's order with the None of the other elements and crashed.
        assert pages[page].index('<nav id="menu"') < pages[page].index(f'id="{page.lower()[:-1]}_heading"')

    # A new page joins the menu and shows it; it needs no button on the main page.
    line = change("add-class", "Publisher", "name:str", "--screen")
    assert line.endswith("added screen Publishers at /publishers (table of Publisher: name; in menu menu)")
    pages, _, _ = app()
    assert all(("/publishers", "Publishers") in _links(page) for page in pages.values())
    assert "to_publishers" not in pages["Authors"]

    # A renamed class's page is renamed in the menu too, and a removed one leaves it.
    assert "GUI titles Publishers -> Editors" in change("rename", "Publisher", "Editor")
    pages, routes, _ = app()
    assert ("/editors", "Editors") in _links(pages["Authors"]) and '<Route path="/editors"' in routes
    removed = change("remove", "Editor")
    assert "screen Editors" in removed
    pages, _, _ = app()
    assert sorted(pages) == ["Authors", "Books"]
    assert _links(pages["Books"]) == [("/Authors", "Authors"), ("/Books", "Books")]

    assert change("add-menu") == "model.py: menu menu links the screens of LibraryApp already (no change)"


def test_a_menu_added_before_the_pages_is_on_each_of_them(starter, change, app):
    change("remove", "Book")
    change("remove", "Author")
    assert change("add-menu") == "model.py: added menu menu (no screen yet: each new one joins it)"
    assert change("add-class", "Company", "name:str", "--screen").endswith(
        "added screen Companies at /companies (table of Company: name; the main page, in menu menu)")
    assert change("add-class", "Deal", "title:str", "--screen").endswith("in menu menu)")
    pages, _, _ = app()
    assert _links(pages["Companies"]) == _links(pages["Deals"]) == [("/companies", "Companies"), ("/deals", "Deals")]


def test_a_column_shows_an_attribute_of_the_related_object(starter, change, besser, app):
    change("add-class", "Publisher", "name:str", "city:str", "--screen")
    change("add-association", "book_publisher", "books:Book[*]", "publisher:Publisher[0..1]")
    assert change("add-column", "Book", "publisher.name") == (
        "model.py: added column Publisher (publisher.name) to table book_table")
    assert change("add-column", "Book", "publisher.city", "--label", "Published in") == (
        "model.py: added column Published in (publisher.city) to table book_table")
    pages, _, _ = app()
    assert _columns(pages["Books"]) == ["Title", "Pages", "Publisher", "Published in"]
    assert re.search(r'"label": "Publisher",\s*"column_type": "lookup"', pages["Books"])
    info = besser("info", "model.py").stdout
    assert "Book table book_table (title, pages, publisher.name, publisher.city)" in info

    for arguments, said in [
        (("Book", "publisher.name"), "no table of Book to add the column to: publisher.name is a column already"),
        (("Book", "authors.name"), "Book.authors is [1..*]: a column shows one object"),
        (("Book", "publisher.country"), "Publisher has no attribute 'country' (its attributes: city, name)"),
        (("Book", "nothing"), "'nothing' is not end.attribute of Book"),
    ]:
        refused = besser("model", "add-column", *arguments)
        assert refused.returncode == 2 and said in refused.stderr, refused.stderr
    # The removed attribute's column goes with it, as a table's own columns do.
    assert "its column from table book_table" in change("remove", "Publisher.city")
    pages, _, _ = app()
    assert _columns(pages["Books"]) == ["Title", "Pages", "Publisher"]


def test_a_page_with_cards_and_a_chart(starter, change, besser, app):
    change("set-enum", "Genre", "NOVEL", "ESSAY")
    change("add-attribute", "Book", "genre:Genre", "price:float", "lent:bool")
    assert change("add-menu").startswith("model.py: added menu menu")
    assert change("add-page", "Library overview", "--route", "/overview", "--main") == (
        "model.py: added page LibraryOverview at /overview (the main page, in menu menu)")
    # The screen is named by its name, its title or its route.
    assert change("add-card", "LibraryOverview", "Books", "count:Book") == (
        "model.py: added card 'Books' to screen LibraryOverview (count of Book)")
    assert change("add-card", "library overview", "Novels value", "sum:Book.price",
                  "--where", "genre = NOVEL and lent == false") == (
        "model.py: added card 'Novels value' to screen LibraryOverview "
        "(sum of Book.price where genre == 'NOVEL' and lent == false)")
    assert change("add-chart", "/overview", "Pages by genre", "sum:Book.pages", "--by", "genre") == (
        "model.py: added bar chart 'Pages by genre' to screen LibraryOverview (sum of Book.pages by genre)")
    assert "added pie chart 'Books by genre'" in change("add-chart", "/overview", "Books by genre", "count:Book",
                                                         "--by", "genre", "--kind", "pie")
    pages, routes, _ = app()
    page = pages["LibraryOverview"]
    assert '<Route path="/overview"' in routes and '<Navigate to="/overview" replace />' in routes
    # In the order they were added, after the menu and the heading.
    order = [page.index(text) for text in ('<nav id="menu"', 'id="library_overview_heading"', 'id="books"',
                                           'id="novels_value"', 'id="pages_by_genre"', 'id="books_by_genre"')]
    assert order == sorted(order)
    assert '"filter_expression": "genre == \'NOVEL\' and lent == false"' in page
    assert re.search(r'"label_field": "genre",\s*"data_field": "pages",\s*"aggregation": "sum"', page)
    assert 'chartType="pie-chart"' in page and '"showTrend": false' in page
    info = besser("info", "model.py").stdout
    assert ("    screen LibraryOverview (main page), /overview: card 'Books' (count:Book); card 'Novels value' "
            "(sum:Book.price where genre == 'NOVEL' and lent == false); bar chart 'Pages by genre' "
            "(sum:Book.pages by genre); pie chart 'Books by genre' (count:Book by genre); menu\n") in info
    assert "    screen Authors, /Authors:" in info  # no longer the main page


@pytest.mark.parametrize("arguments, said", [
    (("Books", "T", "total:Book.pages"), "'total:Book.pages' is not a value: write sum:Class.attribute"),
    (("Books", "T", "sum:Book"), "'sum:Book': sum is of an attribute, as sum:Book.attribute"),
    (("Books", "T", "sum:Book.title"), "Book.title is str: sum is of an int or float attribute"),
    (("Nowhere", "T", "count:Book"), "LibraryApp has no screen 'Nowhere' (its screens: Authors (/Authors), "
                                     "Books (/Books)); 'besser model add-page' adds one"),
    # The generated pages would drop these conditions without a word, and show the unfiltered number.
    (("Books", "T", "count:Book", "--where", "pages > 100 or pages < 10"), "the pages cannot read 'or'"),
    (("Books", "T", "count:Book", "--where", "pages > many"), "pages is a number, 'many' is not"),
    (("Books", "T", "count:Book", "--where", "weight > 1"), "'weight > 1' is not a condition on Book"),
    (("Books", "T", "count:Book", "--where", "pages"), "'pages' is not a condition on Book"),
])
def test_a_card_that_cannot_work_is_refused(starter, besser, arguments, said):
    before = starter.read_text(encoding="utf-8")
    refused = besser("model", "add-card", *arguments)
    assert refused.returncode == 2 and said in refused.stderr, refused.stderr
    assert starter.read_text(encoding="utf-8") == before


def test_a_renamed_class_moves_its_page_unless_told_to_keep_it(starter, change, app):
    change("add-menu")
    change("add-class", "Contact", "name:str", "--screen")
    # Why: "rename the entity, the page keeps its route and title" is an ordinary request, and there was
    # no way to say so: the page moved to /people and the old address opened the main page.
    assert change("rename", "Contact", "Person", "--keep-pages") == (
        "model.py: renamed Contact to Person (variable contact -> person)")
    pages, routes, _ = app()
    assert '<Route path="/contacts"' in routes and 'title="Contacts"' in pages["Contacts"]
    assert ("/contacts", "Contacts") in _links(pages["Authors"])
    assert '"entity": "Person"' in pages["Contacts"]

    # Without it the page follows its class, route included, and the line says how to keep it.
    change("add-class", "Order", "number:str", "--screen")
    line = change("rename", "Order", "Purchase")
    assert line.endswith("GUI titles Orders -> Purchases, route /orders -> /purchases (--keep-pages keeps them)")
    pages, routes, _ = app()
    assert '<Route path="/purchases"' in routes and ("/purchases", "Purchases") in _links(pages["Contacts"])


def test_a_link_on_a_page(starter, change, besser, app):
    assert change("add-link", "Books", "Export CSV", "http://localhost:8000/book/export/csv/") == (
        "model.py: added link 'Export CSV' to screen Books (http://localhost:8000/book/export/csv/)")
    pages, _, _ = app()
    assert re.search(r'<a id="export_csv"[^>]*href="http://localhost:8000/book/export/csv/"', pages["Books"])
    for arguments, said in [(("Books", "Export CSV", "/x"), "screen Books has a link 'Export CSV' already"),
                            (("Books", "Other", "book/export"), "'book/export' is not an address")]:
        refused = besser("model", "add-link", *arguments)
        assert refused.returncode == 2 and said in refused.stderr, refused.stderr


# ── what a measured run still paid for (ten sessions read call by call) ─────


def test_a_derived_attribute_is_the_servers_to_compute(starter, change, besser, write, tmp_path):
    # Why: "fullName is computed by the server, clients never send it". Without --derived the generated
    # test sent fullName and expected it back, and the agent wrote a middleware to get round it.
    assert change("add-attribute", "Author", "fullName:str", "--derived") == (
        "model.py: added Author.fullName:str (derived); column added to table author_table")
    assert "is_derived=True" in (tmp_path / "model.py").read_text(encoding="utf-8")
    assert "class Author: name:str, email:str, fullName:str (derived)" in besser("info", "model.py").stdout
    assert besser("apply").returncode == 0
    write("web_app/backend/extensions.py", '''
        from sqlalchemy import event
        from sql_alchemy import Author


        @event.listens_for(Author, "before_insert")
        @event.listens_for(Author, "before_update")
        def full_name(mapper, connection, target):
            target.fullName = f"{target.name} <{target.email}>"


        def setup(app):
            pass
    ''')
    write("tests/test_full_name.py", '''
        def test_full_name(client):
            made = client.post("/author/", json={"name": "Ada", "email": "a@x", "fullName": "ignored"}).json()
            made = made.get("author", made)
            assert made["fullName"] == "Ada <a@x>"
            assert client.put(f"/author/{made['id']}/", json={"name": "Bo", "email": "b@x"}).status_code == 200
            got = client.get(f"/author/{made['id']}/").json()
            assert got.get("author", got)["fullName"] == "Bo <b@x>"
    ''')
    applied = besser("apply", output=None)
    assert applied.returncode == 0, applied.stdout + applied.stderr
    assert "fullName" not in (tmp_path / "web_app/backend/tests/test_generated_api.py").read_text(encoding="utf-8")


def test_a_line_that_changes_nothing_does_not_undo_its_batch(starter, change, besser):
    change("add-menu")
    # Before: "menu menu links the screens already" was an error, and the lines before it in the batch were lost.
    line = change("batch", "-", stdin="add-class Shelf code:str --screen\nadd-menu\nadd-column Shelf code\n")
    assert line.endswith("menu menu links the screens of LibraryApp already (no change); "
                         "Shelf.code is a column of its table already (no change)")
    assert "screen Shelves" in besser("info", "model.py").stdout


def test_a_link_shows_in_info_and_follows_its_class(starter, change, besser):
    change("add-link", "Books", "Export CSV", "http://localhost:8000/book/export/csv/")
    assert "link 'Export CSV' (http://localhost:8000/book/export/csv/)" in besser("info", "model.py").stdout
    # Before: the link kept /book/ after the class and its routes were renamed, and info did not show it.
    assert "; link Export CSV now to /volume/" in change("rename", "Book", "Volume")
    assert "link 'Export CSV' (http://localhost:8000/volume/export/csv/)" in besser("info", "model.py").stdout


def test_a_rename_or_removal_names_the_hand_written_code_left_behind(starter, besser, write):
    assert besser("apply").returncode == 0  # the generated files are recorded: they are not hand-written
    write("web_app/backend/extensions.py", "from sql_alchemy import Book\n\n\ndef setup(app):\n"
                                           "    app.get('/book/count2/')(lambda: Book.pages)\n")
    write("tests/test_mine.py", "def test_it(client):\n    assert client.get('/book/').status_code == 200\n")
    # Before: nothing was said, and the agent searched the project for the old name (three calls in one run).
    renamed = besser("model", "rename", "Book", "Volume")
    assert renamed.returncode == 0
    # Before the lines were printed: only "tests/test_mine.py:2; web_app/backend/extensions.py:1,5", and the
    # next call printed those lines to see what to change (four sessions of two runs).
    assert renamed.stderr == (
        "warning: hand-written code still names Book, on these lines:\n"
        "  tests/test_mine.py:2: assert client.get('/book/').status_code == 200\n"
        "  web_app/backend/extensions.py:1: from sql_alchemy import Book\n"
        "  web_app/backend/extensions.py:5: app.get('/book/count2/')(lambda: Book.pages)\n")
    removed = besser("model", "remove", "Volume.pages")
    assert ("hand-written code still names Volume.pages, on these lines:\n"
            "  web_app/backend/extensions.py:5: app.get('/book/count2/')(lambda: Book.pages)\n") in removed.stderr



@pytest.mark.skipif(os.name != "nt", reason="an MSYS shell rewrites arguments on Windows only")
def test_a_route_rewritten_by_git_bash_is_read_as_typed(starter, change, monkeypatch):
    # Git Bash turns the argument /overview into C:/Program Files/Git/overview, and /a/home into A:/home.
    # Before: "'C:/Program Files/Git/overview' is not a route", so the agent set MSYS_NO_PATHCONV and retried.
    monkeypatch.setenv("MSYSTEM", "MINGW64")
    monkeypatch.setenv("EXEPATH", "C:\\Program Files\\Git\\bin")
    assert change("add-page", "Overview", "--route", "C:/Program Files/Git/overview") == (
        "model.py: added page Overview at /overview (button on Authors)")
    assert "to screen Overview" in change("add-card", "C:/Program Files/Git/overview", "Books", "count:Book")
    assert "added page Home at /a/home" in change("add-page", "Home", "--route", "A:/home")


def test_the_pages_the_commands_write_look_finished(starter, change, app):
    # Why: beside an application an LLM wrote by hand for the same request, the generated one looked raw: a
    # bare row of links, a title in plain text, cards one under the other with the chart between them. The
    # generated React app ships Tailwind, so the look is classes on the model's elements.
    change("add-menu")
    change("add-class", "Loan", "amount:float", "days:int", "--screen")
    change("add-page", "Library overview", "--route", "/overview", "--main")
    change("add-chart", "/overview", "Loans by days", "sum:Loan.amount", "--by", "days")
    change("add-card", "/overview", "Lent", "sum:Loan.amount")  # after the chart, and still above it
    change("add-card", "/overview", "Loans", "count:Loan")
    change("add-link", "Loans", "Export", "http://localhost:8000/loan/export/")
    pages, _, _ = app()
    overview, loans = pages["LibraryOverview"], pages["Loans"]
    assert '<nav id="menu" className="bg-slate-900 ' in overview
    assert '<div id="library_overview_page" className="max-w-5xl mx-auto ' in overview
    assert '<h1 id="library_overview_heading" className="text-2xl font-bold ' in overview
    assert '<div id="library_overview_cards" className="grid grid-cols-1 md:grid-cols-3 ' in overview
    assert 'id="loans_by_days" className="bg-white rounded-xl shadow ' in overview
    order = [overview.index(text) for text in ('<nav id="menu"', 'id="library_overview_heading"',
                                               'id="library_overview_cards"', 'id="lent"', 'id="loans"',
                                               'id="loans_by_days"')]
    assert order == sorted(order)
    assert '<h1 id="loan_heading" className="text-2xl font-bold ' in loans
    assert loans.index('id="loan_heading"') < loans.index('id="export"') < loans.index('id="loan_table"')
    # The model changes keep working on such a page: its table still follows its class.
    assert "column added to table loan_table" in change("add-attribute", "Loan", "returned:bool")
    assert "GUI titles Loans -> Lendings, route /loans -> /lendings" in change("rename", "Loan", "Lending")
    pages, _, _ = app()
    assert '<h1 id="loan_heading" className="text-2xl font-bold text-slate-900 mb-4">{"Lendings"}</h1>' in pages["Lendings"]


# A GUI written the way a coding agent wrote it in a measured run: one function builds every screen.
BUILT_BY_A_FUNCTION = '''\
from besser.BUML.metamodel.gui import DataBinding, FieldColumn, GUIModel, Module, Screen, Table
from besser.BUML.metamodel.structural import Class, DomainModel, FloatType, Property, StringType

Deal_title = Property(name="title", type=StringType)
Deal_amount = Property(name="amount", type=FloatType)
Deal = Class(name="Deal", attributes={Deal_title, Deal_amount})
crm = DomainModel(name="Crm", types={Deal})


def entity_screen(key, title, cls, columns):
    table = Table(name=f"{key}_table", title=title, columns=columns)
    table.data_binding = DataBinding(domain_concept=cls)
    return Screen(name=title, description=title, is_main_page=True, view_elements={table})


deal_screen = entity_screen("deal", "Deals", Deal, [
    FieldColumn(label="Title", field=Deal_title),
    FieldColumn(label="Amount", field=Deal_amount),
])
crm_gui = GUIModel(name="CrmApp", package="crm", versionCode="1", versionName="1.0", description="A CRM",
                   modules={Module(name="main", screens={deal_screen})})
'''


def test_an_attribute_is_added_to_a_class_whose_table_a_function_builds(write, besser, app, tmp_path):
    write("model.py", BUILT_BY_A_FUNCTION)
    added = besser("model", "add-attribute", "Deal", "priority:str?")
    # Before: refused whole, "cannot find where 'deal_table' is written in model.py; edit it by hand", so
    # the agent wrote the attribute by hand too.
    assert added.returncode == 0, added.stderr
    assert added.stdout == "model.py: added Deal.priority:str?\n"
    assert added.stderr == (
        "warning: table deal_table got no column for priority: model.py does not write its columns where a "
        'command can add to them; add FieldColumn(label="Priority", field=Deal_priority) by hand\n')
    pages, _, _ = app()
    assert _columns(pages["Deals"]) == ["Title", "Amount"]
    # The code of the warning is what works where the columns are written.
    text = (tmp_path / "model.py").read_text(encoding="utf-8")
    assert "Deal_priority = Property(" in text
    write("model.py", text.replace('    FieldColumn(label="Amount", field=Deal_amount),\n',
                                   '    FieldColumn(label="Amount", field=Deal_amount),\n'
                                   '    FieldColumn(label="Priority", field=Deal_priority),\n'))
    pages, _, _ = app()
    assert _columns(pages["Deals"]) == ["Title", "Amount", "Priority"]

def test_columns_are_put_in_the_order_asked(starter, change, besser, app):
    # Before: no command placed a column, and two runs patched model.py with a script to put one first.
    assert change("add-attribute", "Book", "isbn:str").endswith("column added to table book_table")
    assert change("order-columns", "Book", "isbn", "pages") == (
        "model.py: columns of table book_table: isbn, pages, title")
    pages, _, _ = app()
    places = [pages["Books"].index(f'"field": "{name}"') for name in ("isbn", "pages", "title")]
    assert places == sorted(places)
    refused = besser("model", "order-columns", "Book", "nope")
    assert refused.returncode == 2 and "has no column nope (it has: isbn, pages, title;" in refused.stderr


def test_a_screen_takes_its_rows_a_page_and_its_answer_says_its_title(starter, change, app):
    change("add-class", "DealLine", "quantity:int")
    # Before: "added screen DealLines at /deallines", which left a run unsure its title was taken; and no
    # option set the rows, so a run rewrote rowsPerPage in every generated page.
    line = change("add-screen", "DealLine", "--title", "Deal lines", "--rows", "10")
    assert line.startswith("model.py: added screen DealLines titled 'Deal lines' at /deallines "
                           "(table of DealLine: quantity; 10 rows a page;")
    pages, _, _ = app()
    assert '"rowsPerPage": 10' in pages["DealLines"] and '"rowsPerPage": 5' in pages["Books"]


def test_the_model_itself_is_renamed_by_its_name(starter, change, besser):
    # Before: rename took only the elements of a model, so every application built from the starter kept the
    # names Library and LibraryApp, and its API was titled "Library API".
    assert change("rename", "Library", "Shop") == (
        "model.py: renamed the model Library to Shop (the API's title and the database file follow it)")
    assert change("rename", "LibraryApp", "ShopApp") == "model.py: renamed the model LibraryApp to ShopApp"
    info = besser("info", "model.py").stdout
    assert "(DomainModel 'Shop')" in info and "(GUIModel 'ShopApp')" in info


def test_add_class_takes_the_options_of_its_screen(starter, change, besser):
    # Before: "unrecognized arguments: --title Services --route /services", and the batch the line was in
    # was lost with it (a run did the whole batch again).
    line = change("batch", "-", stdin="add-class Service name:str --screen --title Services --route /svc --rows 10\n"
                                      "add-column Book title\n")
    assert "added screen Services at /svc (table of Service: name; 10 rows a page;" in line
    # An option of the screen asks for the screen.
    assert "added screen PriceTags titled 'Price tags' at /pricetags" in change(
        "add-class", "Tag", "label:str", "--title", "Price tags")
