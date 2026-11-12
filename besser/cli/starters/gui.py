"""A starter B-UML GUI model: two screens that show the books and the authors of a library.

    besser validate MODEL_FILE                    check both models
    besser info MODEL_FILE                        what it holds, and the generators that read it
    besser generate web_app MODEL_FILE -o out     generate code ('besser generators' lists them all)
    besser convert MODEL_FILE -o model.json       open it in https://editor.besser-pearl.org (import the project)

B-UML reference: https://besser.readthedocs.io/en/latest/buml_language.html
"""

from besser.BUML.metamodel.gui import (
    Button,
    ButtonActionType,
    ButtonType,
    DataBinding,
    Event,
    FieldColumn,
    GUIModel,
    Module,
    Screen,
    Table,
    Text,
    Transition,
)
from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    IntegerType,
    Multiplicity,
    Property,
    StringType,
)

# The domain model: the classes whose data the screens show
book_title = Property(name="title", type=StringType)
book_pages = Property(name="pages", type=IntegerType)
book = Class(name="Book", attributes={book_title, book_pages})
author_name = Property(name="name", type=StringType)
author_email = Property(name="email", type=StringType)
author = Class(name="Author", attributes={author_name, author_email})
book_authors = BinaryAssociation(name="book_authors", ends={
    Property(name="books", type=book, multiplicity=Multiplicity(0, "*")),
    Property(name="authors", type=author, multiplicity=Multiplicity(1, "*")),
})
library_model = DomainModel(name="Library", types={book, author}, associations={book_authors})

# Components bound to a class: a table lists its objects, one column per attribute.
# component_id is the id a component keeps in the editor and in the generated pages.
author_table = Table(name="author_table", component_id="author_table", title="Authors", action_buttons=True, columns=[
    FieldColumn(label="Name", field=author_name),
    FieldColumn(label="Email", field=author_email),
])
author_table.data_binding = DataBinding(domain_concept=author)
book_table = Table(name="book_table", component_id="book_table", title="Books", show_pagination=True, columns=[
    FieldColumn(label="Title", field=book_title),
    FieldColumn(label="Pages", field=book_pages),
])
book_table.data_binding = DataBinding(domain_concept=book)

# Screens: a page of the application and the components it shows; one of them is the main page
author_screen = Screen(
    name="Authors", component_id="Authors", description="Authors", is_main_page=True,
    view_elements={Text(name="author_heading", component_id="author_heading", content="Authors"), author_table},
)
book_screen = Screen(
    name="Books", component_id="Books", description="Books",
    view_elements={Text(name="book_heading", component_id="book_heading", content="Books"), book_table},
)

# A button that navigates to another screen: the generators read targetScreen, the editor the click event
to_books = Button(
    name="to_books", component_id="to_books", description="Go to the books", label="Books",
    buttonType=ButtonType.CustomizableButton, actionType=ButtonActionType.Navigate, targetScreen=book_screen,
)
to_books.events = {Event(name="to_books_click", actions={Transition(name="go_to_books", target_screen=book_screen)})}
author_screen.view_elements.add(to_books)

# The GUI model: its screens grouped in modules. It is read together with the domain model it shows.
library_gui = GUIModel(
    name="LibraryApp",
    package="org.example.library",
    versionCode="1",
    versionName="1.0",
    description="Browse the books and the authors of a library",
    modules={Module(name="library_module", screens={author_screen, book_screen})},
)
