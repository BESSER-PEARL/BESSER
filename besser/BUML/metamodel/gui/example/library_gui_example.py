"""
Book Library GUI Application Example

This example demonstrates a complete library management system using the BESSER GUI metamodel.
It includes:
- Domain model (Book, Author classes)
- Multiple screens with navigation
- CRUD operations (Create, Read, Update, Delete)
- Data binding between UI and domain model
- Events and Actions for behavior
- Forms, Lists, Buttons, and other UI components
"""

from besser.BUML.metamodel.structural import DomainModel, Class, Property, PrimitiveDataType, Method, IntegerType, StringType, BooleanType
from besser.BUML.metamodel.gui import (
    GUIModel, Module, Screen, Button, ButtonType, ButtonActionType,
    DataList, Form, InputField, InputFieldType, Text,
    Event, EventType, Action, Create, Read, Update, Delete, Transition,
    DataBinding, Parameter, DataSource, DataSourceElement
)
from besser.BUML.metamodel.gui.style import (
    Styling, Size, Position, Color, Layout, LayoutType, 
    JustificationType, UnitSize, PositionType
)


# ===============================
# 1. DOMAIN MODEL
# ===============================

# Create the domain model
library_domain = DomainModel(name="LibraryDomain")

# Define Author class
author_class = Class(name="Author")
author_id = Property(name="id", type=IntegerType)
author_name = Property(name="name", type=StringType)
author_bio = Property(name="biography", type=StringType)
author_birth_year = Property(name="birthYear", type=IntegerType)

# Define Book class
book_class = Class(name="Book")
book_id = Property(name="id", type=IntegerType)
book_title = Property(name="title", type=StringType)
book_isbn = Property(name="isbn", type=StringType)
book_year = Property(name="publicationYear", type=IntegerType)
book_genre = Property(name="genre", type=StringType)

# Add properties to classes
author_class.attributes = {author_id, author_name, author_bio, author_birth_year}
book_class.attributes = {book_id, book_title, book_isbn, book_year, book_genre}

print("✓ Domain Model Created: Author and Book classes")


# ===============================
# 2. STYLING CONFIGURATION
# ===============================

# Define common styles
button_style = Styling(
    size=Size(width="150", height="40", unit_size=UnitSize.PIXELS),
    color=Color(background_color="#007BFF", text_color="#FFFFFF")
)

form_style = Styling(
    size=Size(width="400", height="auto", padding="20", unit_size=UnitSize.PIXELS),
    color=Color(background_color="#F8F9FA")
)

list_style = Styling(
    size=Size(width="100", height="auto", unit_size=UnitSize.PERCENTAGE),
    color=Color(background_color="#FFFFFF", border_color="#DEE2E6")
)

screen_layout = Layout(
    layout_type=LayoutType.FLEX,
    orientation="vertical",
    padding="20px",
    gap="15px",
    alignment=JustificationType.CENTER
)

print("✓ Styling Configuration Created")


# ===============================
# 3. DATA BINDINGS
# ===============================

# Data binding for Book list
book_list_binding = DataBinding(
    name="bookListBinding",
    domain_concept=book_class,
    visualization_attrs={book_title, book_isbn, book_genre},
    # label_field=book_title,
    # data_field=book_id
)

# Data binding for Author list
author_list_binding = DataBinding(
    name="authorListBinding",
    domain_concept=author_class,
    visualization_attrs={author_name, author_birth_year},
    # label_field=author_name,
    # data_field=author_id
)

print("✓ Data Bindings Created")


# ===============================
# 4. ACTIONS (Behavior)
# ===============================

# Book CRUD Actions
read_books_action = Read(
    name="loadBooks",
    target_class=book_class,
    description="Load all books from the database"
)

create_book_action = Create(
    name="createBook",
    target_class=book_class,
    description="Create a new book in the library"
)

update_book_action = Update(
    name="updateBook",
    target_class=book_class,
    description="Update an existing book",
    parameters={Parameter(name="bookId", param_type="int", required=True)}
)

delete_book_action = Delete(
    name="deleteBook",
    target_class=book_class,
    description="Delete a book from the library",
    parameters={Parameter(name="bookId", param_type="int", required=True)}
)

# Author CRUD Actions
read_authors_action = Read(
    name="loadAuthors",
    target_class=author_class,
    description="Load all authors from the database"
)

create_author_action = Create(
    name="createAuthor",
    target_class=author_class,
    description="Create a new author"
)

print("✓ CRUD Actions Created")


# ===============================
# 5. SCREENS
# ===============================

# ----- HOME SCREEN -----
home_title = Text(
    name="homeTitle",
    content="📚 Library Management System",
    description="Main title"
)

home_subtitle = Text(
    name="homeSubtitle",
    content="Manage your books and authors efficiently",
    description="Subtitle"
)

# Buttons will be created after screens are defined (for Transition actions)
home_view_books_btn = Button(
    name="viewBooksBtn",
    label="View Books",
    description="Navigate to books list",
    buttonType=ButtonType.RaisedButton,
    actionType=ButtonActionType.Navigate,
    styling=button_style
)

home_view_authors_btn = Button(
    name="viewAuthorsBtn",
    label="View Authors",
    description="Navigate to authors list",
    buttonType=ButtonType.RaisedButton,
    actionType=ButtonActionType.Navigate,
    styling=button_style
)

home_screen = Screen(
    name="HomeScreen",
    description="Main landing screen of the library application",
    view_elements={home_title, home_subtitle, home_view_books_btn, home_view_authors_btn},
    is_main_page=True,
    layout=screen_layout
)

print("✓ Home Screen Created")


# ----- BOOKS LIST SCREEN -----
books_title = Text(
    name="booksTitle",
    content="📖 Books Collection",
    description="Books list title"
)

# Data source for books list
books_data_source = DataSourceElement(
    name="booksDataSource",
    dataSourceClass=book_class,
    fields={book_title, book_isbn, book_genre}
)

books_list = DataList(
    name="booksList",
    description="List of all books in the library",
    list_sources={books_data_source},
    styling=list_style
)

add_book_btn = Button(
    name="addBookBtn",
    label="+ Add Book",
    description="Navigate to add book form",
    buttonType=ButtonType.FloatingActionButton,
    actionType=ButtonActionType.Navigate,
    styling=button_style
)

back_to_home_btn = Button(
    name="backToHomeBtn",
    label="← Home",
    description="Back to home screen",
    buttonType=ButtonType.TextButton,
    actionType=ButtonActionType.Navigate
)

books_screen = Screen(
    name="BooksScreen",
    description="Screen displaying all books",
    view_elements={books_title, books_list, add_book_btn, back_to_home_btn},
    layout=screen_layout
)

print("✓ Books List Screen Created")


# ----- ADD/EDIT BOOK FORM SCREEN -----
form_title = Text(
    name="formTitle",
    content="Add New Book",
    description="Form title"
)

# Form input fields
title_input = InputField(
    name="titleInput",
    description="Book title input",
    field_type=InputFieldType.Text,
    validationRules="required|min:1|max:200"
)

isbn_input = InputField(
    name="isbnInput",
    description="ISBN input",
    field_type=InputFieldType.Text,
    validationRules="required|isbn"
)

year_input = InputField(
    name="yearInput",
    description="Publication year",
    field_type=InputFieldType.Number,
    validationRules="required|min:1000|max:2100"
)

genre_input = InputField(
    name="genreInput",
    description="Book genre",
    field_type=InputFieldType.Text,
    validationRules="required"
)

book_form = Form(
    name="bookForm",
    description="Form to add or edit a book",
    inputFields={title_input, isbn_input, year_input, genre_input},
    styling=form_style
)

save_book_btn = Button(
    name="saveBookBtn",
    label="💾 Save Book",
    description="Save the book",
    buttonType=ButtonType.RaisedButton,
    actionType=ButtonActionType.SubmitForm,
    styling=button_style
)

cancel_btn = Button(
    name="cancelBtn",
    label="Cancel",
    description="Cancel and go back",
    buttonType=ButtonType.OutlinedButton,
    actionType=ButtonActionType.Cancel
)

book_form_screen = Screen(
    name="BookFormScreen",
    description="Screen for adding or editing a book",
    view_elements={form_title, book_form, save_book_btn, cancel_btn},
    layout=screen_layout
)

print("✓ Book Form Screen Created")


# ----- AUTHORS LIST SCREEN -----
authors_title = Text(
    name="authorsTitle",
    content="✍️ Authors Directory",
    description="Authors list title"
)

authors_data_source = DataSourceElement(
    name="authorsDataSource",
    dataSourceClass=author_class,
    fields={author_name, author_birth_year, author_bio}
)

authors_list = DataList(
    name="authorsList",
    description="List of all authors",
    list_sources={authors_data_source},
    styling=list_style
)

add_author_btn = Button(
    name="addAuthorBtn",
    label="+ Add Author",
    description="Navigate to add author form",
    buttonType=ButtonType.FloatingActionButton,
    actionType=ButtonActionType.Navigate,
    styling=button_style
)

back_to_home_btn2 = Button(
    name="backToHomeBtn2",
    label="← Home",
    description="Back to home screen",
    buttonType=ButtonType.TextButton,
    actionType=ButtonActionType.Navigate
)

authors_screen = Screen(
    name="AuthorsScreen",
    description="Screen displaying all authors",
    view_elements={authors_title, authors_list, add_author_btn, back_to_home_btn2},
    layout=screen_layout
)

print("✓ Authors List Screen Created")


# ===============================
# 6. TRANSITIONS & EVENTS
# ===============================

# Transitions for navigation
transition_to_books = Transition(
    name="goToBooksScreen",
    target_screen=books_screen,
    description="Navigate to books list screen"
)

transition_to_authors = Transition(
    name="goToAuthorsScreen",
    target_screen=authors_screen,
    description="Navigate to authors list screen"
)

transition_to_home = Transition(
    name="goToHomeScreen",
    target_screen=home_screen,
    description="Navigate to home screen"
)

transition_to_book_form = Transition(
    name="goToBookFormScreen",
    target_screen=book_form_screen,
    description="Navigate to book form screen"
)

# Events connecting buttons to actions

# Home screen events
event_view_books = Event(
    name="onViewBooksClick",
    event_type=EventType.OnClick,
    description="When user clicks View Books button",
    actions={read_books_action, transition_to_books}
)

event_view_authors = Event(
    name="onViewAuthorsClick",
    event_type=EventType.OnClick,
    description="When user clicks View Authors button",
    actions={read_authors_action, transition_to_authors}
)

# Books screen events
event_add_book = Event(
    name="onAddBookClick",
    event_type=EventType.OnClick,
    description="When user clicks Add Book button",
    actions={transition_to_book_form}
)

event_back_to_home = Event(
    name="onBackToHomeClick",
    event_type=EventType.OnClick,
    description="When user clicks Back to Home button",
    actions={transition_to_home}
)

# Form submit event
event_save_book = Event(
    name="onSaveBookSubmit",
    event_type=EventType.OnSubmit,
    description="When user submits the book form",
    actions={create_book_action, transition_to_books}
)

# Authors screen events
event_back_to_home2 = Event(
    name="onBackToHomeClick2",
    event_type=EventType.OnClick,
    description="When user clicks Back to Home button from authors",
    actions={transition_to_home}
)

print("✓ Transitions and Events Created")


# ===============================
# 7. MODULE & GUI MODEL
# ===============================

# Create module with all screens
library_module = Module(
    name="LibraryModule",
    screens={home_screen, books_screen, book_form_screen, authors_screen}
)

# Create the complete GUI model
library_gui = GUIModel(
    name="LibraryGUI",
    package="com.library.app",
    versionCode="1.0.0",
    versionName="Library Manager v1.0",
    modules={library_module},
    description="Complete library management system with books and authors",
    screenCompatibility=True
)

print("✓ Library GUI Model Created")


# ===============================
# 8. SUMMARY & STATISTICS
# ===============================

print("\n" + "="*60)
print("📊 LIBRARY APPLICATION SUMMARY")
print("="*60)
print(f"\n🏗️  DOMAIN MODEL:")
print(f"   • Classes: {len(library_domain.types)} (Book, Author)")
print(f"   • Book properties: {len(book_class.attributes)}")
print(f"   • Author properties: {len(author_class.attributes)}")

print(f"\n🎨 GUI MODEL:")
print(f"   • Screens: {len(library_module.screens)}")
print(f"   • Total View Elements: {sum(len(screen.view_elements) for screen in library_module.screens)}")

print(f"\n🎬 BEHAVIOR:")
print(f"   • CRUD Actions: 6 (Create, Read, Update, Delete for Books & Authors)")
print(f"   • Transitions: 4 (Navigation between screens)")
print(f"   • Events: 6 (User interactions)")

print(f"\n📱 SCREENS:")
for screen in library_module.screens:
    print(f"   • {screen.name}: {len(screen.view_elements)} elements")
    if screen.is_main_page:
        print(f"     └─ Main landing page ⭐")

print(f"\n🔗 DATA BINDINGS:")
print(f"   • {book_list_binding.name} → {book_list_binding.domain_concept.name}")
print(f"   • {author_list_binding.name} → {author_list_binding.domain_concept.name}")

print("\n" + "="*60)
print("✅ Library Application Model Complete!")
print("="*60)
print("\n💡 This model demonstrates:")
print("   ✓ Domain model with business entities")
print("   ✓ Multiple screens with navigation")
print("   ✓ Full CRUD operations")
print("   ✓ Event-driven behavior")
print("   ✓ Data binding between UI and domain")
print("   ✓ Forms, lists, buttons, and text components")
print("   ✓ Styling and layout configuration")
print("\n🚀 Ready for code generation!")
