import os
import pytest
from besser.generators.python_classes import PythonGenerator
from besser.BUML.metamodel.structural import (
    Class, DomainModel, Enumeration, EnumerationLiteral,
    DateType, StringType, IntegerType, Property, BinaryAssociation,
    Multiplicity
)

# Define the expected output
enum_output = """
class MemberType(Enum):
    ADULT = "ADULT"
    SENIOR = "SENIOR"
    STUDENT = "STUDENT"
    CHILD = "CHILD"
"""

library_output = """
class Library:

    def __init__(self, name: str, address: str, Book_end: set["Book"] = None):
        self.name = name
        self.address = address
        self.Book_end = Book_end if Book_end is not None else set()
"""

author_output = """
class Author:

    def __init__(self, email: str, member: MemberType, Book_end: set["Book"] = None):
        self.email = email
        self.member = member
        self.Book_end = Book_end if Book_end is not None else set()
"""

book_output = """
class Book:

    def __init__(self, release: date, title: str, pages: int, Library_end: "Library" = None, writtenBy: set["Author"] = None):
        self.release = release
        self.title = title
        self.pages = pages
        self.Library_end = Library_end
        self.writtenBy = writtenBy if writtenBy is not None else set()
"""

@pytest.fixture
def domain_model():
    MemberType: Enumeration = Enumeration(
        name="MemberType",
        literals={
                EnumerationLiteral(name="ADULT"),
                EnumerationLiteral(name="SENIOR"),
                EnumerationLiteral(name="STUDENT"),
                EnumerationLiteral(name="CHILD")
        }
    )

    # Classes
    Book = Class(name="Book")
    Author = Class(name="Author")
    Library = Class(name="Library")

    # Book class attributes and methods
    Book_release: Property = Property(name="release", type=DateType)
    Book_title: Property = Property(name="title", type=StringType)
    Book_pages: Property = Property(name="pages", type=IntegerType)
    Book.attributes={Book_release, Book_title, Book_pages}

    # Author class attributes and methods
    Author_email: Property = Property(name="email", type=StringType)
    Author_member: Property = Property(name="member", type=MemberType)
    Author.attributes={Author_email, Author_member}

    # Library class attributes and methods
    Library_name: Property = Property(name="name", type=StringType)
    Library_address: Property = Property(name="address", type=StringType)
    Library.attributes={Library_name, Library_address}

    # Relationships
    has: BinaryAssociation = BinaryAssociation(
        name="Has",
        ends={
            Property(name="Book_end", type=Book, multiplicity=Multiplicity(0, "*")),
            Property(name="Library_end", type=Library, multiplicity=Multiplicity(1, 1))
        }
    )
    book_author_relation: BinaryAssociation = BinaryAssociation(
        name="BookAuthor_Relation",
        ends={
            Property(name="writtenBy", type=Author, multiplicity=Multiplicity(1, "*")),
            Property(name="Book_end", type=Book, multiplicity=Multiplicity(0, "*"))
        }
    )

    # Domain Model
    model = DomainModel(
        name="Domain_Model",
        types={Book, Author, Library, MemberType},
        associations={has, book_author_relation},
        generalizations={}
    )

    return model

# Define the test function
def test_generator(domain_model, tmpdir):
    # Create an instance of the generator
    output_dir = tmpdir.mkdir("output")
    generator = PythonGenerator(model=domain_model, output_dir=str(output_dir))

    # Generate Python classes
    generator.generate()

    # Check if the file was created
    output_file = os.path.join(str(output_dir), "classes.py")
    assert os.path.isfile(output_file)

    # Read the generated file
    with open(output_file, "r", encoding="utf-8") as f:
        generated_code = f.read()

    # Compare the output with the expected output
    assert enum_output in generated_code
    assert library_output in generated_code
    assert book_output in generated_code
    assert author_output in generated_code

def test_bidirectional_consistency(domain_model, tmpdir):
    # Generate the classes
    output_dir = tmpdir.mkdir("output")
    generator = PythonGenerator(model=domain_model, output_dir=str(output_dir))
    generator.generate()
    output_file = os.path.join(str(output_dir), "classes.py")
    import importlib.util
    from datetime import date
    spec = importlib.util.spec_from_file_location("classes", output_file)
    classes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classes)

    # Create instances
    Library = classes.Library
    Book = classes.Book
    Author = classes.Author

    lib = Library(name="Central", address="Main St")
    book = Book(release=date.today(), title="Python 101", pages=123, Library_end=lib)
    author = Author(email="ab.com", member=classes.MemberType.ADULT)

    # Test: Book's Library_end should be lib, and lib.Book_end should include book
    assert book.Library_end == lib
    assert book in lib.Book_end

    # Test: Add book to library's Book_end, should update book.Library_end
    lib2 = Library(name="Branch", address="2nd St")
    lib2.Book_end = lib2.Book_end | {book}
    assert book.Library_end == lib2
    assert book in lib2.Book_end
    assert book not in lib.Book_end

    # Test: Author writes book
    author.Book_end = author.Book_end | {book}
    assert author in book.writtenBy
    assert book in author.Book_end

    # Test: Set book.writtenBy directly
    author2 = Author(email="abc.com", member=classes.MemberType.ADULT)
    book.writtenBy = book.writtenBy | {author2}
    assert author2 in book.writtenBy
    assert book in author2.Book_end
