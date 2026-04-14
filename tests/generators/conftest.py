"""
Shared test fixtures for generator tests.

This module provides domain model fixtures that are specific to generator
testing and are more elaborate than the base fixtures in ``tests/conftest.py``.

Fixtures defined here are automatically available to all tests under
the ``tests/generators/`` directory without explicit imports.
"""

import pytest
from besser.BUML.metamodel.structural import (
    Class, DomainModel, Property, BinaryAssociation, Multiplicity,
    Generalization, Enumeration, EnumerationLiteral,
    StringType, IntegerType, DateType,
)


# ---------------------------------------------------------------------------
# Library / Book / Author + MemberType enum model
# (used by the Python generator and RDF generator tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def library_model_with_enum():
    """A Library-Book-Author domain model with a MemberType enumeration.

    This fixture builds a model where Author has an ``email`` and a
    ``member`` attribute (typed by MemberType enum) but no ``name``.

    Structure:
        Library  (name: str, address: str)
        Book     (title: str, pages: int, release: date)
        Author   (email: str, member: MemberType)
        MemberType  {ADULT, SENIOR, STUDENT, CHILD}

        Library 1 ---Has--- 0..* Book   (Library_end / Book_end)
        Author  1..* ---BookAuthor_Relation--- 0..* Book  (writtenBy / Book_end)

    Model name: Domain_Model
    """
    MemberType = Enumeration(
        name="MemberType",
        literals={
            EnumerationLiteral(name="ADULT"),
            EnumerationLiteral(name="SENIOR"),
            EnumerationLiteral(name="STUDENT"),
            EnumerationLiteral(name="CHILD"),
        },
    )

    # Classes
    Book = Class(name="Book")
    Author = Class(name="Author")
    Library = Class(name="Library")

    # Book class attributes
    Book_release = Property(name="release", type=DateType)
    Book_title = Property(name="title", type=StringType)
    Book_pages = Property(name="pages", type=IntegerType)
    Book.attributes = {Book_release, Book_title, Book_pages}

    # Author class attributes
    Author_email = Property(name="email", type=StringType)
    Author_member = Property(name="member", type=MemberType)
    Author.attributes = {Author_email, Author_member}

    # Library class attributes
    Library_name = Property(name="name", type=StringType)
    Library_address = Property(name="address", type=StringType)
    Library.attributes = {Library_name, Library_address}

    # Relationships
    has = BinaryAssociation(
        name="Has",
        ends={
            Property(name="Book_end", type=Book, multiplicity=Multiplicity(0, "*")),
            Property(name="Library_end", type=Library, multiplicity=Multiplicity(1, 1)),
        },
    )
    book_author_relation = BinaryAssociation(
        name="BookAuthor_Relation",
        ends={
            Property(name="writtenBy", type=Author, multiplicity=Multiplicity(1, "*")),
            Property(name="Book_end", type=Book, multiplicity=Multiplicity(0, "*")),
        },
    )

    model = DomainModel(
        name="Domain_Model",
        types={Book, Author, Library, MemberType},
        associations={has, book_author_relation},
        generalizations={},
    )
    return model


# ---------------------------------------------------------------------------
# Library / Book / Author + BookType inheritance hierarchy model
# (used by the SQLAlchemy generator tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def library_model_with_inheritance():
    """A Library-Book-Author model extended with a BookType class hierarchy.

    Adds a BookType superclass with Horror, History, and Science subclasses
    (generalizations), plus a BookType-Book association.

    Structure:
        Library   (name: str, address: str)
        Book      (title: str, pages: int, release: date)
        Author    (name: str, email: str)
        BookType  (position: int)
        Horror    (attribute: str)  extends BookType
        History   (attribute: str)  extends BookType
        Science   (attribute: str)  extends BookType

        Library 1 ---Library_Book--- 0..* Book   (locatedIn / has)
        Author  1..* ---Author_Book--- 0..* Book  (writtenBy / publishes)
        BookType 1 ---BookType_Book--- 0..* Book  (book_type / books)

    Model name: Class_Diagram
    """
    # Classes
    Author = Class(name="Author")
    Book = Class(name="Book")
    Library = Class(name="Library")
    BookType = Class(name="BookType")
    Horror = Class(name="Horror")
    History = Class(name="History")
    Science = Class(name="Science")

    # Author class attributes
    Author_name = Property(name="name", type=StringType)
    Author_email = Property(name="email", type=StringType)
    Author.attributes = {Author_name, Author_email}

    # Book class attributes
    Book_pages = Property(name="pages", type=IntegerType)
    Book_title = Property(name="title", type=StringType)
    Book_release = Property(name="release", type=DateType)
    Book.attributes = {Book_release, Book_pages, Book_title}

    # Library class attributes
    Library_name = Property(name="name", type=StringType)
    Library_address = Property(name="address", type=StringType)
    Library.attributes = {Library_address, Library_name}

    # BookType class attributes
    BookType_position = Property(name="position", type=IntegerType)
    BookType.attributes = {BookType_position}

    # Horror class attributes
    Horror_attribute = Property(name="attribute", type=StringType)
    Horror.attributes = {Horror_attribute}

    # History class attributes
    History_attribute = Property(name="attribute", type=StringType)
    History.attributes = {History_attribute}

    # Science class attributes
    Science_attribute = Property(name="attribute", type=StringType)
    Science.attributes = {Science_attribute}

    # Relationships
    Author_Book = BinaryAssociation(
        name="Author_Book",
        ends={
            Property(name="writtenBy", type=Author, multiplicity=Multiplicity(1, 9999)),
            Property(name="publishes", type=Book, multiplicity=Multiplicity(0, 9999)),
        },
    )
    Library_Book = BinaryAssociation(
        name="Library_Book",
        ends={
            Property(name="locatedIn", type=Library, multiplicity=Multiplicity(1, 1)),
            Property(name="has", type=Book, multiplicity=Multiplicity(0, 9999)),
        },
    )
    BookType_Book = BinaryAssociation(
        name="BookType_Book",
        ends={
            Property(name="book_type", type=BookType, multiplicity=Multiplicity(1, 1)),
            Property(name="books", type=Book, multiplicity=Multiplicity(0, 9999)),
        },
    )

    # Generalizations
    gen_History_BookType = Generalization(general=BookType, specific=History)
    gen_Horror_BookType = Generalization(general=BookType, specific=Horror)
    gen_Science_BookType = Generalization(general=BookType, specific=Science)

    # Domain Model
    model = DomainModel(
        name="Class_Diagram",
        types={Author, Book, Library, BookType, Horror, History, Science},
        associations={Author_Book, Library_Book, BookType_Book},
        generalizations={gen_History_BookType, gen_Horror_BookType, gen_Science_BookType},
    )
    return model
