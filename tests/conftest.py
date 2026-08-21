"""
Shared test fixtures for the BESSER test suite.

This module provides commonly used domain model fixtures that are
duplicated across many test files. Using centralized fixtures avoids
copy-paste and ensures consistent test data.

Fixtures defined here are automatically available to all tests under
the ``tests/`` directory without explicit imports.
"""

import pytest
from besser.BUML.metamodel.structural import (
    Class, DomainModel, Property, BinaryAssociation, Multiplicity,
    Enumeration, EnumerationLiteral,
    StringType, IntegerType, FloatType, BooleanType, DateType,
)


# ---------------------------------------------------------------------------
# Library / Book / Author domain model (the most commonly duplicated fixture)
# ---------------------------------------------------------------------------

@pytest.fixture
def library_book_author_model():
    """A Library-Book-Author domain model with two associations.

    This is the canonical "library" model used across many generator tests
    (SQL, JSON Schema, RDF, SQLAlchemy, etc.).

    Structure:
        Library  (name: str, address: str)
        Book     (title: str, pages: int, release: date)
        Author   (name: str, email: str)

        Library 1 ---has--- 0..* Book       (locatedIn / has)
        Author  1..* ---writtenBy--- 0..* Book  (writtenBy / publishes)
    """
    # Classes
    library = Class(name="Library")
    book = Class(name="Book")
    author = Class(name="Author")

    # Library attributes
    library_name = Property(name="name", type=StringType)
    library_address = Property(name="address", type=StringType)
    library.attributes = {library_name, library_address}

    # Book attributes
    book_title = Property(name="title", type=StringType)
    book_pages = Property(name="pages", type=IntegerType)
    book_release = Property(name="release", type=DateType)
    book.attributes = {book_title, book_pages, book_release}

    # Author attributes
    author_name = Property(name="name", type=StringType)
    author_email = Property(name="email", type=StringType)
    author.attributes = {author_name, author_email}

    # Library-Book association
    located_in = Property(name="locatedIn", type=library, multiplicity=Multiplicity(1, 1))
    has = Property(name="has", type=book, multiplicity=Multiplicity(0, "*"))
    lib_book_assoc = BinaryAssociation(name="lib_book_assoc", ends={located_in, has})

    # Book-Author association
    written_by = Property(name="writtenBy", type=author, multiplicity=Multiplicity(1, "*"))
    publishes = Property(name="publishes", type=book, multiplicity=Multiplicity(0, "*"))
    book_author_assoc = BinaryAssociation(name="book_author", ends={written_by, publishes})

    model = DomainModel(
        name="Library_Model",
        types={library, book, author},
        associations={lib_book_assoc, book_author_assoc},
    )
    return model


# ---------------------------------------------------------------------------
# Employee self-association model
# ---------------------------------------------------------------------------

@pytest.fixture
def employee_self_assoc_model():
    """An Employee model with a self-referential manages/subordinates association.

    This is the most common self-association test model, duplicated in
    java, pydantic, rest_api, and django test files.

    Structure:
        Employee (name: str)

        Employee 0..1 ---manager--- 0..* ---subordinates--- Employee
    """
    employee = Class(name="Employee")

    emp_name = Property(name="name", type=StringType)
    employee.attributes = {emp_name}

    manages = BinaryAssociation(
        name="Manages",
        ends={
            Property(name="manager", type=employee, multiplicity=Multiplicity(0, 1)),
            Property(name="subordinates", type=employee, multiplicity=Multiplicity(0, "*")),
        },
    )

    model = DomainModel(
        name="Employee_Model",
        types={employee},
        associations={manages},
        generalizations={},
    )
    return model


# ---------------------------------------------------------------------------
# Simple Library-Book model (no Author, minimal attributes)
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_library_book_model():
    """A minimal Library-Book model without Author.

    Used in pydantic self-assoc and rest_api self-assoc tests as the
    "normal association" counterpart to the self-association models.

    Structure:
        Library (name: str)
        Book    (title: str, pages: int)

        Library 1 ---Has--- 0..* Book
    """
    library = Class(name="Library")
    book = Class(name="Book")

    lib_name = Property(name="name", type=StringType)
    library.attributes = {lib_name}

    book_title = Property(name="title", type=StringType)
    book_pages = Property(name="pages", type=IntegerType)
    book.attributes = {book_title, book_pages}

    has = BinaryAssociation(
        name="Has",
        ends={
            Property(name="books", type=book, multiplicity=Multiplicity(0, "*")),
            Property(name="library", type=library, multiplicity=Multiplicity(1, 1)),
        },
    )

    model = DomainModel(
        name="Library_Model",
        types={library, book},
        associations={has},
        generalizations={},
    )
    return model


# ---------------------------------------------------------------------------
# Player / Team fixtures (used in OCL-related tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def player_class():
    """A Player class with typical attributes for OCL constraint testing."""
    return Class(name="Player", attributes={
        Property(name="age", type=IntegerType),
        Property(name="name", type=StringType),
        Property(name="salary", type=FloatType),
        Property(name="active", type=BooleanType),
        Property(name="jerseyNumber", type=IntegerType),
    })


@pytest.fixture
def team_class():
    """A Team class for OCL constraint testing."""
    return Class(name="Team", attributes={
        Property(name="name", type=StringType),
        Property(name="city", type=StringType),
    })


@pytest.fixture
def player_team_domain_model(player_class, team_class):
    """A domain model containing Player and Team classes.

    This is the base fixture used in ``test_ocl_utils.py`` and
    ``test_ocl_pydantic_integration.py``.
    """
    return DomainModel(
        name="TestModel",
        types={player_class, team_class},
    )
