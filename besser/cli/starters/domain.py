"""A starter B-UML model: a library, its books and their authors.

    besser validate MODEL_FILE                    check the model
    besser info MODEL_FILE                        what it holds, and the generators that read it
    besser generate python MODEL_FILE -o out      generate code ('besser generators' lists them all)
    besser convert MODEL_FILE -o model.json       open it in https://editor.besser-pearl.org (import the project)

B-UML reference: https://besser.readthedocs.io/en/latest/buml_language.html
"""

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DateType,
    DomainModel,
    IntegerType,
    Multiplicity,
    Property,
    StringType,
)

# Classes and their attributes
library = Class(name="Library", attributes={
    Property(name="name", type=StringType),
    Property(name="address", type=StringType),
})
book = Class(name="Book", attributes={
    Property(name="title", type=StringType),
    Property(name="pages", type=IntegerType),
    Property(name="release", type=DateType),
})
author = Class(name="Author", attributes={
    Property(name="name", type=StringType),
    Property(name="email", type=StringType),
})

# Associations: each end names the role of a class and how many of it take part
library_books = BinaryAssociation(name="library_books", ends={
    Property(name="library", type=library, multiplicity=Multiplicity(1, 1)),
    Property(name="books", type=book, multiplicity=Multiplicity(0, "*")),
})
book_authors = BinaryAssociation(name="book_authors", ends={
    Property(name="books", type=book, multiplicity=Multiplicity(0, "*")),
    Property(name="authors", type=author, multiplicity=Multiplicity(1, "*")),
})

# The model. The besser command reads every model bound to a top-level name of this file.
library_model = DomainModel(
    name="Library",
    types={library, book, author},
    associations={library_books, book_authors},
)
