"""A starter B-UML object model: a library with its books, as instances of a small domain model.

    besser validate MODEL_FILE --ocl              check both models, and the constraints on the objects
    besser info MODEL_FILE                        what it holds, and the generators that read it
    besser generate python MODEL_FILE -o out      generate code ('besser generators' lists them all)
    besser convert MODEL_FILE -o model.json       open it in https://editor.besser-pearl.org (import the project)

B-UML reference: https://besser.readthedocs.io/en/latest/buml_language.html
"""

import datetime

from besser.BUML.metamodel.object import ObjectModel
from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    Constraint,
    DateType,
    DomainModel,
    IntegerType,
    Multiplicity,
    Property,
    StringType,
)

# The domain model: the classes the objects are instances of
library = Class(name="Library", attributes={
    Property(name="city", type=StringType),
    Property(name="address", type=StringType),
})
book = Class(name="Book", attributes={
    Property(name="title", type=StringType),
    Property(name="pages", type=IntegerType),
    Property(name="release", type=DateType),
})

# An association: each end names the role of a class and how many of it take part
library_books = BinaryAssociation(name="library_books", ends={
    Property(name="library", type=library, multiplicity=Multiplicity(1, 1)),
    Property(name="books", type=book, multiplicity=Multiplicity(0, "*")),
})

# An OCL invariant: 'besser validate --ocl' evaluates it on every object of its context class
book_has_pages = Constraint(
    name="book_has_pages", context=book, expression="context Book inv book_has_pages: self.pages > 0", language="OCL",
)

library_model = DomainModel(
    name="Library",
    types={library, book},
    associations={library_books},
    constraints={book_has_pages},
)

# Objects: calling a class with the object's name starts a builder; attributes() sets the attribute values
city_library = library("city_library").attributes(city="Esch-sur-Alzette", address="1 Main Street").build()
dune = book("dune").attributes(title="Dune", pages=412, release=datetime.date(1965, 8, 1)).build()
emma = book("emma").attributes(title="Emma", pages=474, release=datetime.date(1815, 12, 23)).build()

# Links: assign the objects at the other end of an association to the name of that end
city_library.books = {dune, emma}

# The object model. It is read together with the domain model its objects are instances of.
library_objects = ObjectModel(name="LibraryObjects", objects={city_library, dune, emma})
