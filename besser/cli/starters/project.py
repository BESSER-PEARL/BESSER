"""A starter B-UML project: a library as a domain model, an object model of it and a state machine.

    besser validate MODEL_FILE --ocl              check every model, and the constraints on the objects
    besser info MODEL_FILE                        what it holds, and the generators that read it
    besser generate python MODEL_FILE -o out      generate code ('besser generators' lists them all)
    besser convert MODEL_FILE -o model.json       open it in https://editor.besser-pearl.org (import the project)

B-UML reference: https://besser.readthedocs.io/en/latest/buml_language.html
"""

from besser.BUML.metamodel.object import ObjectModel
from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.state_machine import (
    Body,
    Event,
    Session,
    StateMachine,
)
from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    Constraint,
    DomainModel,
    IntegerType,
    Metadata,
    Multiplicity,
    Property,
    StringType,
)

# 1. The domain model: classes, an association between them and an OCL invariant
library = Class(name="Library", attributes={Property(name="city", type=StringType)})
book = Class(name="Book", attributes={
    Property(name="title", type=StringType),
    Property(name="pages", type=IntegerType),
})
library_books = BinaryAssociation(name="library_books", ends={
    Property(name="library", type=library, multiplicity=Multiplicity(1, 1)),
    Property(name="books", type=book, multiplicity=Multiplicity(0, "*")),
})
book_has_pages = Constraint(
    name="book_has_pages", context=book, expression="context Book inv book_has_pages: self.pages > 0", language="OCL",
)
library_model = DomainModel(
    name="Library", types={library, book}, associations={library_books}, constraints={book_has_pages},
)

# 2. An object model: instances of the classes above, with attribute values and a link
city_library = library("city_library").attributes(city="Esch-sur-Alzette").build()
dune = book("dune").attributes(title="Dune", pages=412).build()
city_library.books = {dune}
library_objects = ObjectModel(name="LibraryObjects", objects={city_library, dune})

# 3. A state machine: the loan of a book
book_loan = StateMachine(name="BookLoan")
available = book_loan.new_state(name="available", initial=True)
on_loan = book_loan.new_state(name="on_loan")
overdue = book_loan.new_state(name="overdue")


# The body of a state: the function it runs when the machine enters it
def overdue_body(session: Session):
    session.set("reminder_sent", True)


overdue.set_body(Body(name="overdue_body", callable=overdue_body))

# Transitions: from a state, on an event, to another state
book_returned = Event(name="book_returned")
available.when_event(Event(name="book_borrowed")).go_to(on_loan)
on_loan.when_event(book_returned).go_to(available)
on_loan.when_event(Event(name="due_date_passed")).go_to(overdue)
overdue.when_event(book_returned).go_to(available)

# The project: the models that belong together. When a file defines one, the besser command reads only its models.
library_project = Project(
    name="LibraryProject",
    models=[library_model, library_objects, book_loan],
    owner="Library team",
    metadata=Metadata(description="A library, two of its objects and the loan of a book"),
)
