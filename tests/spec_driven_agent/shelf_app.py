"""The Book/Library FastAPI app the #659 regeneration tests generate and change."""
import json

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, IntegerType, Method, MethodImplementationType,
    Multiplicity, Property, StringType,
)
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor

GENERATOR = "generate_fastapi_backend"


def shelf_model(isbn=False, author=False, library="Library", library_method=False) -> DomainModel:
    """Book (+ the library class, named ``library``; None for none) as Phase 1 sees it.

    Book.library is 0..1, so the runtime tests can create a book without a library.
    ``library_method`` gives the library class a method, so the generator also
    writes ``routers/<library>_methods.py``.
    """
    book = Class(name="Book", attributes={Property(name="title", type=StringType),
                                          Property(name="pages", type=IntegerType)})
    if isbn:
        book.attributes = book.attributes | {Property(name="isbn", type=StringType)}
    types, associations = {book}, set()
    if library:
        lib = Class(name=library, attributes={Property(name="name", type=StringType)})
        if library_method:
            opening = Method(name="open", parameters=[], type=None, implementation_type=MethodImplementationType.BAL)
            opening.code = "def open() -> nothing {\n    this.name = this.name\n}"
            lib.add_method(opening)
        types.add(lib)
        associations.add(BinaryAssociation(name="Has", ends={
            Property(name="books", type=book, multiplicity=Multiplicity(0, "*")),
            Property(name="library", type=lib, multiplicity=Multiplicity(0, 1))}))
    if author:
        writer = Class(name="Author", attributes={Property(name="fullName", type=StringType)})
        types.add(writer)
        associations.add(BinaryAssociation(name="Wrote", ends={
            Property(name="works", type=book, multiplicity=Multiplicity(0, "*")),
            Property(name="writer", type=writer, multiplicity=Multiplicity(0, 1))}))
    return DomainModel(name="Shelf", types=types, associations=associations)


def generate_backend(workspace, model) -> None:
    """Run the backend generator into ``workspace`` the way Phase 1 does."""
    reply = json.loads(ToolExecutor(str(workspace), domain_model=model).execute(GENERATOR, {}))
    assert not reply.get("error"), reply
