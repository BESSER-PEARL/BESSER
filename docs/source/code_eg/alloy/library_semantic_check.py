"""Semantic check of a library model with the Alloy Analyzer (see ``examples/semantic_check_example.rst``).

Needs ``JAVA_HOME`` and ``BESSER_ALLOY_JAR`` (see ``utilities/object_diagram_generator.rst``).
"""

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, Constraint, DomainModel, IntegerType, Multiplicity, Property, StringType,
)
from besser.generators.alloy import AlloyGenerator, AlloyTranslationError
from besser.utilities.web_modeling_editor.backend.services.validators.alloy import AlloySolver

# 1. The model: libraries with books, written by authors
library = Class("Library", attributes={Property("name", StringType)})
book = Class("Book", attributes={Property("title", StringType), Property("pages", IntegerType)})
author = Class("Author", attributes={Property("name", StringType)})
located_in = BinaryAssociation("located_in", ends={
    Property("library", library, multiplicity=Multiplicity(1, 1)),
    Property("books", book, multiplicity=Multiplicity(0, "*")),
})
written_by = BinaryAssociation("written_by", ends={
    Property("writes", book, multiplicity=Multiplicity(0, "*")),
    Property("authors", author, multiplicity=Multiplicity(1, "*")),
})

# Its OCL invariants
enough_pages = Constraint("enough_pages", book, "context Book inv enough_pages: self.pages > 10", "OCL")
two_books = Constraint("two_books", library, "context Library inv two_books: self.books->size() >= 2", "OCL")
has_written = Constraint(
    "has_written", author, "context Author inv has_written: self.writes->size() >= 1", "OCL"
)
model = DomainModel(
    "Library", types={library, book, author}, associations={located_in, written_by},
    constraints={enough_pages, two_books, has_written},
)

# 2. The Alloy specification, written to output/model.als
generator = AlloyGenerator(model, output_dir="output")
generator.generate(scope=5)

# 3. The semantic check
solver = AlloySolver(model)
result = solver.check_consistency(scope=5)
print(f"{result.name} within {solver.get_scopes()}")

# 4. An object diagram satisfying the invariants, written to output/alloy_project.py
solver = AlloySolver(model, output_dir="output")
result = solver.generate_project_file(scope=5)
print(result.name)

# 5. No instance: books cannot have more than 10 and fewer than 5 pages
short = Constraint("short", book, "context Book inv short: self.pages < 5", "OCL")
model.constraints = {enough_pages, two_books, has_written, short}
solver = AlloySolver(model)
result = solver.check_consistency(scope=5)
print(f"{result.name} within {solver.get_scopes()}")

# 6. An unsupported construct: strings have no length in the Alloy translation
titled = Constraint("titled", book, "context Book inv titled: self.title.size() > 0", "OCL")
model.constraints = {enough_pages, two_books, has_written, titled}
solver = AlloySolver(model)
try:
    solver.check_consistency(scope=5)
except AlloyTranslationError as error:
    print(error)
