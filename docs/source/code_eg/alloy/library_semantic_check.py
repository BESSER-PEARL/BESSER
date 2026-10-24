"""Semantic check of a library model with the Alloy Analyzer (see ``examples/semantic_check_example.rst``).

Needs ``JAVA_HOME`` and ``BESSER_ALLOY_JAR`` (see ``generators/object_diagram.rst``).
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
AlloyGenerator(model, output_dir="output").generate(scope=5)

# 3. The semantic check
solver = AlloySolver(model)
result = solver.check_consistency(scope=5)
print(f"{result.name} within {solver.get_scopes()}")

# 4. An object diagram satisfying the invariants (name_ is the object's name, name its attribute)
result, object_models = solver.generate_object_models(scope=3)
for obj in sorted(object_models[0].objects, key=lambda obj: obj.name_):
    values = ", ".join(sorted(f"{slot.attribute.name}={slot.value.value!r}" for slot in obj.slots))
    print(f"{obj.name_}: {obj.classifier.name}({values})")
for link in sorted(object_models[0].links, key=lambda link: link.name):
    ends = sorted(link.connections, key=lambda end: end.association_end.name)
    print(" -- ".join(f"{end.object.name_} ({end.association_end.name})" for end in ends))

# 5. No instance: books cannot have more than 10 and fewer than 5 pages
short = Constraint("short", book, "context Book inv short: self.pages < 5", "OCL")
model.constraints = {enough_pages, two_books, has_written, short}
solver = AlloySolver(model)
result = solver.check_consistency(scope=5)
print(f"{result.name} within {solver.get_scopes()}")

# 6. An unsupported construct: strings have no length in the Alloy translation
titled = Constraint("titled", book, "context Book inv titled: self.title.size() > 0", "OCL")
model.constraints = {enough_pages, two_books, has_written, titled}
try:
    AlloySolver(model).check_consistency(scope=5)
except AlloyTranslationError as error:
    print(error)
