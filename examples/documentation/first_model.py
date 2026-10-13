"""The complete example used by the Python getting-started tutorial."""

from besser.BUML.metamodel.structural.structural import (
    Class,
    DomainModel,
    IntegerType,
    Property,
    StringType,
)
from besser.generators.python_classes import PythonGenerator

book = Class(
    name="Book",
    attributes={
        Property(name="title", type=StringType),
        Property(name="pages", type=IntegerType),
    },
)
library = DomainModel(name="Library", types={book}, associations=set())

PythonGenerator(model=library, output_dir="output").generate()
