import os

from besser.BUML.metamodel.structural import DomainModel, Class, Property, \
    Multiplicity, BinaryAssociation, StringType, IntegerType, DateType
from besser.BUML.metamodel.structural.structural import Constraint
from besser.generators.alloy.instance_generator.alloy_analyzer_executor import AlloyResult
from besser.generators.alloy.instance_generator import AlloySolver


# Library attributes definition
library_name: Property = Property(name="name", type=StringType)
address: Property = Property(name="address", type=StringType)
# Library class definition
library: Class = Class(name="Library", attributes={library_name, address})

# Book attributes definition
title: Property = Property(name="title", type=StringType)
pages: Property = Property(name="pages", type=IntegerType)
release: Property = Property(name="release", type=DateType)
# Book class definition
book: Class = Class(name="Book", attributes={title, pages, release})

# Author attributes definition
author_name: Property = Property(name="name", type=StringType)
email: Property = Property(name="email", type=StringType)
# Author class definition
author: Class = Class(name="Author", attributes={author_name, email})

# Library-Book association definition
located_in: Property = Property(name="locatedIn", type=library, multiplicity=Multiplicity(1, 1))
has: Property = Property(name="has", type=book, multiplicity=Multiplicity(0, "*"))
lib_book_association: BinaryAssociation = BinaryAssociation(name="lib_book_assoc", ends={located_in, has})

# Book-Author association definition
publishes: Property = Property(name="publishes", type=book, multiplicity=Multiplicity(0, "*"))
written_by: Property = Property(name="writtenBy", type=author, multiplicity=Multiplicity(1, "*"))
book_author_association: BinaryAssociation = BinaryAssociation(name="book_author_assoc", ends={written_by, publishes})


book_has_title: Constraint = Constraint(
    name="book_has_title",
    context=book,
    expression="context Book inv book_has_title: self.title.size() > 0",
    language="OCL"
)
library_named: Constraint = Constraint(
    name="library_named",
    context=library,
    expression="context Library inv library_named: self.name.size() > 0",
    language="OCL"
)
author_named: Constraint = Constraint(
    name="author_named",
    context=author,
    expression="context Author inv author_named: self.name.size() > 0",
    language="OCL"
)

# Domain model definition
library_model: DomainModel = DomainModel(name="Library_model", types={library, book, author},
                                         associations={lib_book_association, book_author_association},
                                         constraints={book_has_title, library_named, author_named})


# Semantic consistency check
solver = AlloySolver(library_model, output_dir="outdir", scope=3)
result = solver.check_consistency()
assert result == AlloyResult.SAT, "The model is not consistent."

# Generate four BUML object diagrams using Alloy
solver = AlloySolver(library_model, output_dir="outdir", scope=3)
(res, buml_instances) = solver.generate_object_diagrams(num_instances=4)
assert res == AlloyResult.SAT, "The model is not consistent."
assert len(buml_instances) == 4, "The number of generated instances is not correct."

# Generates a complete BUML project code including: The class diagram in BUML, and an instance 
# automatically generated using Alloy.
solver = AlloySolver(library_model, output_dir="ouotro_dir", scope=3)
solver.generate_class_and_object_model()
assert result == AlloyResult.SAT, "The model is not consistent."