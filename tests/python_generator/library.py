from besser.BUML.metamodel.structural import DomainModel, Class, Property, \
    PrimitiveDataType, Multiplicity, BinaryAssociation
from besser.generators.python_classes import PythonGenerator
from besser.generators.test_gen_python import PythonTestCaseGenerator

# Primitive DataTypes
t_int: PrimitiveDataType = PrimitiveDataType("int")
t_str: PrimitiveDataType = PrimitiveDataType("str")
t_datetime: PrimitiveDataType = PrimitiveDataType("datetime")

# Library attributes definition
library_name: Property = Property(name="name", type=t_str)
address: Property = Property(name="address", type=t_str)
# Library class definition
library: Class = Class(name="Library", attributes={library_name, address})

# Book attributes definition
title: Property = Property(name="title", type=t_str)
pages: Property = Property(name="pages", type=t_int)
release: Property = Property(name="release", type=t_datetime)
# Book class definition
book: Class = Class(name="Book", attributes={title, pages, release})

# Author attributes definition
author_name: Property = Property(name="name", type=t_str)
email: Property = Property(name="email", type=t_str)
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

# Domain model definition
library_model: DomainModel = DomainModel(name="Library model", types={library, book, author},
                                         associations={lib_book_association, book_author_association})

# Getting the attributes of the Book class
for attribute in book.attributes:
    print(attribute.name)

# Code Generation

python_model = PythonGenerator(model=library_model)
python_model.generate()


test_case_model =PythonTestCaseGenerator(model = library_model)
test_case_model.generate()