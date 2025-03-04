from besser.BUML.metamodel.structural import DomainModel, Class, Property, \
    Multiplicity, BinaryAssociation, StringType, IntegerType, DateType
from besser.generators.python_classes import PythonGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.generators.sql import SQLGenerator
from besser.generators.rest_api import RESTAPIGenerator
from besser.generators.rdf import RDFGenerator
from besser.generators.backend import BackendGenerator

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

# Domain model definition
library_model: DomainModel = DomainModel(name="Library_model", types={library, book, author},
                                         associations={lib_book_association, book_author_association})

# Getting the attributes of the Book class
for attribute in book.attributes:
    print(attribute.name)

# Code Generation

python_model = PythonGenerator(model=library_model)
python_model.generate()

sql_alchemy = SQLAlchemyGenerator(model=library_model)
sql_alchemy.generate()

sql = SQLGenerator(model=library_model)
sql.generate()

rest_api = RESTAPIGenerator(model=library_model)
rest_api.generate()

backend = BackendGenerator(model=library_model, http_methods=["GET", "POST", "PUT", "DELETE"], nested_creations=False)
backend.generate()

rdf = RDFGenerator(model=library_model)
rdf.generate()
