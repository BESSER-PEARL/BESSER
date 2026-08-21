####################
# STRUCTURAL MODEL #
####################

from besser.BUML.metamodel.structural import (
    Class, Property, Method, Parameter,
    BinaryAssociation, Generalization, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, IntegerType, FloatType, BooleanType,
    TimeType, DateType, DateTimeType, TimeDeltaType,
    AnyType, Constraint, AssociationClass, Metadata, MethodImplementationType
)

# Enumerations
Genre: Enumeration = Enumeration(
    name="Genre",
    literals={
            EnumerationLiteral(name="History"),
			EnumerationLiteral(name="Poetry"),
			EnumerationLiteral(name="Romance"),
			EnumerationLiteral(name="Thriller")
    }
)

# Classes
Author = Class(name="Author")
Book = Class(name="Book")

# Author class attributes and methods
Author_birth: Property = Property(name="birth", type=DateType)
Author_name: Property = Property(name="name", type=StringType)
Author.attributes={Author_birth, Author_name}

# Book class attributes and methods
Book_genre: Property = Property(name="genre", type=Genre)
Book_pages: Property = Property(name="pages", type=IntegerType)
Book_price: Property = Property(name="price", type=FloatType)
Book_title: Property = Property(name="title", type=StringType)
Book.attributes={Book_genre, Book_pages, Book_price, Book_title}

# Relationships
written_by: BinaryAssociation = BinaryAssociation(
    name="written_by",
    ends={
        Property(name="author", type=Author, multiplicity=Multiplicity(1, 9999)),
        Property(name="book", type=Book, multiplicity=Multiplicity(0, 9999))
    }
)


# OCL Constraints
Book_inv_1_1: Constraint = Constraint(
    name="Book_inv_1_1",
    context=Book,
    expression="context Book inv Book_inv_1_1: self.pages > 0",
    language="OCL"
)

# Domain Model
domain_model = DomainModel(
    name="Class_Diagram",
    types={Author, Book, Genre},
    associations={written_by},
    constraints={Book_inv_1_1},
    generalizations={},
    metadata=None
)


######################
# PROJECT DEFINITION #
######################

from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural.structural import Metadata

metadata = Metadata(description="Modern workspace project for UML, GUI and quantum modeling.")
project = Project(
    name="Books",
    models=[domain_model],
    owner="Pablo Ponzio",
    metadata=metadata
)
