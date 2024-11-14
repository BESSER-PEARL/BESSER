# Generated B-UML Model
from besser.BUML.metamodel.structural import (
    Class, Property, Method, Parameter,
    BinaryAssociation, Generalization, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, IntegerType, FloatType, BooleanType,
    TimeType, DateType, DateTimeType, TimeDeltaType
)

# Enumerations
MemberType: Enumeration = Enumeration(
    name="MemberType",
    literals={
            EnumerationLiteral(name="SENIOR"),
			EnumerationLiteral(name="ADULT"),
			EnumerationLiteral(name="CHILD"),
			EnumerationLiteral(name="STUDENT")
    }
)

# Classes
Book = Class(name="Book")
Author = Class(name="Author")
Library = Class(name="Library")

# Book class attributes and methods
Book_pages: Property = Property(name="pages", type=IntegerType)
Book_title: Property = Property(name="title", type=StringType)
Book_release: Property = Property(name="release", type=DateType)
Book.attributes={Book_pages, Book_title, Book_release}

# Author class attributes and methods
Author_email: Property = Property(name="email", type=StringType)
Author_member: Property = Property(name="member", type=MemberType)
Author_m_method: Method = Method(name="method", parameters={})
Author.attributes={Author_email, Author_member}
Author.methods={Author_m_method}

# Library class attributes and methods
Library_address: Property = Property(name="address", type=StringType)
Library_name: Property = Property(name="name", type=StringType)
Library_m_findBook: Method = Method(name="findBook", parameters={}, type=Book)
Library.attributes={Library_address, Library_name}
Library.methods={Library_m_findBook}

# Relationships
BookAuthor_Relation: BinaryAssociation = BinaryAssociation(
    name="BookAuthor_Relation",
    ends={
        Property(name="writtenBy", type=Author, multiplicity=Multiplicity(1, 1)),        Property(name="Book_end", type=Book, multiplicity=Multiplicity(1, 1))
    }
)
Has: BinaryAssociation = BinaryAssociation(
    name="Has",
    ends={
        Property(name="Book_end", type=Book, multiplicity=Multiplicity(1, 1)),        Property(name="Library_end", type=Library, multiplicity=Multiplicity(1, 1))
    }
)

# Domain Model
domain_model = DomainModel(
    name="Generated Model",
    types={Book, Author, Library, MemberType},
    associations={BookAuthor_Relation, Has},
    generalizations={}
)
