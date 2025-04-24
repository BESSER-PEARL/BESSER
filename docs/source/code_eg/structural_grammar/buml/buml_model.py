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
            EnumerationLiteral(name="ADULT"),
			EnumerationLiteral(name="SENIOR"),
			EnumerationLiteral(name="STUDENT"),
			EnumerationLiteral(name="CHILD")
    }
)

# Classes
Book = Class(name="Book")
Author = Class(name="Author")
Library = Class(name="Library")

# Book class attributes and methods
Book_release: Property = Property(name="release", type=DateType)
Book_title: Property = Property(name="title", type=StringType)
Book_pages: Property = Property(name="pages", type=IntegerType)
Book.attributes={Book_release, Book_title, Book_pages}

# Author class attributes and methods
Author_email: Property = Property(name="email", type=StringType)
Author_member: Property = Property(name="member", type=MemberType)
Author_m_method: Method = Method(name="method", parameters={Parameter(name='sms', type=StringType, default_value='message')})
Author.attributes={Author_email, Author_member}
Author.methods={Author_m_method}

# Library class attributes and methods
Library_name: Property = Property(name="name", type=StringType)
Library_address: Property = Property(name="address", type=StringType)
Library_m_findBook: Method = Method(name="findBook", parameters={Parameter(name='title', type=StringType)}, type=Book)
Library.attributes={Library_name, Library_address}
Library.methods={Library_m_findBook}

# Relationships
Has: BinaryAssociation = BinaryAssociation(
    name="Has",
    ends={
        Property(name="Book_end", type=Book, multiplicity=Multiplicity(0, "*")),
        Property(name="Library_end", type=Library, multiplicity=Multiplicity(1, 1))
    }
)
BookAuthor_Relation: BinaryAssociation = BinaryAssociation(
    name="BookAuthor_Relation",
    ends={
        Property(name="writtenBy", type=Author, multiplicity=Multiplicity(1, "*")),
        Property(name="Book_end", type=Book, multiplicity=Multiplicity(0, "*"))
    }
)

# Domain Model
domain_model = DomainModel(
    name="Generated_Model",
    types={Book, Author, Library, MemberType},
    associations={Has, BookAuthor_Relation},
    generalizations={}
)
