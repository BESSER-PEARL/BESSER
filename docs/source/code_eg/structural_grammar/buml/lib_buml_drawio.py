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
            EnumerationLiteral(name="STUDENT"),
			EnumerationLiteral(name="SENIOR"),
			EnumerationLiteral(name="ADULT"),
			EnumerationLiteral(name="CHILD")
    }
)

# Classes
Author = Class(name="Author")
Book = Class(name="Book")
Library = Class(name="Library")

# Author class attributes and methods
Author_member: Property = Property(name="member", type=MemberType)
Author_email: Property = Property(name="email", type=StringType)
Author_m_method: Method = Method(name="method", parameters={'sms': Parameter(name='sms', type=StringType, default_value='message')})
Author.attributes={Author_member, Author_email}
Author.methods={Author_m_method}

# Book class attributes and methods
Book_pages: Property = Property(name="pages", type=IntegerType)
Book_title: Property = Property(name="title", type=StringType)
Book_release: Property = Property(name="release", type=DateType)
Book.attributes={Book_pages, Book_title, Book_release}

# Library class attributes and methods
Library_address: Property = Property(name="address", type=StringType)
Library_name: Property = Property(name="name", type=StringType)
Library_m_findBook: Method = Method(name="findBook", parameters={'title': Parameter(name='title', type=StringType)}, type=Book)
Library.attributes={Library_address, Library_name}
Library.methods={Library_m_findBook}

# Relationships
Has: BinaryAssociation = BinaryAssociation(
    name="Has",
    ends={
        Property(name="Library_end", type=Library, multiplicity=Multiplicity(1, 1)),        Property(name="Book_end", type=Book, multiplicity=Multiplicity(1, 1))
    }
)
BookAuthor_Relation: BinaryAssociation = BinaryAssociation(
    name="BookAuthor_Relation",
    ends={
        Property(name="Book_end", type=Book, multiplicity=Multiplicity(1, 1)),        Property(name="writtenBy", type=Author, multiplicity=Multiplicity(1, 1))
    }
)

# Domain Model
domain_model = DomainModel(
    name="Generated Model",
    types={Author, Book, Library, MemberType},
    associations={Has, BookAuthor_Relation},
    generalizations={}
)
