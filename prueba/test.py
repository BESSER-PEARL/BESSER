# Generated B-UML Model
from besser.BUML.metamodel.structural import (
    Class, Property, Method, Parameter,
    BinaryAssociation, Generalization, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, IntegerType, FloatType, BooleanType,
    TimeType, DateType, DateTimeType, TimeDeltaType
)

# Enumerations
ContactM = Enumeration(name="ContactM", literals={EnumerationLiteral(name="EMAIL"), EnumerationLiteral(name="LETTER"), EnumerationLiteral(name="TELEPHONE")})

# Classes
Library: Class = Class(name="Library")
Platform: Class = Class(name="Platform")
Science: Class = Class(name="Science")
Book: Class = Class(name="Book")
Literature: Class = Class(name="Literature")
Fantasy: Class = Class(name="Fantasy")
Author: Class = Class(name="Author")

# Library class attributes and methods
Library_address: Property = Property(name="address", type=StringType, visibility="public")
Library_name: Property = Property(name="name", type=StringType, visibility="public")
Library.attributes={Library_address, Library_name}

# Book class attributes and methods
Book_title: Property = Property(name="title", type=StringType, visibility="public")
Book_pages: Property = Property(name="pages", type=IntegerType, visibility="public")
Book_edition: Property = Property(name="edition", type=DateType, visibility="public")
Book.attributes={Book_title, Book_pages, Book_edition}

# Author class attributes and methods
Author_name: Property = Property(name="name", type=StringType, visibility="public")
Author_email: Property = Property(name="email", type=StringType, visibility="public")
Author_m_func: Method = Method(name="func", visibility="public", parameters={}, type=StringType)
Author_m_notify: Method = Method(name="notify", visibility="public", parameters={Parameter(name="contact_method", type=ContactM), Parameter(name="sms", type=StringType, default_value="message")})
Author.attributes={Author_name, Author_email}
Author.methods={Author_m_func, Author_m_notify}

# Relationships
has: BinaryAssociation = BinaryAssociation(name="has", ends={Property(name="Library", type=Library, multiplicity=Multiplicity(1, 1)), Property(name="Book", type=Book, multiplicity=Multiplicity(1, 1))})
writtenBy: BinaryAssociation = BinaryAssociation(name="writtenBy", ends={Property(name="Author", type=Author, multiplicity=Multiplicity(1, 9999)), Property(name="Book", type=Book, multiplicity=Multiplicity(0, 9999))})

# Generalizations
gen_Science_Book = Generalization(general=Book, specific=Science)
gen_Fantasy_Book = Generalization(general=Book, specific=Fantasy)
gen_Literature_Book = Generalization(general=Book, specific=Literature)

# Domain Model
domain_model = DomainModel(
    name="Generated Model",
    types={Platform, Science, Author, ContactM, Library, Book, Literature, Fantasy},
    associations={has, writtenBy},
    generalizations={gen_Science_Book, gen_Fantasy_Book, gen_Literature_Book}
)
