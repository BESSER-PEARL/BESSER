from besser.BUML.metamodel.structural import *  

# Primitive Data Types
str_type = PrimitiveDataType("str")
int_type = PrimitiveDataType("int")
date_type = PrimitiveDataType("date")

# Enumerations
MemberType = Enumeration(name="MemberType", literals = {
			EnumerationLiteral(name="CHILD"),
			EnumerationLiteral(name="ADULT"),
			EnumerationLiteral(name="SENIOR"),
			EnumerationLiteral(name="STUDENT")})

# Classes
Library: Class = Class(name="Library")
Book: Class = Class(name="Book")
Author: Class = Class(name="Author")

# Library class attributes and methods
Library_name: Property = Property(name="name", type=str_type, visibility="public")
Library_address: Property = Property(name="address", type=str_type, visibility="public")
Library_m_findBook: Method = Method(name="findBook", visibility="public", parameters={Parameter(name="title", type=str_type)}, type=Book)
Library.attributes={Library_name, Library_address}
Library.methods={Library_m_findBook}

# Book class attributes and methods
Book_title: Property = Property(name="title", type=str_type, visibility="public")
Book_pages: Property = Property(name="pages", type=int_type, visibility="public")
Book_release: Property = Property(name="release", type=date_type, visibility="public")
Book.attributes={Book_title, Book_pages, Book_release}

# Author class attributes and methods
Author_name: Property = Property(name="name", type=str_type, visibility="public")
Author_email: Property = Property(name="email", type=str_type, visibility="public")
Author_member: Property = Property(name="member", type=MemberType, visibility="public")
Author_m_notify: Method = Method(name="notify", visibility="public", parameters={Parameter(name="sms", type=str_type, default_value="hello")}, type=None)
Author.attributes={Author_name, Author_email, Author_member}
Author.methods={Author_m_notify}

# Relationships
writtenBy: BinaryAssociation = BinaryAssociation(name="writtenBy", ends={
        Property(name="writtenBy", type=Book, multiplicity=Multiplicity(0, "*")),
        Property(name="writtenBy", type=Author, multiplicity=Multiplicity(1, "*"))})
has: BinaryAssociation = BinaryAssociation(name="has", ends={
        Property(name="has", type=Library, multiplicity=Multiplicity(1, 1)),
        Property(name="has", type=Book, multiplicity=Multiplicity(0, "*"))})


# Domain Model
domain: DomainModel = DomainModel(
				name="Domain Model",
				types={Library, Book, Author},
				associations={writtenBy, has},
				generalizations=set(),
				enumerations={MemberType}
				)
