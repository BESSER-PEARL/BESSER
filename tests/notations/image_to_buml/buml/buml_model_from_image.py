from besser.BUML.metamodel.structural import NamedElement, DomainModel, Type, Class, \
        Property, PrimitiveDataType, Multiplicity, Association, BinaryAssociation, Generalization, \
        GeneralizationSet, AssociationClass 

# Primitive Data Types 
date_type = PrimitiveDataType("date")
int_type = PrimitiveDataType("int")
str_type = PrimitiveDataType("str")

# Library class definition 
Library_name: Property = Property(name="name", type=str_type)
Library_address: Property = Property(name="address", type=str_type)
Library: Class = Class(name="Library", attributes={Library_name, Library_address})

# Author class definition 
Author_name: Property = Property(name="name", type=str_type)
Author_email: Property = Property(name="email", type=str_type)
Author: Class = Class(name="Author", attributes={Author_name, Author_email})

# Book class definition 
Book_title: Property = Property(name="title", type=str_type)
Book_pages: Property = Property(name="pages", type=int_type)
Book_release: Property = Property(name="release", type=date_type)
Book: Class = Class(name="Book", attributes={Book_title, Book_pages, Book_release})

# Relationships
has: BinaryAssociation = BinaryAssociation(name="has", ends={
        Property(name="has", type=Library, multiplicity=Multiplicity(1, 1)),
        Property(name="has", type=Book, multiplicity=Multiplicity(0, "*"))})
writtenBy: BinaryAssociation = BinaryAssociation(name="writtenBy", ends={
        Property(name="writtenBy", type=Book, multiplicity=Multiplicity(0, "*")),
        Property(name="writtenBy", type=Author, multiplicity=Multiplicity(1, "*"))})


# Domain Model
domain: DomainModel = DomainModel(name="Domain Model", types={Library, Author, Book}, associations={has, writtenBy}, generalizations=set())