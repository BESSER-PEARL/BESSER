from besser.BUML.metamodel.structural import NamedElement, DomainModel, Type, Class, \
        Property, PrimitiveDataType, Multiplicity, Association, BinaryAssociation, Generalization, \
        GeneralizationSet, AssociationClass 

# Primitive Data Types 
date_type = PrimitiveDataType("date")
str_type = PrimitiveDataType("str")
int_type = PrimitiveDataType("int")

# Library class definition 
Library_name: Property = Property(name="name", property_type=str_type)
Library_address: Property = Property(name="address", property_type=str_type)
Library: Class = Class(name="Library", attributes={Library_name, Library_address})

# Book class definition 
Book_title: Property = Property(name="title", property_type=str_type)
Book_pages: Property = Property(name="pages", property_type=int_type)
Book_release: Property = Property(name="release", property_type=date_type)
Book: Class = Class(name="Book", attributes={Book_title, Book_pages, Book_release})

# Author class definition 
Author_name: Property = Property(name="name", property_type=str_type)
Author_email: Property = Property(name="email", property_type=str_type)
Author: Class = Class(name="Author", attributes={Author_name, Author_email})

# Relationships
has: BinaryAssociation = BinaryAssociation(name="has", ends={
        Property(name="has", property_type=Library, multiplicity=Multiplicity(1, 1)),
        Property(name="has", property_type=Book, multiplicity=Multiplicity(0, "*"))})
writtenBy: BinaryAssociation = BinaryAssociation(name="writtenBy", ends={
        Property(name="writtenBy", property_type=Author, multiplicity=Multiplicity(1, "*")),
        Property(name="writtenBy", property_type=Book, multiplicity=Multiplicity(0, "*"))})


# Domain Model
domain: DomainModel = DomainModel(name="Domain Model", types={Library, Book, Author}, associations={has, writtenBy}, generalizations=set())