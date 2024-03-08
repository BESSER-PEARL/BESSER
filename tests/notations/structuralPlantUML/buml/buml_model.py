from besser.BUML.metamodel.structural import NamedElement, DomainModel, Type, Class, \
        Property, PrimitiveDataType, Multiplicity, Association, BinaryAssociation, Generalization, \
        GeneralizationSet, AssociationClass 

# Primitive Data Types 
date_type = PrimitiveDataType("date")
int_type = PrimitiveDataType("int")
str_type = PrimitiveDataType("str")

# Library class definition 
Library_name: Property = Property(name="name", property_type=str_type, visibility="public")
Library_address: Property = Property(name="address", property_type=str_type, visibility="public")
Library: Class = Class(name="Library", attributes={Library_name, Library_address})

# Book class definition 
Book_title: Property = Property(name="title", property_type=str_type, visibility="public")
Book_pages: Property = Property(name="pages", property_type=int_type, visibility="public")
Book_release: Property = Property(name="release", property_type=date_type, visibility="public")
Book: Class = Class(name="Book", attributes={Book_title, Book_pages, Book_release})

# Author class definition 
Author_name: Property = Property(name="name", property_type=str_type, visibility="public")
Author_email: Property = Property(name="email", property_type=str_type, visibility="public")
Author: Class = Class(name="Author", attributes={Author_name, Author_email})

# Relationships
writedBy: BinaryAssociation = BinaryAssociation(name="writedBy", ends={
        Property(name="writedBy", property_type=Book, multiplicity=Multiplicity(0, "*")),
        Property(name="writedBy", property_type=Author, multiplicity=Multiplicity(1, "*"))})
has: BinaryAssociation = BinaryAssociation(name="has", ends={
        Property(name="has", property_type=Library, multiplicity=Multiplicity(1, 1)),
        Property(name="has", property_type=Book, multiplicity=Multiplicity(0, "*"))})


# Domain Model
domain: DomainModel = DomainModel(name="Domain Model", types={Library, Book, Author}, associations={writedBy, has}, generalizations=set())