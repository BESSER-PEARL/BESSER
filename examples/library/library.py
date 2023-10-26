from BUML.metamodel.structural.structural import DomainModel, Class, Property, \
    PrimitiveDataType, Multiplicity, BinaryAssociation

# Library attributes definition
library_name: Property = Property(name="name", owner=None, property_type=PrimitiveDataType("str"))
address: Property = Property(name="address", owner=None, property_type=PrimitiveDataType("str"))
# Library class definition
library: Class = Class (name="Library", attributes={library_name, address})

# Book attributes definition
title: Property = Property(name="title", owner=None, property_type=PrimitiveDataType("str"))
pages: Property = Property(name="pages", owner=None, property_type=PrimitiveDataType("int"))
release: Property = Property(name="release", owner=None, property_type=PrimitiveDataType("date"))
# Book class definition
book: Class = Class (name="Book", attributes={title, pages, release})

# Author attributes definition
author_name: Property = Property(name="name", owner=None, property_type=PrimitiveDataType("str"))
email: Property = Property(name="email", owner=None, property_type=PrimitiveDataType("str"))
# Author class definition
author: Class = Class (name="Author", attributes={author_name, email})

# Library-Book association definition
located_in_end: Property = Property(name="locatedIn", owner=None, property_type=library, multiplicity=Multiplicity(1, 1))
has_end: Property = Property(name="has", owner=None, property_type=book, multiplicity=Multiplicity(0, "*"))
lib_book_association: BinaryAssociation = BinaryAssociation(name="lib-book-assoc", ends={located_in_end, has_end})

# Book-Author association definition
publishes_end: Property = Property(name="publishes", owner=None, property_type=book, multiplicity=Multiplicity(0, "*"))
writed_by_end: Property = Property(name="writedBy", owner=None, property_type=author, multiplicity=Multiplicity(1, "*"))
book_author_association: BinaryAssociation = BinaryAssociation(name="book-author-assoc", ends={writed_by_end, publishes_end})

# Domain model definition
library_model : DomainModel = DomainModel(name="Library model", types={library, book, author}, 
                                          associations={lib_book_association, book_author_association})