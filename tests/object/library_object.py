from besser.BUML.metamodel.structural import DomainModel, Class, Property, \
    PrimitiveDataType, Multiplicity, BinaryAssociation
from besser.BUML.metamodel.object import *
import datetime

#############################################
#   Library - structural model definition   #
#############################################

# Primitive DataTypes
t_int: PrimitiveDataType = PrimitiveDataType("int")
t_str: PrimitiveDataType = PrimitiveDataType("str")
t_date: PrimitiveDataType = PrimitiveDataType("date")

# Library attributes definition
library_name: Property = Property(name="name", property_type=t_str)
address: Property = Property(name="address", property_type=t_str)
# Library class definition
library: Class = Class (name="Library", attributes={library_name, address})

# Book attributes definition
title: Property = Property(name="title", property_type=t_str)
pages: Property = Property(name="pages", property_type=t_int)
release: Property = Property(name="release", property_type=t_date)
# Book class definition
book: Class = Class (name="Book", attributes={title, pages, release})

# Author attributes definition
author_name: Property = Property(name="name", property_type=t_str)
email: Property = Property(name="email", property_type=t_str)
# Author class definition
author: Class = Class (name="Author", attributes={author_name, email})

# Library-Book association definition
located_in: Property = Property(name="locatedIn",property_type=library, multiplicity=Multiplicity(1, 1))
has: Property = Property(name="has", property_type=book, multiplicity=Multiplicity(0, "*"))
lib_book_association: BinaryAssociation = BinaryAssociation(name="lib_book_assoc", ends={located_in, has})

# Book-Author association definition
publishes: Property = Property(name="publishes", property_type=book, multiplicity=Multiplicity(0, "*"))
writed_by: Property = Property(name="writedBy", property_type=author, multiplicity=Multiplicity(1, "*"))
book_author_association: BinaryAssociation = BinaryAssociation(name="book_author_assoc", ends={writed_by, publishes})

# Domain model definition
library_model : DomainModel = DomainModel(name="Library model", types={library, book, author}, 
                                          associations={lib_book_association, book_author_association})


#########################################
#   Library - object model definition   #
#########################################


# Library object attributes
library_obj_name: AttributeLink = AttributeLink(attribute=library_name, value=DataValue(classifier=t_str, value="Library test"))
library_obj_address: AttributeLink = AttributeLink(attribute=address, value=DataValue(classifier=t_str, value="street 123"))
# Library object
library_obj: Object = Object(name="Library Object", classifier=library, slots=[library_obj_name, library_obj_address])

# Book object attributes
book_obj_name: AttributeLink = AttributeLink(attribute=title, value=DataValue(classifier=t_str, value="Book tittle"))
book_obj_pages: AttributeLink = AttributeLink(attribute=pages, value=DataValue(classifier=t_int, value=300))
book_obj_release: AttributeLink = AttributeLink(attribute=release, value=DataValue(classifier=t_date, value=datetime.datetime(2020, 3, 15)))
# Book object
book_obj: Object = Object(name="Book Object", classifier=book, slots=[book_obj_name, book_obj_pages])

# Author object attributes
author_obj_name: AttributeLink = AttributeLink(attribute=author_name, value=DataValue(classifier=t_str, value="John Doe"))
author_obj_email: AttributeLink = AttributeLink(attribute=email, value=DataValue(classifier=t_str, value="john@doe.com"))
# Author object
author_obj: Object = Object(name="Author Object", classifier=author, slots=[author_obj_name, author_obj_email])

# Book object and Author object link
book_link_end1: LinkEnd = LinkEnd(name="book_end1", association_end=publishes, object=book_obj)
author_link_end: LinkEnd = LinkEnd(name="author_end", association_end=writed_by, object=author_obj)
author_book_link: Link = Link(name="author_book_link", association=book_author_association, connections=[book_link_end1,author_link_end])

# Book Library and Book object link
book_link_end2: LinkEnd = LinkEnd(name="book_end2", association_end=has, object=book_obj)
library_link_end: LinkEnd = LinkEnd(name="library_end", association_end=located_in, object=library_obj)
library_book_link: Link = Link(name="library_book_link", association=book_author_association, connections=[book_link_end2,library_link_end])

# Object model definition
object_model: ObjectModel = ObjectModel(name="Object model", instances={library_obj, author_obj, book_obj}, links={author_book_link, library_book_link})