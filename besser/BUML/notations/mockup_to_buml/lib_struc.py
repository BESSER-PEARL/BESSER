from besser.BUML.metamodel.structural import DomainModel, Class, Property, \
    Multiplicity, BinaryAssociation, StringType, IntegerType, DateType


# Library attributes definition
library_name: Property = Property(name="name", type=StringType)
address: Property = Property(name="address", type=StringType)
# Library class definition
library: Class = Class(name="Library", attributes={library_name, address})

# Book attributes definition
title: Property = Property(name="title", type=StringType)
pages: Property = Property(name="pages", type=IntegerType)
release: Property = Property(name="release", type=DateType)
# Book class definition
book: Class = Class(name="Book", attributes={title, pages, release})

# Author attributes definition
author_name: Property = Property(name="name", type=StringType)
email: Property = Property(name="email", type=StringType)
# Author class definition
author: Class = Class(name="Author", attributes={author_name, email})

# Library-Book association definition
located_in: Property = Property(name="locatedIn", type=library, multiplicity=Multiplicity(1, 1))
has: Property = Property(name="has", type=book, multiplicity=Multiplicity(0, "*"))
lib_book_association: BinaryAssociation = BinaryAssociation(name="lib_book_assoc", ends={located_in, has})

# Book-Author association definition
publishes: Property = Property(name="publishes", type=book, multiplicity=Multiplicity(0, "*"))
written_by: Property = Property(name="writtenBy", type=author, multiplicity=Multiplicity(1, "*"))
book_author_association: BinaryAssociation = BinaryAssociation(name="book_author_assoc", ends={written_by, publishes})

# Domain model definition
library_model: DomainModel = DomainModel(name="Library_model", types={library, book, author},
                                         associations={lib_book_association, book_author_association})

