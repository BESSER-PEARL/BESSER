
from besser.BUML.metamodel.structural import DomainModel, Class, Property, \
    Multiplicity, BinaryAssociation,Constraint,\
    StringType, IntegerType, DateType
from besser.BUML.metamodel.object import *
import datetime

#############################################
#   Library - structural model definition   #
#############################################

# Library attributes definition
library_name: Property = Property(name="name", type=StringType)
address: Property = Property(name="address", type=StringType)
# Library class definition
library: Class = Class (name="Library", attributes={library_name, address})

# Book attributes definition
title: Property = Property(name="title", type=StringType)
pages: Property = Property(name="pages", type=IntegerType)
release: Property = Property(name="release", type=DateType)
# Book class definition
book: Class = Class (name="Book", attributes={title, pages, release})

# Author attributes definition
author_name: Property = Property(name="name", type=StringType)
email: Property = Property(name="email", type=StringType)
# Author class definition
author: Class = Class (name="Author", attributes={author_name, email})

# Library-Book association definition
located_in: Property = Property(name="locatedIn",type=library, multiplicity=Multiplicity(1, 1))
has: Property = Property(name="has", type=book, multiplicity=Multiplicity(0, "*"))
lib_book_association: BinaryAssociation = BinaryAssociation(name="lib_book_assoc", ends={located_in, has})

# Book-Author association definition
publishes: Property = Property(name="publishes", type=book, multiplicity=Multiplicity(0, "*"))
written_by: Property = Property(name="writtenBy", type=author, multiplicity=Multiplicity(1, "*"))
book_author_association: BinaryAssociation = BinaryAssociation(name="book_author_assoc", ends={written_by, publishes})

constraintPageNumber: Constraint = Constraint(name = "libraryPageNumber",context=library,expression="context Library inv inv1: self.has ->forAll(b:Book|b.pages>0)",language="OCL")

constraintBookPageNumber: Constraint = Constraint(name = "BookPageNumber",context=book,expression="context Book inv inv2: self.pages>0",language="OCL")

constraintTitleIncorrect: Constraint = Constraint(name = "BookTitleInc",context=book,expression="context Book inv inv2: self.title= 'NI')",language="OCL")

constraintTitleCorrect: Constraint = Constraint(name = "BookTitle",context=book,expression="context Book inv inv2: self.title <> 'NI'",language="OCL")

constraintLibraryExists: Constraint = Constraint(name = "LibarayExistsConst",context=library,expression="context Library inv inv3: self.has->exists( i_book : Book | i_book.pages <= 110 )",language="OCL")

# constraintLibraryExclude: Constraint = Constraint(name = "LibarayExclude",context=library,expression="context Library inv inv3: self.has->excludes()",language="OCL")
constraintLibrarySize: Constraint = Constraint(name = "LibaraySize",context=library,expression="context Library inv inv3: self.has->size()>1",language="OCL")

constraintLibraryCollect: Constraint = Constraint(name = "LibaryCollect",context=library,
expression="context Library inv inv3: self.has->collect(i_book : Book | i_book.pages <= 110)"
           "->size()>0",language="OCL")

constraintLibraryIf: Constraint = Constraint(name ="LibaryIf", context=library,
expression="context Library inv inv3: if self.name <> 'NI' then self.has->exists"
           "( i_book : Book | i_book.pages <= 110 ) else self.has->forAll(b:Book|b.pages>0)"
           " endif", language="OCL")

constraintLibraryElse: Constraint = Constraint(name ="LibaryElse", context=library,
expression="context Library inv inv3: if self.name = 'NI' then self.has->exists"
           "( i_book : Book | i_book.pages <= 110 ) else self.has->forAll(b:Book|b.pages>0)"
           " endif", language="OCL")

constraintLibraryElseFalse: Constraint = Constraint(name ="LibaryElseFalse", context=library, expression="context Library inv inv3: if self.name = 'NI' then self.has->exists( i_book : Book | i_book.pages <= 110 ) else self.has->forAll(b:Book|b.pages<0) endif", language="OCL")


constraintNameOCLIsTypeOf: Constraint = Constraint(name ="BookIsTypeOfStr", context=book, expression="context Book inv inv3: self.title.oclIsTypeOf(String)", language="OCL")

constraintNameOCLIsTypeOfInt: Constraint = Constraint(name ="BookIsTypeOfInt", context=book, expression="context Book inv inv3: self.pages.oclIsTypeOf(Integer)", language="OCL")

constraintNameOCLIsTypeOfIntFalse: Constraint = Constraint(name ="BookIsTypeOfIntFalse", context=book, expression="context Book inv inv3: self.pages.oclIsTypeOf(String)", language="OCL")
constraintifElseSize: Constraint = Constraint(name ="constraintifElseSize", context=library, expression="context Library inv inv3: if self.name <> 'NI' then self.has->exists( i_book : Book | i_book.pages <= 110 )->size()<3 else self.has->forAll(b:Book|b.pages>0) endif", language="OCL")


# Domain model definition
library_model : DomainModel = DomainModel(name="Library_model", types={library, book, author}, 
                                          associations={lib_book_association, book_author_association},
                                          constraints={
                                              constraintBookPageNumber,
                                                       # constraintTitleIncorrect,
                                                       constraintPageNumber,
                                                       constraintTitleCorrect,
                                                       constraintLibraryExists,
                                                       constraintLibrarySize,
                                                       constraintLibraryCollect,
                                                       constraintLibraryIf,
                                                       constraintLibraryElse,
                                                       constraintLibraryElseFalse,
                                                       constraintNameOCLIsTypeOf,
                                                       constraintNameOCLIsTypeOfInt,
                                                       constraintNameOCLIsTypeOfIntFalse
                                          }

                                            # constraints={constraintLibrarySize}
                                          )


#########################################
#   Library - object model definition   #
#########################################


# Library object attributes
library_obj_name: AttributeLink = AttributeLink(attribute=library_name, value=DataValue(classifier=StringType, value="Library test"))
library_obj_address: AttributeLink = AttributeLink(attribute=address, value=DataValue(classifier=StringType, value="street 123"))
# Library object
library_obj: Object = Object(name="Library_Object", classifier=library, slots=[library_obj_name, library_obj_address])

# Book object attributes
book_obj_name: AttributeLink = AttributeLink(attribute=title, value=DataValue(classifier=StringType, value="Book tittle"))
book_obj_pages: AttributeLink = AttributeLink(attribute=pages, value=DataValue(classifier=IntegerType, value=100))
book_obj_release: AttributeLink = AttributeLink(attribute=release, value=DataValue(classifier=DateType, value=datetime.datetime(2020, 3, 15)))
# Book object
book_obj: Object = Object(name="Book_Object", classifier=book, slots=[book_obj_name, book_obj_pages])

# Book_2 object attributes

book_obj_name_2: AttributeLink = AttributeLink(attribute=title, value=DataValue(classifier=StringType, value="Book tittle_2"))
book_obj_pages_2: AttributeLink = AttributeLink(attribute=pages, value=DataValue(classifier=IntegerType, value=400))
book_obj_release_2: AttributeLink = AttributeLink(attribute=release, value=DataValue(classifier=DateType, value=datetime.datetime(2024, 3, 15)))
# Book object
book_obj_2: Object = Object(name="Book_2_Object", classifier=book, slots=[book_obj_name_2, book_obj_pages_2])

# Author object attributes
author_obj_name: AttributeLink = AttributeLink(attribute=author_name, value=DataValue(classifier=StringType, value="John Doe"))
author_obj_email: AttributeLink = AttributeLink(attribute=email, value=DataValue(classifier=StringType, value="john@doe.com"))
# Author object
author_obj: Object = Object(name="Author_Object", classifier=author, slots=[author_obj_name, author_obj_email])

# Book object and Author object link
book_link_end1: LinkEnd = LinkEnd(name="book_end1", association_end=publishes, object=book_obj)
author_link_end: LinkEnd = LinkEnd(name="author_end", association_end=written_by, object=author_obj)
author_book_link: Link = Link(name="author_book_link", association=book_author_association, connections=[book_link_end1,author_link_end])

# Book Library and Book object link
book_link_end2: LinkEnd = LinkEnd(name="book_end2", association_end=has, object=book_obj)
library_link_end: LinkEnd = LinkEnd(name="library_end", association_end=located_in, object=library_obj)
library_book_link: Link = Link(name="library_book_link", association=book_author_association, connections=[book_link_end2,library_link_end])

# Book object and Author object link
book_link_end3: LinkEnd = LinkEnd(name="book_end3", association_end=publishes, object=book_obj_2)
author_link_end2: LinkEnd = LinkEnd(name="author_end2", association_end=written_by, object=author_obj)
author_book_link2: Link = Link(name="author_book_link2", association=book_author_association, connections=[book_link_end2,author_link_end2])

# Book Library and Book object link
book_link_end4: LinkEnd = LinkEnd(name="book_end4", association_end=has, object=book_obj_2)
library_link_end3: LinkEnd = LinkEnd(name="library_end3", association_end=located_in, object=library_obj)
library_book_link3: Link = Link(name="library_book_link3", association=book_author_association, connections=[book_link_end3,library_link_end3])



# Object model definition
object_model: ObjectModel = ObjectModel(name="Object_model", objects={library_obj, author_obj, book_obj,book_obj_2})