from besser.BUML.metamodel.structural import DomainModel, MethodImplementationType, Method, Class

from besser.BUML.metamodel.action_language.action_language import *
from besser.BUML.notations.action_language.ActionLanguageASTBuilder import parse_bal


def get_model() -> DomainModel:
    from besser.BUML.metamodel.structural import Property, IntegerType, FloatType, DateType, \
        TimeType, StringType, BinaryAssociation, Multiplicity, Parameter
    # Classes
    Book = Class(name="Book")
    Library = Class(name="Library")
    Author = Class(name="Author")

    # Book class attributes and methods
    Book_title: Property = Property(name="title", type=StringType)
    Book_pages: Property = Property(name="pages", type=IntegerType)
    Book_stock: Property = Property(name="stock", type=IntegerType)
    Book_price: Property = Property(name="price", type=FloatType)
    Book_release: Property = Property(name="release", type=DateType)
    Book_time: Property = Property(name="time", type=TimeType)
    Book_m_decrease_stock: Method = Method(name="decrease_stock", parameters=[Parameter(name='qty', type=IntegerType)], implementation_type=MethodImplementationType.NONE)

    Book.attributes={Book_title, Book_release, Book_pages, Book_stock, Book_time, Book_price}
    Book.methods={Book_m_decrease_stock}

    # Library class attributes and methods
    Library_name: Property = Property(name="name", type=StringType)
    Library.attributes={Library_name}

    # Author class attributes and methods
    Author_name: Property = Property(name="name", type=StringType)
    Author.attributes={Author_name}

    # Relationships
    books: BinaryAssociation = BinaryAssociation(
        name="books",
        ends={
            Property(name="library", type=Library, multiplicity=Multiplicity(0, 9999)),
            Property(name="books", type=Book, multiplicity=Multiplicity(0, 9999))
        }
    )
    books_1: BinaryAssociation = BinaryAssociation(
        name="books_1",
        ends={
            Property(name="authors", type=Author, multiplicity=Multiplicity(0, 9999)),
            Property(name="books", type=Book, multiplicity=Multiplicity(1, 9999))
        }
    )

    return DomainModel(
        name="Class_Diagram",
        types={Book, Library, Author},
        associations={books, books_1},
        generalizations={},
        metadata=None
    )

### TODO : Test the typechecker