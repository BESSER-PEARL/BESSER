
from besser.BUML.metamodel.structural import (
    Class, Property, Method, Parameter,
    BinaryAssociation, Generalization, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, IntegerType, FloatType, BooleanType,
    TimeType, DateType, DateTimeType, TimeDeltaType,
    AnyType, Constraint, AssociationClass, Metadata
)
from besser.generators.backend import BackendGenerator
# Classes
Author = Class(name="Author")
Book = Class(name="Book")
Library = Class(name="Library")

# Author class attributes and methods
Author_name: Property = Property(name="name", type=StringType)
Author_email: Property = Property(name="email", type=StringType)
Author.attributes={Author_email, Author_name}

# Book class attributes and methods
Book_pages: Property = Property(name="pages", type=IntegerType)
Book_title: Property = Property(name="title", type=StringType)
Book_release: Property = Property(name="release", type=DateType)

# Instance method - operates on a single book
Book_m_hello_world: Method = Method(
    name="hello_world", 
    parameters={}, 
    code="def hello_world(self):\n    print(f\"Hello from {self.title}!\")\n    return self.title\n"
)

# Instance method with parameter - calculate discounted price
Book_m_calculate_discount: Method = Method(
    name="calculate_discount",
    parameters={Parameter(name="discount", type=FloatType)},
    code="def calculate_discount(self, discount):\n    discounted = int(self.pages * (1 - discount))\n    print(f\"Original pages: {self.pages}, After {discount*100}% discount: {discounted}\")\n    return discounted\n"
)

# Instance check method - check if book is long
Book_m_is_long: Method = Method(
    name="is_long",
    parameters={Parameter(name="min_pages", type=IntegerType, default_value=300)},
    code="def is_long(self, min_pages=300):\n    is_long = self.pages > min_pages\n    print(f\"Book '{self.title}' has {self.pages} pages. Is long ({min_pages}+): {is_long}\")\n    return is_long\n"
)

# Instance check method - check if book is old
Book_m_is_old: Method = Method(
    name="is_old",
    parameters={Parameter(name="years", type=IntegerType, default_value=10)},
    code="def is_old(self, years=10):\n    from datetime import datetime\n    current_year = datetime.now().year\n    book_age = current_year - self.release.year\n    is_old = book_age >= years\n    print(f\"Book '{self.title}' published {book_age} years ago. Is old ({years}+ years): {is_old}\")\n    return is_old\n"
)

# Class method - get longest book
Book_m_get_longest: Method = Method(
    name="get_longest",
    parameters={},
    code="def get_longest(session):\n    return session.query(Book).order_by(Book.pages.desc()).first()\n"
)

# Class method - count books with min pages
Book_m_count_long_books: Method = Method(
    name="count_long_books",
    parameters={Parameter(name="min_pages", type=IntegerType, default_value=300)},
    code="def count_long_books(session, min_pages=300):\n    count = session.query(Book).filter(Book.pages > min_pages).count()\n    print(f\"Found {count} books with more than {min_pages} pages\")\n    return count\n"
)

# Class method - get average pages
Book_m_average_pages: Method = Method(
    name="average_pages",
    parameters={},
    code="def average_pages(session):\n    from sqlalchemy import func\n    avg = session.query(func.avg(Book.pages)).scalar()\n    avg = float(avg) if avg else 0\n    print(f\"Average pages across all books: {avg:.2f}\")\n    return avg\n"
)

Book.attributes={Book_title, Book_release, Book_pages}
Book.methods={Book_m_hello_world, Book_m_calculate_discount, Book_m_is_long, Book_m_is_old, Book_m_get_longest, Book_m_count_long_books, Book_m_average_pages}

# Library class attributes and methods
Library_name: Property = Property(name="name", type=StringType)
Library_address: Property = Property(name="address", type=StringType)

# Instance method - get library info
Library_m_info: Method = Method(
    name="info",
    parameters={},
    code="def info(self):\n    print(f\"Library: {self.name}\")\n    print(f\"Address: {self.address}\")\n    return {'name': self.name, 'address': self.address}\n"
)

# Instance check method - check if library has address
Library_m_has_address: Method = Method(
    name="has_address",
    parameters={},
    code="def has_address(self):\n    has_it = self.address is not None and len(self.address) > 0\n    print(f\"Library '{self.name}' has address: {has_it}\")\n    return has_it\n"
)

# Class method - count libraries
Library_m_count_libraries: Method = Method(
    name="count_libraries",
    parameters={},
    code="def count_libraries(session):\n    count = session.query(Library).count()\n    print(f\"Total libraries in database: {count}\")\n    return count\n"
)

# Class method - find library by name
Library_m_find_by_name: Method = Method(
    name="find_by_name",
    parameters={Parameter(name="name", type=StringType, default_value=None)},
    code="def find_by_name(session, name):\n    library = session.query(Library).filter(Library.name.ilike(f'%{name}%')).first()\n    if library:\n        print(f\"Found library: {library.name}\")\n    else:\n        print(f\"Library with name containing '{name}' not found\")\n    return library\n"
)

Library.attributes={Library_address, Library_name}
Library.methods={Library_m_info, Library_m_has_address, Library_m_count_libraries, Library_m_find_by_name}

# Relationships
Author_Book: BinaryAssociation = BinaryAssociation(
    name="Author_Book",
    ends={
        Property(name="writtenBy", type=Author, multiplicity=Multiplicity(1, 9999)),
        Property(name="publishes", type=Book, multiplicity=Multiplicity(0, 9999))
    }
)
Library_Book: BinaryAssociation = BinaryAssociation(
    name="Library_Book",
    ends={
        Property(name="locatedIn", type=Library, multiplicity=Multiplicity(1, 1)),
        Property(name="has", type=Book, multiplicity=Multiplicity(0, 9999))
    }
)

# Domain Model
domain_model = DomainModel(
    name="Class_Diagram",
    types={Author, Book, Library},
    associations={Author_Book, Library_Book},
    generalizations={},
    metadata=None
)

x = BackendGenerator(domain_model)
x.generate()

