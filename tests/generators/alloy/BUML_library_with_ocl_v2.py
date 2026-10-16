import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    Constraint,
    DateType,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    FloatType,
    IntegerType,
    Method,
    MethodImplementationType,
    Multiplicity,
    Parameter,
    Property,
    StringType,
)

# ---------------------------------------------------------------------------
# Fixture: library domain model with OCL constraints (v2)
# ---------------------------------------------------------------------------

@pytest.fixture
def library_with_ocl_v2():
    """Library / Book / Author / Genre.

    Built from ``tests/BUML/metamodel/structural/library_v2/library_with_ocl_v2.py``:

    - two bidirectional many-to-many associations: Library 1..* -- 0..* Book
      and Author 1..* -- 0..* Book;
    - String, Integer, Float, Date and enumeration attributes;
    - OCL invariants using integer bounds, String ``size()``, collection
      ``size()``, ``forAll`` with date comparison, ``implies`` with string
      literals, ``if-then-else`` with ``select``.
    """
    # --- Enumeration ---
    genre_literals = {
        EnumerationLiteral(name="Poetry"),
        EnumerationLiteral(name="Thriller"),
        EnumerationLiteral(name="History"),
        EnumerationLiteral(name="Technology"),
        EnumerationLiteral(name="Romance"),
        EnumerationLiteral(name="Horror"),
        EnumerationLiteral(name="Adventure"),
        EnumerationLiteral(name="Philosophy"),
        EnumerationLiteral(name="Cookbooks"),
        EnumerationLiteral(name="Fantasy"),
    }
    Genre = Enumeration(name="Genre", literals=genre_literals)

    # --- Classes ---
    Author = Class(name="Author")
    Author.attributes = {
        Property(name="name", type=StringType),
        Property(name="birth", type=DateType),
    }

    Book = Class(name="Book")
    Book.attributes = {
        Property(name="title", type=StringType),
        Property(name="pages", type=IntegerType),
        Property(name="stock", type=IntegerType),
        Property(name="price", type=FloatType),
        Property(name="release", type=DateType),
        Property(name="genre", type=Genre),
    }
    Book.methods = {
        Method(name="decrease_stock", parameters={Parameter(name="qty", type=IntegerType)},
               implementation_type=MethodImplementationType.NONE),
    }

    Library = Class(name="Library")
    Library.attributes = {
        Property(name="name", type=StringType),
        Property(name="web_page", type=StringType),
        Property(name="address", type=StringType),
        Property(name="telephone", type=StringType),
    }
    cheapest_book_by = Method(name="cheapest_book_by", parameters={Parameter(name="author", type=Author)},
                              type=StringType, implementation_type=MethodImplementationType.BAL)
    cheapest_book_by.code = """def cheapest_book_by(author:Author) -> str {
    cheapest:Book = null;
    price = 1000000000.0;
    for(book in this.books){
        if(book.authors.contains(author)
            && book.price <= price){
            cheapest = book;
            price = book.price;
        }
    }
    return cheapest.title;
}"""
    Library.methods = {cheapest_book_by}

    # --- Associations (bidirectional) ---
    # Library <-> Book: a library holds any number of books, a book is in 1..* libraries
    library_books = BinaryAssociation(
        name="books",
        ends={
            Property(name="library", type=Library, multiplicity=Multiplicity(1, "*"),
                      is_navigable=True),
            Property(name="books", type=Book, multiplicity=Multiplicity(0, "*"),
                      is_navigable=True),
        },
    )

    # Author <-> Book: an author writes any number of books, a book has 1..* authors
    author_books = BinaryAssociation(
        name="books_1",
        ends={
            Property(name="authors", type=Author, multiplicity=Multiplicity(1, "*"),
                      is_navigable=True),
            Property(name="books", type=Book, multiplicity=Multiplicity(0, "*"),
                      is_navigable=True),
        },
    )

    # --- OCL Constraints ---
    # A book has a positive number of pages
    book_positive_pages = Constraint(
        name="book_positive_pages",
        context=Book,
        expression="context Book inv book_positive_pages: self.pages > 0",
        language="OCL",
    )

    # Stock is never negative
    book_nonneg_stock = Constraint(
        name="book_nonneg_stock",
        context=Book,
        expression="context Book inv book_nonneg_stock: self.stock >= 0",
        language="OCL",
    )

    # String size(): a book has a non-empty title
    book_has_title = Constraint(
        name="book_has_title",
        context=Book,
        expression="context Book inv book_has_title: self.title.size() > 0",
        language="OCL",
    )

    # Price is never negative
    book_nonneg_price = Constraint(
        name="book_nonneg_price",
        context=Book,
        expression="context Book inv book_nonneg_price: self.price >= 0",
        language="OCL",
    )

    # Collection size(): a library holds at least one book
    library_has_books = Constraint(
        name="library_has_books",
        context=Library,
        expression="context Library inv library_has_books: self.books->size() > 0",
        language="OCL",
    )

    # String size(): a library has a non-empty name
    library_named = Constraint(
        name="library_named",
        context=Library,
        expression="context Library inv library_named: self.name.size() > 0",
        language="OCL",
    )

    # String size(): an author has a non-empty name
    author_named = Constraint(
        name="author_named",
        context=Author,
        expression="context Author inv author_named: self.name.size() > 0",
        language="OCL",
    )

    # forAll + date comparison: books are released after the author's birth
    Author_inv_8_1 = Constraint(
        name="Author_inv_8_1",
        context=Author,
        expression=(
            "context Author inv Author_inv_8_1 : "
            "self.books->forAll(b |b.release>=self.birth)"
        ),
        language="OCL",
    )

    # implies + string literal: 'borges' has written at least one book
    Author_inv_9_1 = Constraint(
        name="Author_inv_9_1",
        context=Author,
        expression=(
            "context Author inv Author_inv_9_1 : "
            "self.name='borges' implies  self.books->size()>0"
        ),
        language="OCL",
    )

    # if-then-else + select + size: only 'cortazar' is allowed, with more than
    # 2 books of over 10 pages
    Author_inv_10_1 = Constraint(
        name="Author_inv_10_1",
        context=Author,
        expression=(
            "context Author inv Author_inv_10_1 : "
            "if self.name='cortazar' then self.books->select(b|b.pages>10)->size()>2 "
            "else false endif"
        ),
        language="OCL",
    )

    # collect returns a Bag, whose size() has no Alloy translation: this invariant is disabled
    # collect + size: the books of a library have more than 5 authors in total
    # Library_inv_11_1 = Constraint(
    #     name="Library_inv_11_1",
    #     context=Library,
    #     expression=(
    #         "context Library inv Library_inv_11_1 : "
    #         "self.books-> collect(b |b.authors) ->size()>5"
    #     ),
    #     language="OCL",
    # )

    model = DomainModel(
        name="Library_with_OCL",
        types={Book, Library, Author, Genre},
        associations={library_books, author_books},
        constraints={
            book_positive_pages, book_nonneg_stock, book_has_title, book_nonneg_price,
            library_has_books, library_named, author_named,
            Author_inv_8_1, Author_inv_9_1, Author_inv_10_1,
        },
    )
    return model
