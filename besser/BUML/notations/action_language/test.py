from besser.BUML.metamodel.structural import Class, Property, StringType, FloatType, BinaryAssociation, Multiplicity, \
    DomainModel, Enumeration, EnumerationLiteral, Method, Parameter
from besser.generators.backend import BackendGenerator

text = '''
def set_book_type(book_type: BookType = BookType::Fantasy) -> nothing {
    def get_default() -> BookType[]{
        return BookType[] {BookType::SciFi, BookType::Fantasy};
    }
    default: BookType = get_default()[0];
    if(book_type != null){
        this.book_type = book_type;
    } else {
        this.book_type = default;
    }
}
'''

#####################
##### METAMODEL #####
#####################

# Classes
Author =    Class(name="Author")
Book =      Class(name="Book")
Library =   Class(name="Library")
Model =     Class(name="Model")

# Enums

BookType = Enumeration(name="BookType", literals={EnumerationLiteral("SciFi"), EnumerationLiteral("Fantasy")})

# Author class attributes and methods
Author_name: Property = Property(name="name", type=StringType)
Author_firstname: Property = Property(name="firstname", type=StringType)
Author.attributes={Author_firstname, Author_name}

# Book class attributes and methods
Book_name: Property = Property(name="name", type=StringType)
Book_isbn: Property = Property(name="isbn", type=StringType)
Book_price: Property = Property(name="price", type=FloatType)
Book_BookType: Property = Property(name="book_type", type=BookType)
Book.attributes={Book_name, Book_isbn, Book_price, Book_BookType}
set_book_type = Method("set_book_type", parameters=[Parameter("book_type",BookType)], type=None, code=text, code_lang="BAL")
Book.methods={set_book_type}

# Library class attributes and methods
Library_name: Property = Property(name="name", type=StringType)
Library.attributes={Library_name}

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
Model_Author: BinaryAssociation = BinaryAssociation(
    name="Model_Author",
    ends={
        Property(name="model", type=Model, multiplicity=Multiplicity(1, 1), is_navigable=False),
        Property(name="authors", type=Author, multiplicity=Multiplicity(0, 9999))
    }
)
Model_Book: BinaryAssociation = BinaryAssociation(
    name="Model_Book",
    ends={
        Property(name="model_1", type=Model, multiplicity=Multiplicity(1, 1), is_navigable=False),
        Property(name="books", type=Book, multiplicity=Multiplicity(0, 9999))
    }
)
Model_Library: BinaryAssociation = BinaryAssociation(
    name="Model_Library",
    ends={
        Property(name="model_2", type=Model, multiplicity=Multiplicity(1, 1), is_navigable=False),
        Property(name="libraries", type=Library, multiplicity=Multiplicity(0, 9999))
    }
)

# Domain Model
domain_model = DomainModel(
    name="Library",
    types={Author, Book, Library, Model, BookType},
    associations={Author_Book, Library_Book, Model_Author, Model_Book, Model_Library},
    generalizations={}
)

# ast = parse_bal(domain_model, Book, text)
generator = BackendGenerator(domain_model, http_methods=["POST"], output_dir="./test_gen")
generator.generate()