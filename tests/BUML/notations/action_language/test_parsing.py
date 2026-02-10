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

def test_parse_function_def():
    # setup
    model = get_model()
    library:Class = model.get_type_by_name("Library")
    method: Method = Method(name="method",
                            parameters=[],
                            type=None,
                            implementation_type=MethodImplementationType.BAL
                    )
    method.code = """def method() -> nothing {}"""
    library.add_method(method)

    # parsing
    function:FunctionDefinition = parse_bal(model, library, method.code)

    expected_type = FunctionType([None], Nothing())

    assert function is not None
    assert function.name == "method"
    assert function.return_type == Nothing()
    assert function.parameters[0].name == "self"
    assert function.declared_type == expected_type

def test_parse_types():
    # setup
    model = get_model()
    library:Class = model.get_type_by_name("Library")
    method: Method = Method(name="method",
                            parameters=[],
                            type=None,
                            implementation_type=MethodImplementationType.BAL
                    )
    method.code = """def method() -> nothing {
        var1:bool = null;
        var2:int = null;
        var3:float = null;
        var4:str = null;
        var5:any = null;
        var6:int[] = null;
        var7:Author = null;
        var8:[Book, Library] -> nothing = null;
        var9:[Book?] -> nothing = null;
    }"""
    library.add_method(method)

    # parsing
    function:FunctionDefinition = parse_bal(model, library, method.code)

    expected_type = FunctionType([None], Nothing())

    statements:list[Assignment] = function.body.statements
    targets:list[ExplicitDecl] = [s.target for s in statements]
    types:list[Type] = [t.declared_type for t in targets]
    assert types[0] == BoolType()
    assert types[1] == IntType()
    assert types[2] == RealType()
    assert types[3] == StringType()
    assert types[4] == AnyType()
    assert types[5] == SequenceType(IntType())
    assert types[6] == ObjectType(model.get_type_by_name("Author"))
    var8_type = FunctionType([
                                    ObjectType(model.get_type_by_name("Book")),
                                    ObjectType(model.get_type_by_name("Library"))
                                ],
                                Nothing()
                            )
    var9_type = FunctionType([
                                    OptionalType(ObjectType(model.get_type_by_name("Book")))
                                ],
                                Nothing()
                            )
    assert types[7] == var8_type
    assert types[8] == var9_type