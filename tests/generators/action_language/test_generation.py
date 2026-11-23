import pytest

from besser.BUML.metamodel.structural import *
from besser.BUML.notations.action_language.ActionLanguageASTBuilder import parse_bal
from besser.generators.action_language.PythonGenerator import bal_to_python
from besser.generators.action_language.RESTGenerator import bal_to_rest


def get_model() -> DomainModel:
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
    Book_m_decrease_stock: Method = Method(name="decrease_stock", parameters={Parameter(name='qty', type=IntegerType)}, implementation_type=MethodImplementationType.NONE)

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

def test_REST_API_Generation():
    # setup
    model = get_model()
    library:Class = model.get_type_by_name("Library")
    rename: Method = Method(name="rename",
                            parameters=[
                                Parameter(name='new_name', type=StringType)
                            ],
                            type=None,
                            implementation_type=MethodImplementationType.BAL
                    )
    rename.code = """def rename(new_name:str) -> nothing {
            this.name = new_name;
        }"""
    library.add_method(rename)

    # parse and generate
    function = parse_bal(model, library, rename.code)
    code = bal_to_rest(function, library.name)

    # oracle
    signature = "async def rename(self, new_name: str) -> None :"
    instance_save = "inst_to_update = _library_object"
    update_call = "await update_library(inst_to_update.id, LibraryCreate(name = new_name, books = [x.id for x in inst_to_update.books]), database)"

    # checks
    assert signature in code
    assert instance_save in code
    assert update_call in code


NESTED_LOOPS = """def count_long(min_pages:int) -> int {
    count = 0;
    for(book in this.books){
        for(other in this.books){
            if(book.pages > min_pages){
                count = count + 1;
            }
        }
    }
    return count;
}"""

TWO_ITERATORS = """def count_pairs() -> int {
    count = 0;
    for(first in this.books, second in this.books){
        count = count + 1;
    }
    return count;
}"""


def _generate_twice(code: str, to_text):
    """Two parses of one method, both alive, so no object identity can repeat."""
    model = get_model()
    library: Class = model.get_type_by_name("Library")
    functions = [parse_bal(model, library, code) for _ in range(2)]
    return [to_text(function, library) for function in functions]


@pytest.mark.parametrize("to_text", [
    lambda function, cls: bal_to_rest(function, cls.name),
    lambda function, cls: bal_to_python(function),
], ids=["rest", "python"])
def test_for_loop_code_is_the_same_on_every_parse(to_text):
    """Loop variables were named after the For node's object hash, so each
    generation wrote different code for the same method."""
    first, second = _generate_twice(NESTED_LOOPS, to_text)

    assert first == second
    assert "_book_seq1 = " in first
    assert "_other_seq2 = " in first


@pytest.mark.parametrize("to_text", [
    lambda function, cls: bal_to_rest(function, cls.name),
    lambda function, cls: bal_to_python(function),
], ids=["rest", "python"])
def test_for_loop_iterators_keep_their_written_order(to_text):
    for code in _generate_twice(TWO_ITERATORS, to_text):
        assigned = [line.strip().split(" = ")[0] for line in code.splitlines() if "_seq" in line]
        assert assigned[:2] == ["_first_seq1", "_second_seq1"], code