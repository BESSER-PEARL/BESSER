import os
import sys
import importlib.util
import pytest
from sqlalchemy import create_engine, inspect, MetaData
from sqlalchemy.orm import sessionmaker, declarative_base
import datetime
import re

from besser.generators.sql_alchemy import SQLAlchemyGenerator


# Use the shared library_model_with_inheritance fixture from
# tests/generators/conftest.py


@pytest.fixture
def domain_model(library_model_with_inheritance):
    """Alias the shared fixture so existing test signatures stay unchanged."""
    return library_model_with_inheritance

@pytest.fixture
def generated_sqlalchemy_module(domain_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = SQLAlchemyGenerator(model=domain_model, output_dir=str(output_dir))
    generator.generate(dbms="sqlite")
    file_path = os.path.join(str(output_dir), "sql_alchemy.py")

    # Read and patch the generated file to use in-memory DB
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Remove or replace the database connection and table creation section
    code = re.sub(
        r"# Database connection.*?Base\.metadata\.create_all\(engine, checkfirst=True\)",
        (
            "# Database connection (patched for testing)\n"
            "DATABASE_URL = 'sqlite:///:memory:'\n"
            "engine = create_engine(DATABASE_URL, echo=False)\n"
            "# Table creation will be handled by the test"
        ),
        code,
        flags=re.DOTALL,
    )

    # Write the patched code to a new temp file
    patched_file_path = os.path.join(str(output_dir), "sql_alchemy_patched.py")
    with open(patched_file_path, "w", encoding="utf-8") as f:
        f.write(code)

    # Dynamically import the patched module
    spec = importlib.util.spec_from_file_location("sql_alchemy", patched_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sql_alchemy"] = module
    spec.loader.exec_module(module)
    return module

def test_tables_exist(generated_sqlalchemy_module):
    engine = create_engine("sqlite:///:memory:")
    generated_sqlalchemy_module.Base.metadata.create_all(engine)
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "library" in tables
    assert "book" in tables
    assert "author" in tables
    assert any(tbl in tables for tbl in ["book_author", "author_book", "author_book_assoc"])

def test_columns_exist(generated_sqlalchemy_module):
    engine = create_engine("sqlite:///:memory:")
    generated_sqlalchemy_module.Base.metadata.create_all(engine)
    inspector = inspect(engine)
    # Library columns
    columns = [col["name"] for col in inspector.get_columns("library")]
    assert "id" in columns
    assert "name" in columns
    assert "address" in columns
    # Book columns
    columns = [col["name"] for col in inspector.get_columns("book")]
    assert "id" in columns
    assert "title" in columns
    assert "pages" in columns
    assert "release" in columns
    # Author columns
    columns = [col["name"] for col in inspector.get_columns("author")]
    assert "id" in columns
    assert "name" in columns
    assert "email" in columns

def test_primary_keys(generated_sqlalchemy_module):
    engine = create_engine("sqlite:///:memory:")
    generated_sqlalchemy_module.Base.metadata.create_all(engine)
    inspector = inspect(engine)
    for table in ["library", "book", "author"]:
        pk = inspector.get_pk_constraint(table)
        assert "id" in pk["constrained_columns"]

def test_foreign_keys(generated_sqlalchemy_module):
    engine = create_engine("sqlite:///:memory:")
    generated_sqlalchemy_module.Base.metadata.create_all(engine)
    inspector = inspect(engine)
    # Check that the foreign key constraints exist and reference the correct tables
    fk_found = False
    assoc_table = next((tbl for tbl in inspector.get_table_names() if tbl in ["book_author", "author_book", "author_book_assoc"]), None)
    assert assoc_table is not None
    for fk in inspector.get_foreign_keys(assoc_table):
        if ("writtenBy" in fk["constrained_columns"] and fk["referred_table"] == "author") or \
           ("publishes" in fk["constrained_columns"] and fk["referred_table"] == "book"):
            fk_found = True
    assert fk_found

def test_relationships(generated_sqlalchemy_module):
    # Check that relationships exist on the mapped classes
    Library = getattr(generated_sqlalchemy_module, "Library")
    Book = getattr(generated_sqlalchemy_module, "Book")
    Author = getattr(generated_sqlalchemy_module, "Author")
    # Check that the ORM relationships exist (attributes are present)
    assert hasattr(Library, "has")
    assert hasattr(Book, "locatedIn")
    assert hasattr(Book, "writtenBy")
    assert hasattr(Author, "publishes")

def test_can_insert_and_query(generated_sqlalchemy_module):
    engine = create_engine("sqlite:///:memory:")
    generated_sqlalchemy_module.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    Library = getattr(generated_sqlalchemy_module, "Library")
    Book = getattr(generated_sqlalchemy_module, "Book")
    Author = getattr(generated_sqlalchemy_module, "Author")
    BookType = getattr(generated_sqlalchemy_module, "BookType")

    lib = Library(name="Central", address="Main St")
    session.add(lib)
    session.commit()

    book_type = BookType(position=1)
    session.add(book_type)
    session.commit()

    book = Book(title="Python 101", pages=200, release=datetime.date(2023, 1, 1), locatedIn=lib, book_type=book_type)
    session.add(book)
    session.commit()

    author = Author(name="Alice", email="alice@example.com")
    session.add(author)
    session.commit()

    assert session.query(Library).filter_by(name="Central").one()
    assert session.query(Book).filter_by(title="Python 101").one()
    assert session.query(Author).filter_by(name="Alice").one()

def test_inheritance_and_subclasses(generated_sqlalchemy_module):
    BookType = getattr(generated_sqlalchemy_module, "BookType")
    Horror = getattr(generated_sqlalchemy_module, "Horror")
    History = getattr(generated_sqlalchemy_module, "History")
    Science = getattr(generated_sqlalchemy_module, "Science")
    # Subclasses should inherit from BookType
    assert issubclass(Horror, BookType)
    assert issubclass(History, BookType)
    assert issubclass(Science, BookType)

def test_booktype_relationships(generated_sqlalchemy_module):
    BookType = getattr(generated_sqlalchemy_module, "BookType")
    Book = getattr(generated_sqlalchemy_module, "Book")
    # BookType should have 'books' relationship, Book should have 'book_type'
    assert hasattr(BookType, "books")
    assert hasattr(Book, "book_type")

def test_crud_booktype_and_subclasses(generated_sqlalchemy_module):
    engine = create_engine("sqlite:///:memory:")
    generated_sqlalchemy_module.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    BookType = getattr(generated_sqlalchemy_module, "BookType")
    Horror = getattr(generated_sqlalchemy_module, "Horror")
    Book = getattr(generated_sqlalchemy_module, "Book")
    Library = getattr(generated_sqlalchemy_module, "Library")

    # Create a Library (required for Book)
    lib = Library(name="Central", address="Main St")
    session.add(lib)
    session.commit()

    # Create a Horror book type and a Book linked to it
    horror_type = Horror(attribute="Scary", position=1)
    session.add(horror_type)
    session.commit()
    book = Book(title="Dracula", pages=300, release=datetime.date(1897, 5, 26), book_type=horror_type, locatedIn=lib)
    session.add(book)
    session.commit()

    # Query and check
    horror_db = session.query(Horror).filter_by(attribute="Scary").one()
    assert horror_db.position == 1
    book_db = session.query(Book).filter_by(title="Dracula").one()
    assert book_db.book_type == horror_db
    # Check reverse relationship
    assert book_db in horror_db.books

def test_polymorphic_query(generated_sqlalchemy_module):
    engine = create_engine("sqlite:///:memory:")
    generated_sqlalchemy_module.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    BookType = getattr(generated_sqlalchemy_module, "BookType")
    Horror = getattr(generated_sqlalchemy_module, "Horror")
    History = getattr(generated_sqlalchemy_module, "History")
    Science = getattr(generated_sqlalchemy_module, "Science")

    horror = Horror(attribute="Scary", position=1)
    history = History(attribute="Ancient", position=2)
    science = Science(attribute="Physics", position=3)
    session.add_all([horror, history, science])
    session.commit()

    # Query all BookType and check polymorphic results
    all_types = session.query(BookType).all()
    type_names = {type(obj).__name__ for obj in all_types}
    assert "Horror" in type_names
    assert "History" in type_names
    assert "Science" in type_names

def test_booktype_book_association(generated_sqlalchemy_module):
    engine = create_engine("sqlite:///:memory:")
    generated_sqlalchemy_module.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    BookType = getattr(generated_sqlalchemy_module, "BookType")
    Book = getattr(generated_sqlalchemy_module, "Book")
    Library = getattr(generated_sqlalchemy_module, "Library")

    # Create a Library (required for Book)
    lib = Library(name="Central", address="Main St")
    session.add(lib)
    session.commit()

    book_type = BookType(position=10)
    session.add(book_type)
    session.commit()
    book1 = Book(title="Book1", pages=100, release=datetime.date(2000, 1, 1), book_type=book_type, locatedIn=lib)
    book2 = Book(title="Book2", pages=150, release=datetime.date(2005, 2, 2), book_type=book_type, locatedIn=lib)
    session.add_all([book1, book2])
    session.commit()

    # Check that BookType.books contains both books
    bt_db = session.query(BookType).filter_by(position=10).one()
    book_titles = {b.title for b in bt_db.books}
    assert "Book1" in book_titles
    assert "Book2" in book_titles

def test_reserved_name_validation_class():
    """Test that reserved class names raise a ValueError with a clear message."""
    from besser.BUML.metamodel.structural import DomainModel, Class
    
    # Create a model with a reserved class name
    model = DomainModel(name="test_model")
    reserved_class = Class(name="Base")  # Reserved name
    model.add_type(reserved_class)
    
    generator = SQLAlchemyGenerator(model=model)
    
    with pytest.raises(ValueError) as exc_info:
        generator.generate(dbms="sqlite")
    
    error_message = str(exc_info.value)
    assert "Base" in error_message
    assert "reserved" in error_message.lower()
    assert "cannot be used" in error_message.lower()


def test_reserved_name_validation_attribute():
    """Test that reserved attribute names raise a ValueError with a clear message."""
    from besser.BUML.metamodel.structural import DomainModel, Class, Property, PrimitiveDataType, Multiplicity
    
    # Create a model with a class containing a reserved attribute name
    model = DomainModel(name="test_model")
    base_class = Class(name="MyClass")
    reserved_attr = Property(name="Base", type=PrimitiveDataType("str"), multiplicity=Multiplicity(1, 1))
    base_class.attributes = {reserved_attr}
    model.add_type(base_class)
    
    generator = SQLAlchemyGenerator(model=model)
    
    with pytest.raises(ValueError) as exc_info:
        generator.generate(dbms="sqlite")
    
    error_message = str(exc_info.value)
    assert "Base" in error_message
    assert "MyClass" in error_message
    assert "reserved" in error_message.lower()


def test_reserved_name_validation_enumeration():
    """Test that reserved enumeration names raise a ValueError with a clear message."""
    from besser.BUML.metamodel.structural import DomainModel, Enumeration, EnumerationLiteral
    
    # Create a model with a reserved enumeration name
    model = DomainModel(name="test_model")
    literal1 = EnumerationLiteral(name="VALUE1")
    literal2 = EnumerationLiteral(name="VALUE2")
    reserved_enum = Enumeration(name="Enum", literals={literal1, literal2})
    model.add_type(reserved_enum)
    
    generator = SQLAlchemyGenerator(model=model)
    
    with pytest.raises(ValueError) as exc_info:
        generator.generate(dbms="sqlite")
    
    error_message = str(exc_info.value)
    assert "Enum" in error_message
    assert "reserved" in error_message.lower()


def test_non_reserved_names_allowed():
    """Test that non-underscore SQLAlchemy type names are allowed as class names."""
    from besser.BUML.metamodel.structural import DomainModel, Class, Property, PrimitiveDataType, Multiplicity
    
    # These names (without underscores) are allowed because the imports use underscore aliases
    model = DomainModel(name="test_model")
    
    # Class named "Table" should work (import uses Table_)
    table_class = Class(name="Table")
    model.add_type(table_class)
    
    # Class named "Column" should work (import uses Column_)
    column_class = Class(name="Column")
    model.add_type(column_class)
    
    # Class named "Boolean" should work (import uses Boolean_)
    boolean_class = Class(name="Boolean")
    attr1 = Property(name="value", type=PrimitiveDataType("bool"), multiplicity=Multiplicity(1, 1))
    boolean_class.attributes = {attr1}
    model.add_type(boolean_class)
    
    # Class named "String" should work (import uses String_)
    string_class = Class(name="String")
    model.add_type(string_class)
    
    # Class named "Integer" should work (import uses Integer_)
    integer_class = Class(name="Integer")
    model.add_type(integer_class)
    
    generator = SQLAlchemyGenerator(model=model)
    
    # Should not raise any validation errors
    generator.validate_model()  # This should pass without errors


def test_reserved_underscore_aliases():
    """Test that underscore-aliased names are properly reserved."""
    from besser.BUML.metamodel.structural import DomainModel, Class
    
    # Test that underscore variants are reserved
    model = DomainModel(name="test_model")
    reserved_class = Class(name="Boolean_")  # Reserved: underscore alias
    model.add_type(reserved_class)
    
    generator = SQLAlchemyGenerator(model=model)
    
    with pytest.raises(ValueError) as exc_info:
        generator.generate(dbms="sqlite")
    
    error_message = str(exc_info.value)
    assert "Boolean_" in error_message
    assert "reserved" in error_message.lower()