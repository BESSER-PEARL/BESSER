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


def test_association_class_links_are_deleted_with_their_entities(tmpdir):
    """Deleting an entity must cascade to its association-class rows: their FK
    is part of the composite primary key, so without delete-orphan SQLAlchemy
    tries to null it out and the delete fails with an AssertionError."""
    from besser.BUML.metamodel.structural import (
        AssociationClass, BinaryAssociation, Class, DomainModel, FloatType,
        IntegerType, Multiplicity, Property, StringType,
    )

    trip = Class(name="Trip", attributes={
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="reference", type=StringType),
    })
    seat = Class(name="Seat", attributes={Property(name="code", type=IntegerType, is_id=True)})
    trip_seat = BinaryAssociation(name="trip_seat", ends={
        Property(name="trips", type=trip, multiplicity=Multiplicity(0, "*")),
        Property(name="seats", type=seat, multiplicity=Multiplicity(0, "*")),
    })
    reservation = AssociationClass(
        name="Reservation", attributes={Property(name="price", type=FloatType)}, association=trip_seat,
    )
    model = DomainModel(name="TripModel", types={trip, seat, reservation}, associations={trip_seat})

    output_dir = tmpdir.mkdir("assoc_cascade")
    SQLAlchemyGenerator(model=model, output_dir=str(output_dir)).generate(dbms="sqlite")
    with open(os.path.join(str(output_dir), "sql_alchemy.py"), encoding="utf-8") as f:
        code = f.read()

    # Entity -> links: owned, deleted with the entity (from both ends)
    assert ('Trip.reservations: Mapped_[List_["Reservation"]] = relationship("Reservation", '
            'back_populates="trips", cascade="all, delete-orphan")') in code
    assert ('Seat.reservations: Mapped_[List_["Reservation"]] = relationship("Reservation", '
            'back_populates="seats", cascade="all, delete-orphan")') in code
    # Link -> entity: no cascade, deleting a link never deletes the entity
    assert 'Reservation.trips: Mapped_["Trip"] = relationship("Trip", back_populates="reservations")' in code
    assert 'Reservation.seats: Mapped_["Seat"] = relationship("Seat", back_populates="reservations")' in code


def _load_generated_module(file_path, module_name):
    """Import a generated sql_alchemy.py without letting it open its own database."""
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    code = re.sub(
        r"# Database connection.*?Base\.metadata\.create_all\(engine, checkfirst=True\)",
        "engine = create_engine('sqlite:///:memory:', echo=False)",
        code,
        flags=re.DOTALL,
    )
    patched_path = os.path.join(os.path.dirname(file_path), f"{module_name}.py")
    with open(patched_path, "w", encoding="utf-8") as f:
        f.write(code)
    spec = importlib.util.spec_from_file_location(module_name, patched_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_association_class_ends_are_reachable_under_their_model_names(tmpdir):
    """An association class must not hide the association: method bodies and OCL
    constraints are written against the end names from the diagram (self.seats),
    so those must resolve on the entity, not only the link rows."""
    from besser.BUML.metamodel.structural import (
        AssociationClass, BinaryAssociation, Class, DomainModel, FloatType,
        IntegerType, Multiplicity, Property, StringType,
    )

    trip = Class(name="Trip", attributes={
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="reference", type=StringType),
    })
    seat = Class(name="Seat", attributes={Property(name="code", type=IntegerType, is_id=True)})
    trip_seat = BinaryAssociation(name="trip_seat", ends={
        Property(name="trips", type=trip, multiplicity=Multiplicity(0, "*")),
        Property(name="seats", type=seat, multiplicity=Multiplicity(0, "*")),
    })
    reservation = AssociationClass(
        name="Reservation", attributes={Property(name="price", type=FloatType)}, association=trip_seat,
    )
    model = DomainModel(name="TripModel", types={trip, seat, reservation}, associations={trip_seat})

    output_dir = tmpdir.mkdir("assoc_ends")
    SQLAlchemyGenerator(model=model, output_dir=str(output_dir)).generate(dbms="sqlite")
    with open(os.path.join(str(output_dir), "sql_alchemy.py"), encoding="utf-8") as f:
        code = f.read()

    # The direct navigations are views over the link table: writes go through the
    # association-class rows, so they must not compete for the same foreign keys.
    assert ('Trip.seats: Mapped_[List_["Seat"]] = relationship("Seat", '
            'secondary=Reservation.__table__, viewonly=True)') in code
    assert ('Seat.trips: Mapped_[List_["Trip"]] = relationship("Trip", '
            'secondary=Reservation.__table__, viewonly=True)') in code

    module = _load_generated_module(os.path.join(str(output_dir), "sql_alchemy.py"), "assoc_ends_module")
    engine = create_engine("sqlite:///:memory:")
    module.Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    a_trip = module.Trip(id=1, reference="TRIP-1")
    a_seat = module.Seat(code=12)
    session.add_all([a_trip, a_seat])
    session.commit()
    session.add(module.Reservation(trips_id=1, seats_id=12, price=25.0))
    session.commit()
    session.expire_all()

    assert [s.code for s in session.query(module.Trip).one().seats] == [12]
    assert [t.id for t in session.query(module.Seat).one().trips] == [1]
    session.close()


def test_optional_single_ended_association_maps_to_a_scalar(tmpdir):
    """A 0..1 end must read back as the object or None. These relationships are
    assigned after the class body, so SQLAlchemy never sees the Mapped_ annotation
    and infers a collection from the foreign-key direction unless uselist says
    otherwise - and `if self.invoice is not None` is then always true."""
    from besser.BUML.metamodel.structural import (
        BinaryAssociation, Class, DomainModel, FloatType, IntegerType,
        Multiplicity, Property,
    )

    booking = Class(name="Booking", attributes={Property(name="id", type=IntegerType, is_id=True)})
    invoice = Class(name="Invoice", attributes={
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="amount", type=FloatType),
    })
    booking_invoice = BinaryAssociation(name="booking_invoice", ends={
        Property(name="booking", type=booking, multiplicity=Multiplicity(1, 1)),
        Property(name="invoice", type=invoice, multiplicity=Multiplicity(0, 1)),
    })
    model = DomainModel(name="BillingModel", types={booking, invoice},
                        associations={booking_invoice})

    output_dir = tmpdir.mkdir("one_to_one")
    SQLAlchemyGenerator(model=model, output_dir=str(output_dir)).generate(dbms="sqlite")
    with open(os.path.join(str(output_dir), "sql_alchemy.py"), encoding="utf-8") as f:
        code = f.read()
    assert "uselist=False" in code

    module = _load_generated_module(os.path.join(str(output_dir), "sql_alchemy.py"), "one_to_one_module")
    engine = create_engine("sqlite:///:memory:")
    module.Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add_all([module.Booking(id=1), module.Booking(id=2)])
    session.commit()
    session.add(module.Invoice(id=10, amount=99.0, booking_id=1))
    session.commit()
    session.expire_all()

    billed = session.query(module.Booking).filter(module.Booking.id == 1).one()
    unbilled = session.query(module.Booking).filter(module.Booking.id == 2).one()
    assert billed.invoice.id == 10
    assert unbilled.invoice is None
    session.close()


def test_class_methods_are_emitted_on_the_entity(tmpdir):
    """A method body that calls another method of the same object only resolves
    if the methods live on the class, not only inside the endpoints that expose
    them. A method named after a column is skipped: defining both would leave the
    column unreadable."""
    from besser.BUML.metamodel.structural import (
        Class, DateType, DomainModel, FloatType, IntegerType, Method, Property,
    )

    booking = Class(
        name="Booking",
        attributes={
            Property(name="id", type=IntegerType, is_id=True),
            Property(name="price", type=FloatType),
            Property(name="check_in", type=DateType),
        },
        methods={
            Method(name="total", code="def total(self):\n    return self.price * 2\n"),
            Method(name="bill", code="def bill(self):\n    return self.total()\n"),
            # Shares its name with the column above.
            Method(name="check_in", code="def check_in(self):\n    return True\n"),
        },
    )
    model = DomainModel(name="BookingModel", types={booking})

    output_dir = tmpdir.mkdir("methods")
    SQLAlchemyGenerator(model=model, output_dir=str(output_dir)).generate(dbms="sqlite")
    with open(os.path.join(str(output_dir), "sql_alchemy.py"), encoding="utf-8") as f:
        code = f.read()

    assert "    def total(self):" in code
    assert "    def bill(self):" in code
    assert "    def check_in(self):" not in code

    module = _load_generated_module(os.path.join(str(output_dir), "sql_alchemy.py"), "methods_module")
    engine = create_engine("sqlite:///:memory:")
    module.Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(module.Booking(id=1, price=10.0, check_in=datetime.date(2026, 10, 1)))
    session.commit()

    stored = session.query(module.Booking).one()
    # The one calling the other is the point of putting them on the class.
    assert stored.bill() == 20.0
    # The column kept its name.
    assert stored.check_in == datetime.date(2026, 10, 1)
    session.close()
