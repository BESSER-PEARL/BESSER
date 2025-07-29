import os
import pytest
from besser.generators.sql import SQLGenerator
from besser.BUML.metamodel.structural import (
    Class, DomainModel, DateType, StringType, IntegerType,
    Property, BinaryAssociation, Multiplicity
)

@pytest.fixture
def domain_model():
    # Library class
    library_name: Property = Property(name="name", type=StringType)
    address: Property = Property(name="address", type=StringType)
    library: Class = Class(name="Library", attributes={library_name, address})

    # Book class
    title: Property = Property(name="title", type=StringType)
    pages: Property = Property(name="pages", type=IntegerType)
    release: Property = Property(name="release", type=DateType)
    book: Class = Class(name="Book", attributes={title, pages, release})

    # Author class
    author_name: Property = Property(name="name", type=StringType)
    email: Property = Property(name="email", type=StringType)
    author: Class = Class(name="Author", attributes={author_name, email})

    # Library-Book association definition
    located_in: Property = Property(name="locatedIn", type=library, multiplicity=Multiplicity(1, 1))
    has: Property = Property(name="has", type=book, multiplicity=Multiplicity(0, "*"))
    lib_book_association: BinaryAssociation = BinaryAssociation(name="lib_book_assoc", ends={located_in, has})

    # Book-Author association definition
    publishes: Property = Property(name="publishes", type=book, multiplicity=Multiplicity(0, "*"))
    written_by: Property = Property(name="writtenBy", type=author, multiplicity=Multiplicity(1, "*"))
    book_author_association: BinaryAssociation = BinaryAssociation(name="book_author", ends={written_by, publishes})

    # Domain model definition
    model: DomainModel = DomainModel(name="Library_model", types={library, book, author},
                                    associations={lib_book_association, book_author_association})

    return model

def test_tables_exist(domain_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = SQLGenerator(model=domain_model, output_dir=str(output_dir), sql_dialect="postgresql")
    generator.generate()

    output_file = os.path.join(str(output_dir), "tables_postgresql.sql")
    assert os.path.isfile(output_file)

    with open(output_file, "r", encoding="utf-8") as f:
        generated_code = f.read()

    # Verify CREATE TABLE sentences
    assert "CREATE TABLE library" in generated_code
    assert "CREATE TABLE book" in generated_code
    assert "CREATE TABLE author" in generated_code

def test_columns_exist(domain_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = SQLGenerator(model=domain_model, output_dir=str(output_dir), sql_dialect="postgresql")
    generator.generate()

    output_file = os.path.join(str(output_dir), "tables_postgresql.sql")
    assert os.path.isfile(output_file)

    with open(output_file, "r", encoding="utf-8") as f:
        generated_code = f.read()

    # Verify columns in the library table
    assert "id SERIAL NOT NULL" in generated_code
    assert "name VARCHAR(100) NOT NULL" in generated_code
    assert "address VARCHAR(100) NOT NULL" in generated_code

    # Verify columns in the book table
    assert "title VARCHAR(100) NOT NULL" in generated_code
    assert "pages INTEGER NOT NULL" in generated_code
    assert "release DATE NOT NULL" in generated_code

    # Verify columns in the author table
    assert "email VARCHAR(100) NOT NULL" in generated_code

def test_primary_keys(domain_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = SQLGenerator(model=domain_model, output_dir=str(output_dir), sql_dialect="postgresql")
    generator.generate()

    output_file = os.path.join(str(output_dir), "tables_postgresql.sql")
    assert os.path.isfile(output_file)

    with open(output_file, "r", encoding="utf-8") as f:
        generated_code = f.read()

    # Verify primary keys
    assert "PRIMARY KEY (id)" in generated_code

def test_foreign_keys(domain_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = SQLGenerator(model=domain_model, output_dir=str(output_dir), sql_dialect="postgresql")
    generator.generate()

    output_file = os.path.join(str(output_dir), "tables_postgresql.sql")
    assert os.path.isfile(output_file)

    with open(output_file, "r", encoding="utf-8") as f:
        generated_code = f.read()

    # Check that the foreign key constraints exist and reference the correct tables
    assert 'FOREIGN KEY("locatedIn_id") REFERENCES library (id)' in generated_code

def test_many_to_many_foreign_keys(domain_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = SQLGenerator(model=domain_model, output_dir=str(output_dir), sql_dialect="postgresql")
    generator.generate()

    output_file = os.path.join(str(output_dir), "tables_postgresql.sql")
    assert os.path.isfile(output_file)

    with open(output_file, "r", encoding="utf-8") as f:
        generated_code = f.read()

    # Check that the many-to-many relationship table 'author_book' has two foreign keys
    assert 'CREATE TABLE book_author (' in generated_code
    assert 'PRIMARY KEY ("writtenBy", publishes)' in generated_code \
        or 'PRIMARY KEY (publishes, "writtenBy")' in generated_code
    assert 'FOREIGN KEY("writtenBy") REFERENCES author (id)' in generated_code
    assert 'FOREIGN KEY(publishes) REFERENCES book (id)' in generated_code

    # Optional: Check that the correct table names are referenced in the foreign keys
    assert 'FOREIGN KEY("writtenBy") REFERENCES author (id)' in generated_code
    assert 'FOREIGN KEY(publishes) REFERENCES book (id)' in generated_code

def test_sql_dialect_postgresql(domain_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = SQLGenerator(model=domain_model, output_dir=str(output_dir), sql_dialect="postgresql")
    generator.generate()
    output_file = os.path.join(str(output_dir), "tables_postgresql.sql")

    with open(output_file, "r", encoding="utf-8") as f:
        code = f.read()

    assert "id SERIAL NOT NULL" in code  # PostgreSQL specific syntax

def test_output_file_not_empty(domain_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = SQLGenerator(model=domain_model, output_dir=str(output_dir), sql_dialect="postgresql")
    generator.generate()
    output_file = os.path.join(str(output_dir), "tables_postgresql.sql")

    assert os.path.getsize(output_file) > 0
