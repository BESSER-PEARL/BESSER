import os
import pytest
from besser.generators.sql import SQLGenerator


# Use the shared library_book_author_model fixture from tests/conftest.py
# (aliased to domain_model via indirect parametrization for minimal diff)


@pytest.fixture
def domain_model(library_book_author_model):
    """Alias the shared fixture so existing test signatures stay unchanged."""
    return library_book_author_model


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
