import os
import pytest
from besser.generators.sql import SQLGenerator
from besser.BUML.metamodel.structural import (
    Class, DomainModel, DateType, StringType, IntegerType,
    Property, BinaryAssociation, Multiplicity, Enumeration, EnumerationLiteral
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

@pytest.fixture
def domain_model_with_enums():
    """Domain model with enumerations for testing enum handling"""
    # Create enumerations
    member_type: Enumeration = Enumeration(
        name="MemberType",
        literals={
            EnumerationLiteral(name="CHILD"),
            EnumerationLiteral(name="ADULT"),
            EnumerationLiteral(name="SENIOR"),
            EnumerationLiteral(name="STUDENT")
        }
    )
    
    status_type: Enumeration = Enumeration(
        name="StatusType",
        literals={
            EnumerationLiteral(name="ACTIVE"),
            EnumerationLiteral(name="INACTIVE"),
            EnumerationLiteral(name="PENDING")
        }
    )
    
    # Library class
    library_name: Property = Property(name="name", type=StringType)
    address: Property = Property(name="address", type=StringType)
    library: Class = Class(name="Library", attributes={library_name, address})

    # Author class with enum
    author_name: Property = Property(name="name", type=StringType)
    member_type_prop: Property = Property(name="memberType", type=member_type)
    author: Class = Class(name="Author", attributes={author_name, member_type_prop})

    # Book class with enum
    title: Property = Property(name="title", type=StringType)
    pages: Property = Property(name="pages", type=IntegerType)
    release: Property = Property(name="release", type=DateType)
    status_prop: Property = Property(name="status", type=status_type)
    book: Class = Class(name="Book", attributes={title, pages, release, status_prop})

    # Library-Book association definition
    located_in: Property = Property(name="locatedIn", type=library, multiplicity=Multiplicity(1, 1))
    has: Property = Property(name="has", type=book, multiplicity=Multiplicity(0, "*"))
    lib_book_association: BinaryAssociation = BinaryAssociation(name="lib_book_assoc", ends={located_in, has})

    # Domain model definition
    model: DomainModel = DomainModel(
        name="Library_model_with_enums",
        types={library, book, author, member_type, status_type},
        associations={lib_book_association}
    )

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


# Oracle-specific tests for enum handling


def test_oracle_no_create_type(domain_model_with_enums, tmpdir):
    """Verify that Oracle does NOT generate CREATE TYPE ... AS ENUM statements"""
    output_dir = tmpdir.mkdir("output")
    generator = SQLGenerator(model=domain_model_with_enums, output_dir=str(output_dir), sql_dialect="oracle")
    generator.generate()
    
    output_file = os.path.join(str(output_dir), "tables_oracle.sql")
    assert os.path.isfile(output_file)
    
    with open(output_file, "r", encoding="utf-8") as f:
        generated_code = f.read()
    
    # Oracle should NOT have CREATE TYPE statements
    assert "CREATE TYPE" not in generated_code
    assert "AS ENUM" not in generated_code


def test_oracle_check_constraints_for_enums(domain_model_with_enums, tmpdir):
    """Verify that Oracle generates CHECK constraints for enum columns"""
    output_dir = tmpdir.mkdir("output")
    generator = SQLGenerator(model=domain_model_with_enums, output_dir=str(output_dir), sql_dialect="oracle")
    generator.generate()
    
    output_file = os.path.join(str(output_dir), "tables_oracle.sql")
    assert os.path.isfile(output_file)
    
    with open(output_file, "r", encoding="utf-8") as f:
        generated_code = f.read()
    
    # Verify CHECK constraints exist for enum columns
    assert "CHECK" in generated_code
    
    # Check for memberType enum values in CHECK constraint
    assert "CHILD" in generated_code
    assert "ADULT" in generated_code
    assert "SENIOR" in generated_code
    assert "STUDENT" in generated_code
    
    # Check for status enum values in CHECK constraint
    assert "ACTIVE" in generated_code
    assert "INACTIVE" in generated_code
    assert "PENDING" in generated_code


def test_oracle_check_constraint_syntax(domain_model_with_enums, tmpdir):
    """Verify that Oracle CHECK constraints have correct syntax"""
    output_dir = tmpdir.mkdir("output")
    generator = SQLGenerator(model=domain_model_with_enums, output_dir=str(output_dir), sql_dialect="oracle")
    generator.generate()
    
    output_file = os.path.join(str(output_dir), "tables_oracle.sql")
    assert os.path.isfile(output_file)
    
    with open(output_file, "r", encoding="utf-8") as f:
        generated_code = f.read()
    
    # Verify CHECK constraint syntax: CHECK (column IN ('value1', 'value2', ...))
    # The column reference should match the column definition (quoted or unquoted)
    assert "CHECK (" in generated_code
    assert " IN (" in generated_code
    
    # Verify that enum columns use VARCHAR
    assert "VARCHAR" in generated_code


def test_oracle_vs_postgresql_enum_handling(domain_model_with_enums, tmpdir):
    """Verify that PostgreSQL still uses CREATE TYPE while Oracle uses CHECK constraints"""
    output_dir = tmpdir.mkdir("output")
    
    # Generate for PostgreSQL
    pg_generator = SQLGenerator(model=domain_model_with_enums, output_dir=str(output_dir), sql_dialect="postgresql")
    pg_generator.generate()
    
    pg_file = os.path.join(str(output_dir), "tables_postgresql.sql")
    with open(pg_file, "r", encoding="utf-8") as f:
        pg_code = f.read()
    
    # Generate for Oracle
    oracle_generator = SQLGenerator(model=domain_model_with_enums, output_dir=str(output_dir), sql_dialect="oracle")
    oracle_generator.generate()
    
    oracle_file = os.path.join(str(output_dir), "tables_oracle.sql")
    with open(oracle_file, "r", encoding="utf-8") as f:
        oracle_code = f.read()
    
    # PostgreSQL should have CREATE TYPE
    assert "CREATE TYPE" in pg_code
    assert "AS ENUM" in pg_code
    
    # Oracle should NOT have CREATE TYPE but should have CHECK
    assert "CREATE TYPE" not in oracle_code
    assert "AS ENUM" not in oracle_code
    assert "CHECK" in oracle_code


def test_oracle_enum_column_varchar_type(domain_model_with_enums, tmpdir):
    """Verify that Oracle uses VARCHAR for enum columns, not the enum type itself"""
    output_dir = tmpdir.mkdir("output")
    generator = SQLGenerator(model=domain_model_with_enums, output_dir=str(output_dir), sql_dialect="oracle")
    generator.generate()
    
    output_file = os.path.join(str(output_dir), "tables_oracle.sql")
    assert os.path.isfile(output_file)
    
    with open(output_file, "r", encoding="utf-8") as f:
        generated_code = f.read()
    
    # Oracle should use VARCHAR for enum columns, not custom types
    assert "VARCHAR" in generated_code or "VARCHAR2" in generated_code
    # Should not reference the enum type names as column types
    assert "membertype NOT NULL" not in generated_code.lower()
    assert "statustype NOT NULL" not in generated_code.lower()


def test_oracle_tables_and_columns_exist(domain_model_with_enums, tmpdir):
    """Verify that all expected tables and columns are present in Oracle DDL"""
    output_dir = tmpdir.mkdir("output")
    generator = SQLGenerator(model=domain_model_with_enums, output_dir=str(output_dir), sql_dialect="oracle")
    generator.generate()
    
    output_file = os.path.join(str(output_dir), "tables_oracle.sql")
    assert os.path.isfile(output_file)
    
    with open(output_file, "r", encoding="utf-8") as f:
        generated_code = f.read()
    
    # Verify tables exist (more flexible matching)
    assert "library" in generated_code.lower()
    assert "book" in generated_code.lower()
    assert "author" in generated_code.lower()
    assert "CREATE TABLE" in generated_code
    
    # Verify enum columns exist
    assert "memberType" in generated_code or "membertype" in generated_code
    assert "status" in generated_code
    
    # Verify primary keys
    assert "PRIMARY KEY" in generated_code
    
    # Verify foreign keys
    assert "FOREIGN KEY" in generated_code


def test_oracle_output_not_empty(domain_model_with_enums, tmpdir):
    """Verify that Oracle SQL output file is not empty"""
    output_dir = tmpdir.mkdir("output")
    generator = SQLGenerator(model=domain_model_with_enums, output_dir=str(output_dir), sql_dialect="oracle")
    generator.generate()
    
    output_file = os.path.join(str(output_dir), "tables_oracle.sql")
    assert os.path.isfile(output_file)
    assert os.path.getsize(output_file) > 0

