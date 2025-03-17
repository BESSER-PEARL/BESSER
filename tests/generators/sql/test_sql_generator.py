import os
import pytest
from besser.generators.sql import SQLGenerator
from besser.BUML.metamodel.structural import (
    Class, DomainModel, DateType, StringType, IntegerType,
    Property, BinaryAssociation, Multiplicity
)

# Define the expected output
association_output = """
CREATE TABLE book_author_assoc"""

library_output = """
CREATE TABLE library (
	id SERIAL NOT NULL, 
	name VARCHAR(100) NOT NULL, 
	address VARCHAR(100) NOT NULL, 
	PRIMARY KEY (id)
)

;"""

author_output = """
CREATE TABLE author (
	id SERIAL NOT NULL, 
	name VARCHAR(100) NOT NULL, 
	email VARCHAR(100) NOT NULL, 
	PRIMARY KEY (id)
)

;"""

book_output = """
CREATE TABLE book (
	id SERIAL NOT NULL, 
	title VARCHAR(100) NOT NULL, 
	pages INTEGER NOT NULL, 
	release DATE NOT NULL, 
	"locatedIn_id" INTEGER NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY("locatedIn_id") REFERENCES library (id)
)

;"""

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
    book_author_association: BinaryAssociation = BinaryAssociation(name="book_author_assoc", ends={written_by, publishes})

    # Domain model definition
    model: DomainModel = DomainModel(name="Library_model", types={library, book, author},
                                    associations={lib_book_association, book_author_association})

    return model

# Define the test function
def test_generator(domain_model, tmpdir):
    # Create an instance of the generator
    output_dir = tmpdir.mkdir("output")
    generator = SQLGenerator(model=domain_model, output_dir=str(output_dir), sql_dialect="postgresql")

    # Generate SQL statements
    generator.generate()

    # Check if the file was created
    output_file = os.path.join(str(output_dir), "tables_postgresql.sql")
    assert os.path.isfile(output_file)

    # Read the generated file
    with open(output_file, "r", encoding="utf-8") as f:
        generated_code = f.read()

    # Compare the output with the expected output
    assert association_output in generated_code
    assert library_output in generated_code
    assert book_output in generated_code
    assert author_output in generated_code
