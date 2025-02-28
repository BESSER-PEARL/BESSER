import os
import pytest
from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Multiplicity,
    BinaryAssociation, StringType, IntegerType, DateType
)
from besser.generators.json import JSONSchemaGenerator

@pytest.fixture
def domain_model():
    # Create a sample domain model for testing
    # Library attributes definition
    library_name = Property(name="name", type=StringType)
    address = Property(name="address", type=StringType)
    # Library class definition
    library = Class(name="Library", attributes={library_name, address})

    # Book attributes definition
    title = Property(name="title", type=StringType)
    pages = Property(name="pages", type=IntegerType)
    release = Property(name="release", type=DateType)
    # Book class definition
    book = Class(name="Book", attributes={title, pages, release})

    # Author attributes definition
    author_name = Property(name="name", type=StringType)
    email = Property(name="email", type=StringType)
    # Author class definition
    author = Class(name="Author", attributes={author_name, email})

    # Library-Book association definition
    located_in = Property(name="locatedIn", type=library, multiplicity=Multiplicity(1, 1))
    has = Property(name="has", type=book, multiplicity=Multiplicity(0, "*"))
    lib_book_association = BinaryAssociation(name="lib_book_assoc", ends={located_in, has})

    # Book-Author association definition
    publishes = Property(name="publishes", type=book, multiplicity=Multiplicity(0, "*"))
    written_by = Property(name="writtenBy", type=author, multiplicity=Multiplicity(1, "*"))
    book_author_association = BinaryAssociation(name="book_author_assoc",
                                                ends={written_by, publishes})

    # Domain model definition
    model = DomainModel(name="Library_Model", types={library, book, author},
                        associations={lib_book_association, book_author_association})

    return model

def test_json_schema_generator(domain_model, tmpdir):
    # Create an instance of the generator
    output_dir = tmpdir.mkdir("output")
    generator = JSONSchemaGenerator(model=domain_model, output_dir=str(output_dir))

    # Generate the JSON schema
    generator.generate()

    # Check if the file was created
    output_file = os.path.join(str(output_dir), "json_schema.json")
    assert os.path.isfile(output_file)

    # Read the generated file
    with open(output_file, "r") as f:
        generated_code = f.read()

    # Check if the generated code contains expected content
    assert '"Library"' in generated_code
    assert '"Book"' in generated_code
    assert '"Author"' in generated_code
