import os
import pytest
from besser.generators.json import JSONSchemaGenerator


# Use the shared library_book_author_model fixture from tests/conftest.py


@pytest.fixture
def domain_model(library_book_author_model):
    """Alias the shared fixture so existing test signatures stay unchanged."""
    return library_book_author_model


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
