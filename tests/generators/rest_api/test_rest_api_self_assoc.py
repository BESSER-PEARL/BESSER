import os
import pytest
from besser.generators.rest_api import RESTAPIGenerator


# Use shared fixtures from tests/conftest.py


@pytest.fixture
def self_association_model(employee_self_assoc_model):
    """Alias the shared Employee self-association fixture."""
    return employee_self_assoc_model


@pytest.fixture
def library_book_model(simple_library_book_model):
    """Alias the shared simple Library-Book fixture."""
    return simple_library_book_model


def test_self_association_does_not_crash(self_association_model, tmp_path):
    """Generating a REST API from a model with a self-association should not crash
    and should produce at least one output file."""
    output_dir = str(tmp_path / "output_self")
    generator = RESTAPIGenerator(
        model=self_association_model,
        output_dir=output_dir
    )
    generator.generate()

    generated_files = os.listdir(output_dir)
    assert len(generated_files) > 0, "Expected at least one generated file in output directory"

    # The generator should produce rest_api.py (non-backend mode)
    rest_api_file = os.path.join(output_dir, "rest_api.py")
    assert os.path.isfile(rest_api_file), "rest_api.py should be generated"

    with open(rest_api_file, "r", encoding="utf-8") as f:
        content = f.read()

    # The generated code should reference the Employee class
    assert "employee" in content.lower(), "Generated REST API should contain Employee references"
    # Should contain FastAPI endpoint definitions
    assert "app" in content, "Generated code should define a FastAPI app"


def test_normal_association_does_not_crash(library_book_model, tmp_path):
    """Generating a REST API from a normal Library-Book model should not crash
    and should produce at least one output file."""
    output_dir = str(tmp_path / "output_normal")
    generator = RESTAPIGenerator(
        model=library_book_model,
        output_dir=output_dir
    )
    generator.generate()

    generated_files = os.listdir(output_dir)
    assert len(generated_files) > 0, "Expected at least one generated file in output directory"

    # The generator should produce rest_api.py (non-backend mode)
    rest_api_file = os.path.join(output_dir, "rest_api.py")
    assert os.path.isfile(rest_api_file), "rest_api.py should be generated"

    with open(rest_api_file, "r", encoding="utf-8") as f:
        content = f.read()

    # The generated code should reference both Library and Book
    assert "library" in content.lower(), "Generated REST API should contain Library references"
    assert "book" in content.lower(), "Generated REST API should contain Book references"
    # Should contain FastAPI endpoint definitions
    assert "app" in content, "Generated code should define a FastAPI app"
