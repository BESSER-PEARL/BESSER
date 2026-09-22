import os
import pytest
from besser.generators.pydantic_classes import PydanticGenerator


# Use shared fixtures from tests/conftest.py


@pytest.fixture
def employee_model(employee_self_assoc_model):
    """Alias the shared Employee self-association fixture."""
    return employee_self_assoc_model


@pytest.fixture
def library_model(simple_library_book_model):
    """Alias the shared simple Library-Book fixture."""
    return simple_library_book_model


def test_self_association_does_not_crash(employee_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = PydanticGenerator(model=employee_model, output_dir=str(output_dir))
    generator.generate()

    output_file = os.path.join(str(output_dir), "pydantic_classes.py")
    assert os.path.isfile(output_file)


def test_self_association_generates_both_fields(employee_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = PydanticGenerator(model=employee_model, output_dir=str(output_dir))
    generator.generate()

    output_file = os.path.join(str(output_dir), "pydantic_classes.py")
    with open(output_file, "r", encoding="utf-8") as f:
        generated_code = f.read()

    assert "manager" in generated_code, "Expected 'manager' field in generated code"
    assert "subordinates" in generated_code, "Expected 'subordinates' field in generated code"


def test_normal_association(library_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = PydanticGenerator(model=library_model, output_dir=str(output_dir))
    generator.generate()

    output_file = os.path.join(str(output_dir), "pydantic_classes.py")
    assert os.path.isfile(output_file)

    with open(output_file, "r", encoding="utf-8") as f:
        generated_code = f.read()

    assert "books" in generated_code, "Expected 'books' field in generated code"
