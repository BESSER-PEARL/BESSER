import os
import pytest
from besser.generators.python_classes import PythonGenerator


# Use the shared library_model_with_enum fixture from tests/generators/conftest.py


@pytest.fixture
def domain_model(library_model_with_enum):
    """Alias the shared fixture so existing test signatures stay unchanged."""
    return library_model_with_enum


# Define the test function
def test_generator(domain_model, tmpdir):
    # Create an instance of the generator
    output_dir = tmpdir.mkdir("output")
    generator = PythonGenerator(model=domain_model, output_dir=str(output_dir))

    # Generate Python classes
    generator.generate()

    # Check if the file was created
    output_file = os.path.join(str(output_dir), "classes.py")
    assert os.path.isfile(output_file)

    # Read the generated file
    with open(output_file, "r", encoding="utf-8") as f:
        generated_code = f.read()

    # Check enum
    assert "class MemberType(Enum):" in generated_code
    assert 'ADULT = "ADULT"' in generated_code
    assert 'SENIOR = "SENIOR"' in generated_code
    assert 'STUDENT = "STUDENT"' in generated_code
    assert 'CHILD = "CHILD"' in generated_code

    # Check classes exist
    assert "class Library:" in generated_code
    assert "class Book:" in generated_code
    assert "class Author:" in generated_code

    # Check Library attributes
    assert "self.name = name" in generated_code
    assert "self.address = address" in generated_code

    # Check Book attributes
    assert "self.release = release" in generated_code
    assert "self.title = title" in generated_code
    assert "self.pages = pages" in generated_code

    # Check Author attributes
    assert "self.email = email" in generated_code
    assert "self.member = member" in generated_code

    # Check relationship ends
    assert 'Library_end: "Library" = None' in generated_code
    assert 'writtenBy: set["Author"] = None' in generated_code
    assert 'Book_end: set["Book"] = None' in generated_code

def test_bidirectional_consistency(domain_model, tmpdir):
    # Generate the classes
    output_dir = tmpdir.mkdir("output")
    generator = PythonGenerator(model=domain_model, output_dir=str(output_dir))
    generator.generate()
    output_file = os.path.join(str(output_dir), "classes.py")
    import importlib.util
    from datetime import date
    spec = importlib.util.spec_from_file_location("classes", output_file)
    classes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classes)

    # Create instances
    Library = classes.Library
    Book = classes.Book
    Author = classes.Author

    lib = Library(name="Central", address="Main St")
    book = Book(release=date.today(), title="Python 101", pages=123, Library_end=lib)
    author = Author(email="ab.com", member=classes.MemberType.ADULT)

    # Test: Book's Library_end should be lib, and lib.Book_end should include book
    assert book.Library_end == lib
    assert book in lib.Book_end

    # Test: Add book to library's Book_end, should update book.Library_end
    lib2 = Library(name="Branch", address="2nd St")
    lib2.Book_end = lib2.Book_end | {book}
    assert book.Library_end == lib2
    assert book in lib2.Book_end
    assert book not in lib.Book_end

    # Test: Author writes book
    author.Book_end = author.Book_end | {book}
    assert author in book.writtenBy
    assert book in author.Book_end

    # Test: Set book.writtenBy directly
    author2 = Author(email="abc.com", member=classes.MemberType.ADULT)
    book.writtenBy = book.writtenBy | {author2}
    assert author2 in book.writtenBy
    assert book in author2.Book_end
