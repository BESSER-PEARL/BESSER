"""Tests for the Spring Boot backend generator.

The models come from the centralized fixtures in ``tests/conftest.py``; the
Spring-specific fixtures below only add what a JPA mapping additionally needs
(identifiers, a ``time`` attribute and an inheritance pair).
"""

import copy
import filecmp
import os
from pathlib import Path

import pytest

from besser.BUML.metamodel.structural import (
    Class,
    Generalization,
    IntegerType,
    Property,
    StringType,
    TimeType,
)
from besser.generators.spring import SpringBackendGenerator
from besser.generators.spring.java_types import (
    JAVA_TYPES,
    to_java_identifier,
    validate_java_package,
)

PACKAGE_DIR = Path("src", "main", "java", "com", "example")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spring_library_model(library_book_author_model):
    """``library_book_author_model`` completed with what a JPA mapping needs.

    * every concrete class gets an identifier (JPA entities require one),
    * ``Library`` gets a ``time`` attribute (to pin the ``TimeType`` mapping),
    * ``Author`` gets an abstract parent (to exercise ``@MappedSuperclass``).
    """
    model = library_book_author_model

    library = model.get_class_by_name("Library")
    library.add_attribute(Property(name="id", type=IntegerType, is_id=True))
    library.add_attribute(Property(name="openingTime", type=TimeType))

    book = model.get_class_by_name("Book")
    book.add_attribute(Property(name="isbn", type=StringType, is_id=True))

    person = Class(name="Person", is_abstract=True)
    person.add_attribute(Property(name="id", type=IntegerType, is_id=True))
    model.add_type(person)
    model.add_generalization(
        Generalization(general=person, specific=model.get_class_by_name("Author"))
    )

    return model


@pytest.fixture
def generated_library(spring_library_model, tmp_path):
    """Path: A generated Spring Boot project for the library model."""
    output_dir = tmp_path / "project"
    SpringBackendGenerator(spring_library_model, output_dir=str(output_dir),
                           app_name="Library").generate()
    return output_dir


def read(project_dir: Path, *parts: str) -> str:
    return (project_dir.joinpath(*parts)).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_generator_is_constructible_like_every_other_generator(spring_library_model, tmp_path):
    """The web editor instantiates generators as ``(model, output_dir=...)``."""
    generator = SpringBackendGenerator(spring_library_model, output_dir=str(tmp_path))
    generator.generate()

    assert (tmp_path / "pom.xml").exists()


def test_generator_is_registered_and_buildable_by_the_backend(spring_library_model, tmp_path):
    """The registry entry has to be instantiable exactly as ``_generate_standard`` does."""
    from besser.utilities.web_modeling_editor.backend.config.generators import SUPPORTED_GENERATORS

    generator_info = SUPPORTED_GENERATORS["spring"]
    generator = generator_info.generator_class(spring_library_model, output_dir=str(tmp_path))
    generator.generate()

    assert generator_info.output_type == "zip"
    assert (tmp_path / "pom.xml").exists()


def test_defaults_are_shared_with_the_backend_constants():
    """The backend constants must mirror the generator-owned defaults."""
    from besser.generators.spring import (
        DEFAULT_JAVA_VERSION,
        DEFAULT_SPRING_APP_NAME,
        DEFAULT_SPRING_BOOT_VERSION,
    )
    from besser.utilities.web_modeling_editor.backend.constants import constants

    assert constants.DEFAULT_SPRING_BOOT_VERSION == DEFAULT_SPRING_BOOT_VERSION
    assert constants.DEFAULT_JAVA_VERSION == DEFAULT_JAVA_VERSION
    assert constants.DEFAULT_SPRING_APP_NAME == DEFAULT_SPRING_APP_NAME


def test_relative_output_dir_is_not_joined_twice(spring_library_model, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    SpringBackendGenerator(spring_library_model, output_dir="generated/myapp").generate()

    assert (tmp_path / "generated" / "myapp" / "pom.xml").exists()
    assert not (tmp_path / "generated" / "myapp" / "generated").exists()


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_project_scaffolding_is_where_maven_expects_it(generated_library):
    assert (generated_library / "pom.xml").is_file()
    # mvnw/mvnw.cmd belong at the project root, only the properties file does not.
    assert (generated_library / "mvnw").is_file()
    assert (generated_library / "mvnw.cmd").is_file()
    assert (generated_library / ".mvn" / "wrapper" / "maven-wrapper.properties").is_file()
    assert not (generated_library / ".mvn" / "mvnw").exists()

    assert (generated_library / PACKAGE_DIR / "Library.java").is_file()
    assert (generated_library / "src" / "main" / "resources" / "application.properties").is_file()


def test_maven_wrapper_is_a_usable_script(generated_library):
    mvnw = generated_library / "mvnw"
    content = mvnw.read_bytes()

    assert content.startswith(b"#!/bin/sh\n"), "the POSIX wrapper must not carry CRLF endings"
    assert b"{{" not in content and b"{%" not in content
    if os.name != "nt":
        assert os.access(mvnw, os.X_OK), "the POSIX wrapper must be executable"


@pytest.mark.parametrize("class_name", ["Library", "Book", "Author"])
def test_every_concrete_class_gets_a_full_layer_stack(generated_library, class_name):
    assert (generated_library / PACKAGE_DIR / "entity" / f"{class_name}.java").is_file()
    assert (generated_library / PACKAGE_DIR / "repository" / f"I{class_name}Repository.java").is_file()
    assert (generated_library / PACKAGE_DIR / "service" / "interfaces" / f"I{class_name}Service.java").is_file()
    assert (generated_library / PACKAGE_DIR / "service" / "impl" / f"{class_name}Service.java").is_file()
    assert (generated_library / PACKAGE_DIR / "controller" / f"{class_name}Controller.java").is_file()


def test_abstract_class_has_no_repository_service_or_controller(generated_library):
    assert (generated_library / PACKAGE_DIR / "entity" / "Person.java").is_file()
    assert not (generated_library / PACKAGE_DIR / "repository" / "IPersonRepository.java").exists()
    assert not (generated_library / PACKAGE_DIR / "controller" / "PersonController.java").exists()


def test_http_samples_stay_out_of_the_compiled_source_tree(generated_library):
    assert (generated_library / "src" / "test" / "resources" / "http" / "book.http").is_file()
    assert not (generated_library / PACKAGE_DIR / "http_test").exists()


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

def test_concrete_class_is_mapped_as_an_entity(generated_library):
    book = read(generated_library, *PACKAGE_DIR.parts, "entity", "Book.java")

    assert "@Entity" in book
    assert '@Table(name = "books")' in book
    assert "public class Book {" in book


def test_abstract_parent_is_a_mapped_superclass_the_child_extends(generated_library):
    person = read(generated_library, *PACKAGE_DIR.parts, "entity", "Person.java")
    author = read(generated_library, *PACKAGE_DIR.parts, "entity", "Author.java")

    assert "@MappedSuperclass" in person
    assert "@Entity" not in person
    assert "public abstract class Person {" in person
    assert "public class Author extends Person {" in author


def test_many_to_many_has_exactly_one_owning_side(generated_library):
    book = read(generated_library, *PACKAGE_DIR.parts, "entity", "Book.java")
    author = read(generated_library, *PACKAGE_DIR.parts, "entity", "Author.java")

    assert book.count("@ManyToMany") == 1
    assert author.count("@ManyToMany") == 1
    # Exactly one side declares mappedBy, the other one owns the join table.
    mapped_by_sides = [text for text in (book, author) if 'mappedBy = "' in text]
    join_table_sides = [text for text in (book, author) if "@JoinTable" in text]
    assert len(mapped_by_sides) == 1
    assert len(join_table_sides) == 1
    assert mapped_by_sides[0] is not join_table_sides[0]
    assert '@JoinTable(name = "book_author"' in join_table_sides[0]


def test_many_to_one_side_owns_the_join_column(generated_library):
    book = read(generated_library, *PACKAGE_DIR.parts, "entity", "Book.java")
    library = read(generated_library, *PACKAGE_DIR.parts, "entity", "Library.java")

    # Book is the "many" side of Library 1 --- 0..* Book: it holds the FK.
    assert "@ManyToOne" in book
    assert '@JoinColumn(name = "library_id")' in book
    assert "private Library locatedIn;" in book

    # The inverse side is a mapped-by one-to-many, never a second owner.
    assert '@OneToMany(mappedBy = "locatedIn")' in library
    assert "@JoinColumn" not in library


def test_self_association_produces_both_directed_ends(employee_self_assoc_model, tmp_path):
    employee = employee_self_assoc_model.get_class_by_name("Employee")
    employee.add_attribute(Property(name="id", type=IntegerType, is_id=True))
    SpringBackendGenerator(employee_self_assoc_model, output_dir=str(tmp_path)).generate()

    entity = read(tmp_path, *PACKAGE_DIR.parts, "entity", "Employee.java")

    assert '@OneToMany(mappedBy = "manager")' in entity
    assert "private List<Employee> subordinates" in entity
    assert "@ManyToOne" in entity
    assert "private Employee manager;" in entity


def test_time_type_is_the_same_java_type_in_every_layer(generated_library):
    """The regression that made the generated project uncompilable: an entity
    field typed ``LocalTime`` with a repository/service method taking
    ``LocalDateTime``."""
    entity = read(generated_library, *PACKAGE_DIR.parts, "entity", "Library.java")
    repository = read(generated_library, *PACKAGE_DIR.parts, "repository", "ILibraryRepository.java")
    service = read(generated_library, *PACKAGE_DIR.parts, "service", "interfaces", "ILibraryService.java")
    service_impl = read(generated_library, *PACKAGE_DIR.parts, "service", "impl", "LibraryService.java")

    assert "LocalTime openingTime;" in entity
    for layer in (repository, service, service_impl):
        assert "findAllByOpeningTime(LocalTime openingTime)" in layer
        assert "LocalDateTime" not in layer


def test_controller_uses_the_real_identifier_setter(generated_library):
    """``Book`` is identified by ``isbn``, so ``setId`` would not exist."""
    controller = read(generated_library, *PACKAGE_DIR.parts, "controller", "BookController.java")

    assert "entity.setIsbn(id);" in controller
    assert "entity.setId(id);" not in controller
    assert "public ResponseEntity<Book> getById(@PathVariable String id)" in controller


def test_java_types_are_defined_once(generated_library):
    """The four sub-generators must share one mapping, not four copies."""
    from besser.generators.spring import (
        spring_controller_generator,
        spring_entity_generator,
        spring_repository_generator,
        spring_service_generator,
    )

    for module in (spring_entity_generator, spring_repository_generator,
                   spring_service_generator, spring_controller_generator):
        assert not hasattr(module, "JAVA_TYPES"), f"{module.__name__} redefines JAVA_TYPES"
    assert JAVA_TYPES["time"] == "LocalTime"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def _tree(root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*") if path.is_file()
    )


def test_output_is_byte_identical_across_runs(spring_library_model, tmp_path):
    """Sets (``enum.literals``, ``model.associations``) iterate in an order that
    depends on object identity, so a second, freshly built copy of the same model
    is what actually catches unsorted iteration."""
    first, second = tmp_path / "first", tmp_path / "second"
    SpringBackendGenerator(spring_library_model, output_dir=str(first)).generate()
    SpringBackendGenerator(copy.deepcopy(spring_library_model), output_dir=str(second)).generate()

    assert _tree(first) == _tree(second)
    for relative in _tree(first):
        assert filecmp.cmp(first / relative, second / relative, shallow=False), relative


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

def test_traversal_in_a_class_name_stays_inside_the_output_dir(spring_library_model, tmp_path):
    evil = Class(name="../../evil")
    evil.add_attribute(Property(name="id", type=IntegerType, is_id=True))
    spring_library_model.add_type(evil)

    output_dir = tmp_path / "project"
    SpringBackendGenerator(spring_library_model, output_dir=str(output_dir)).generate()

    written = list(tmp_path.rglob("*"))
    assert all(output_dir in path.parents or path == output_dir for path in written)
    assert not (tmp_path / "evil.java").exists()
    assert (output_dir / PACKAGE_DIR / "entity" / "evil.java").is_file()


def test_java_keywords_and_odd_names_become_legal_identifiers():
    assert to_java_identifier("../../evil") == "evil"
    assert to_java_identifier("a/b\\c") == "a_b_c"
    assert to_java_identifier("class") == "class_"
    assert to_java_identifier("new", capitalize=True) == "New"
    assert to_java_identifier("2fast") == "_2fast"
    assert to_java_identifier("..") == "Unnamed"


def test_invalid_package_names_are_rejected():
    assert validate_java_package("com.example.app") == "com.example.app"
    for invalid in ("../../etc", "com..example", "com.Example", "com.new", ""):
        with pytest.raises(ValueError):
            validate_java_package(invalid)


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------

def test_missing_identifier_reports_the_class(player_team_domain_model, tmp_path):
    """Neither Player nor Team has an ``is_id`` attribute."""
    with pytest.raises(ValueError, match="Player|Team"):
        SpringBackendGenerator(player_team_domain_model, output_dir=str(tmp_path)).generate()
