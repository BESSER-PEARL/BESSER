"""The generated ``models.py`` must be parseable Python.

Three defects, all of them in the *whitespace control* of
``besser/generators/django/templates/models.py.j2``. The Jinja environment is
built with ``trim_blocks=True, lstrip_blocks=True`` (which eats the newline
*after* a block tag), so a ``{%-`` marker — which eats the whitespace *before*
the tag — leaves nothing at all between the two pieces of rendered code:

* Every enumeration in the model collapsed onto a single line: the class
  header, all its literals, and the next enum class ran together as
  ``class MemberStatus(models.TextChoices):    ACTIVE = 'ACTIVE', 'ACTIVE'…
  class LoanStatus(…)``. Any model with an ``Enumeration`` therefore shipped a
  ``models.py`` that ``ast.parse`` rejects — Django could not even import it.
* Two consecutive relationship fields on the same class joined at the
  statement level: ``…null=True)    book = models.ForeignKey(``. Keyword
  arguments may share a line, but two assignments may not.
* ``manage.py startapp`` imports the settings module it just wrote, so
  ``<project>/<project>/__pycache__/*.pyc`` landed in the generated tree and
  was packaged into the user's download.

The pre-existing Django compile tests all use enum-free, relationship-free
models, which is exactly why the first two shipped.
"""

import ast
import os
import shutil

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Enumeration, EnumerationLiteral,
    IntegerType, Multiplicity, Property, StringType,
)
from besser.generators.django import DjangoGenerator


def _model_with_enums_and_relationships() -> DomainModel:
    """Two enumerations, and a class carrying two ForeignKeys."""
    member_status = Enumeration(name="MemberStatus", literals={
        EnumerationLiteral(name="ACTIVE"),
        EnumerationLiteral(name="EXPIRED"),
        EnumerationLiteral(name="SUSPENDED"),
    })
    loan_status = Enumeration(name="LoanStatus", literals={
        EnumerationLiteral(name="OPEN"),
        EnumerationLiteral(name="CLOSED"),
    })

    member = Class(name="Member")
    member.attributes = {
        Property(name="email", type=StringType),
        Property(name="status", type=member_status),
    }
    book = Class(name="Book")
    book.attributes = {Property(name="title", type=StringType)}
    loan = Class(name="Loan")
    loan.attributes = {
        Property(name="days", type=IntegerType),
        Property(name="state", type=loan_status),
    }

    # Loan *--1 Member and Loan *--0..1 Book: both become ForeignKeys on Loan.
    member_loans = BinaryAssociation(name="MemberLoans", ends={
        Property(name="loans", type=loan, multiplicity=Multiplicity(0, "*")),
        Property(name="member", type=member, multiplicity=Multiplicity(1, 1)),
    })
    book_loans = BinaryAssociation(name="BookLoans", ends={
        Property(name="loans_of_book", type=loan, multiplicity=Multiplicity(0, "*")),
        Property(name="book", type=book, multiplicity=Multiplicity(0, 1)),
    })

    return DomainModel(
        name="Library",
        types={member, book, loan, member_status, loan_status},
        associations={member_loans, book_loans},
    )


def _generate(model: DomainModel, tmp_path) -> str:
    """Run the full DjangoGenerator; return the generated project directory."""
    if shutil.which("django-admin") is None:
        pytest.skip("django-admin is not on PATH")

    DjangoGenerator(
        model=model,
        project_name="new_project",
        app_name="core_app",
        output_dir=str(tmp_path),
    ).generate()
    return os.path.join(str(tmp_path), "new_project")


@pytest.fixture
def generated_project(tmp_path):
    return _generate(_model_with_enums_and_relationships(), tmp_path)


@pytest.fixture
def generated_project_without_enums(tmp_path):
    """The same relationships, with the enumerations stripped out.

    Isolates the ForeignKey defect: with enums present the file already fails
    to parse on line 3, which would mask it.
    """
    model = _model_with_enums_and_relationships()
    for cls in model.get_classes():
        cls.attributes = {
            attr for attr in cls.attributes
            if not isinstance(attr.type, Enumeration)
        }
    model.types = {
        t for t in model.types if not isinstance(t, Enumeration)
    }
    assert not model.get_enumerations()
    return _generate(model, tmp_path)


def _models_source(project_dir: str) -> str:
    path = os.path.join(project_dir, "core_app", "models.py")
    assert os.path.isfile(path), f"{path} was not generated"
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def test_generated_models_with_enumerations_is_valid_python(generated_project):
    """``models.py`` must compile. Enumerations used to collapse onto one line."""
    source = _models_source(generated_project)

    try:
        ast.parse(source)
    except SyntaxError as exc:
        offending = source.splitlines()[exc.lineno - 1] if exc.lineno else ""
        pytest.fail(
            f"generated models.py is not valid Python: line {exc.lineno}: "
            f"{exc.msg}\n{offending[:300]}"
        )

    compile(source, "models.py", "exec")


def test_each_enum_class_and_literal_is_on_its_own_line(generated_project):
    """The header, every literal, and the next class must be separate lines."""
    source = _models_source(generated_project)
    tree = ast.parse(source)

    enums = {
        node.name: node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(getattr(base, "attr", None) == "TextChoices" for base in node.bases)
    }
    assert set(enums) == {"MemberStatus", "LoanStatus"}

    literals = {
        target.id
        for stmt in enums["MemberStatus"].body
        if isinstance(stmt, ast.Assign)
        for target in stmt.targets
        if isinstance(target, ast.Name)
    }
    assert literals == {"ACTIVE", "EXPIRED", "SUSPENDED"}


def test_consecutive_relationship_fields_are_separate_statements(
        generated_project_without_enums):
    """Two ForeignKeys used to join as ``…null=True)    book = models.…``.

    Keyword arguments may legally share a line inside a call, but the closing
    paren of one assignment and the start of the next may not: this is a
    statement-level join, and it is a syntax error on its own — the enum
    defect is not needed to trigger it.
    """
    source = _models_source(generated_project_without_enums)
    tree = ast.parse(source)

    loan = next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "Loan")
    fields = {
        target.id
        for stmt in loan.body
        if isinstance(stmt, ast.Assign)
        for target in stmt.targets
        if isinstance(target, ast.Name)
    }
    assert {"member", "book"} <= fields


def test_generated_project_ships_no_compiled_bytecode(generated_project):
    """``manage.py startapp`` used to leave its own ``__pycache__`` behind."""
    leaked = [
        os.path.join(root, name)
        for root, _, files in os.walk(generated_project)
        for name in files
        if name.endswith((".pyc", ".pyo")) or "__pycache__" in root
    ]
    assert not leaked, f"compiled bytecode leaked into the download: {leaked}"
