"""OCL invariants -> CHECK constraints (GitHub issue #191).

Simple invariants (``self.<attribute> <op> <literal>``, optionally joined by
``and``) become ``CheckConstraint`` table arguments; everything else is skipped
without an error.
"""

import importlib.util
import os
import sys

import pytest
from sqlalchemy import create_engine, insert
from sqlalchemy.exc import IntegrityError

from besser.BUML.metamodel.structural import (
    AssociationClass, BinaryAssociation, BooleanType, Class, Constraint, DomainModel,
    FloatType, Generalization, IntegerType, Multiplicity, Property, StringType,
)
from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.generators.sql_alchemy.ocl_check_constraints import get_check_constraints

TYPES = {"int": "Integer_", "str": "String_(100)", "float": "Float_", "bool": "Boolean_"}


def _book_model(*invariants, language="OCL"):
    """Book(pages:int, title:str, price:float, inPrint:bool, total:int derived) + invariants."""
    book = Class(name="Book", attributes={
        Property(name="pages", type=IntegerType),
        Property(name="title", type=StringType),
        Property(name="price", type=FloatType),
        Property(name="inPrint", type=BooleanType),
        Property(name="total", type=IntegerType, is_derived=True),
    })
    constraints = {
        Constraint(name=f"c{i}", context=book, expression=f"context Book inv: {body}", language=language)
        for i, body in enumerate(invariants)
    }
    return DomainModel(name="Lib", types={book}, constraints=constraints), book


def _translate(*invariants, **kwargs):
    model, book = _book_model(*invariants, **kwargs)
    return [c["expression"] for c in get_check_constraints(model, TYPES, [book]).get("Book", [])]


@pytest.mark.parametrize("body, expected", [
    ("self.pages > 0", "_sa_column('pages', Integer_) > 0"),
    ("self.pages >= 1 and self.pages <= 10",
     "_sa_and(_sa_column('pages', Integer_) >= 1, _sa_column('pages', Integer_) <= 10)"),
    ("0 < self.pages", "_sa_column('pages', Integer_) > 0"),
    ("self.price > 1.5", "_sa_column('price', Float_) > 1.5"),
    ("self.price <> 2", "_sa_column('price', Float_) != 2"),
    ("self.title <> ''", "_sa_column('title', String_(100)) != ''"),
    ("self.title = 'a\"b'", "_sa_column('title', String_(100)) == 'a\"b'"),
    ("self.inPrint = true", "_sa_column('inPrint', Boolean_) == True"),
])
def test_simple_invariants_are_translated(body, expected):
    assert _translate(body) == [expected]


@pytest.mark.parametrize("body", [
    "self.pages > 0 or self.pages < 10",          # or
    "not (self.pages > 0)",                         # not
    "self.title.size() > 0",                        # operation call
    "self.pages > self.price",                      # attribute vs attribute
    "self.title < 'm'",                             # string ordering is collation-dependent
    "self.pages > 'x'",                             # type mismatch
    "self.inPrint > true",                          # ordering on booleans
    "self.total > 0",                               # derived attribute
    "self.pages > -1",                              # not parseable by the OCL parser today
    "self.pages > 0 and self.title.size() > 0",     # partially translatable -> skipped whole
    "this is not OCL",                              # syntax error
])
def test_untranslatable_invariants_are_skipped(body):
    assert _translate(body) == []


def test_non_ocl_constraints_are_skipped():
    assert _translate("self.pages > 0", language="Python") == []


def test_inherited_attribute_is_skipped():
    parent = Class(name="Item", attributes={Property(name="price", type=FloatType)})
    child = Class(name="Book", attributes={Property(name="pages", type=IntegerType)})
    constraints = {
        Constraint(name="inherited", context=child, expression="context Book inv: self.price > 0", language="OCL"),
        Constraint(name="own", context=child, expression="context Book inv: self.pages > 0", language="OCL"),
    }
    model = DomainModel(name="M", types={parent, child}, generalizations={Generalization(general=parent, specific=child)},
                        constraints=constraints)
    result = get_check_constraints(model, TYPES, [parent, child])
    assert [c["name"] for c in result["Book"]] == ["ck_book_own"]


def test_constraint_names_are_deterministic_short_and_unique():
    model, book = _book_model("self.pages > 0")
    long_name = "a" * 80
    for name in (long_name, long_name + "!"):
        model.constraints = model.constraints | {
            Constraint(name=name, context=book, expression="context Book inv: self.pages < 9", language="OCL")
        }
    names = [c["name"] for c in get_check_constraints(model, TYPES, [book])["Book"]]
    assert len(names) == len(set(names)) == 3
    assert all(len(n) <= 30 for n in names)
    assert names == [c["name"] for c in get_check_constraints(model, TYPES, [book])["Book"]]


def _generate(model, tmpdir, **kwargs):
    output_dir = str(tmpdir.mkdir("out"))
    SQLAlchemyGenerator(model=model, output_dir=output_dir).generate(**kwargs)
    path = os.path.join(output_dir, "sql_alchemy.py")
    with open(path, "r", encoding="utf-8") as f:
        return path, f.read()


def _import(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_sqlalchemy_output_unchanged_by_default(tmpdir):
    model, _ = _book_model("self.pages > 0")
    _, code = _generate(model, tmpdir)
    assert "CheckConstraint" not in code


def test_sqlalchemy_check_constraints_are_enforced(tmpdir):
    model, _ = _book_model("self.pages > 0", "self.title <> ''", "self.title.size() > 0")
    path, code = _generate(model, tmpdir, ocl_check_constraints=True)
    assert 'name="ck_book_c0"' in code and 'name="ck_book_c1"' in code
    assert "ck_book_c2" not in code

    module = _import(path, "_ocl_checks_generated")
    engine = create_engine("sqlite://")
    module.Base.metadata.create_all(engine)
    table = module.Base.metadata.tables["book"]
    row = {"pages": 10, "title": "ok", "price": 1.0, "inPrint": True, "total": 0}
    with engine.begin() as conn:
        conn.execute(insert(table).values(**row))
    for bad in ({"pages": 0}, {"title": ""}):
        with pytest.raises(IntegrityError, match="CHECK constraint failed"):
            with engine.begin() as conn:
                conn.execute(insert(table).values(**{**row, **bad}))
    engine.dispose()


def test_association_class_keeps_unique_constraint_next_to_checks(tmpdir):
    book = Class(name="Book", attributes={Property(name="pages", type=IntegerType)})
    reader = Class(name="Reader", attributes={Property(name="name", type=StringType)})
    reads = BinaryAssociation(name="reads", ends={
        Property(name="book", type=book, multiplicity=Multiplicity(0, "*")),
        Property(name="reader", type=reader, multiplicity=Multiplicity(0, "*")),
    })
    loan = AssociationClass(name="Loan", association=reads, attributes={
        Property(name="code", type=StringType, is_id=True),
        Property(name="days", type=IntegerType),
    })
    constraint = Constraint(name="maxDays", context=loan, expression="context Loan inv: self.days <= 30",
                            language="OCL")
    model = DomainModel(name="Lib", types={book, reader, loan}, associations={reads}, constraints={constraint})
    path, code = _generate(model, tmpdir, ocl_check_constraints=True)
    assert '_UniqueConstraint("book_id", "reader_id"),' in code
    assert '_CheckConstraint(_sa_column(\'days\', Integer_) <= 30, name="ck_loan_maxdays"),' in code

    module = _import(path, "_ocl_checks_assoc_generated")
    constraint_types = {type(c).__name__ for c in module.Base.metadata.tables["loan"].constraints}
    assert {"UniqueConstraint", "CheckConstraint"} <= constraint_types
