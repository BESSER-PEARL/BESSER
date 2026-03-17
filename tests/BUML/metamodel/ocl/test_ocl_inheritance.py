"""Tests for OCL constraints referencing inherited (superclass) attributes.

Reproduces issue #198: OCL Constraint fails when referencing attributes of a superclass.
"""
import pytest
import datetime
from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Multiplicity, BinaryAssociation,
    Constraint, Generalization, StringType, IntegerType, FloatType, DateType,
)
from besser.BUML.metamodel.object import (
    ObjectModel, Object, AttributeLink, DataValue, LinkEnd, Link,
)
from besser.BUML.notations.ocl.OCLParserWrapper import OCLParserWrapper


# ---------------------------------------------------------------------------
# Fixtures: a model with inheritance
#
#   Person (name: str, salary: int)
#     ^
#     |
#   Employee (department_name: str)
#
#   Department (size: int) --boss--> Employee
#                          --members--> Employee (0..*)
# ---------------------------------------------------------------------------

@pytest.fixture
def inheritance_model():
    # Parent class
    Person = Class(name="Person")
    person_name = Property(name="name", type=StringType)
    person_salary = Property(name="salary", type=IntegerType)
    Person.attributes = {person_name, person_salary}

    # Child class
    Employee = Class(name="Employee")
    emp_dept_name = Property(name="department_name", type=StringType)
    Employee.attributes = {emp_dept_name}

    # Generalization
    gen = Generalization(general=Person, specific=Employee)

    # Department class
    Department = Class(name="Department")
    dept_size = Property(name="headcount", type=IntegerType)
    dept_name = Property(name="dept_name", type=StringType)
    Department.attributes = {dept_size, dept_name}

    # Department -> Employee (boss, 1..1)
    boss_end = Property(name="boss", type=Employee, multiplicity=Multiplicity(1, 1))
    dept_end = Property(name="dept_of_boss", type=Department, multiplicity=Multiplicity(0, 9999))
    boss_assoc = BinaryAssociation(name="dept_boss", ends={boss_end, dept_end})

    # Department -> Employee (members, 0..*)
    members_end = Property(name="members", type=Employee, multiplicity=Multiplicity(0, 9999))
    dept_end2 = Property(name="dept_of_member", type=Department, multiplicity=Multiplicity(1, 1))
    members_assoc = BinaryAssociation(name="dept_members", ends={members_end, dept_end2})

    return {
        "Person": Person,
        "Employee": Employee,
        "Department": Department,
        "person_name": person_name,
        "person_salary": person_salary,
        "emp_dept_name": emp_dept_name,
        "dept_size": dept_size,
        "dept_name": dept_name,
        "boss_end": boss_end,
        "members_end": members_end,
        "gen": gen,
        "boss_assoc": boss_assoc,
        "members_assoc": members_assoc,
    }


@pytest.fixture
def object_model(inheritance_model):
    m = inheritance_model
    emp_obj = Object(
        name="emp1",
        classifier=m["Employee"],
        slots=[
            AttributeLink(attribute=m["emp_dept_name"],
                          value=DataValue(classifier=StringType, value="Engineering")),
            AttributeLink(attribute=m["person_name"],
                          value=DataValue(classifier=StringType, value="Alice")),
            AttributeLink(attribute=m["person_salary"],
                          value=DataValue(classifier=IntegerType, value=3000)),
        ],
    )
    dept_obj = Object(
        name="dept1",
        classifier=m["Department"],
        slots=[
            AttributeLink(attribute=m["dept_size"],
                          value=DataValue(classifier=IntegerType, value=10)),
            AttributeLink(attribute=m["dept_name"],
                          value=DataValue(classifier=StringType, value="Engineering")),
        ],
    )
    return ObjectModel(name="test_obj_model", objects={emp_obj, dept_obj})


def _make_model_with_constraint(inheritance_model, expression, context_class):
    """Build a DomainModel with a single OCL constraint and return (model, constraint)."""
    m = inheritance_model
    constraint = Constraint(
        name="test_constraint",
        context=context_class,
        expression=expression,
        language="OCL",
    )
    model = DomainModel(
        name="TestModel",
        types={m["Person"], m["Employee"], m["Department"]},
        associations={m["boss_assoc"], m["members_assoc"]},
        generalizations={m["gen"]},
        constraints={constraint},
    )
    return model, constraint


def _parse_ok(model, constraint, object_model):
    """Parse the constraint and return True if it succeeds."""
    parser = OCLParserWrapper(model, object_model)
    return parser.parse(constraint)


# ===== Tests for direct attributes (should already work) =====

def test_direct_attribute_on_context(inheritance_model, object_model):
    """self.headcount on Department - direct attribute, should always work."""
    model, c = _make_model_with_constraint(
        inheritance_model,
        "context Department inv inv1: self.headcount>0",
        inheritance_model["Department"],
    )
    assert _parse_ok(model, c, object_model)


def test_direct_attribute_on_child_class(inheritance_model, object_model):
    """self.department_name on Employee - direct attribute of child class."""
    model, c = _make_model_with_constraint(
        inheritance_model,
        "context Employee inv inv1: self.department_name <> 'none'",
        inheritance_model["Employee"],
    )
    assert _parse_ok(model, c, object_model)


# ===== Tests for inherited attributes (issue #198) =====

def test_inherited_attribute_on_self(inheritance_model, object_model):
    """self.salary on Employee - salary is defined on Person (parent).
    This is the core of issue #198."""
    model, c = _make_model_with_constraint(
        inheritance_model,
        "context Employee inv inv1: self.salary>0",
        inheritance_model["Employee"],
    )
    assert _parse_ok(model, c, object_model)


def test_inherited_attribute_on_self_name(inheritance_model, object_model):
    """self.name on Employee - name is defined on Person (parent)."""
    model, c = _make_model_with_constraint(
        inheritance_model,
        "context Employee inv inv1: self.name <> 'none'",
        inheritance_model["Employee"],
    )
    assert _parse_ok(model, c, object_model)


def test_inherited_attribute_after_navigation(inheritance_model, object_model):
    """self.boss.salary - navigate Department->Employee then access
    salary which is on Person (parent of Employee).
    This is the exact scenario from issue #198."""
    model, c = _make_model_with_constraint(
        inheritance_model,
        "context Department inv inv1: self.boss.salary>2000",
        inheritance_model["Department"],
    )
    assert _parse_ok(model, c, object_model)


def test_inherited_attribute_after_navigation_name(inheritance_model, object_model):
    """self.boss.name - navigate then access inherited attribute."""
    model, c = _make_model_with_constraint(
        inheritance_model,
        "context Department inv inv1: self.boss.name <> 'none'",
        inheritance_model["Department"],
    )
    assert _parse_ok(model, c, object_model)


def test_inherited_attribute_in_if_then_else(inheritance_model, object_model):
    """if self.headcount>5 then self.boss.salary>2000 else true endif
    - the exact constraint from issue #198."""
    model, c = _make_model_with_constraint(
        inheritance_model,
        "context Department inv inv1: if self.headcount>5 then self.boss.salary>2000 else self.headcount>0 endif",
        inheritance_model["Department"],
    )
    assert _parse_ok(model, c, object_model)


def test_inherited_attribute_in_forall(inheritance_model, object_model):
    """self.members->forAll(e:Employee|e.salary>0) - inherited attr in iterator."""
    model, c = _make_model_with_constraint(
        inheritance_model,
        "context Department inv inv1: self.members->forAll(e:Employee|e.salary>0)",
        inheritance_model["Department"],
    )
    assert _parse_ok(model, c, object_model)


def test_inherited_attribute_in_exists(inheritance_model, object_model):
    """self.members->exists(e:Employee|e.name<>'') - inherited attr in exists."""
    model, c = _make_model_with_constraint(
        inheritance_model,
        "context Department inv inv1: self.members->exists(e:Employee|e.name <> 'none')",
        inheritance_model["Department"],
    )
    assert _parse_ok(model, c, object_model)


def test_direct_and_inherited_mixed(inheritance_model, object_model):
    """self.boss.department_name <> '' - direct attr of child after navigation."""
    model, c = _make_model_with_constraint(
        inheritance_model,
        "context Department inv inv1: self.boss.department_name <> 'none'",
        inheritance_model["Department"],
    )
    assert _parse_ok(model, c, object_model)


def test_inherited_attribute_comparison(inheritance_model, object_model):
    """self.boss.salary>1000 - compare inherited attribute to literal."""
    model, c = _make_model_with_constraint(
        inheritance_model,
        "context Department inv inv1: self.boss.salary>1000",
        inheritance_model["Department"],
    )
    assert _parse_ok(model, c, object_model)
