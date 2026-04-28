"""Shared fixture for B-OCL notation tests.

The model mirrors the trimmed paper Fig. 1 from PRD §6.1: ``Department`` and
``Employee`` linked by a ``WorksIn`` association with multiplicity 1..1 on the
employer end and 0..* on the employee end. Sufficient to exercise every
parse/print round-trip and the multiplicity-based normalization rules later.
"""

import pytest

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, BinaryAssociation, Multiplicity,
    StringType, IntegerType, FloatType,
)


@pytest.fixture(scope="module")
def model() -> DomainModel:
    department = Class("Department", attributes={
        Property("name", StringType),
        Property("minSalary", FloatType),
        Property("maxJuniorSal", FloatType),
    })
    employee = Class("Employee", attributes={
        Property("name", StringType),
        Property("age", IntegerType),
        Property("salary", FloatType),
    })
    works_in = BinaryAssociation("WorksIn", ends={
        Property("employer", department, multiplicity=Multiplicity(1, 1)),
        Property("employee", employee, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel(
        "CompanyModel",
        types={department, employee},
        associations={works_in},
    )
