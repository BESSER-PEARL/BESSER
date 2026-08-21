####################
# STRUCTURAL MODEL #
####################

from besser.BUML.metamodel.structural import (
    Class, Property, Method, Parameter,
    BinaryAssociation, Generalization, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, IntegerType, FloatType, BooleanType,
    TimeType, DateType, DateTimeType, TimeDeltaType,
    AnyType, Constraint, AssociationClass, Metadata, MethodImplementationType
)

# Classes
Person = Class(name="Person")

# Person class attributes and methods
Person_name: Property = Property(name="name", type=StringType)
Person_dni: Property = Property(name="dni", type=StringType)
Person_birthDate: Property = Property(name="birthDate", type=DateType)
Person.attributes={Person_birthDate, Person_dni, Person_name}


# OCL Constraints
Person_inv_1_1: Constraint = Constraint(
    name="Person_inv_1_1",
    context=Person,
    expression="context Person inv Person_inv_1_1 : self.birthDate>\'2/20/2000\'",
    language="OCL"
)

# Domain Model
domain_model = DomainModel(
    name="Class_Diagram",
    types={Person},
    associations={},
    constraints={Person_inv_1_1},
    generalizations={},
    metadata=None
)


######################
# PROJECT DEFINITION #
######################

from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural.structural import Metadata

metadata = Metadata(description="Modern workspace project for UML, GUI and quantum modeling.")
project = Project(
    name="Bank_date",
    models=[domain_model],
    owner="BESSER User",
    metadata=metadata
)
