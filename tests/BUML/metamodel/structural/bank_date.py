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

# Enumerations
TCategory: Enumeration = Enumeration(
    name="TCategory",
    literals={
            EnumerationLiteral(name="EXECUTIVE"),
			EnumerationLiteral(name="JUNIOR")
    }
)

# Classes
Employee = Class(name="Employee")
Person = Class(name="Person")

# Employee class attributes and methods
Employee_category: Property = Property(name="category", type=TCategory)
Employee_idEmployee: Property = Property(name="idEmployee", type=IntegerType)
Employee.attributes={Employee_category, Employee_idEmployee}

# Person class attributes and methods
Person_birthDate: Property = Property(name="birthDate", type=DateType)
Person_dni: Property = Property(name="dni", type=StringType)
Person_name: Property = Property(name="name", type=StringType)
Person.attributes={Person_birthDate, Person_dni, Person_name}

# Generalizations
gen_Employee_Person = Generalization(general=Person, specific=Employee)


# OCL Constraints
Person_inv_1_1: Constraint = Constraint(
    name="Person_inv_1_1",
    context=Person,
    expression="context Person inv Person_inv_1_1 : self.birthDate>\'2/20/2000\'",
    language="OCL"
)
Person_inv_1_2: Constraint = Constraint(
    name="Person_inv_1_2",
    context=Employee,
    expression="context Employee inv Person_inv_1_2 : self.category=TCategory::JUNIOR",
    language="OCL"
)

# Domain Model
domain_model = DomainModel(
    name="Class_Diagram",
    types={Employee, Person, TCategory},
    associations={},
    constraints={Person_inv_1_1, Person_inv_1_2},
    generalizations={gen_Employee_Person},
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
