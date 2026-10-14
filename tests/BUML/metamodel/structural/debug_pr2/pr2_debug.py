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
A = Class(name="A")
B = Class(name="B")
NC = Class(name="NC")
ZZ = Class(name="ZZ")

# A class attributes and methods
A_b: Property = Property(name="b", type=StringType)
A_d: Property = Property(name="d", type=IntegerType)
A_z: Property = Property(name="z", type=IntegerType)
A_pages: Property = Property(name="pages", type=IntegerType)
A.attributes={A_b, A_d, A_pages, A_z}

# B class attributes and methods
B_a: Property = Property(name="a", type=BooleanType)
B_b: Property = Property(name="b", type=BooleanType)
B_g: Property = Property(name="g", type=BooleanType)
B.attributes={B_a, B_b, B_g}

# NC class attributes and methods
NC_n: Property = Property(name="n", type=IntegerType)
NC_k: Property = Property(name="k", type=IntegerType)
NC.attributes={NC_k, NC_n}

# ZZ class attributes and methods
ZZ_Attribute: Property = Property(name="Attribute", type=StringType)
ZZ_f: Property = Property(name="f", type=DateType)
ZZ.attributes={ZZ_Attribute, ZZ_f}

# Relationships
NC_B: BinaryAssociation = BinaryAssociation(
    name="NC_B",
    ends={
        Property(name="nc", type=NC, multiplicity=Multiplicity(1, 1), is_navigable=False),
        Property(name="bs", type=B, multiplicity=Multiplicity(1, 1))
    }
)
A_ZZ: BinaryAssociation = BinaryAssociation(
    name="A_ZZ",
    ends={
        Property(name="a", type=A, multiplicity=Multiplicity(1, 1), is_navigable=False),
        Property(name="zs", type=ZZ, multiplicity=Multiplicity(1, 9999))
    }
)

# Association Classes
ab: BinaryAssociation = BinaryAssociation(
    name="ab",
    ends={
        Property(name="aes", type=A, multiplicity=Multiplicity(2, 2)),
        Property(name="bes", type=B, multiplicity=Multiplicity(1, 3))
    }
)

AC_cv: Property = Property(name="cv", type=StringType)
AC = AssociationClass(
    name="AC",
    attributes={AC_cv}, association=ab
)



# OCL Constraints
A_inv_1_1: Constraint = Constraint(
    name="A_inv_1_1",
    context=A,
    expression="context A inv A_inv_1_1 :     self.d>0 implies false",
    language="OCL"
)
A_inv_2_1: Constraint = Constraint(
    name="A_inv_2_1",
    context=A,
    expression="context A inv A_inv_2_1 : self.z *2 >10",
    language="OCL"
)
B_inv_3_1: Constraint = Constraint(
    name="B_inv_3_1",
    context=B,
    expression="context B inv B_inv_3_1 : self.a xor self.b",
    language="OCL"
)
B_inv_4_1: Constraint = Constraint(
    name="B_inv_4_1",
    context=B,
    expression="context B inv B_inv_4_1 : self.a implies self.b or self.g",
    language="OCL"
)
NC_inv_5_1: Constraint = Constraint(
    name="NC_inv_5_1",
    context=NC,
    expression="context NC inv NC_inv_5_1 : self.k=self.n.abs()",
    language="OCL"
)
A_inv_6_1: Constraint = Constraint(
    name="A_inv_6_1",
    context=A,
    expression="context A inv A_inv_6_1 : self.pages > 20",
    language="OCL"
)
A_inv_7_1: Constraint = Constraint(
    name="A_inv_7_1",
    context=A,
    expression="context A inv A_inv_7_1 : self.zs->size()>3",
    language="OCL"
)

# Domain Model
domain_model = DomainModel(
    name="Class_Diagram",
    types={A, B, NC, ZZ, AC},
    associations={NC_B, A_ZZ, ab},
    constraints={A_inv_1_1, A_inv_2_1, B_inv_3_1, B_inv_4_1, NC_inv_5_1, A_inv_6_1, A_inv_7_1},
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
    name="New_Project",
    models=[domain_model],
    owner="BESSER User",
    metadata=metadata
)
