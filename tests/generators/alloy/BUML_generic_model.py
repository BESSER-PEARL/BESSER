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
ClaseA0 = Class(name="ClaseA0", is_abstract=True)
ClaseA1 = Class(name="ClaseA1", is_abstract=True)
ClaseA = Class(name="ClaseA")
ClaseB = Class(name="ClaseB")
ClaseD = Class(name="ClaseD")

# ClaseA0 class attributes and methods
ClaseA0_ida0: Property = Property(name="ida0", type=IntegerType)
ClaseA0.attributes={ClaseA0_ida0}

# ClaseA1 class attributes and methods
ClaseA1_ida1: Property = Property(name="ida1", type=IntegerType)
ClaseA1.attributes={ClaseA1_ida1}

# ClaseA class attributes and methods
ClaseA_at1: Property = Property(name="at1", type=IntegerType)
ClaseA_at2: Property = Property(name="at2", type=StringType)
ClaseA_atr3: Property = Property(name="atr3", type=FloatType)
ClaseA_at4: Property = Property(name="at4", type=IntegerType)
ClaseA_at5: Property = Property(name="at5", type=DateType)
ClaseA.attributes={ClaseA_at1, ClaseA_at2, ClaseA_atr3, ClaseA_at4, ClaseA_at5}

# ClaseB class attributes and methods
ClaseB_bt1: Property = Property(name="bt1", type=IntegerType)
ClaseB_bt2: Property = Property(name="bt2", type=StringType)
ClaseB_btr3: Property = Property(name="btr3", type=FloatType)
ClaseB_bt4: Property = Property(name="bt4", type=DateType)
ClaseB.attributes={ClaseB_bt1, ClaseB_bt2, ClaseB_btr3, ClaseB_bt4}

# ClaseD class attributes and methods
ClaseD_dt1: Property = Property(name="dt1", type=IntegerType)
ClaseD_dt2: Property = Property(name="dt2", type=IntegerType)
ClaseD.attributes={ClaseD_dt1, ClaseD_dt2}

# Relationships
D_B_1: BinaryAssociation = BinaryAssociation(
    name="D_B_1",
    ends={
        Property(name="reld", type=ClaseD, multiplicity=Multiplicity(1, 1), is_navigable=False),
        Property(name="rolb", type=ClaseB, multiplicity=Multiplicity(0, 1))
    }
)
D_B_2: BinaryAssociation = BinaryAssociation(
    name="D_B_2",
    ends={
        Property(name="reld1", type=ClaseD, multiplicity=Multiplicity(1, 1), is_navigable=False),
        Property(name="reld2", type=ClaseB, multiplicity=Multiplicity(1, 9999))
    }
)
D_A0: BinaryAssociation = BinaryAssociation(
    name="D_A0",
    ends={
        Property(name="d", type=ClaseD, multiplicity=Multiplicity(1, 1)),
        Property(name="a0s", type=ClaseA0, multiplicity=Multiplicity(2, 2))
    }
)

# Association Classes
ab: BinaryAssociation = BinaryAssociation(
    name="ab",
    ends={
        Property(name="aes", type=ClaseA, multiplicity=Multiplicity(2, 2)),
        Property(name="bes", type=ClaseB, multiplicity=Multiplicity(3, 5))
    }
)

ClaseC_ctr1: Property = Property(name="ctr1", type=IntegerType)
ClaseC_ctr2: Property = Property(name="ctr2", type=StringType)
ClaseC = AssociationClass(
    name="ClaseC",
    attributes={ClaseC_ctr1, ClaseC_ctr2}, association=ab
)

# Generalizations
gen_ClaseA1_ClaseA0 = Generalization(general=ClaseA0, specific=ClaseA1)
gen_ClaseA_ClaseA1 = Generalization(general=ClaseA1, specific=ClaseA)

# OCL Constraints
ClaseA_inv_1_1: Constraint = Constraint(
    name="ClaseA_inv_1_1",
    context=ClaseA,
    expression="context ClaseA inv ClaseA_inv_1_1 : self.at1>0 implies self.at4<0",
    language="OCL"
)
ClaseA_inv_2_1: Constraint = Constraint(
    name="ClaseA_inv_2_1",
    context=ClaseA,
    expression="context ClaseA inv ClaseA_inv_2_1 : self.at1<0 implies false",
    language="OCL"
)
ClaseA_inv_3_1: Constraint = Constraint(
    name="ClaseA_inv_3_1",
    context=ClaseA,
    expression="context ClaseA inv ClaseA_inv_3_1 : if self.at1>0 then self.at4>self.at1 else self.at4<=0 endif",
    language="OCL"
)
ClaseD_inv_4_1: Constraint = Constraint(
    name="ClaseD_inv_4_1",
    context=ClaseD,
    expression="context ClaseD inv ClaseD_inv_4_1 : self.dt1=2 xor self.dt2='pepe'",
    language="OCL"
)
ClaseB_inv_5_1: Constraint = Constraint(
    name="ClaseB_inv_5_1",
    context=ClaseB,
    expression="context ClaseB inv ClaseB_inv_5_1 : self.bt1=self.bt2.abs()",
    language="OCL"
)
ClaseD_inv_6_1: Constraint = Constraint(
    name="ClaseD_inv_6_1",
    context=ClaseD,
    expression="context ClaseD inv ClaseD_inv_6_1 : self.reld2->size()>3",
    language="OCL"
)
ClaseA_inv_7_1: Constraint = Constraint(
    name="ClaseA_inv_7_1",
    context=ClaseA,
    expression="context ClaseA inv ClaseA_inv_7_1 : ClaseA.allInstances()->select(a | a.at1>0)->size()>=1",
    language="OCL"
)
ClaseD_inv_8_1: Constraint = Constraint(
    name="ClaseD_inv_8_1",
    context=ClaseD,
    expression="context ClaseD inv ClaseD_inv_8_1 : ClaseD.allInstances()->forAll(d | d.dt1>0 and d.dt2>0)",
    language="OCL"
)

# Domain Model
domain_model = DomainModel(
    name="Class_Diagram",
    types={ClaseA0, ClaseA1, ClaseA, ClaseB, ClaseD, ClaseC},
    associations={D_B_1, D_B_2, D_A0, ab},
    constraints={ClaseA_inv_1_1, ClaseA_inv_2_1, ClaseA_inv_3_1, ClaseD_inv_4_1, ClaseB_inv_5_1,
                 ClaseD_inv_6_1, ClaseA_inv_7_1, ClaseD_inv_8_1},
    generalizations={gen_ClaseA1_ClaseA0, gen_ClaseA_ClaseA1},
    metadata=None
)


######################
# PROJECT DEFINITION #
######################

from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural.structural import Metadata

metadata = Metadata(description="Generic model for OCL/Alloy testing.")
project = Project(
    name="Generic_Project",
    models=[domain_model],
    owner="BESSER User",
    metadata=metadata
)
