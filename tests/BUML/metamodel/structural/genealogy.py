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
Adam = Class(name="Adam")
Eve = Class(name="Eve")
Man = Class(name="Man")
Person = Class(name="Person", is_abstract=True)
Woman = Class(name="Woman")

# Adam class attributes and methods

# Eve class attributes and methods

# Man class attributes and methods

# Person class attributes and methods

# Woman class attributes and methods

# Relationships
is_child_of: BinaryAssociation = BinaryAssociation(
    name="is_child_of",
    ends={
        Property(name="person_1", type=Person, multiplicity=Multiplicity(1, 1), is_navigable=False),
        Property(name="parents", type=Person, multiplicity=Multiplicity(0, 9999))
    }
)
is_coupled_with: BinaryAssociation = BinaryAssociation(
    name="is_coupled_with",
    ends={
        Property(name="person", type=Person, multiplicity=Multiplicity(1, 1), is_navigable=False),
        Property(name="spouse", type=Person, multiplicity=Multiplicity(0, 1))
    }
)

# Generalizations
gen_Adam_Man = Generalization(general=Man, specific=Adam)
gen_Eve_Woman = Generalization(general=Woman, specific=Eve)
gen_Man_Person = Generalization(general=Person, specific=Man)
gen_Woman_Person = Generalization(general=Person, specific=Woman)


# OCL Constraints
Adam_inv_1_1: Constraint = Constraint(
    name="Adam_inv_1_1",
    context=Adam,
    expression="context Adam inv Adam_inv_1_1 :  Adam.allInstances()->size() = 1",
    language="OCL"
)
Adam_inv_4_1: Constraint = Constraint(
    name="Adam_inv_4_1",
    context=Adam,
    expression="context Adam inv Adam_inv_4_1 : self.spouse.oclIsTypeOf(Eve) and self.spouse->size()=1",
    language="OCL"
)
Adam_inv_8_1: Constraint = Constraint(
    name="Adam_inv_8_1",
    context=Adam,
    expression="context Adam inv Adam_inv_8_1 : self.parents->isEmpty()",
    language="OCL"
)
Eve_inv_2_1: Constraint = Constraint(
    name="Eve_inv_2_1",
    context=Eve,
    expression="context Eve inv Eve_inv_2_1 :  Eve.allInstances()->size() = 1",
    language="OCL"
)
Eve_inv_6_1: Constraint = Constraint(
    name="Eve_inv_6_1",
    context=Eve,
    expression="context Eve inv Eve_inv_6_1 : self.spouse.oclIsTypeOf(Adam) and self.spouse->size()=1",
    language="OCL"
)
Eve_inv_7_1: Constraint = Constraint(
    name="Eve_inv_7_1",
    context=Eve,
    expression="context Eve inv Eve_inv_7_1 : self.parents ->isEmpty()",
    language="OCL"
)
Man_inv_11_1: Constraint = Constraint(
    name="Man_inv_11_1",
    context=Man,
    expression="context Man inv Man_inv_11_1 : self.spouse.oclIsKindOf(Woman)",
    language="OCL"
)
Person_inv_10_1: Constraint = Constraint(
    name="Person_inv_10_1",
    context=Person,
    expression="context Person inv Person_inv_10_1 : self.parents->size()=2   implies self.parents->exists(p,q: Person | p.oclIsKindOf(Woman) and q.oclIsKindOf(Man))",
    language="OCL"
)
Person_inv_13_1: Constraint = Constraint(
    name="Person_inv_13_1",
    context=Person,
    expression="context Person inv Person_inv_13_1 : self.spouse.parents->intersection(self.parents)->isEmpty()",
    language="OCL"
)
Person_inv_14_1: Constraint = Constraint(
    name="Person_inv_14_1",
    context=Person,
    expression="context Person inv Person_inv_14_1  : self.parents->intersection(self.spouse)->isEmpty()",
    language="OCL"
)
Person_inv_3_1: Constraint = Constraint(
    name="Person_inv_3_1",
    context=Person,
    expression="context Person inv Person_inv_3_1 :  self.parents->closure(parents)->excludes(self)",
    language="OCL"
)
Person_inv_5_1: Constraint = Constraint(
    name="Person_inv_5_1",
    context=Person,
    expression="context Person inv Person_inv_5_1 : (self.spouse<>null) implies (self= self.spouse.spouse)",
    language="OCL"
)
Woman_inv_12_1: Constraint = Constraint(
    name="Woman_inv_12_1",
    context=Woman,
    expression="context Woman inv Woman_inv_12_1 : self.spouse.oclIsKindOf(Man)",
    language="OCL"
)
Person_inv_9_1: Constraint = Constraint(
    name="Person_inv_9_1",
    context=Person,
    expression="context Person inv Person_inv_9_1 :  not (self.oclIsTypeOf(Adam) or self.oclIsTypeOf(Eve)) implies (self.parents->size()=2)",
    language="OCL"
)

# Domain Model
domain_model = DomainModel(
    name="Class_Diagram",
    types={Adam, Eve, Man, Person, Woman},
    associations={is_child_of, is_coupled_with},
    constraints={Adam_inv_1_1, Adam_inv_4_1, Adam_inv_8_1, Eve_inv_2_1, Eve_inv_6_1, Eve_inv_7_1, Man_inv_11_1, Person_inv_10_1, Person_inv_13_1, Person_inv_14_1, Person_inv_3_1, Person_inv_5_1, Woman_inv_12_1, Person_inv_9_1},
    generalizations={gen_Adam_Man, gen_Eve_Woman, gen_Man_Person, gen_Woman_Person},
    metadata=None
)


######################
# PROJECT DEFINITION #
######################

from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural.structural import Metadata

metadata = Metadata(description="")
project = Project(
    name="Genealogy_besser",
    models=[domain_model],
    owner="BESSER User",
    metadata=metadata
)
