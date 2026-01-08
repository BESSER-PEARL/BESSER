####################
# STRUCTURAL MODEL #
####################

from besser.BUML.metamodel.structural import (
    Class, Property, Method, Parameter,
    BinaryAssociation, Generalization, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, IntegerType, FloatType, BooleanType,
    TimeType, DateType, DateTimeType, TimeDeltaType,
    AnyType, Constraint, AssociationClass, Metadata
)

# Classes
Team = Class(name="Team")
Player = Class(name="Player")

# Team class attributes and methods
Team_city: Property = Property(name="city", type=StringType)
Team_name: Property = Property(name="name", type=StringType)
Team_division: Property = Property(name="division", type=StringType)
Team.attributes={Team_division, Team_name, Team_city}

# Player class attributes and methods
Player_position: Property = Property(name="position", type=StringType)
Player_age: Property = Property(name="age", type=IntegerType)
Player_name: Property = Property(name="name", type=StringType)
Player_jerseyNumber: Property = Property(name="jerseyNumber", type=IntegerType)
Player.attributes={Player_jerseyNumber, Player_age, Player_name, Player_position}

# Relationships
team_player: BinaryAssociation = BinaryAssociation(
    name="team_player",
    ends={
        Property(name="team", type=Team, multiplicity=Multiplicity(1, 1)),
        Property(name="players", type=Player, multiplicity=Multiplicity(0, 9999))
    }
)


# OCL Constraints
constraint_Player_0_1: Constraint = Constraint(
    name="constraint_Player_0_1",
    context=Player,
    expression="context Player inv inv1: self.age > 10",
    language="OCL"
)

# Domain Model
domain_model = DomainModel(
    name="Class_Diagram",
    types={Team, Player},
    associations={team_player},
    constraints={constraint_Player_0_1},
    generalizations={},
    metadata=None
)


######################
# PROJECT DEFINITION #
######################

from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural.structural import Metadata

metadata = Metadata(description="New project")
project = Project(
    name="UnnamedProject",
    models=[domain_model],
    owner="User",
    metadata=metadata
)
