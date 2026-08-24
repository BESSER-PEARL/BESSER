####################
# STRUCTURAL MODEL #
####################
from besser.BUML.metamodel.structural import (
    Class, Property, BinaryAssociation, Generalization, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, IntegerType, DateType, Constraint, Metadata
)
from besser.generators.alloy_generator import AlloyGenerator

# Enumerations
Tposition: Enumeration = Enumeration(
    name="Tposition",
    literals={
            EnumerationLiteral(name="GOALKEEPEAR"),
            EnumerationLiteral(name="DEFENDER"),
            EnumerationLiteral(name="MIDFIELDER"),
            EnumerationLiteral(name="FORWARD")
    }
)

# Classes
Persona = Class(name="Persona")
Fan = Class(name="Fan")
Team = Class(name="Team")
Player = Class(name="Player")
City = Class(name="City")

# Persona class attributes and methods
Persona_fdn: Property = Property(name="fnac", type=DateType)
Persona_ndp: Property = Property(name="ndp", type=StringType)
Persona_id: Property = Property(name="id", type=IntegerType)
Persona.attributes={Persona_ndp, Persona_id,Persona_fdn}

# Fan class attributes and methods
Fan_name: Property = Property(name="name", type=StringType)
Fan_age: Property = Property(name="age", type=IntegerType)
Fan_position: Property = Property(name="position", type=StringType)
Fan_jerseyNumber: Property = Property(name="jerseyNumber", type=IntegerType)
Fan.attributes={Fan_age, Fan_name, Fan_position, Fan_jerseyNumber}

# Team class attributes and methods
Team_division: Property = Property(name="division", type=StringType)
Team_name: Property = Property(name="name", type=StringType)
Team.attributes={Team_name, Team_division}

# Player class attributes and methods
Player_nickname: Property = Property(name="nickname", type=StringType)
Player_year_in: Property = Property(name="year_in", type=IntegerType)
Player_position: Property = Property(name="position", type=Tposition)
Player.attributes={Player_year_in, Player_position, Player_nickname}

# City class attributes and methods
City_zipCode: Property = Property(name="zipCode", type=StringType)
City_name: Property = Property(name="name", type=StringType)
City.attributes={City_zipCode, City_name}

# Relationships
team_player: BinaryAssociation = BinaryAssociation(
    name="team_player",
    ends={
        Property(name="team", type=Team, multiplicity=Multiplicity(1, 1)),
        Property(name="fans", type=Fan, multiplicity=Multiplicity(2, 4))
    }
)
Player_Team: BinaryAssociation = BinaryAssociation(
    name="Player_Team",
    ends={
        Property(name="contains", type=Player, multiplicity=Multiplicity(1, 11)),
        Property(name="plays_in", type=Team, multiplicity=Multiplicity(1, 1))
    }
)
Team_City: BinaryAssociation = BinaryAssociation(
    name="Team_City",
    ends={
        Property(name="have", type=Team, multiplicity=Multiplicity(0, 9999)),
        Property(name="belongs_to", type=City, multiplicity=Multiplicity(1, 1))
    }
)
Team_Player: BinaryAssociation = BinaryAssociation(
    name="Team_Player",
    ends={
        Property(name="team_1", type=Team, multiplicity=Multiplicity(1, 1)),
        Property(name="captain", type=Player, multiplicity=Multiplicity(1, 1))
    }
)

# Generalizations
gen_Fan_Persona = Generalization(general=Persona, specific=Fan)
gen_Player_Persona = Generalization(general=Persona, specific=Player)


# OCL Constraints
constraint_Team_0_1: Constraint = Constraint(
    name="constraint_Team_0_1",
    context=Persona,
    expression='context Persona inv: self.fnac = "02/05/2024"',
    language="OCL"
)

constraint_Team_1_1: Constraint = Constraint(
    name="constraint_Team_1_1",
    context=Team,
    expression='context Team inv: self.fans ->select(p|p.fnac > "1/1/2000" )->size()=4',
    language="OCL"
)

#ocl='self.Fan ->select(p|p.fdnac > "1/1/2000" )->size()=4'   



# Domain Model
domain_model = DomainModel(
    name="Class_Diagram",
    types={Persona, Fan, Team, Player, City, Tposition},
    associations={team_player, Player_Team, Team_City, Team_Player},
    #constraints={constraint_Team_0_1,constraint_Team_1_1},
    constraints={constraint_Team_1_1},
   
    generalizations={gen_Fan_Persona, gen_Player_Persona},
    metadata=None
)



######################
# PROJECT DEFINITION #
######################

from besser.BUML.metamodel.project import Project

metadata = Metadata(description="Liga AFA")
project = Project(
    name="t",
    models=[domain_model],
    owner="User",
    metadata=metadata
)
alloy_model = AlloyGenerator(model=domain_model)
alloy_model.generate()

