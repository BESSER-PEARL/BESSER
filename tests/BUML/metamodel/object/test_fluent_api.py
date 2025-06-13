import pytest
from besser.BUML.metamodel.object import *
from besser.BUML.metamodel.structural import (
    Class, Property, Method, Parameter,
    BinaryAssociation, Generalization, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, IntegerType, FloatType, BooleanType,
    TimeType, DateType, DateTimeType, TimeDeltaType,
    AnyType, Constraint, AssociationClass
)

# Domain model definition

# Classes
Person = Class(name="Person")
Player = Class(name="Player")
Team = Class(name="Team")
Coach = Class(name="Coach")

# Person class attributes and methods
Person_country: Property = Property(name="country", type=StringType)
Person.attributes={Person_country}

# Player class attributes and methods
Player_age: Property = Property(name="age", type=IntegerType)
Player_jerseyNumber: Property = Property(name="jerseyNumber", type=IntegerType)
Player_position: Property = Property(name="position", type=StringType)
Player_name: Property = Property(name="name", type=StringType)
Player.attributes={Player_jerseyNumber, Player_position, Player_name, Player_age}

# Team class attributes and methods
Team_name: Property = Property(name="name", type=StringType)
Team_city: Property = Property(name="city", type=StringType)
Team_division: Property = Property(name="division", type=StringType)
Team.attributes={Team_city, Team_name, Team_division}

# Coach class attributes and methods
Coach_name: Property = Property(name="name", type=StringType)
Coach_salary: Property = Property(name="salary", type=IntegerType)
Coach.attributes={Coach_name, Coach_salary}

# Relationships
person_team_asso: BinaryAssociation = BinaryAssociation(
    name="person_team_asso",
    ends={
        Property(name="follows", type=Team, multiplicity=Multiplicity(1, 9999)),
        Property(name="fans", type=Person, multiplicity=Multiplicity(0, 9999))
    }
)
team_player_asso: BinaryAssociation = BinaryAssociation(
    name="team_player_asso",
    ends={
        Property(name="plays_for", type=Team, multiplicity=Multiplicity(1, 1)),
        Property(name="players", type=Player, multiplicity=Multiplicity(0, 9999))
    }
)
Team_Coach: BinaryAssociation = BinaryAssociation(
    name="Team_Coach",
    ends={
        Property(name="leads", type=Team, multiplicity=Multiplicity(1, 1)),
        Property(name="lead_by", type=Coach, multiplicity=Multiplicity(1, 9999))
    }
)

# Generalizations
gen_Player_Person = Generalization(general=Person, specific=Player)
gen_Coach_Person = Generalization(general=Person, specific=Coach)

# Domain Model
domain_model = DomainModel(
    name="Team-Playermodel",
    types={Person, Player, Team, Coach},
    associations={person_team_asso, team_player_asso, Team_Coach},
    generalizations={gen_Player_Person, gen_Coach_Person}
)

# Tests for the fluent API of the object model

def test_team_creation():
    team = Team("team1").attributes(name="Lakers", city="LA", division="West").build()
    assert team.name == "Lakers"
    assert team.city == "LA"
    assert team.division == "West"

def test_player_creation():
    player = Player("player1").attributes(name="LeBron", age=38, position="Forward", jerseyNumber=6, country="USA").build()
    assert player.name == "LeBron"
    assert player.age == 38
    assert player.position == "Forward"
    assert player.jerseyNumber == 6
    assert player.country == "USA"

def test_coach_creation():
    coach = Coach("coach1").attributes(name="Vogel", salary=100, country="USA").build()
    assert coach.name == "Vogel"
    assert coach.salary == 100
    assert coach.country == "USA"

def test_player_plays_for_team():
    team = Team("team2").attributes(name="Warriors", city="SF", division="West").build()
    player = Player("player2").attributes(name="Curry", age=34, position="Guard", jerseyNumber=30, country="USA") \
        .link(team, "plays_for").build()
    assert player.plays_for == team
    assert player.plays_for.division == "West"

def test_player_follows_team():
    team = Team("team3").attributes(name="Bulls", city="Chicago", division="East").build()
    player = Player("player3").attributes(name="Jordan", age=59, position="Guard", jerseyNumber=23, country="USA") \
        .link(team, "follows").build()
    assert player.follows == team
    assert player.follows.city == "Chicago"

def test_coach_leads_team():
    team = Team("team4").attributes(name="Celtics", city="Boston", division="East").build()
    coach = Coach("coach2").attributes(name="Stevens", salary=120, country="USA") \
        .link(team, "leads").build()
    assert coach.leads == team
    assert coach.leads.city == "Boston"

def test_coach_follows_team():
    team = Team("team5").attributes(name="Heat", city="Miami", division="East").build()
    team2 = Team("team2").attributes(name="Test", city="Lux", division="East").build()
    coach = Coach("coach3").attributes(name="Spoelstra", salary=110, country="USA") \
        .link({team,team2}, "follows").build()
    assert team in coach.follows
    assert any(t.city == "Miami" for t in coach.follows)

def test_object_model_creation():
    team = Team("team6").attributes(name="Knicks", city="NY", division="East").build()
    player = Player("player4").attributes(name="Barrett", age=22, position="Guard", jerseyNumber=9, country="Canada").build()
    coach = Coach("coach4").attributes(name="Thibodeau", salary=90, country="USA").build()
    obj_model = ObjectModel(name="NBA_Model", objects={team, player, coach})
    assert team in obj_model.objects
    assert player in obj_model.objects
    assert coach in obj_model.objects

def test_multiple_links():
    team1 = Team("team8").attributes(name="Nets", city="Brooklyn", division="East").build()
    team2 = Team("team9").attributes(name="Raptors", city="Toronto", division="East").build()
    player = Player("player6").attributes(name="Durant", age=33, position="Forward", jerseyNumber=7, country="USA") \
        .link(team1, "plays_for") \
        .link({team1, team2}, "follows").build()
    assert player.plays_for == team1
    assert team2 in player.follows
    assert team1 in player.follows

def test_update_attributes_after_build():
    team = Team("team10").attributes(name="Spurs", city="San Antonio", division="West").build()
    team.city = "Austin"
    assert team.city == "Austin"

def test_object_internal_name():
    team = Team("team11").attributes(name="Kings", city="Sacramento", division="West").build()
    assert team.name == "Kings"
    assert team.name_ == "team11"
    team.name_ = "new_team_internal_name"
    assert team.name_ == "new_team_internal_name"
    assert team.name == "Kings"