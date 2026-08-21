from besser.BUML.metamodel.structural import (
    Class, Property, Method, Parameter,
    BinaryAssociation, Generalization, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, IntegerType, FloatType, BooleanType,
    TimeType, DateType, DateTimeType, TimeDeltaType,
    AnyType, Constraint, AssociationClass, Metadata, MethodImplementationType
)

# --- Enumeraciones ---
MatchStage: Enumeration = Enumeration(
    name="MatchStage",
    literals={
        EnumerationLiteral(name="GROUP_STAGE"),
        EnumerationLiteral(name="ROUND_OF_16"),
        EnumerationLiteral(name="QUARTER_FINAL"),
        EnumerationLiteral(name="SEMI_FINAL"),
        EnumerationLiteral(name="THIRD_PLACE_PLAYOFF"),
        EnumerationLiteral(name="FINAL")
    }
)

PlayerPosition: Enumeration = Enumeration(
    name="PlayerPosition",
    literals={
        EnumerationLiteral(name="GOALKEEPER"),
        EnumerationLiteral(name="DEFENDER"),
        EnumerationLiteral(name="MIDFIELDER"),
        EnumerationLiteral(name="FORWARD")
    }
)

# --- Clases y atributos ---
Tournament = Class(name="Tournament")
Tournament_name: Property = Property(name="name", type=StringType, visibility="public")
Tournament_year: Property = Property(name="year", type=IntegerType, visibility="public")
Tournament_hostCountry: Property = Property(name="hostCountry", type=StringType, visibility="public")
Tournament_startDate: Property = Property(name="startDate", type=DateType, visibility="public")
Tournament_endDate: Property = Property(name="endDate", type=DateType, visibility="public")
Tournament.attributes = {Tournament_name, Tournament_year, Tournament_hostCountry, Tournament_startDate, Tournament_endDate}

Group = Class(name="Group")
Group_name: Property = Property(name="name", type=StringType, visibility="public")
Group.attributes = {Group_name}

Team = Class(name="Team")
Team_name: Property = Property(name="name", type=StringType, visibility="public")
Team_fifaCode: Property = Property(name="fifaCode", type=StringType, visibility="public")
Team_fifaRanking: Property = Property(name="fifaRanking", type=IntegerType, visibility="public")
Team.attributes = {Team_name, Team_fifaCode, Team_fifaRanking}

Stadium = Class(name="Stadium")
Stadium_name: Property = Property(name="name", type=StringType, visibility="public")
Stadium_city: Property = Property(name="city", type=StringType, visibility="public")
Stadium_capacity: Property = Property(name="capacity", type=IntegerType, visibility="public")
Stadium.attributes = {Stadium_name, Stadium_city, Stadium_capacity}

Match = Class(name="Match")
Match_matchDate: Property = Property(name="matchDate", type=DateTimeType, visibility="public")
Match_stage: Property = Property(name="stage", type=MatchStage, visibility="public")
Match_homeScore: Property = Property(name="homeScore", type=IntegerType, visibility="public")
Match_awayScore: Property = Property(name="awayScore", type=IntegerType, visibility="public")
Match.attributes = {Match_matchDate, Match_stage, Match_homeScore, Match_awayScore}

Person = Class(name="Person", is_abstract=True)
Person_firstName: Property = Property(name="firstName", type=StringType, visibility="public")
Person_lastName: Property = Property(name="lastName", type=StringType, visibility="public")
Person_nationality: Property = Property(name="nationality", type=StringType, visibility="public")
Person_birthDate: Property = Property(name="birthDate", type=DateType, visibility="public")
Person.attributes = {Person_firstName, Person_lastName, Person_nationality, Person_birthDate}

Player = Class(name="Player")
Player_position: Property = Property(name="position", type=PlayerPosition, visibility="public")
Player_shirtNumber: Property = Property(name="shirtNumber", type=IntegerType, visibility="public")
Player_goalsScored: Property = Property(name="goalsScored", type=IntegerType, visibility="public")
Player.attributes = {Player_position, Player_shirtNumber, Player_goalsScored}

Coach = Class(name="Coach")
Coach_yearsOfExperience: Property = Property(name="yearsOfExperience", type=IntegerType, visibility="public")
Coach.attributes = {Coach_yearsOfExperience}

Referee = Class(name="Referee")
Referee_matchesOfficiated: Property = Property(name="matchesOfficiated", type=IntegerType, visibility="public")
Referee.attributes = {Referee_matchesOfficiated}

# --- Generalizaciones ---
gen_player_person: Generalization = Generalization(general=Person, specific=Player)
gen_coach_person: Generalization = Generalization(general=Person, specific=Coach)
gen_referee_person: Generalization = Generalization(general=Person, specific=Referee)

# --- Asociaciones ---
TournamentTeams: BinaryAssociation = BinaryAssociation(
    name="TournamentTeams",
    ends={
        Property(name="tournament", type=Tournament, multiplicity=Multiplicity(1, 1)),
        Property(name="participatingTeams", type=Team, multiplicity=Multiplicity(0, 9999))
    }
)

TournamentGroups: BinaryAssociation = BinaryAssociation(
    name="TournamentGroups",
    ends={
        Property(name="tournament", type=Tournament, multiplicity=Multiplicity(1, 1)),
        Property(name="groups", type=Group, multiplicity=Multiplicity(0, 9999))
    }
)

GroupTeams: BinaryAssociation = BinaryAssociation(
    name="GroupTeams",
    ends={
        Property(name="group", type=Group, multiplicity=Multiplicity(1, 1)),
        Property(name="teams", type=Team, multiplicity=Multiplicity(2, 8))
    }
)

TournamentMatches: BinaryAssociation = BinaryAssociation(
    name="TournamentMatches",
    ends={
        Property(name="tournament", type=Tournament, multiplicity=Multiplicity(1, 1)),
        Property(name="matches", type=Match, multiplicity=Multiplicity(0, 9999))
    }
)

MatchStadium: BinaryAssociation = BinaryAssociation(
    name="MatchStadium",
    ends={
        Property(name="matches", type=Match, multiplicity=Multiplicity(0, 9999)),
        Property(name="stadium", type=Stadium, multiplicity=Multiplicity(1, 1))
    }
)

MatchHomeTeam: BinaryAssociation = BinaryAssociation(
    name="MatchHomeTeam",
    ends={
        Property(name="homeMatches", type=Match, multiplicity=Multiplicity(0, 9999)),
        Property(name="homeTeam", type=Team, multiplicity=Multiplicity(1, 1))
    }
)

MatchAwayTeam: BinaryAssociation = BinaryAssociation(
    name="MatchAwayTeam",
    ends={
        Property(name="awayMatches", type=Match, multiplicity=Multiplicity(0, 9999)),
        Property(name="awayTeam", type=Team, multiplicity=Multiplicity(1, 1))
    }
)

TeamSquad: BinaryAssociation = BinaryAssociation(
    name="TeamSquad",
    ends={
        Property(name="team", type=Team, multiplicity=Multiplicity(1, 1)),
        Property(name="players", type=Player, multiplicity=Multiplicity(1, 26))
    }
)

TeamCoach: BinaryAssociation = BinaryAssociation(
    name="TeamCoach",
    ends={
        Property(name="team", type=Team, multiplicity=Multiplicity(0, 1)),
        Property(name="coach", type=Coach, multiplicity=Multiplicity(0, 1))
    }
)

MatchReferees: BinaryAssociation = BinaryAssociation(
    name="MatchReferees",
    ends={
        Property(name="matches", type=Match, multiplicity=Multiplicity(0, 9999)),
        Property(name="referees", type=Referee, multiplicity=Multiplicity(1, 3))
    }
)

# --- Restricciones activas (todas confirmadas) ---
constraint_match_different_teams: Constraint = Constraint(
    name="Match_inv_DifferentTeams",
    context=Match,
    expression="context Match inv Match_inv_DifferentTeams: self.homeTeam <> self.awayTeam",
    language="OCL"
)

constraint_match_valid_scores: Constraint = Constraint(
    name="Match_inv_ValidScores",
    context=Match,
    expression="context Match inv Match_inv_ValidScores: self.homeScore >= 0 and self.awayScore >= 0",
    language="OCL"
)

constraint_group_teams_count: Constraint = Constraint(
    name="Group_inv_GroupTeamsCount",
    context=Group,
    expression="context Group inv Group_inv_GroupTeamsCount: self.teams->size() = 4",
    language="OCL"
)

constraint_tournament_has_final: Constraint = Constraint(
    name="Tournament_inv_HasFinal",
    context=Tournament,
    expression="context Tournament inv Tournament_inv_HasFinal: self.matches->exists(m | m.stage = MatchStage::FINAL)",
    language="OCL"
)

constraint_player_valid_shirt_number: Constraint = Constraint(
    name="Player_inv_ValidShirtNumber",
    context=Player,
    expression="context Player inv Player_inv_ValidShirtNumber: self.shirtNumber > 0 and self.shirtNumber <= 99",
    language="OCL"
)

constraint_team_unique_shirt_numbers: Constraint = Constraint(
    name="Team_inv_UniqueShirtNumbers",
    context=Team,
    expression="context Team inv Team_inv_UniqueShirtNumbers: self.players->forAll(p1, p2 | p1 <> p2 implies p1.shirtNumber <> p2.shirtNumber)",
    language="OCL"
)

domain_model: DomainModel = DomainModel(
    name="FIFAWorldCup",
    types={
        Tournament, Group, Team, Stadium, Match,
        Person, Player, Coach, Referee,
        MatchStage, PlayerPosition
    },
    associations={
        TournamentTeams, TournamentGroups, GroupTeams, TournamentMatches,
        MatchStadium, MatchHomeTeam, MatchAwayTeam, TeamSquad, TeamCoach, MatchReferees
    },
    generalizations={gen_player_person, gen_coach_person, gen_referee_person},
    constraints={
        constraint_match_different_teams, constraint_match_valid_scores,
        constraint_group_teams_count, constraint_tournament_has_final,
        constraint_player_valid_shirt_number, constraint_team_unique_shirt_numbers
    },
    metadata=None
)

from besser.BUML.metamodel.project import Project

metadata = Metadata(description="A model of a FIFA World Cup tournament, including teams, matches, stadiums, and tournament personnel.")
project = Project(
    name="FIFAWorldCup",
    models=[domain_model],
    owner="BESSER User",
    metadata=metadata
)
