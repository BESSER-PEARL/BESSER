####################
# STRUCTURAL MODEL #
####################

from besser.BUML.metamodel.structural import (
    Class, Property, BinaryAssociation, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, IntegerType, FloatType, BooleanType,
    DateType, DateTimeType, Constraint
)

# ─────────────────────────────────────────────
# ENUMERATIONS
# ─────────────────────────────────────────────

CardType = Enumeration(name="CardType")
CardType_YELLOW = EnumerationLiteral(name="YELLOW", owner=CardType)
CardType_RED = EnumerationLiteral(name="RED", owner=CardType)
CardType_YELLOW_RED = EnumerationLiteral(name="YELLOW_RED", owner=CardType)
CardType.literals = {CardType_YELLOW, CardType_RED, CardType_YELLOW_RED}

GoalType = Enumeration(name="GoalType")
GoalType_REGULAR = EnumerationLiteral(name="REGULAR", owner=GoalType)
GoalType_PENALTY = EnumerationLiteral(name="PENALTY", owner=GoalType)
GoalType_OWN_GOAL = EnumerationLiteral(name="OWN_GOAL", owner=GoalType)
GoalType_FREE_KICK = EnumerationLiteral(name="FREE_KICK", owner=GoalType)
GoalType.literals = {GoalType_REGULAR, GoalType_PENALTY, GoalType_OWN_GOAL, GoalType_FREE_KICK}

MatchStatus = Enumeration(name="MatchStatus")
MatchStatus_SCHEDULED = EnumerationLiteral(name="SCHEDULED", owner=MatchStatus)
MatchStatus_IN_PROGRESS = EnumerationLiteral(name="IN_PROGRESS", owner=MatchStatus)
MatchStatus_FINISHED = EnumerationLiteral(name="FINISHED", owner=MatchStatus)
MatchStatus_CANCELLED = EnumerationLiteral(name="CANCELLED", owner=MatchStatus)
MatchStatus.literals = {MatchStatus_SCHEDULED, MatchStatus_IN_PROGRESS, MatchStatus_FINISHED, MatchStatus_CANCELLED}

PhaseType = Enumeration(name="PhaseType")
PhaseType_GROUP_STAGE = EnumerationLiteral(name="GROUP_STAGE", owner=PhaseType)
PhaseType_ROUND_OF_16 = EnumerationLiteral(name="ROUND_OF_16", owner=PhaseType)
PhaseType_QUARTER_FINAL = EnumerationLiteral(name="QUARTER_FINAL", owner=PhaseType)
PhaseType_SEMI_FINAL = EnumerationLiteral(name="SEMI_FINAL", owner=PhaseType)
PhaseType_THIRD_PLACE = EnumerationLiteral(name="THIRD_PLACE", owner=PhaseType)
PhaseType_FINAL = EnumerationLiteral(name="FINAL", owner=PhaseType)
PhaseType.literals = {PhaseType_GROUP_STAGE, PhaseType_ROUND_OF_16, PhaseType_QUARTER_FINAL,
                      PhaseType_SEMI_FINAL, PhaseType_THIRD_PLACE, PhaseType_FINAL}

PlayerPosition = Enumeration(name="PlayerPosition")
PlayerPosition_GOALKEEPER = EnumerationLiteral(name="GOALKEEPER", owner=PlayerPosition)
PlayerPosition_DEFENDER = EnumerationLiteral(name="DEFENDER", owner=PlayerPosition)
PlayerPosition_MIDFIELDER = EnumerationLiteral(name="MIDFIELDER", owner=PlayerPosition)
PlayerPosition_FORWARD = EnumerationLiteral(name="FORWARD", owner=PlayerPosition)
PlayerPosition.literals = {PlayerPosition_GOALKEEPER, PlayerPosition_DEFENDER,
                            PlayerPosition_MIDFIELDER, PlayerPosition_FORWARD}

RefereeRole = Enumeration(name="RefereeRole")
RefereeRole_MAIN = EnumerationLiteral(name="MAIN", owner=RefereeRole)
RefereeRole_ASSISTANT = EnumerationLiteral(name="ASSISTANT", owner=RefereeRole)
RefereeRole_VAR = EnumerationLiteral(name="VAR", owner=RefereeRole)
RefereeRole_FOURTH_OFFICIAL = EnumerationLiteral(name="FOURTH_OFFICIAL", owner=RefereeRole)
RefereeRole.literals = {RefereeRole_MAIN, RefereeRole_ASSISTANT, RefereeRole_VAR, RefereeRole_FOURTH_OFFICIAL}

# ─────────────────────────────────────────────
# CLASSES
# ─────────────────────────────────────────────

Country = Class(name="Country")
Stadium = Class(name="Stadium")
TournamentEdition = Class(name="TournamentEdition")
TournamentPhase = Class(name="TournamentPhase")
Group = Class(name="Group")
GroupStanding = Class(name="GroupStanding")
Team = Class(name="Team")
Player = Class(name="Player")
Referee = Class(name="Referee")
MatchReferee = Class(name="MatchReferee")
Match = Class(name="Match")
Goal = Class(name="Goal")
Card = Class(name="Card")
Substitution = Class(name="Substitution")

# ─────────────────────────────────────────────
# ATTRIBUTES
# ─────────────────────────────────────────────

# Country
Country_name: Property = Property(name="name", type=StringType)
Country_isoCode: Property = Property(name="isoCode", type=StringType)
Country_confederation: Property = Property(name="confederation", type=StringType)
Country.attributes = {Country_name, Country_isoCode, Country_confederation}

# Stadium
Stadium_name: Property = Property(name="name", type=StringType)
Stadium_city: Property = Property(name="city", type=StringType)
Stadium_capacity: Property = Property(name="capacity", type=IntegerType)
Stadium.attributes = {Stadium_name, Stadium_city, Stadium_capacity}

# TournamentEdition
TournamentEdition_year: Property = Property(name="year", type=IntegerType)
TournamentEdition_name: Property = Property(name="name", type=StringType)
TournamentEdition_startDate: Property = Property(name="startDate", type=DateType)
TournamentEdition_endDate: Property = Property(name="endDate", type=DateType)
TournamentEdition.attributes = {TournamentEdition_year, TournamentEdition_name,
                                 TournamentEdition_startDate, TournamentEdition_endDate}

# TournamentPhase
TournamentPhase_name: Property = Property(name="name", type=StringType)
TournamentPhase_phaseOrder: Property = Property(name="phaseOrder", type=IntegerType)
TournamentPhase_phaseType: Property = Property(name="phaseType", type=PhaseType)
TournamentPhase.attributes = {TournamentPhase_name, TournamentPhase_phaseOrder, TournamentPhase_phaseType}

# Group
Group_name: Property = Property(name="name", type=StringType)
Group.attributes = {Group_name}

# GroupStanding
GroupStanding_played: Property = Property(name="played", type=IntegerType)
GroupStanding_won: Property = Property(name="won", type=IntegerType)
GroupStanding_drawn: Property = Property(name="drawn", type=IntegerType)
GroupStanding_lost: Property = Property(name="lost", type=IntegerType)
GroupStanding_goalsFor: Property = Property(name="goalsFor", type=IntegerType)
GroupStanding_goalsAgainst: Property = Property(name="goalsAgainst", type=IntegerType)
GroupStanding_goalDifference: Property = Property(name="goalDifference", type=IntegerType)
GroupStanding_points: Property = Property(name="points", type=IntegerType)
GroupStanding_position: Property = Property(name="position", type=IntegerType)
GroupStanding.attributes = {GroupStanding_played, GroupStanding_won, GroupStanding_drawn,
                             GroupStanding_lost, GroupStanding_goalsFor, GroupStanding_goalsAgainst,
                             GroupStanding_goalDifference, GroupStanding_points, GroupStanding_position}

# Team
Team_name: Property = Property(name="name", type=StringType)
Team_nickname: Property = Property(name="nickname", type=StringType)
Team_headCoach: Property = Property(name="headCoach", type=StringType)
Team_jerseyColor: Property = Property(name="jerseyColor", type=StringType)
Team.attributes = {Team_name, Team_nickname, Team_headCoach, Team_jerseyColor}

# Player
Player_firstName: Property = Property(name="firstName", type=StringType)
Player_lastName: Property = Property(name="lastName", type=StringType)
Player_birthDate: Property = Property(name="birthDate", type=DateType)
Player_birthCountry: Property = Property(name="birthCountry", type=StringType)
Player_jerseyNumber: Property = Property(name="jerseyNumber", type=IntegerType)
Player_heightCm: Property = Property(name="heightCm", type=FloatType)
Player_weightKg: Property = Property(name="weightKg", type=FloatType)
Player_clubName: Property = Property(name="clubName", type=StringType)
Player_isCaptain: Property = Property(name="isCaptain", type=BooleanType)
Player_position: Property = Property(name="position", type=PlayerPosition)
Player.attributes = {Player_firstName, Player_lastName, Player_birthDate, Player_birthCountry,
                     Player_jerseyNumber, Player_heightCm, Player_weightKg,
                     Player_clubName, Player_isCaptain, Player_position}

# Referee
Referee_firstName: Property = Property(name="firstName", type=StringType)
Referee_lastName: Property = Property(name="lastName", type=StringType)
Referee_birthDate: Property = Property(name="birthDate", type=DateType)
Referee_licenseNumber: Property = Property(name="licenseNumber", type=StringType)
Referee.attributes = {Referee_firstName, Referee_lastName, Referee_birthDate, Referee_licenseNumber}

# MatchReferee
MatchReferee_role: Property = Property(name="role", type=RefereeRole)
MatchReferee.attributes = {MatchReferee_role}

# Match
Match_matchDate: Property = Property(name="matchDate", type=DateTimeType)
Match_homeScore: Property = Property(name="homeScore", type=IntegerType)
Match_awayScore: Property = Property(name="awayScore", type=IntegerType)
Match_wentToExtraTime: Property = Property(name="wentToExtraTime", type=BooleanType)
Match_wentToPenaltyShootout: Property = Property(name="wentToPenaltyShootout", type=BooleanType)
Match_homePenaltyScore: Property = Property(name="homePenaltyScore", type=IntegerType)
Match_awayPenaltyScore: Property = Property(name="awayPenaltyScore", type=IntegerType)
Match_attendance: Property = Property(name="attendance", type=IntegerType)
Match_status: Property = Property(name="status", type=MatchStatus)
Match.attributes = {Match_matchDate, Match_homeScore, Match_awayScore, Match_wentToExtraTime,
                    Match_wentToPenaltyShootout, Match_homePenaltyScore, Match_awayPenaltyScore,
                    Match_attendance, Match_status}

# Goal
Goal_minute: Property = Property(name="minute", type=IntegerType)
Goal_extraTimeMinute: Property = Property(name="extraTimeMinute", type=IntegerType)
Goal_goalType: Property = Property(name="goalType", type=GoalType)
Goal.attributes = {Goal_minute, Goal_extraTimeMinute, Goal_goalType}

# Card
Card_minute: Property = Property(name="minute", type=IntegerType)
Card_reason: Property = Property(name="reason", type=StringType)
Card_cardType: Property = Property(name="cardType", type=CardType)
Card.attributes = {Card_minute, Card_reason, Card_cardType}

# Substitution
Substitution_minute: Property = Property(name="minute", type=IntegerType)
Substitution.attributes = {Substitution_minute}

# ─────────────────────────────────────────────
# ASSOCIATIONS
# ─────────────────────────────────────────────

# Stadium -- Country
StadiumCountry: BinaryAssociation = BinaryAssociation(
    name="StadiumCountry",
    ends={
        Property(name="locatedIn", type=Country, multiplicity=Multiplicity(1, 1)),
        Property(name="stadiums", type=Stadium, multiplicity=Multiplicity(0, "*"), is_navigable=False)
    }
)

# TournamentEdition -- Country (host)
EditionHostCountry: BinaryAssociation = BinaryAssociation(
    name="EditionHostCountry",
    ends={
        Property(name="hostedBy", type=Country, multiplicity=Multiplicity(1, "*")),
        Property(name="hosts", type=TournamentEdition, multiplicity=Multiplicity(0, "*"), is_navigable=False)
    }
)

# TournamentPhase -- TournamentEdition
EditionPhase: BinaryAssociation = BinaryAssociation(
    name="EditionPhase",
    ends={
        Property(name="edition", type=TournamentEdition, multiplicity=Multiplicity(1, 1)),
        Property(name="phases", type=TournamentPhase, multiplicity=Multiplicity(1, "*"), is_navigable=False)
    }
)

# Group -- TournamentPhase
GroupPhase: BinaryAssociation = BinaryAssociation(
    name="GroupPhase",
    ends={
        Property(name="phase", type=TournamentPhase, multiplicity=Multiplicity(1, 1)),
        Property(name="groups", type=Group, multiplicity=Multiplicity(0, "*"), is_navigable=False)
    }
)

# Team -- Country
TeamCountry: BinaryAssociation = BinaryAssociation(
    name="TeamCountry",
    ends={
        Property(name="represents", type=Country, multiplicity=Multiplicity(1, 1)),
        Property(name="team", type=Team, multiplicity=Multiplicity(0, 1), is_navigable=False)
    }
)

# Team -- TournamentEdition
TeamEdition: BinaryAssociation = BinaryAssociation(
    name="TeamEdition",
    ends={
        Property(name="edition", type=TournamentEdition, multiplicity=Multiplicity(0, "*")),
        Property(name="teams", type=Team, multiplicity=Multiplicity(16, "*"), is_navigable=False)
    }
)

# Team -- Group
TeamGroup: BinaryAssociation = BinaryAssociation(
    name="TeamGroup",
    ends={
        Property(name="group", type=Group, multiplicity=Multiplicity(0, 1)),
        Property(name="teams", type=Team, multiplicity=Multiplicity(4, 4), is_navigable=False)
    }
)

# Player -- Team
PlayerTeam: BinaryAssociation = BinaryAssociation(
    name="PlayerTeam",
    ends={
        Property(name="team", type=Team, multiplicity=Multiplicity(1, 1)),
        Property(name="players", type=Player, multiplicity=Multiplicity(11, 26), is_navigable=False)
    }
)

# Referee -- Country
RefereeCountry: BinaryAssociation = BinaryAssociation(
    name="RefereeCountry",
    ends={
        Property(name="nationality", type=Country, multiplicity=Multiplicity(1, 1)),
        Property(name="referees", type=Referee, multiplicity=Multiplicity(0, "*"), is_navigable=False)
    }
)

# MatchReferee -- Match
MatchRefereesMatch: BinaryAssociation = BinaryAssociation(
    name="MatchRefereesMatch",
    ends={
        Property(name="match", type=Match, multiplicity=Multiplicity(1, 1)),
        Property(name="refereeAssignments", type=MatchReferee, multiplicity=Multiplicity(2, 5), is_navigable=False)
    }
)

# MatchReferee -- Referee
MatchRefereesReferee: BinaryAssociation = BinaryAssociation(
    name="MatchRefereesReferee",
    ends={
        Property(name="referee", type=Referee, multiplicity=Multiplicity(1, 1)),
        Property(name="matchAssignments", type=MatchReferee, multiplicity=Multiplicity(0, "*"), is_navigable=False)
    }
)

# Match -- TournamentPhase
MatchPhase: BinaryAssociation = BinaryAssociation(
    name="MatchPhase",
    ends={
        Property(name="phase", type=TournamentPhase, multiplicity=Multiplicity(1, 1)),
        Property(name="matches", type=Match, multiplicity=Multiplicity(0, "*"), is_navigable=False)
    }
)

# Match -- Stadium
MatchStadium: BinaryAssociation = BinaryAssociation(
    name="MatchStadium",
    ends={
        Property(name="stadium", type=Stadium, multiplicity=Multiplicity(1, 1)),
        Property(name="matches", type=Match, multiplicity=Multiplicity(0, "*"), is_navigable=False)
    }
)

# Match -- Team (home)
MatchHomeTeam: BinaryAssociation = BinaryAssociation(
    name="MatchHomeTeam",
    ends={
        Property(name="homeTeam", type=Team, multiplicity=Multiplicity(1, 1)),
        Property(name="homeMatches", type=Match, multiplicity=Multiplicity(0, "*"), is_navigable=False)
    }
)

# Match -- Team (away)
MatchAwayTeam: BinaryAssociation = BinaryAssociation(
    name="MatchAwayTeam",
    ends={
        Property(name="awayTeam", type=Team, multiplicity=Multiplicity(1, 1)),
        Property(name="awayMatches", type=Match, multiplicity=Multiplicity(0, "*"), is_navigable=False)
    }
)

# Match -- Group
MatchGroup: BinaryAssociation = BinaryAssociation(
    name="MatchGroup",
    ends={
        Property(name="group", type=Group, multiplicity=Multiplicity(0, 1)),
        Property(name="matches", type=Match, multiplicity=Multiplicity(0, "*"), is_navigable=False)
    }
)

# Goal -- Match
GoalMatch: BinaryAssociation = BinaryAssociation(
    name="GoalMatch",
    ends={
        Property(name="match", type=Match, multiplicity=Multiplicity(1, 1)),
        Property(name="goals", type=Goal, multiplicity=Multiplicity(0, "*"), is_navigable=False)
    }
)

# Goal -- Player (scorer)
GoalScorer: BinaryAssociation = BinaryAssociation(
    name="GoalScorer",
    ends={
        Property(name="scorer", type=Player, multiplicity=Multiplicity(1, 1)),
        Property(name="goalsScored", type=Goal, multiplicity=Multiplicity(0, "*"), is_navigable=False)
    }
)

# Goal -- Player (assist, optional)
GoalAssist: BinaryAssociation = BinaryAssociation(
    name="GoalAssist",
    ends={
        Property(name="assist", type=Player, multiplicity=Multiplicity(0, 1)),
        Property(name="assists", type=Goal, multiplicity=Multiplicity(0, "*"), is_navigable=False)
    }
)

# Card -- Match
CardMatch: BinaryAssociation = BinaryAssociation(
    name="CardMatch",
    ends={
        Property(name="match", type=Match, multiplicity=Multiplicity(1, 1)),
        Property(name="cards", type=Card, multiplicity=Multiplicity(0, "*"), is_navigable=False)
    }
)

# Card -- Player
CardPlayer: BinaryAssociation = BinaryAssociation(
    name="CardPlayer",
    ends={
        Property(name="player", type=Player, multiplicity=Multiplicity(1, 1)),
        Property(name="cardsReceived", type=Card, multiplicity=Multiplicity(0, "*"), is_navigable=False)
    }
)

# Substitution -- Match
SubstitutionMatch: BinaryAssociation = BinaryAssociation(
    name="SubstitutionMatch",
    ends={
        Property(name="match", type=Match, multiplicity=Multiplicity(1, 1)),
        Property(name="substitutions", type=Substitution, multiplicity=Multiplicity(0, "*"), is_navigable=False)
    }
)

# Substitution -- Team
SubstitutionTeam: BinaryAssociation = BinaryAssociation(
    name="SubstitutionTeam",
    ends={
        Property(name="team", type=Team, multiplicity=Multiplicity(1, 1)),
        Property(name="substitutions", type=Substitution, multiplicity=Multiplicity(0, "*"), is_navigable=False)
    }
)

# Substitution -- Player (out)
SubstitutionPlayerOut: BinaryAssociation = BinaryAssociation(
    name="SubstitutionPlayerOut",
    ends={
        Property(name="playerOut", type=Player, multiplicity=Multiplicity(1, 1)),
        Property(name="substitutedOut", type=Substitution, multiplicity=Multiplicity(0, "*"), is_navigable=False)
    }
)

# Substitution -- Player (in)
SubstitutionPlayerIn: BinaryAssociation = BinaryAssociation(
    name="SubstitutionPlayerIn",
    ends={
        Property(name="playerIn", type=Player, multiplicity=Multiplicity(1, 1)),
        Property(name="substitutedIn", type=Substitution, multiplicity=Multiplicity(0, "*"), is_navigable=False)
    }
)

# GroupStanding -- Group
GroupStandingGroup: BinaryAssociation = BinaryAssociation(
    name="GroupStandingGroup",
    ends={
        Property(name="group", type=Group, multiplicity=Multiplicity(1, 1)),
        Property(name="standings", type=GroupStanding, multiplicity=Multiplicity(0, 4), is_navigable=False)
    }
)

# GroupStanding -- Team
GroupStandingTeam: BinaryAssociation = BinaryAssociation(
    name="GroupStandingTeam",
    ends={
        Property(name="team", type=Team, multiplicity=Multiplicity(1, 1)),
        Property(name="groupStandings", type=GroupStanding, multiplicity=Multiplicity(0, "*"), is_navigable=False)
    }
)

# ─────────────────────────────────────────────
# OCL CONSTRAINTS
# ─────────────────────────────────────────────

ocl_player_age: Constraint = Constraint(
    name="PlayerMinimumAge",
    context=Player,
    expression="context Player inv PlayerMinimumAge: self.birthDate <= today() - 15",
    language="OCL"
)

ocl_squad_size: Constraint = Constraint(
    name="MaxSquadSize",
    context=Team,
    expression="context Team inv MaxSquadSize: self.players->size() <= 26",
    language="OCL"
)

ocl_sub_same_team: Constraint = Constraint(
    name="SubstitutionSameTeam",
    context=Substitution,
    expression="context Substitution inv SubstitutionSameTeam: "
               "self.playerIn.team = self.team and self.playerOut.team = self.team",
    language="OCL"
)

ocl_different_teams: Constraint = Constraint(
    name="DifferentTeams",
    context=Match,
    expression="context Match inv DifferentTeams: self.homeTeam <> self.awayTeam",
    language="OCL"
)

ocl_score_non_negative: Constraint = Constraint(
    name="ScoreNonNegative",
    context=Match,
    expression="context Match inv ScoreNonNegative: self.homeScore >= 0 and self.awayScore >= 0",
    language="OCL"
)

ocl_referee_neutrality: Constraint = Constraint(
    name="RefereeNeutrality",
    context=MatchReferee,
    expression="context MatchReferee inv RefereeNeutrality: "
               "self.referee.nationality <> self.match.homeTeam.represents and "
               "self.referee.nationality <> self.match.awayTeam.represents",
    language="OCL"
)

ocl_card_minute: Constraint = Constraint(
    name="CardMinuteValid",
    context=Card,
    expression="context Card inv CardMinuteValid: self.minute >= 1 and self.minute <= 120",
    language="OCL"
)

ocl_goal_minute: Constraint = Constraint(
    name="GoalMinutePositive",
    context=Goal,
    expression="context Goal inv GoalMinutePositive: self.minute >= 1",
    language="OCL"
)

ocl_max_subs: Constraint = Constraint(
    name="MaxSubstitutionsPerTeam",
    context=Match,
    expression="context Match inv MaxSubstitutionsPerTeam: "
               "self.substitutions->select(s | s.team = self.homeTeam)->size() <= 5 and "
               "self.substitutions->select(s | s.team = self.awayTeam)->size() <= 5",
    language="OCL"
)

ocl_knockout_winner: Constraint = Constraint(
    name="KnockoutMustHaveWinner",
    context=Match,
    expression="context Match inv KnockoutMustHaveWinner: "
               "self.phase.phaseOrder > 1 implies "
               "(self.homeScore <> self.awayScore or self.wentToPenaltyShootout = true)",
    language="OCL"
)

# ─────────────────────────────────────────────
# DOMAIN MODEL
# ─────────────────────────────────────────────

domain_model = DomainModel(
    name="WorldCupModel",
    types={
        CardType, GoalType, MatchStatus, PhaseType, PlayerPosition, RefereeRole,
        Country, Stadium, TournamentEdition, TournamentPhase,
        Group, GroupStanding, Team, Player, Referee,
        MatchReferee, Match, Goal, Card, Substitution
    },
    associations={
        StadiumCountry, EditionHostCountry, EditionPhase,
        GroupPhase, TeamCountry, TeamEdition, TeamGroup,
        PlayerTeam, RefereeCountry,
        MatchRefereesMatch, MatchRefereesReferee,
        MatchPhase, MatchStadium, MatchHomeTeam, MatchAwayTeam, MatchGroup,
        GoalMatch, GoalScorer, GoalAssist,
        CardMatch, CardPlayer,
        SubstitutionMatch, SubstitutionTeam, SubstitutionPlayerOut, SubstitutionPlayerIn,
        GroupStandingGroup, GroupStandingTeam
    },
    constraints={
        ocl_player_age, ocl_squad_size, ocl_sub_same_team,
        ocl_different_teams, ocl_score_non_negative, ocl_referee_neutrality,
        ocl_card_minute, ocl_goal_minute, ocl_max_subs, ocl_knockout_winner
    },
    generalizations={},
    metadata=None
)
