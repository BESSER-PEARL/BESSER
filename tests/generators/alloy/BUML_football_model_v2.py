import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    Constraint,
    DateType,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    Generalization,
    IntegerType,
    Multiplicity,
    Property,
    StringType,
)

# ---------------------------------------------------------------------------
# Fixture: football domain model (v2)
# ---------------------------------------------------------------------------

@pytest.fixture
def football_model_v2():
    """Country / Team / Person / Player / GoalKeeper / HeadCoach / Championship / Match / Fan.

    Extends the v1 football model with:

    - an abstract ``Person`` class holding the data shared by ``Player``, ``Fan``
      and ``HeadCoach`` (name and birth date);
    - a ``HeadCoach`` with a coaching start date and a ``CoachStatus``;
    - date attributes on ``Team`` (foundation) and ``Match`` (day it is played);
    - a minimum of 11 players per team (11..*);
    - OCL invariants using ``forAll``, ``select``/``size``, ``null`` checks,
      date comparisons and integer bounds.
    """
    # --- Enumerations ---
    position_literals = {
        EnumerationLiteral(name="GOALKEEPER"),
        EnumerationLiteral(name="DEFENDER"),
        EnumerationLiteral(name="MIDFIELDER"),
        EnumerationLiteral(name="FORWARD"),
    }
    Position = Enumeration(name="Position", literals=position_literals)

    coach_status_literals = {
        EnumerationLiteral(name="ACTIVE"),
        EnumerationLiteral(name="INACTIVE"),
        EnumerationLiteral(name="RETIRED"),
    }
    CoachStatus = Enumeration(name="CoachStatus", literals=coach_status_literals)

    # --- Classes ---
    Country = Class(name="Country")
    Country.attributes = {
        Property(name="name", type=StringType),
        Property(name="isoCode", type=StringType),
    }

    Team = Class(name="Team")
    Team.attributes = {
        Property(name="name", type=StringType),
        Property(name="foundationDate", type=DateType),
    }

    Person = Class(name="Person", is_abstract=True)
    Person.attributes = {
        Property(name="name", type=StringType),
        Property(name="birthDate", type=DateType),
    }

    Player = Class(name="Player")
    Player.attributes = {
        Property(name="age", type=IntegerType),
        Property(name="position", type=Position),
    }
    player_generalization = Generalization(general=Person, specific=Player)

    GoalKeeper = Class(name="GoalKeeper")
    GoalKeeper.attributes = {
        Property(name="penaltySaves", type=IntegerType),
    }
    gk_generalization = Generalization(general=Player, specific=GoalKeeper)

    HeadCoach = Class(name="HeadCoach")
    HeadCoach.attributes = {
        Property(name="coachingStartDate", type=DateType),
        Property(name="status", type=CoachStatus),
    }
    coach_generalization = Generalization(general=Person, specific=HeadCoach)

    Championship = Class(name="Championship")
    Championship.attributes = {
        Property(name="name", type=StringType),
        Property(name="year", type=IntegerType),
    }

    Match = Class(name="Match")
    Match.attributes = {
        Property(name="homeGoals", type=IntegerType),
        Property(name="awayGoals", type=IntegerType),
        Property(name="matchDate", type=DateType),
    }

    Fan = Class(name="Fan")
    fan_generalization = Generalization(general=Person, specific=Fan)

    # --- Associations (one-way navigable unless stated otherwise) ---
    # Team -> Country (1..1)
    team_country = BinaryAssociation(
        name="TeamCountry",
        ends={
            Property(name="country", type=Country, multiplicity=Multiplicity(1, 1),
                      is_navigable=True),
            Property(name="teams", type=Team, multiplicity=Multiplicity(0, "*"),
                      is_navigable=False),
        },
    )

    # Championship -> Match (0..*)
    championship_match = BinaryAssociation(
        name="ChampionshipMatch",
        ends={
            Property(name="matches", type=Match, multiplicity=Multiplicity(0, "*"),
                      is_navigable=True),
            Property(name="championship", type=Championship, multiplicity=Multiplicity(1, 1),
                      is_navigable=False),
        },
    )

    # Match -> Team, home side (1..1)
    match_home_team = BinaryAssociation(
        name="MatchHomeTeam",
        ends={
            Property(name="homeTeam", type=Team, multiplicity=Multiplicity(1, 1),
                      is_navigable=True),
            Property(name="homeMatches", type=Match, multiplicity=Multiplicity(0, "*"),
                      is_navigable=False),
        },
    )

    # Match -> Team, away side (1..1)
    match_away_team = BinaryAssociation(
        name="MatchAwayTeam",
        ends={
            Property(name="awayTeam", type=Team, multiplicity=Multiplicity(1, 1),
                      is_navigable=True),
            Property(name="awayMatches", type=Match, multiplicity=Multiplicity(0, "*"),
                      is_navigable=False),
        },
    )

    # Team -> Player (11..*) -- a team has at least 11 players
    team_player = BinaryAssociation(
        name="TeamPlayer",
        ends={
            Property(name="players", type=Player, multiplicity=Multiplicity(11, "*"),
                      is_navigable=True),
            Property(name="team", type=Team, multiplicity=Multiplicity(1, 1),
                      is_navigable=False),
        },
    )

    # Team <-> HeadCoach (bidirectional): every team has exactly one current head
    # coach, and a head coach may or may not be in charge of a team (0..1)
    team_head_coach = BinaryAssociation(
        name="TeamHeadCoach",
        ends={
            Property(name="headCoach", type=HeadCoach, multiplicity=Multiplicity(1, 1),
                      is_navigable=True),
            Property(name="currentTeam", type=Team, multiplicity=Multiplicity(0, 1),
                      is_navigable=True),
        },
    )

    # Fan -> Team (2..4) -- a fan follows between 2 and 4 teams
    fan_team = BinaryAssociation(
        name="FanTeam",
        ends={
            Property(name="favoriteTeams", type=Team, multiplicity=Multiplicity(2, 4),
                      is_navigable=True),
            Property(name="fans", type=Fan, multiplicity=Multiplicity(0, "*"),
                      is_navigable=False),
        },
    )

    # Fan -> Country, country of birth (1..1)
    fan_country = BinaryAssociation(
        name="FanCountry",
        ends={
            Property(name="birthCountry", type=Country, multiplicity=Multiplicity(1, 1),
                      is_navigable=True),
            Property(name="fansBorn", type=Fan, multiplicity=Multiplicity(0, "*"),
                      is_navigable=False),
        },
    )

    # --- OCL Constraints ---
    # forAll: every player of a team must be older than 16
    Team_inv_players_older_than_16 = Constraint(
        name="Team_inv_players_older_than_16",
        context=Team,
        expression=(
            "context Team inv Team_inv_players_older_than_16 : "
            "self.players->forAll(p | p.age > 16)"
        ),
        language="OCL",
    )

    # select + size: a team cannot have more than 3 goalkeepers
    Team_inv_max_3_goalkeepers = Constraint(
        name="Team_inv_max_3_goalkeepers",
        context=Team,
        expression=(
            "context Team inv Team_inv_max_3_goalkeepers : "
            "self.players->select(p | p.position = Position::GOALKEEPER)->size() <= 3"
        ),
        language="OCL",
    )

    # A retired head coach is not in charge of any team
    HeadCoach_inv_retired_has_no_team = Constraint(
        name="HeadCoach_inv_retired_has_no_team",
        context=HeadCoach,
        expression=(
            "context HeadCoach inv HeadCoach_inv_retired_has_no_team : "
            "(self.status = CoachStatus::RETIRED) implies (self.currentTeam = null)"
        ),
        language="OCL",
    )

    # A team is founded before every match it plays (home or away)
    Match_inv_teams_founded_before_match = Constraint(
        name="Match_inv_teams_founded_before_match",
        context=Match,
        expression=(
            "context Match inv Match_inv_teams_founded_before_match : "
            "(self.homeTeam.foundationDate < self.matchDate) "
            "and (self.awayTeam.foundationDate < self.matchDate)"
        ),
        language="OCL",
    )

    # Goals scored by each side are never negative
    Match_inv_non_negative_goals = Constraint(
        name="Match_inv_non_negative_goals",
        context=Match,
        expression=(
            "context Match inv Match_inv_non_negative_goals : "
            "(self.homeGoals >= 0) and (self.awayGoals >= 0)"
        ),
        language="OCL",
    )

    model = DomainModel(
        name="FootballModelV2",
        types={
            Country, Team, Person, Player, GoalKeeper, HeadCoach, Championship, Match, Fan,
            Position, CoachStatus,
        },
        associations={
            team_country, championship_match, match_home_team, match_away_team,
            team_player, team_head_coach, fan_team, fan_country,
        },
        generalizations={
            player_generalization, gk_generalization, coach_generalization, fan_generalization,
        },
        constraints={
            Team_inv_players_older_than_16, Team_inv_max_3_goalkeepers,
            HeadCoach_inv_retired_has_no_team, Match_inv_teams_founded_before_match,
            Match_inv_non_negative_goals,
        },
    )
    return model
