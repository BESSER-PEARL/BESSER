import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
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
# Fixture: football domain model
# ---------------------------------------------------------------------------

@pytest.fixture
def football_model():
    """Country / Team / Player / GoalKeeper / Championship / Match / Fan.

    See module docstring for the full structure and the multiplicities
    each association exercises (1..1, 0..*, 2..4).
    """
    # --- Enumeration ---
    position_literals = {
        EnumerationLiteral(name="GOALKEEPER"),
        EnumerationLiteral(name="DEFENDER"),
        EnumerationLiteral(name="MIDFIELDER"),
        EnumerationLiteral(name="FORWARD"),
    }
    Position = Enumeration(name="Position", literals=position_literals)

    # --- Classes ---
    Country = Class(name="Country")
    Country.attributes = {
        Property(name="name", type=StringType),
        Property(name="isoCode", type=StringType),
    }

    Team = Class(name="Team")
    Team.attributes = {
        Property(name="name", type=StringType),
        Property(name="foundationYear", type=IntegerType),
    }

    Player = Class(name="Player")
    Player.attributes = {
        Property(name="name", type=StringType),
        Property(name="age", type=IntegerType),
        Property(name="position", type=Position),
    }

    GoalKeeper = Class(name="GoalKeeper")
    GoalKeeper.attributes = {
        Property(name="penaltySaves", type=IntegerType),
    }
    gk_generalization = Generalization(general=Player, specific=GoalKeeper)

    Championship = Class(name="Championship")
    Championship.attributes = {
        Property(name="name", type=StringType),
        Property(name="year", type=IntegerType),
    }

    Match = Class(name="Match")
    Match.attributes = {
        Property(name="homeGoals", type=IntegerType),
        Property(name="awayGoals", type=IntegerType),
    }

    Fan = Class(name="Fan")
    Fan.attributes = {
        Property(name="name", type=StringType),
    }

    # --- Associations (all one-way navigable) ---
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

    # Team -> Player (0..*)
    team_player = BinaryAssociation(
        name="TeamPlayer",
        ends={
            Property(name="players", type=Player, multiplicity=Multiplicity(0, "*"),
                      is_navigable=True),
            Property(name="team", type=Team, multiplicity=Multiplicity(1, 1),
                      is_navigable=False),
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

    model = DomainModel(
        name="FootballModel",
        types={Country, Team, Player, GoalKeeper, Championship, Match, Fan, Position},
        associations={
            team_country, championship_match, match_home_team, match_away_team,
            team_player, fan_team, fan_country,
        },
        generalizations={gk_generalization},
    )
    return model