import os
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
from besser.generators.alloy.alloy_generator import AlloyGenerator
from tests.generators.alloy.utils_alloy_re_parser import AlloySpec

# ---------------------------------------------------------------------------
# Fixture: football domain model
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
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


@pytest.fixture
def spec(model, tmpdir):
    tmpdir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=model, output_dir=str(tmpdir))
    generator.generate()
    als_file = generator.get_specification_path()
    assert os.path.isfile(als_file), f"Alloy specification not generated at {als_file}"
    with open(als_file, "r", encoding="utf-8") as f:
        content = f.read()
    return AlloySpec(content)


def test_signatures(spec):
    assert "sig Country" in spec.headers
    assert "sig Team" in spec.headers
    assert "sig Player" in spec.headers
    assert "sig GoalKeeper extends Player" in spec.headers
    assert "sig Championship" in spec.headers
    assert "sig Match" in spec.headers
    assert "sig Fan" in spec.headers


def test_signature_fields(spec):
    assert len(spec.fields["Country"]) == 2
    assert "Country_name: Str" in spec.fields["Country"]
    assert "Country_isoCode: Str" in spec.fields["Country"]

    assert len(spec.fields["Team"]) == 4
    assert "Team_name: Str" in spec.fields["Team"]
    assert "Team_foundationYear: Int" in spec.fields["Team"]
    assert "Team_country: one Country" in spec.fields["Team"]
    assert "Team_players: set Player" in spec.fields["Team"]

    assert len(spec.fields["Player"]) == 3
    assert "Player_name: Str" in spec.fields["Player"]
    assert "Player_age: Int" in spec.fields["Player"]
    assert "Player_position: Position" in spec.fields["Player"]

    assert len(spec.fields["GoalKeeper"]) == 1
    assert "GoalKeeper_penaltySaves: Int" in spec.fields["GoalKeeper"]

    assert len(spec.fields["Championship"]) == 3
    assert "Championship_name: Str" in spec.fields["Championship"]
    assert "Championship_year: Int" in spec.fields["Championship"]
    assert "Championship_matches: set Match" in spec.fields["Championship"]

    assert len(spec.fields["Match"]) == 4
    assert "Match_homeGoals: Int" in spec.fields["Match"]
    assert "Match_awayGoals: Int" in spec.fields["Match"]
    assert "Match_homeTeam: one Team" in spec.fields["Match"]
    assert "Match_awayTeam: one Team" in spec.fields["Match"]

    assert len(spec.fields["Fan"]) == 3
    assert "Fan_name: Str" in spec.fields["Fan"]
    assert "Fan_birthCountry: one Country" in spec.fields["Fan"]
    assert "Fan_favoriteTeams: set Team" in spec.fields["Fan"]

    assert len(spec.fields["Position"]) == 0
    assert len(spec.fields["ENUM_Position_GOALKEEPER"]) == 0
    assert len(spec.fields["ENUM_Position_DEFENDER"]) == 0
    assert len(spec.fields["ENUM_Position_MIDFIELDER"]) == 0
    assert len(spec.fields["ENUM_Position_FORWARD"]) == 0


def test_enumerations(spec):
    assert len(spec.subsigs_of("Position")) == 4
    assert "abstract sig Position" in spec.headers
    assert "one sig ENUM_Position_GOALKEEPER extends Position" in spec.headers
    assert "one sig ENUM_Position_DEFENDER extends Position" in spec.headers
    assert "one sig ENUM_Position_MIDFIELDER extends Position" in spec.headers
    assert "one sig ENUM_Position_FORWARD extends Position" in spec.headers


def test_facts(spec):
    # Fan -> Team (2..4): the bounds are enforced by facts, the field itself is a plain set
    assert len(spec.facts) == 2
    assert "all a: Fan | #(a.Fan_favoriteTeams) >= 2" in spec.facts
    assert "all a: Fan | #(a.Fan_favoriteTeams) <= 4" in spec.facts


def test_run_command(spec):
    assert spec.run_pred == "instance_model"
    assert "5 Country" in spec.run_scopes
    assert "5 Team" in spec.run_scopes
    assert "5 Player" in spec.run_scopes
    assert "5 GoalKeeper" in spec.run_scopes
    assert "5 Championship" in spec.run_scopes
    assert "5 Match" in spec.run_scopes
    assert "5 Fan" in spec.run_scopes
    assert "5 Int" in spec.run_scopes
    assert "5 Str" in spec.run_scopes
    assert "5 seq" in spec.run_scopes


