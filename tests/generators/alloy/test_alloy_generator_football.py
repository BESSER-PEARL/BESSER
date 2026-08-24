"""Tests for AlloyGenerator using a richer football domain model.

------------------------------------------------------------------------
Model under test: football domain
------------------------------------------------------------------------

Classes (all attribute names/types in English, per request):

    Country       (name: str, isoCode: str)
    Team          (name: str, foundationYear: int)
    Player        (name: str, age: int, position: Position[enum])
    GoalKeeper    (penaltySaves: int)            -- extends Player
    Championship  (name: str, year: int)
    Match         (homeGoals: int, awayGoals: int)
    Fan           (name: str)

Enumeration:

    Position = { GOALKEEPER, DEFENDER, MIDFIELDER, FORWARD }

Associations (all unidirectional -- navigable in one direction only,
matching how each relationship is actually queried in this domain):

    Team        --(1..1)-->  Country        "country"          (Team -> Country)
    Championship --(0..*)--> Match          "matches"          (Championship -> Match)
    Match       --(1..1)-->  Team           "homeTeam"          (Match -> Team)
    Match       --(1..1)-->  Team           "awayTeam"          (Match -> Team)
    Team        --(0..*)-->  Player         "players"          (Team -> Player)
    Fan         --(2..4)-->  Team           "favoriteTeams"     (Fan -> Team)
    Fan         --(1..1)-->  Country        "birthCountry"      (Fan -> Country)

This model covers all requested multiplicities (1..1, 0..*, 2..4) and the presence of an enum-typed attribute (Player.position),:
  - 1..1   : Team->Country, Match->homeTeam, Match->awayTeam, Fan->birthCountry
  - 0..*   : Championship->Match, Team->Player
  - 2..4   : Fan->favoriteTeams

"""
import glob
import os
import re
import sys
from pathlib import Path


import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from besser.BUML.metamodel.structural import (
    Class, Property, BinaryAssociation, Multiplicity, Generalization,
    Enumeration, EnumerationLiteral, Constraint,
    DomainModel, StringType, IntegerType,
)
from besser.generators.alloy_generator.alloy_generator import AlloyGenerator


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


def _generated_als_path(output_dir):
    """Locate the single .als file produced in output_dir.
    """
    matches = glob.glob(os.path.join(output_dir, "model.als"))
    assert len(matches) == 1, (
        f"Expected exactly one model.als file in {output_dir}, found {matches}"
    )
    return matches[0]


def _generate_and_print(model, tmpdir, label=""):
    """Run AlloyGenerator on `model` and print the resulting .als to
    stdout (run pytest with -s to see it), then return its contents.

    """
    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=model, output_dir=str(output_dir))
    generator.generate()

    als_path = _generated_als_path(str(output_dir))
    with open(als_path, "r", encoding="utf-8") as f:
        spec = f.read()

    header = f" Generated Alloy spec ({label}) — {als_path} "
    print("\n" + header.center(100, "="))
    print(spec)
    print("=" * 100 + "\n")

    return spec


# ---------------------------------------------------------------------------
# Generation tests
# ---------------------------------------------------------------------------

def test_generator_creates_als_file(football_model, tmpdir):
    spec = _generate_and_print(football_model, tmpdir, "creates_als_file")
    assert spec  # non-empty file was produced


def test_all_class_signatures_present(football_model, tmpdir):
    spec = _generate_and_print(football_model, tmpdir, "all_class_signatures_present")

    for sig_name in ["Country", "Team", "Player", "GoalKeeper", "Championship", "Match", "Fan"]:
        assert f"sig {sig_name}" in spec


def test_inheritance_renders_with_extends(football_model, tmpdir):
    """GoalKeeper has exactly one parent (Player).
    """
    spec = _generate_and_print(football_model, tmpdir, "inheritance_renders_with_extends")

    assert "sig GoalKeeper extends Player" in spec


def test_enumeration_literals_present(football_model, tmpdir):
    spec = _generate_and_print(football_model, tmpdir, "enumeration_literals_present")

    assert "abstract sig Position" in spec
    for literal in ["GOALKEEPER", "DEFENDER", "MIDFIELDER", "FORWARD"]:
        assert f"ENUM_Position_{literal}" in spec


def test_enum_typed_attribute_present(football_model, tmpdir):
    """Player.position is typed with the Position enum.
    """
    spec = _generate_and_print(football_model, tmpdir, "enum_typed_attribute_present")

    assert "Player_position: Position" in spec


def test_one_to_one_ends_render_with_one_keyword(football_model, tmpdir):
    """Every 1..1 navigable end (Team->Country, Match->homeTeam,
    Match->awayTeam, Fan->birthCountry) must use Alloy's `one` keyword."""
    spec = _generate_and_print(football_model, tmpdir, "one_to_one_ends_render_with_one_keyword")

    assert "Team_country: one Country" in spec
    assert "Match_homeTeam: one Team" in spec
    assert "Match_awayTeam: one Team" in spec
    assert "Fan_birthCountry: one Country" in spec


def test_zero_to_many_ends_render_with_set_keyword(football_model, tmpdir):
    """0..* navigable ends (Championship->Match, Team->Player) must use
    Alloy's `set` keyword, same as a bounded multi-valued end would."""
    spec = _generate_and_print(football_model, tmpdir, "zero_to_many_ends_render_with_set_keyword")

    assert "Championship_matches: set Match" in spec
    assert "Team_players: set Player" in spec


def test_two_to_four_end_renders_with_set_keyword(football_model, tmpdir):
    """Fan->favoriteTeams (2..4, max > 1) must use Alloy's `set` keyword."""
    spec = _generate_and_print(football_model, tmpdir, "two_to_four_end_renders_with_set_keyword")

    assert "Fan_favoriteTeams: set Team" in spec


def test_two_to_four_multiplicity_has_lower_and_upper_bound_facts(football_model, tmpdir):
    """Fan->favoriteTeams is the only association in this model whose
    bounds fall outside the (skipped) 1..1 and 0..* shapes, so it's the
    only one that should produce explicit cardinality facts: >=2 and <=4.

    """
    spec = _generate_and_print(
        football_model, tmpdir, "two_to_four_multiplicity_has_lower_and_upper_bound_facts"
    )

    assert (
        "#(a.Fan_favoriteTeams)>=2" in spec or "#(b.Fan_favoriteTeams)>=2" in spec
    ), spec
    assert (
        "#(a.Fan_favoriteTeams)<=4" in spec or "#(b.Fan_favoriteTeams)<=4" in spec
    ), spec


def test_one_to_one_and_zero_to_many_ends_produce_no_cardinality_facts(football_model, tmpdir):
    spec = _generate_and_print(
        football_model, tmpdir,
        "one_to_one_and_zero_to_many_ends_produce_no_cardinality_facts",
    )

    cardinality_facts = [line for line in spec.splitlines() if "#(" in line]
    assert len(cardinality_facts) == 2, (
        f"Expected exactly 2 cardinality facts (Fan_favoriteTeams >=2 and <=4), "
        f"got {len(cardinality_facts)}: {cardinality_facts}"
    )
    for line in cardinality_facts:
        assert "Fan_favoriteTeams" in line


def test_all_associations_are_unidirectional_no_inverse_facts(football_model, tmpdir):
    """Every association in this model is deliberately one-way navigable
    (e.g. you query a Team's Country, not a Country's Teams), so the
    generator must NOT emit any inverse-relation equality fact (`~`)
    anywhere in the spec."""
    spec = _generate_and_print(
        football_model, tmpdir, "all_associations_are_unidirectional_no_inverse_facts"
    )

    assert "~" not in spec


def test_non_navigable_ends_are_omitted_from_owning_sig(football_model, tmpdir):
    """The non-navigable side of each association must not appear as a
    field anywhere -- e.g. Country has no `Country_teams` field, even
    though Team->Country is modeled as a BinaryAssociation with a
    (non-navigable) reverse end."""
    spec = _generate_and_print(
        football_model, tmpdir, "non_navigable_ends_are_omitted_from_owning_sig"
    )

    for absent_field in [
        "Country_teams", "Match_championship", "Team_homeMatches",
        "Team_awayMatches", "Player_team", "Team_fans", "Fan_fansBorn",
    ]:
        assert absent_field not in spec


def test_generic_instance_model_predicate_and_run(football_model, tmpdir):
    """The generic alloy_final_als.j2 (alloy_final_als_old.j2 content) must
    emit one `some <Sig>` line and one `5 <Sig>` per-signature scope entry per
    class, including the inherited GoalKeeper."""
    spec = _generate_and_print(
        football_model, tmpdir, "generic_instance_model_predicate_and_run"
    )
    spec = re.sub(r"\s+", " ", spec)

    assert "pred instance_model[]{" in spec
    assert "run instance_model for" in spec
    for sig_name in ["Country", "Team", "Player", "GoalKeeper", "Championship", "Match", "Fan"]:
        assert f"some {sig_name}" in spec
        assert f"5 {sig_name}" in spec


def test_ocl_constraint_on_football_model_is_translated(football_model, tmpdir):
    """A real OCL invariant on Match ("self.homeGoals >= 0") must be
    translated end-to-end via the real translate_ocl_alloy.ocl_to_alloy
    pipeline against this richer model, not just the minimal Team/Player
    one used in test_alloy_generator.py."""
    Match = next(c for c in football_model.types if getattr(c, "name", None) == "Match")
    goals_non_negative = Constraint(
        name="GoalsNonNegative",
        context=Match,
        expression="context Match inv GoalsNonNegative: self.homeGoals >= 0",
        language="OCL",
    )
    football_model.constraints = {goals_non_negative}

    spec = _generate_and_print(
        football_model, tmpdir, "ocl_constraint_on_football_model_is_translated"
    )

    assert "this/Match" in spec
    assert "Match_homeGoals >= 0" in spec
    assert "fact{" in spec


def test_allinstances_constraint_on_football_model(football_model, tmpdir):
    """A cross-class allInstances invariant on the richer model: every Team
    (regardless of which Championship constrains it) must have been founded
    in or after 1850.

    context Championship inv:
        Team.allInstances()->forAll(t | t.foundationYear >= 1850)

    The iterator variable must resolve its attribute through the Team class,
    not the Championship context, even though the receiver of allInstances()
    differs from the constraint context.
    """
    Championship = next(
        c for c in football_model.types if getattr(c, "name", None) == "Championship"
    )
    teams_founded_modern = Constraint(
        name="TeamsFoundedModern",
        context=Championship,
        expression=(
            "context Championship inv TeamsFoundedModern: "
            "Team.allInstances()->forAll(t | t.foundationYear >= 1850)"
        ),
        language="OCL",
    )
    football_model.constraints = {teams_founded_modern}

    spec = _generate_and_print(
        football_model, tmpdir, "allinstances_constraint_on_football_model"
    )

    assert "all self:this/Championship" in spec
    assert "all t : Team" in spec
    assert "t.Team_foundationYear >= 1850" in spec
    assert "fact{" in spec
