import os
import pytest
from tests.generators.alloy.BUML_football_model import football_model
from besser.BUML.metamodel.structural import AnyType, Class, DomainModel, Property, TimeDeltaType, TimeType
from besser.generators.alloy.alloy_generator import AlloyGenerator
from besser.generators.alloy.alloy_translator import BumlToAlloy
from tests.generators.alloy.utils_alloy_translation import AlloyTestSpec


@pytest.fixture
def spec(football_model, tmpdir):
    tmpdir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=football_model, output_dir=str(tmpdir))
    generator.generate()
    als_file = generator.get_specification_path()
    assert os.path.isfile(als_file), f"Alloy specification not generated at {als_file}"
    # String attributes require the strings library
    assert sorted(os.listdir(str(tmpdir))) == ["model.als", "strings.als"]
    with open(als_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "open strings" in content
    return AlloyTestSpec(content)


def test_signatures(spec):
    assert "sig Country" in spec.sigs
    assert "sig Team" in spec.sigs
    assert "sig Player" in spec.sigs
    assert "sig GoalKeeper extends Player" in spec.sigs
    assert "sig Championship" in spec.sigs
    assert "sig Match" in spec.sigs
    assert "sig Fan" in spec.sigs
    # No OCL string literals: the scope is filled with other strings
    assert len(spec.subsigs_of("Str")) == 5
    assert "one sig Str_str0 extends Str" in spec.sigs
    assert "one sig Str_str1 extends Str" in spec.sigs
    assert "one sig Str_str2 extends Str" in spec.sigs
    assert "one sig Str_str3 extends Str" in spec.sigs
    assert "one sig Str_str4 extends Str" in spec.sigs


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
    assert "abstract sig Position" in spec.sigs
    assert "one sig ENUM_Position_GOALKEEPER extends Position" in spec.sigs
    assert "one sig ENUM_Position_DEFENDER extends Position" in spec.sigs
    assert "one sig ENUM_Position_MIDFIELDER extends Position" in spec.sigs
    assert "one sig ENUM_Position_FORWARD extends Position" in spec.sigs


def test_facts(spec):
    # Fan -> Team (2..4): the bounds are enforced by facts, the field itself is a plain set
    assert len(spec.facts) == 6
    assert "all a: Fan | #(a.Fan_favoriteTeams) >= 2" in spec.facts
    assert "all a: Fan | #(a.Fan_favoriteTeams) <= 4" in spec.facts
    # Player -> Team (1..1) and Match -> Championship (1..1) are not navigable, so there is no
    # `one` field enforcing them: the bounds are checked navigating the opposite field backwards
    assert "all a: Player | #(Team_players.a) >= 1" in spec.facts
    assert "all a: Player | #(Team_players.a) <= 1" in spec.facts
    assert "all a: Match | #(Championship_matches.a) >= 1" in spec.facts
    assert "all a: Match | #(Championship_matches.a) <= 1" in spec.facts


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
    # Uninterpreted strings have no length to bound
    # assert "5 seq" in spec.run_scopes
    assert not any(scope.endswith(" seq") for scope in spec.run_scopes)