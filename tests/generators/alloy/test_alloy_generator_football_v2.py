import os
import pytest
from tests.generators.alloy.BUML_football_model_v2 import football_model_v2
from besser.generators.alloy.alloy_generator import AlloyGenerator
from tests.generators.alloy.utils_alloy_re_parser import AlloySpec


@pytest.fixture
def spec(football_model_v2, tmpdir):
    tmpdir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=football_model_v2, output_dir=str(tmpdir))
    generator.generate()
    als_file = generator.get_specification_path()
    assert os.path.isfile(als_file), f"Alloy specification not generated at {als_file}"
    # Date attributes require the date library next to the specification
    assert os.path.isfile(os.path.join(str(tmpdir), "date.als"))
    with open(als_file, "r", encoding="utf-8") as f:
        content = f.read()
    return AlloySpec(content)


def test_signatures(spec):
    assert "sig Country" in spec.sigs
    assert "sig Team" in spec.sigs
    assert "abstract sig Person" in spec.sigs
    assert "sig Player extends Person" in spec.sigs
    assert "sig GoalKeeper extends Player" in spec.sigs
    assert "sig HeadCoach extends Person" in spec.sigs
    assert "sig Fan extends Person" in spec.sigs
    assert "sig Championship" in spec.sigs
    assert "sig Match" in spec.sigs


def test_signature_fields(spec):
    assert len(spec.fields["Country"]) == 2
    assert "Country_name: Str" in spec.fields["Country"]
    assert "Country_isoCode: Str" in spec.fields["Country"]

    assert len(spec.fields["Team"]) == 5
    assert "Team_name: Str" in spec.fields["Team"]
    assert "Team_foundationDate: Date" in spec.fields["Team"]
    assert "Team_country: one Country" in spec.fields["Team"]
    assert "Team_players: set Player" in spec.fields["Team"]
    assert "Team_headCoach: one HeadCoach" in spec.fields["Team"]

    assert len(spec.fields["Person"]) == 2
    assert "Person_name: Str" in spec.fields["Person"]
    assert "Person_birthDate: Date" in spec.fields["Person"]

    assert len(spec.fields["Player"]) == 2
    assert "Player_age: Int" in spec.fields["Player"]
    assert "Player_position: Position" in spec.fields["Player"]

    assert len(spec.fields["GoalKeeper"]) == 1
    assert "GoalKeeper_penaltySaves: Int" in spec.fields["GoalKeeper"]

    assert len(spec.fields["HeadCoach"]) == 3
    assert "HeadCoach_coachingStartDate: Date" in spec.fields["HeadCoach"]
    assert "HeadCoach_status: CoachStatus" in spec.fields["HeadCoach"]
    assert "HeadCoach_currentTeam: lone Team" in spec.fields["HeadCoach"]

    assert len(spec.fields["Fan"]) == 2
    assert "Fan_birthCountry: one Country" in spec.fields["Fan"]
    assert "Fan_favoriteTeams: set Team" in spec.fields["Fan"]

    assert len(spec.fields["Championship"]) == 3
    assert "Championship_name: Str" in spec.fields["Championship"]
    assert "Championship_year: Int" in spec.fields["Championship"]
    assert "Championship_matches: set Match" in spec.fields["Championship"]

    assert len(spec.fields["Match"]) == 5
    assert "Match_homeGoals: Int" in spec.fields["Match"]
    assert "Match_awayGoals: Int" in spec.fields["Match"]
    assert "Match_matchDate: Date" in spec.fields["Match"]
    assert "Match_homeTeam: one Team" in spec.fields["Match"]
    assert "Match_awayTeam: one Team" in spec.fields["Match"]

    assert len(spec.fields["Position"]) == 0
    assert len(spec.fields["ENUM_Position_GOALKEEPER"]) == 0
    assert len(spec.fields["ENUM_Position_DEFENDER"]) == 0
    assert len(spec.fields["ENUM_Position_MIDFIELDER"]) == 0
    assert len(spec.fields["ENUM_Position_FORWARD"]) == 0

    assert len(spec.fields["CoachStatus"]) == 0
    assert len(spec.fields["ENUM_CoachStatus_ACTIVE"]) == 0
    assert len(spec.fields["ENUM_CoachStatus_INACTIVE"]) == 0
    assert len(spec.fields["ENUM_CoachStatus_RETIRED"]) == 0


def test_enumerations(spec):
    assert len(spec.subsigs_of("Position")) == 4
    assert "abstract sig Position" in spec.sigs
    assert "one sig ENUM_Position_GOALKEEPER extends Position" in spec.sigs
    assert "one sig ENUM_Position_DEFENDER extends Position" in spec.sigs
    assert "one sig ENUM_Position_MIDFIELDER extends Position" in spec.sigs
    assert "one sig ENUM_Position_FORWARD extends Position" in spec.sigs

    assert len(spec.subsigs_of("CoachStatus")) == 3
    assert "abstract sig CoachStatus" in spec.sigs
    assert "one sig ENUM_CoachStatus_ACTIVE extends CoachStatus" in spec.sigs
    assert "one sig ENUM_CoachStatus_INACTIVE extends CoachStatus" in spec.sigs
    assert "one sig ENUM_CoachStatus_RETIRED extends CoachStatus" in spec.sigs


def test_facts(spec):
    assert len(spec.facts) == 8

    # Fan -> Team (2..4)
    assert "all b: Fan | #(b.Fan_favoriteTeams) >= 2" in spec.facts
    assert "all b: Fan | #(b.Fan_favoriteTeams) <= 4" in spec.facts
    # Team -> Player (11..*): only the lower bound is enforced
    assert "all b: Team | #(b.Team_players) >= 11" in spec.facts
    # HeadCoach -> Team (0..1)
    assert "all a: HeadCoach | #(a.HeadCoach_currentTeam) <= 1" in spec.facts
    # HeadCoach <-> Team: bidirectional association
    assert "HeadCoach_currentTeam = ~Team_headCoach" in spec.facts

    # OCL: players older than 16 (forAll -> universal quantifier)
    assert "all self:this/Team|all p : self.Team_players | (p.Player_age > 16)" in spec.facts
    # Disabled with their invariants: enumeration literals are not in the B-OCL grammar
    # # OCL: at most 3 goalkeepers (select -> auxiliary function, size -> cardinality)
    # assert "all self:this/Team|(#({select0[self.Team_players]}) <= 3)" in spec.facts
    # # OCL: a retired coach has no team
    # assert "all self:this/HeadCoach|((self.HeadCoach_status = ENUM_CoachStatus_RETIRED) " \
    #     "=> ( no(self.HeadCoach_currentTeam)))" in spec.facts
    # OCL: both teams were founded before the match
    assert "all self:this/Match|((dateLt[self.Match_homeTeam.Team_foundationDate,self.Match_matchDate]) " \
        "and (dateLt[self.Match_awayTeam.Team_foundationDate,self.Match_matchDate]))" in spec.facts
    # OCL: non-negative goals
    assert "all self:this/Match|((self.Match_homeGoals >= 0) and (self.Match_awayGoals >= 0))" in spec.facts


def test_run_command(spec):
    assert spec.run_pred == "instance_model"
    assert "5 Country" in spec.run_scopes
    assert "5 Team" in spec.run_scopes
    assert "5 Person" in spec.run_scopes
    assert "5 Player" in spec.run_scopes
    assert "5 GoalKeeper" in spec.run_scopes
    assert "5 HeadCoach" in spec.run_scopes
    assert "5 Fan" in spec.run_scopes
    assert "5 Championship" in spec.run_scopes
    assert "5 Match" in spec.run_scopes
    assert "5 Date" in spec.run_scopes
    assert "5 Int" in spec.run_scopes
    assert "5 Str" in spec.run_scopes
    assert "5 seq" in spec.run_scopes
