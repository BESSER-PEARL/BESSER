import os
import pytest
from tests.generators.alloy.BUML_football_model_v2 import football_model_v2
from besser.generators.alloy.alloy_generator import AlloyGenerator
from tests.generators.alloy.utils_alloy_re_parser import AlloySpec


@pytest.fixture
def generated(football_model_v2, tmpdir):
    tmpdir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=football_model_v2, output_dir=str(tmpdir))
    generator.generate()
    als_file = generator.get_specification_path()
    assert os.path.isfile(als_file), f"Alloy specification not generated at {als_file}"
    with open(als_file, "r", encoding="utf-8") as f:
        content = f.read()
    return str(tmpdir), content


@pytest.fixture
def spec(generated):
    _, content = generated
    return AlloySpec(content)


@pytest.fixture
def raw(generated):
    _, content = generated
    return content


def _has_fact(spec, template):
    """The quantified variable (``a``/``b``) depends on the association end order."""
    return any(template.format(v=v) in spec.facts for v in ("a", "b"))


def test_signatures(spec):
    assert "sig Country" in spec.headers
    assert "sig Team" in spec.headers
    assert "abstract sig Person" in spec.headers
    assert "sig Player extends Person" in spec.headers
    assert "sig GoalKeeper extends Player" in spec.headers
    assert "sig HeadCoach extends Person" in spec.headers
    assert "sig Fan extends Person" in spec.headers
    assert "sig Championship" in spec.headers
    assert "sig Match" in spec.headers


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


def test_enumerations(spec):
    assert len(spec.subsigs_of("Position")) == 4
    assert "abstract sig Position" in spec.headers
    assert "one sig ENUM_Position_GOALKEEPER extends Position" in spec.headers
    assert "one sig ENUM_Position_DEFENDER extends Position" in spec.headers
    assert "one sig ENUM_Position_MIDFIELDER extends Position" in spec.headers
    assert "one sig ENUM_Position_FORWARD extends Position" in spec.headers

    assert len(spec.subsigs_of("CoachStatus")) == 3
    assert "abstract sig CoachStatus" in spec.headers
    assert "one sig ENUM_CoachStatus_ACTIVE extends CoachStatus" in spec.headers
    assert "one sig ENUM_CoachStatus_INACTIVE extends CoachStatus" in spec.headers
    assert "one sig ENUM_CoachStatus_RETIRED extends CoachStatus" in spec.headers


def test_date_support(generated, raw):
    output_dir, _ = generated
    assert "open util/ordering[Date]" in raw
    assert "open date" in raw
    assert os.path.isfile(os.path.join(output_dir, "date.als"))


def test_multiplicity_facts(spec):
    # Fan -> Team (2..4)
    assert _has_fact(spec, "all {v}: Fan | #({v}.Fan_favoriteTeams) >= 2")
    assert _has_fact(spec, "all {v}: Fan | #({v}.Fan_favoriteTeams) <= 4")
    # Team -> Player (11..*): only the lower bound is enforced
    assert _has_fact(spec, "all {v}: Team | #({v}.Team_players) >= 11")
    # HeadCoach -> Team (0..1)
    assert _has_fact(spec, "all {v}: HeadCoach | #({v}.HeadCoach_currentTeam) <= 1")


def test_bidirectional_head_coach_fact(spec):
    assert (
        "HeadCoach_currentTeam = ~Team_headCoach" in spec.facts
        or "Team_headCoach = ~HeadCoach_currentTeam" in spec.facts
    )


def test_ocl_players_older_than_16(raw):
    # forAll -> universal quantifier over the team's players
    assert "all self:this/Team|all p : self.Team_players | (p.Player_age > 16)" in raw


def test_ocl_max_3_goalkeepers(raw):
    # select -> auxiliary function, size -> cardinality
    assert "fun select0[suva:set univ]: set univ" in raw
    assert "(___x___.Player_position = ENUM_Position_GOALKEEPER)" in raw
    assert "all self:this/Team|(#({select0[self.Team_players]}) <= 3)" in raw


def test_ocl_retired_coach_has_no_team(raw):
    assert (
        "all self:this/HeadCoach|((self.HeadCoach_status = ENUM_CoachStatus_RETIRED) "
        "=> ( no(self.HeadCoach_currentTeam)))"
    ) in raw


def test_ocl_teams_founded_before_match(raw):
    assert (
        "all self:this/Match|((dateLt[self.Match_homeTeam.Team_foundationDate,self.Match_matchDate]) "
        "and (dateLt[self.Match_awayTeam.Team_foundationDate,self.Match_matchDate]))"
    ) in raw


def test_ocl_non_negative_goals(raw):
    assert "all self:this/Match|((self.Match_homeGoals >= 0) and (self.Match_awayGoals >= 0))" in raw


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
