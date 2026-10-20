import datetime

import pytest

from besser.BUML.metamodel.object import ObjectModel
from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, BinaryAssociation, Generalization, Multiplicity, StringType,
    IntegerType, DateType, Enumeration, EnumerationLiteral,
)
from besser.utilities.buml_code_builder.domain_model_builder import object_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
    class_buml_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.object_diagram_converter import (
    object_buml_to_json,
)
from tests.utilities.web_modeling_editor.backend.services.converters.utils_buml_to_json_tests import (
    object_attributes, object_links, object_names,
)


@pytest.fixture()
def model() -> DomainModel:
    """Countries, teams, championships and matches, and the abstract ``Person`` specialised into
    ``Player`` (and its ``GoalKeeper``), ``HeadCoach`` and ``Fan``."""
    position = Enumeration("Position", literals={
        EnumerationLiteral("GOALKEEPER"), EnumerationLiteral("DEFENDER"),
        EnumerationLiteral("MIDFIELDER"), EnumerationLiteral("FORWARD"),
    })
    coach_status = Enumeration("CoachStatus", literals={
        EnumerationLiteral("ACTIVE"), EnumerationLiteral("INACTIVE"), EnumerationLiteral("RETIRED"),
    })
    country = Class("Country", attributes={
        Property("name", StringType),
        Property("isoCode", StringType),
    })
    team = Class("Team", attributes={
        Property("name", StringType),
        Property("foundationDate", DateType),
    })
    person = Class("Person", is_abstract=True, attributes={
        Property("name", StringType),
        Property("birthDate", DateType),
    })
    player = Class("Player", attributes={
        Property("age", IntegerType),
        Property("position", position),
    })
    goal_keeper = Class("GoalKeeper", attributes={
        Property("penaltySaves", IntegerType),
    })
    head_coach = Class("HeadCoach", attributes={
        Property("coachingStartDate", DateType),
        Property("status", coach_status),
    })
    championship = Class("Championship", attributes={
        Property("name", StringType),
        Property("year", IntegerType),
    })
    match = Class("Match", attributes={
        Property("homeGoals", IntegerType),
        Property("awayGoals", IntegerType),
        Property("matchDate", DateType),
    })
    fan = Class("Fan")
    generalizations = {
        Generalization(general=person, specific=player),
        Generalization(general=player, specific=goal_keeper),
        Generalization(general=person, specific=head_coach),
        Generalization(general=person, specific=fan),
    }
    team_country = BinaryAssociation("TeamCountry", ends={
        Property("country", country, multiplicity=Multiplicity(1, 1)),
        Property("teams", team, multiplicity=Multiplicity(0, "*"), is_navigable=False),
    })
    championship_match = BinaryAssociation("ChampionshipMatch", ends={
        Property("matches", match, multiplicity=Multiplicity(0, "*")),
        Property("championship", championship, multiplicity=Multiplicity(1, 1), is_navigable=False),
    })
    match_home_team = BinaryAssociation("MatchHomeTeam", ends={
        Property("homeTeam", team, multiplicity=Multiplicity(1, 1)),
        Property("homeMatches", match, multiplicity=Multiplicity(0, "*"), is_navigable=False),
    })
    match_away_team = BinaryAssociation("MatchAwayTeam", ends={
        Property("awayTeam", team, multiplicity=Multiplicity(1, 1)),
        Property("awayMatches", match, multiplicity=Multiplicity(0, "*"), is_navigable=False),
    })
    team_player = BinaryAssociation("TeamPlayer", ends={
        Property("players", player, multiplicity=Multiplicity(11, "*")),
        Property("team", team, multiplicity=Multiplicity(1, 1), is_navigable=False),
    })
    team_head_coach = BinaryAssociation("TeamHeadCoach", ends={
        Property("headCoach", head_coach, multiplicity=Multiplicity(1, 1)),
        Property("currentTeam", team, multiplicity=Multiplicity(0, 1)),
    })
    fan_team = BinaryAssociation("FanTeam", ends={
        Property("favoriteTeams", team, multiplicity=Multiplicity(2, 4)),
        Property("fans", fan, multiplicity=Multiplicity(0, "*"), is_navigable=False),
    })
    fan_country = BinaryAssociation("FanCountry", ends={
        Property("birthCountry", country, multiplicity=Multiplicity(1, 1)),
        Property("fansBorn", fan, multiplicity=Multiplicity(0, "*"), is_navigable=False),
    })
    return DomainModel(
        "Football",
        types={position, coach_status, country, team, person, player, goal_keeper, head_coach,
               championship, match, fan},
        associations={team_country, championship_match, match_home_team, match_away_team,
                      team_player, team_head_coach, fan_team, fan_country},
        generalizations=generalizations,
    )


@pytest.fixture()
def object(model) -> ObjectModel:
    """``boca`` and ``river`` from ``argentina`` play the ``final`` of ``liga``; ``boca`` has a
    ``coach`` and the players ``keeper`` and ``striker``; ``supporter`` follows both teams."""
    country = model.get_class_by_name("Country")
    team = model.get_class_by_name("Team")
    player = model.get_class_by_name("Player")
    goal_keeper = model.get_class_by_name("GoalKeeper")
    head_coach = model.get_class_by_name("HeadCoach")
    championship = model.get_class_by_name("Championship")
    match = model.get_class_by_name("Match")
    fan = model.get_class_by_name("Fan")
    position = model.get_type_by_name("Position")
    coach_status = model.get_type_by_name("CoachStatus")
    argentina = country("argentina").attributes(name="Argentina", isoCode="AR").build()
    boca = team("boca").attributes(name="Boca", foundationDate=datetime.date(1905, 4, 3)).build()
    river = team("river").attributes(name="River", foundationDate=datetime.date(1901, 5, 25)).build()
    coach = head_coach("coach").attributes(name="Coach", birthDate=datetime.date(1960, 1, 1),
                                           coachingStartDate=datetime.date(2020, 1, 1),
                                           status=coach_status.ACTIVE).build()
    keeper = goal_keeper("keeper").attributes(name="Keeper", birthDate=datetime.date(1994, 2, 2), age=30,
                                              position=position.GOALKEEPER, penaltySaves=5).build()
    striker = player("striker").attributes(name="Striker", birthDate=datetime.date(1999, 3, 3), age=25,
                                           position=position.FORWARD).build()
    liga = championship("liga").attributes(name="Liga", year=2024).build()
    final = match("final").attributes(homeGoals=2, awayGoals=1, matchDate=datetime.date(2024, 12, 8)).build()
    supporter = fan("supporter").attributes(name="Supporter", birthDate=datetime.date(1980, 6, 6)).build()
    boca.country = argentina
    river.country = argentina
    boca.headCoach = coach
    boca.players = {keeper, striker}
    liga.matches = {final}
    final.homeTeam = boca
    final.awayTeam = river
    supporter.favoriteTeams = {boca, river}
    supporter.birthCountry = argentina
    return ObjectModel("Football", objects={argentina, boca, river, coach, keeper, striker, liga, final, supporter})


def test_to_json_keeps_information(object, model, tmp_path):
    code_file = tmp_path / "football_objects.py"
    object_model_to_code(object, str(code_file))
    result = object_buml_to_json(code_file.read_text(), class_buml_to_json(model))

    names = object_names(result).values()
    attributes = object_attributes(result)
    links = object_links(result)
    # The importer used to drop everything below
    assert "boca.foundationDate = 1905-04-03" in attributes
    assert "river.foundationDate = 1901-05-25" in attributes
    assert "coach.birthDate = 1960-01-01" in attributes
    assert "coach.coachingStartDate = 2020-01-01" in attributes
    assert "coach.status = ACTIVE" in attributes
    assert "keeper.birthDate = 1994-02-02" in attributes
    assert "keeper.position = GOALKEEPER" in attributes
    assert "striker.birthDate = 1999-03-03" in attributes
    assert "striker.position = FORWARD" in attributes
    assert "final.matchDate = 2024-12-08" in attributes
    assert "supporter.birthDate = 1980-06-06" in attributes
    assert "boca -players-> keeper" in links
    assert "boca -players-> striker" in links
    assert "supporter -favoriteTeams-> boca" in links
    assert "supporter -favoriteTeams-> river" in links
    # The importer already kept everything below
    assert "argentina" in names
    assert "boca" in names
    assert "river" in names
    assert "coach" in names
    assert "keeper" in names
    assert "striker" in names
    assert "liga" in names
    assert "final" in names
    assert "supporter" in names
    assert "argentina.name = Argentina" in attributes
    assert "argentina.isoCode = AR" in attributes
    assert "boca.name = Boca" in attributes
    assert "river.name = River" in attributes
    assert "coach.name = Coach" in attributes
    assert "keeper.name = Keeper" in attributes
    assert "keeper.age = 30" in attributes
    assert "keeper.penaltySaves = 5" in attributes
    assert "striker.name = Striker" in attributes
    assert "striker.age = 25" in attributes
    assert "liga.name = Liga" in attributes
    assert "liga.year = 2024" in attributes
    assert "final.homeGoals = 2" in attributes
    assert "final.awayGoals = 1" in attributes
    assert "supporter.name = Supporter" in attributes
    assert "boca -country-> argentina" in links
    assert "river -country-> argentina" in links
    assert "boca -headCoach-> coach" in links
    assert "liga -matches-> final" in links
    assert "final -homeTeam-> boca" in links
    assert "final -awayTeam-> river" in links
    assert "supporter -birthCountry-> argentina" in links


def test_inherited_attributes_keep_their_class_attribute(object, model, tmp_path):
    # GoalKeeper inherits age from Player and name from Person; they used to get
    # attributeId None and type "str".
    code_file = tmp_path / "football_objects.py"
    object_model_to_code(object, str(code_file))
    result = object_buml_to_json(code_file.read_text(), class_buml_to_json(model))

    [keeper_node] = [n for n in result["nodes"] if n["data"].get("name") == "keeper"]
    keeper = {f"{row['name']} = {row['value']}": row for row in keeper_node["data"]["attributes"]}
    assert keeper["age = 30"]["attributeId"] is not None
    assert keeper["age = 30"]["attributeType"] == "int"
    assert keeper["name = Keeper"]["attributeId"] is not None
