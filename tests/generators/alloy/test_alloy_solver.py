import os

import pytest

from besser.BUML.metamodel.object import ObjectModel
from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, BinaryAssociation, Multiplicity, Constraint, StringType, IntegerType,
)
from besser.generators.alloy.instance_generator import AlloySolver
from besser.generators.alloy.instance_generator.alloy_analyzer_executor import AlloyResult
from tests.generators.alloy.utils_alloy_translation import describe

# The tests run the Alloy Analyzer, located through these environment variables
pytestmark = pytest.mark.skipif(
    not (os.getenv("JAVA_HOME") and os.getenv("BESSER_ALLOY_JAR")),
    reason="JAVA_HOME and BESSER_ALLOY_JAR must be set to run the Alloy Analyzer",
)


def _team_player_model(player_age: str) -> DomainModel:
    """``Team`` 1 -- 1 ``Player``, the team named ``'Barcelona'`` and the player constrained by *player_age*.
    At scope 1 the model has a single instance, if any."""
    team = Class("Team", attributes={Property("name", StringType)})
    player = Class("Player", attributes={Property("age", IntegerType)})
    plays_in = BinaryAssociation("plays_in", ends={
        Property("team", team, multiplicity=Multiplicity(1, 1)),
        Property("player", player, multiplicity=Multiplicity(1, 1)),
    })
    team_name = Constraint("team_name", team, "context Team inv team_name: self.name = 'Barcelona'", "OCL")
    age = Constraint("player_age", player, f"context Player inv player_age: {player_age}", "OCL")
    return DomainModel(
        "Team_model", types={team, player}, associations={plays_in}, constraints={team_name, age}
    )


@pytest.fixture
def consistent_model() -> DomainModel:
    return _team_player_model("self.age = 2")


@pytest.fixture
def inconsistent_model() -> DomainModel:
    return _team_player_model("self.age = 2 and self.age = 3")


def test_generate_object_models_consistent(consistent_model, tmp_path):
    solver = AlloySolver(consistent_model, output_dir=str(tmp_path))
    result, object_models = solver.generate_object_models(scope=1)
    assert result == AlloyResult.SAT
    assert len(object_models) == 1

    object_model = object_models[0]
    description = describe(object_model)
    assert object_model.validate(raise_exception=False)["errors"] == []
    assert "Team_0: Team" in description
    assert "Player_0: Player" in description
    assert "Team_0.name = 'Barcelona'" in description
    assert "Player_0.age = 2" in description
    assert "Team_0.player = Player_0" in description
    assert "Player_0.team = Team_0" in description
    assert len(description) == 6


def test_generate_object_models_inconsistent(inconsistent_model, tmp_path):
    solver = AlloySolver(inconsistent_model, output_dir=str(tmp_path))
    result, object_models = solver.generate_object_models(scope=1)
    assert result == AlloyResult.UNSAT
    assert object_models == []
