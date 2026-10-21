import os

import pytest

from besser.BUML.metamodel.object import ObjectModel
from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, BinaryAssociation, Multiplicity, Constraint, StringType, IntegerType,
)
from besser.utilities.web_modeling_editor.backend.services.validators.alloy import AlloyResult, AlloySolver
from tests.utilities.web_modeling_editor.backend.services.validators.alloy.utils_alloy_solver import describe

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


@pytest.fixture
def too_large_model() -> DomainModel:
    """The literal 40000 needs a 17-bit ``Int``: its 131072 atoms are too many for the Alloy Analyzer
    to represent the relation ``age``, even at scope 1."""
    return _team_player_model("self.age = 40000")


def test_check_consistency_consistent(consistent_model, tmp_path):
    solver = AlloySolver(consistent_model, output_dir=str(tmp_path))
    assert solver.check_consistency(scope=1) == AlloyResult.SAT


def test_check_consistency_inconsistent(inconsistent_model, tmp_path):
    solver = AlloySolver(inconsistent_model, output_dir=str(tmp_path))
    assert solver.check_consistency(scope=1) == AlloyResult.UNSAT


def test_check_consistency_scopes_too_large(too_large_model, tmp_path):
    solver = AlloySolver(too_large_model, output_dir=str(tmp_path))
    assert solver.check_consistency(scope=1) == AlloyResult.SCOPES_TOO_LARGE


def test_generate_object_models_scopes_too_large(too_large_model, tmp_path):
    solver = AlloySolver(too_large_model, output_dir=str(tmp_path))
    result, object_models = solver.generate_object_models(scope=1)
    assert result == AlloyResult.SCOPES_TOO_LARGE
    assert object_models == []


def test_generate_object_models_consistent(consistent_model, tmp_path):
    solver = AlloySolver(consistent_model, output_dir=str(tmp_path))
    result, object_models = solver.generate_object_models(scope=1)
    assert result == AlloyResult.SAT
    assert len(object_models) == 1

    scopes = solver.get_scopes()
    assert scopes.objects == 1
    assert scopes.integers == (-4, 3)  # the literal 2 needs 2 bits, plus the sign bit
    assert scopes.strings == 1  # the literal 'Barcelona'
    assert scopes.dates is None

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


def test_scopes_before_any_analysis(consistent_model, tmp_path):
    solver = AlloySolver(consistent_model, output_dir=str(tmp_path))
    assert solver.get_scopes() is None
