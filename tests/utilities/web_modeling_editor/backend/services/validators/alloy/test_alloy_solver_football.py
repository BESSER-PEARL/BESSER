import pytest

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, BinaryAssociation, Multiplicity, Constraint, StringType, IntegerType,
)
from besser.utilities.web_modeling_editor.backend.services.validators.alloy import AlloyResult, AlloySolver
from tests.utilities.web_modeling_editor.backend.services.validators.alloy.utils_alloy_solver import (
    check_consistency, describe, generate_object_models, needs_alloy,
)

# The tests run the Alloy Analyzer
pytestmark = needs_alloy


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


@pytest.fixture
def squad_model() -> DomainModel:
    """``Team`` 1 -- 0..11 ``Player``: the bound 11 is outside Alloy's default integers, [-8, 7]."""
    team = Class("Team")
    player = Class("Player")
    squad = BinaryAssociation("squad", ends={
        Property("team", team, multiplicity=Multiplicity(1, 1)),
        Property("players", player, multiplicity=Multiplicity(0, 11)),
    })
    return DomainModel("Football", types={team, player}, associations={squad})


@pytest.fixture
def ageless_player_model() -> DomainModel:
    """``Player(age: int, optional)``, whose age must be null and less than 100: inconsistent in OCL,
    since comparing null with ``<`` is invalid."""
    player = Class("Player", attributes={Property("age", IntegerType, is_optional=True)})
    no_age = Constraint("no_age", player, "context Player inv no_age: self.age = null", "OCL")
    age_limit = Constraint("age_limit", player, "context Player inv age_limit: self.age < 100", "OCL")
    return DomainModel("Football", types={player}, constraints={no_age, age_limit})


@pytest.fixture
def ageless_player_arithmetic_model() -> DomainModel:
    """``Player(age: int, optional)``, whose age must be null and its successor positive: inconsistent in OCL,
    since null plus one is invalid."""
    player = Class("Player", attributes={Property("age", IntegerType, is_optional=True)})
    no_age = Constraint("no_age", player, "context Player inv no_age: self.age = null", "OCL")
    next_age = Constraint("next_age", player, "context Player inv next_age: self.age + 1 > 0", "OCL")
    return DomainModel("Football", types={player}, constraints={no_age, next_age})


@pytest.fixture
def ageless_player_inequality_model() -> DomainModel:
    """``Player(age: int, optional)``, whose age must be null and different from 0: consistent in OCL,
    since null <> 0 is true."""
    player = Class("Player", attributes={Property("age", IntegerType, is_optional=True)})
    no_age = Constraint("no_age", player, "context Player inv no_age: self.age = null", "OCL")
    nonzero_age = Constraint("nonzero_age", player, "context Player inv nonzero_age: self.age <> 0", "OCL")
    return DomainModel("Football", types={player}, constraints={no_age, nonzero_age})


def test_check_consistency_consistent(consistent_model):
    result, scopes = check_consistency(consistent_model, scope=1)
    assert result == AlloyResult.SAT
    assert scopes.objects == 1
    assert scopes.integers == (-4, 3)  # the literal 2 needs 2 bits, plus the sign bit
    assert scopes.strings == 1  # the literal 'Barcelona'
    assert scopes.dates is None


def test_check_consistency_inconsistent(inconsistent_model):
    result, _ = check_consistency(inconsistent_model, scope=1)
    assert result == AlloyResult.UNSAT


def test_check_consistency_scopes_too_large(too_large_model):
    result, scopes = check_consistency(too_large_model, scope=1)
    assert result == AlloyResult.SCOPES_TOO_LARGE
    assert scopes.bitwidth == 17  # the literal 40000 needs 16 bits, plus the sign bit


def test_generate_object_models_scopes_too_large(too_large_model):
    result, object_models = generate_object_models(too_large_model, scope=1)
    assert result == AlloyResult.SCOPES_TOO_LARGE
    assert object_models == []


def test_generate_object_models_consistent(consistent_model):
    result, object_models = generate_object_models(consistent_model, scope=1)
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


def test_generate_object_models_inconsistent(inconsistent_model):
    result, object_models = generate_object_models(inconsistent_model, scope=1)
    assert result == AlloyResult.UNSAT
    assert object_models == []


def test_check_consistency_squad_of_eleven(squad_model):
    result, scopes = check_consistency(squad_model, scope=5)
    assert result == AlloyResult.SAT
    assert scopes.integers == (-16, 15)  # the bound 11 needs 4 bits, plus the sign bit


def test_check_consistency_ageless_player(ageless_player_model):
    result, _ = check_consistency(ageless_player_model, scope=1)
    assert result == AlloyResult.UNSAT


def test_check_consistency_ageless_player_arithmetic(ageless_player_arithmetic_model):
    result, _ = check_consistency(ageless_player_arithmetic_model, scope=1)
    assert result == AlloyResult.UNSAT


def test_check_consistency_ageless_player_inequality(ageless_player_inequality_model):
    result, _ = check_consistency(ageless_player_inequality_model, scope=1)
    assert result == AlloyResult.SAT


def test_scopes_before_any_analysis(consistent_model, tmp_path):
    solver = AlloySolver(consistent_model, output_dir=str(tmp_path))
    assert solver.get_scopes() is None
