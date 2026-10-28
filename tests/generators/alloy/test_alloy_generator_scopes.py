import pytest
from besser.BUML.metamodel.structural import (
    BinaryAssociation, BooleanType, Class, DomainModel, Multiplicity, Property,
)
from tests.generators.alloy.BUML_football_model import football_model  # noqa: F401
from tests.generators.alloy.BUML_library_with_ocl_v2 import library_with_ocl_v2  # noqa: F401
from tests.generators.alloy.utils_alloy_translation import translate


@pytest.fixture
def football_spec_s3(football_model):  # noqa: F811
    return translate(football_model, scope=3)


def test_run_command_football_s3(football_spec_s3):
    assert football_spec_s3.run_pred == "instance_model"
    assert "3 Country" in football_spec_s3.run_scopes
    assert "3 Team" in football_spec_s3.run_scopes
    assert "3 Player" in football_spec_s3.run_scopes
    assert "3 GoalKeeper" in football_spec_s3.run_scopes
    assert "3 Championship" in football_spec_s3.run_scopes
    assert "3 Match" in football_spec_s3.run_scopes
    assert "3 Fan" in football_spec_s3.run_scopes
    assert "4 Int" in football_spec_s3.run_scopes  # the bound 4 of favoriteTeams needs 3 bits, plus the sign bit
    assert "3 Str" in football_spec_s3.run_scopes
    assert len(football_spec_s3.run_scopes) == 9


@pytest.fixture
def football_spec_s7(football_model):  # noqa: F811
    return translate(football_model, scope=7)


def test_run_command_football_s7(football_spec_s7):
    assert football_spec_s7.run_pred == "instance_model"
    assert "7 Country" in football_spec_s7.run_scopes
    assert "7 Team" in football_spec_s7.run_scopes
    assert "7 Player" in football_spec_s7.run_scopes
    assert "7 GoalKeeper" in football_spec_s7.run_scopes
    assert "7 Championship" in football_spec_s7.run_scopes
    assert "7 Match" in football_spec_s7.run_scopes
    assert "7 Fan" in football_spec_s7.run_scopes
    assert "7 Int" in football_spec_s7.run_scopes
    assert "7 Str" in football_spec_s7.run_scopes
    assert len(football_spec_s7.run_scopes) == 9


@pytest.fixture
def library_spec_s1(library_with_ocl_v2):  # noqa: F811
    return translate(library_with_ocl_v2, scope=1)


def test_run_command_library_s1(library_spec_s1):
    assert library_spec_s1.run_pred == "instance_model"
    assert "1 Author" in library_spec_s1.run_scopes
    assert "1 Book" in library_spec_s1.run_scopes
    assert "1 Library" in library_spec_s1.run_scopes
    assert "1 Date" in library_spec_s1.run_scopes
    assert "2 Str" in library_spec_s1.run_scopes  # the literals 'borges' and 'cortazar'
    assert "5 Int" in library_spec_s1.run_scopes  # the literal 10 needs 4 bits, plus the sign bit
    assert len(library_spec_s1.run_scopes) == 6


@pytest.fixture
def switch_model() -> DomainModel:
    """``Switch(on: bool)``: a model without integers, strings or dates."""
    switch = Class("Switch", attributes={Property("on", BooleanType)})
    return DomainModel("Switch_model", types={switch})


@pytest.fixture
def switch_spec_s3(switch_model):
    return translate(switch_model, scope=3)


def test_run_command_switch_s3(switch_spec_s3):
    assert switch_spec_s3.run_pred == "instance_model"
    assert "3 Switch" in switch_spec_s3.run_scopes
    assert "3 Int" in switch_spec_s3.run_scopes  # the scope
    # No strings or dates: no Str or Date scope
    assert len(switch_spec_s3.run_scopes) == 2


@pytest.fixture
def squad_model() -> DomainModel:
    """``Team`` 1 -- 0..11 ``Player``: a model without integers, but with the multiplicity bound 11."""
    team = Class("Team")
    player = Class("Player")
    squad = BinaryAssociation("squad", ends={
        Property("team", team, multiplicity=Multiplicity(1, 1)),
        Property("players", player, multiplicity=Multiplicity(0, 11)),
    })
    return DomainModel("Football", types={team, player}, associations={squad})


@pytest.fixture
def squad_spec_s5(squad_model):
    return translate(squad_model, scope=5)


def test_run_command_squad_s5(squad_spec_s5):
    assert squad_spec_s5.run_pred == "instance_model"
    assert "5 Team" in squad_spec_s5.run_scopes
    assert "5 Player" in squad_spec_s5.run_scopes
    assert "5 Int" in squad_spec_s5.run_scopes  # the bound 11 needs 4 bits, plus the sign bit
    assert len(squad_spec_s5.run_scopes) == 3
