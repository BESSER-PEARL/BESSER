import os
import pytest
from besser.BUML.metamodel.structural import (
    BinaryAssociation, BooleanType, Class, DomainModel, Multiplicity, Property,
)
from tests.generators.alloy.BUML_football_model import football_model
from tests.generators.alloy.BUML_library_with_ocl_v2 import library_with_ocl_v2
from besser.generators.alloy.alloy_generator import AlloyGenerator
from tests.generators.alloy.utils_alloy_translation import AlloyTestSpec

@pytest.fixture
def football_spec_s3(football_model, tmpdir):
    tmpdir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=football_model, output_dir=str(tmpdir))
    generator.generate(scope=3)
    als_file = generator.get_specification_path()
    assert os.path.isfile(als_file), f"Alloy specification not generated at {als_file}"
    with open(als_file, "r", encoding="utf-8") as f:
        content = f.read()
    return AlloyTestSpec(content)


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
def football_spec_s7(football_model, tmpdir):
    tmpdir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=football_model, output_dir=str(tmpdir))
    generator.generate(scope=7)
    als_file = generator.get_specification_path()
    assert os.path.isfile(als_file), f"Alloy specification not generated at {als_file}"
    with open(als_file, "r", encoding="utf-8") as f:
        content = f.read()
    return AlloyTestSpec(content)


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
def library_spec_s1(library_with_ocl_v2, tmpdir):
    tmpdir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=library_with_ocl_v2, output_dir=str(tmpdir))
    generator.generate(scope=1)
    als_file = generator.get_specification_path()
    assert os.path.isfile(als_file), f"Alloy specification not generated at {als_file}"
    with open(als_file, "r", encoding="utf-8") as f:
        content = f.read()
    return AlloyTestSpec(content)


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


def test_scopes_football_s3(football_model, tmpdir):
    generator = AlloyGenerator(model=football_model, output_dir=str(tmpdir))
    generator.generate(scope=3)
    scopes = generator.get_scopes()
    assert scopes.objects == 3
    assert scopes.bitwidth == 4  # the bound 4 of favoriteTeams needs 3 bits, plus the sign bit
    assert scopes.integers == (-8, 7)
    assert scopes.strings == 3
    assert scopes.dates is None


def test_scopes_library_s1(library_with_ocl_v2, tmpdir):
    generator = AlloyGenerator(model=library_with_ocl_v2, output_dir=str(tmpdir))
    generator.generate(scope=1)
    scopes = generator.get_scopes()
    assert scopes.objects == 1
    assert scopes.bitwidth == 5  # the literal 10 needs 4 bits, plus the sign bit
    assert scopes.integers == (-16, 15)
    assert scopes.strings == 2  # the literals 'borges' and 'cortazar'
    assert scopes.dates == 1


def test_scopes_without_values(switch_model, tmpdir):
    generator = AlloyGenerator(model=switch_model, output_dir=str(tmpdir))
    generator.generate(scope=3)
    scopes = generator.get_scopes()
    assert scopes.bitwidth == 3  # the scope
    assert scopes.integers == (-4, 3)
    assert scopes.strings is None
    assert scopes.dates is None


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


def test_scopes_squad_s5(squad_model, tmpdir):
    generator = AlloyGenerator(model=squad_model, output_dir=str(tmpdir))
    generator.generate(scope=5)
    scopes = generator.get_scopes()
    assert scopes.objects == 5
    assert scopes.bitwidth == 5  # the bound 11 needs 4 bits, plus the sign bit
    assert scopes.integers == (-16, 15)
