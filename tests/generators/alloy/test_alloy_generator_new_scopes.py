import os
import pytest
from tests.generators.alloy.BUML_football_model import football_model
from besser.generators.alloy.alloy_generator import AlloyGenerator
from tests.generators.alloy.utils_alloy_re_parser import AlloySpec

@pytest.fixture
def spec_s3(football_model, tmpdir):
    tmpdir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=football_model, output_dir=str(tmpdir), scope=3)
    generator.generate()
    als_file = generator.get_specification_path()
    assert os.path.isfile(als_file), f"Alloy specification not generated at {als_file}"
    with open(als_file, "r", encoding="utf-8") as f:
        content = f.read()
    return AlloySpec(content)


def test_run_command_s3(spec_s3):
    assert spec_s3.run_pred == "instance_model"
    assert "3 Country" in spec_s3.run_scopes
    assert "3 Team" in spec_s3.run_scopes
    assert "3 Player" in spec_s3.run_scopes
    assert "3 GoalKeeper" in spec_s3.run_scopes
    assert "3 Championship" in spec_s3.run_scopes
    assert "3 Match" in spec_s3.run_scopes
    assert "3 Fan" in spec_s3.run_scopes
    assert "3 Int" in spec_s3.run_scopes
    assert "3 Str" in spec_s3.run_scopes
    assert "3 seq" in spec_s3.run_scopes


@pytest.fixture
def spec_s7(football_model, tmpdir):
    tmpdir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=football_model, output_dir=str(tmpdir), scope=7)
    generator.generate()
    als_file = generator.get_specification_path()
    assert os.path.isfile(als_file), f"Alloy specification not generated at {als_file}"
    with open(als_file, "r", encoding="utf-8") as f:
        content = f.read()
    return AlloySpec(content)


def test_run_command_s7(spec_s7):
    assert spec_s7.run_pred == "instance_model"
    assert "7 Country" in spec_s7.run_scopes
    assert "7 Team" in spec_s7.run_scopes
    assert "7 Player" in spec_s7.run_scopes
    assert "7 GoalKeeper" in spec_s7.run_scopes
    assert "7 Championship" in spec_s7.run_scopes
    assert "7 Match" in spec_s7.run_scopes
    assert "7 Fan" in spec_s7.run_scopes
    assert "7 Int" in spec_s7.run_scopes
    assert "7 Str" in spec_s7.run_scopes
    assert "7 seq" in spec_s7.run_scopes


