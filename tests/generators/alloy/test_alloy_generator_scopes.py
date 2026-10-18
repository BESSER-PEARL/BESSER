import os
import pytest
from tests.generators.alloy.BUML_football_model import football_model
from tests.generators.alloy.BUML_library_with_ocl_v2 import library_with_ocl_v2
from besser.generators.alloy.alloy_generator_new import AlloyGeneratorNew
from tests.generators.alloy.utils_alloy_translation import AlloyTestSpec

@pytest.fixture
def football_spec_s3(football_model, tmpdir):
    tmpdir = tmpdir.mkdir("output")
    generator = AlloyGeneratorNew(model=football_model, output_dir=str(tmpdir))
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
    assert "3 Int" in football_spec_s3.run_scopes
    assert "3 Str" in football_spec_s3.run_scopes
    assert len(football_spec_s3.run_scopes) == 9


@pytest.fixture
def football_spec_s7(football_model, tmpdir):
    tmpdir = tmpdir.mkdir("output")
    generator = AlloyGeneratorNew(model=football_model, output_dir=str(tmpdir))
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
    generator = AlloyGeneratorNew(model=library_with_ocl_v2, output_dir=str(tmpdir))
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
