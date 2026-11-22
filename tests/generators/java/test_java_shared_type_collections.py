"""Two associations to the same class must not produce two identical add helpers."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Multiplicity, Property, StringType,
)
from besser.generators.java_classes import JavaGenerator


def _model(roles):
    country = Class(name="Country", attributes={Property(name="code", type=StringType)})
    obs = Class(name="Observation", attributes={Property(name="value", type=StringType)})
    assocs = set()
    for role, back in roles:
        assocs.add(BinaryAssociation(name=f"obs_{role}", ends={
            Property(name=role, type=country, multiplicity=Multiplicity(1, 1)),
            Property(name=back, type=obs, multiplicity=Multiplicity(0, "*")),
        }))
    return DomainModel(name="Trade", types={country, obs}, associations=assocs)


def _country_source(tmp_path, roles):
    JavaGenerator(model=_model(roles), output_dir=str(tmp_path)).generate()
    return open(os.path.join(str(tmp_path), "Country.java"), encoding="utf-8").read()


def test_collections_of_the_same_type_get_role_based_helpers(tmp_path):
    source = _country_source(tmp_path, [("reporter", "observations_as_reporter"),
                                        ("partner", "observations_as_partner")])
    assert "public void addObservation(" not in source
    assert "public void addToObservations_as_reporter(Observation observation)" in source
    assert "public void addToObservations_as_partner(Observation observation)" in source


def test_a_single_collection_keeps_the_type_based_helper(tmp_path):
    source = _country_source(tmp_path, [("partner", "observations")])
    assert "public void addObservation(Observation observation)" in source


def test_role_based_helpers_compile(tmp_path):
    """Compile the two-collection output when a JDK is on the machine, as
    test_java_generator.test_generated_java_compiles does."""
    javac = shutil.which("javac")
    if javac is None:
        pytest.skip("no JDK available")

    _country_source(tmp_path, [("reporter", "observations_as_reporter"),
                               ("partner", "observations_as_partner")])
    sources = [str(p) for p in Path(str(tmp_path)).glob("*.java")]
    result = subprocess.run([javac, *sources], capture_output=True, text=True, cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
