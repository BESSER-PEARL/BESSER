import os

import pytest

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, BinaryAssociation, Multiplicity, Constraint,
)
from besser.utilities.web_modeling_editor.backend.services.validators.alloy import AlloyResult, AlloySolver
from tests.generators.alloy.BUML_genealogy_model import domain_model as genealogy_fixture

# The tests run the Alloy Analyzer, located through these environment variables
pytestmark = pytest.mark.skipif(
    not (os.getenv("JAVA_HOME") and os.getenv("BESSER_ALLOY_JAR")),
    reason="JAVA_HOME and BESSER_ALLOY_JAR must be set to run the Alloy Analyzer",
)


@pytest.fixture
def genealogy_model() -> DomainModel:
    """``Person`` with ``parents`` 0..*, where nobody is their own ancestor (``closure`` with an iterator)."""
    person = Class("Person")
    is_child_of = BinaryAssociation("is_child_of", ends={
        Property("children", person, multiplicity=Multiplicity(0, "*")),
        Property("parents", person, multiplicity=Multiplicity(0, "*")),
    })
    acyclic = Constraint(
        "acyclic", person, "context Person inv acyclic: self.parents->closure(p | p.parents)->excludes(self)", "OCL"
    )
    return DomainModel("Genealogy", types={person}, associations={is_child_of}, constraints={acyclic})


def test_acyclic_genealogy_is_consistent(genealogy_model, tmp_path):
    solver = AlloySolver(genealogy_model, output_dir=str(tmp_path))
    assert solver.check_consistency(scope=3) == AlloyResult.SAT


def test_everyone_with_a_parent_has_no_acyclic_instance(genealogy_model, tmp_path):
    """In a finite genealogy where everyone has a parent, someone is their own ancestor."""
    person = genealogy_model.get_class_by_name("Person")
    genealogy_model.constraints = {
        Constraint("has_parent", person, "context Person inv has_parent: self.parents->size() >= 1", "OCL"),
        # closure in the short form
        Constraint(
            "acyclic", person, "context Person inv acyclic: self.parents->closure(parents)->excludes(self)", "OCL"
        ),
    }
    solver = AlloySolver(genealogy_model, output_dir=str(tmp_path))
    assert solver.check_consistency(scope=3) == AlloyResult.UNSAT


def test_genealogy_fixture_is_consistent(tmp_path):
    """The genealogy of the Alloy generator tests, with Adam, Eve and their descendants, has an instance."""
    solver = AlloySolver(genealogy_fixture, output_dir=str(tmp_path))
    assert solver.check_consistency(scope=5) == AlloyResult.SAT
