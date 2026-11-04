import pytest

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, BinaryAssociation, Multiplicity, Constraint,
)
from besser.utilities.web_modeling_editor.backend.services.validators.alloy import AlloyResult
from tests.generators.alloy.BUML_genealogy_model import genealogy_model  # noqa: F401
from tests.utilities.web_modeling_editor.backend.services.validators.alloy.utils_alloy_solver import (
    check_consistency, needs_alloy,
)

# The tests run the Alloy Analyzer
pytestmark = needs_alloy


@pytest.fixture
def parents_model() -> DomainModel:
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


def test_acyclic_genealogy_is_consistent(parents_model):
    result, _ = check_consistency(parents_model, scope=3)
    assert result == AlloyResult.SAT


def test_everyone_with_a_parent_has_no_acyclic_instance(parents_model):
    """In a finite genealogy where everyone has a parent, someone is their own ancestor."""
    person = parents_model.get_class_by_name("Person")
    parents_model.constraints = {
        Constraint("has_parent", person, "context Person inv has_parent: self.parents->size() >= 1", "OCL"),
        # closure in the short form
        Constraint(
            "acyclic", person, "context Person inv acyclic: self.parents->closure(parents)->excludes(self)", "OCL"
        ),
    }
    result, _ = check_consistency(parents_model, scope=3)
    assert result == AlloyResult.UNSAT


def test_genealogy_fixture_is_consistent(genealogy_model):  # noqa: F811
    """The genealogy of the Alloy generator tests, with Adam, Eve and their descendants, has an instance."""
    result, _ = check_consistency(genealogy_model, scope=5)
    assert result == AlloyResult.SAT
