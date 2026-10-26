import pytest

from besser.BUML.metamodel.structural import BinaryAssociation, Class, DomainModel, Multiplicity, Property
from tests.generators.alloy.utils_alloy_translation import translate


@pytest.fixture
def genealogy_model() -> DomainModel:
    """``Person`` with ``parents`` and ``children`` 0..*."""
    person = Class("Person")
    is_child_of = BinaryAssociation("is_child_of", ends={
        Property("children", person, multiplicity=Multiplicity(0, "*")),
        Property("parents", person, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel("Genealogy", types={person}, associations={is_child_of})


# a body navigating from the iterator closes its fields

def test_closure_short_form(genealogy_model):
    spec = translate(genealogy_model, "Person", "self.parents->closure(parents)->excludes(self)", scope=5)
    assert "all self:this/Person|(self not in ((self.Person_parents).*(Person_parents)))" in spec.facts


def test_closure_explicit_iterator(genealogy_model):
    spec = translate(genealogy_model, "Person", "self.parents->closure(p | p.parents)->excludes(self)", scope=5)
    assert "all self:this/Person|(self not in ((self.Person_parents).*(Person_parents)))" in spec.facts


def test_closure_of_a_chain(genealogy_model):
    spec = translate(genealogy_model, "Person", "self.parents->closure(parents.parents)->excludes(self)", scope=5)
    assert "all self:this/Person|(self not in ((self.Person_parents).*(Person_parents.Person_parents)))" \
        in spec.facts


# and so do unions and intersections of such navigations

def test_closure_of_a_union(genealogy_model):
    spec = translate(
        genealogy_model, "Person", "self.parents->closure(p | p.parents->union(p.children))->excludes(self)", scope=5
    )
    assert "all self:this/Person|(self not in ((self.Person_parents).*((Person_parents + Person_children))))" \
        in spec.facts


def test_closure_of_an_intersection(genealogy_model):
    spec = translate(
        genealogy_model, "Person", "self.parents->closure(p | p.parents->intersection(p.children))->excludes(self)",
        scope=5,
    )
    assert "all self:this/Person|(self not in ((self.Person_parents).*((Person_parents & Person_children))))" \
        in spec.facts


# any other body closes the relation from each element to the values of the body

def test_closure_of_another_body(genealogy_model):
    spec = translate(
        genealogy_model, "Person",
        "self.parents->closure(p | p.parents->select(q | not q.children->isEmpty()))->excludes(self)", scope=5,
    )
    assert "all self:this/Person|(self not in ((self.Person_parents).*({p: Person, reached0: Person | " \
        "reached0 in {select0[p.Person_parents]}})))" in spec.facts
