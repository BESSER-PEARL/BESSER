import pytest

from besser.BUML.metamodel.structural import BinaryAssociation, Class, Constraint, DomainModel, Multiplicity, Property
from besser.generators.alloy.alloy_translator import BumlToAlloy


@pytest.fixture
def genealogy_model() -> DomainModel:
    """``Person`` with ``parents`` and ``children`` 0..*."""
    person = Class("Person")
    is_child_of = BinaryAssociation("is_child_of", ends={
        Property("children", person, multiplicity=Multiplicity(0, "*")),
        Property("parents", person, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel("Genealogy", types={person}, associations={is_child_of})


@pytest.mark.parametrize("closure, translation", [
    # a body navigating from the iterator closes its fields
    ("closure(parents)", "(self.Person_parents).*(Person_parents)"),
    ("closure(p | p.parents)", "(self.Person_parents).*(Person_parents)"),
    ("closure(parents.parents)", "(self.Person_parents).*(Person_parents.Person_parents)"),
    # and so do unions and intersections of such navigations
    ("closure(p | p.parents->union(p.children))", "(self.Person_parents).*((Person_parents + Person_children))"),
    ("closure(p | p.parents->intersection(p.children))",
     "(self.Person_parents).*((Person_parents & Person_children))"),
    # any other body closes the relation from each element to the values of the body
    ("closure(p | p.parents->select(q | not q.children->isEmpty()))",
     "(self.Person_parents).*({p: Person, reached0: Person | reached0 in {select0[p.Person_parents]}})"),
])
def test_closure(genealogy_model, closure, translation):
    person = genealogy_model.get_class_by_name("Person")
    genealogy_model.constraints = {
        Constraint("acyclic", person, f"context Person inv acyclic: self.parents->{closure}->excludes(self)", "OCL")
    }
    specification = BumlToAlloy(genealogy_model, scope=5, seed=0).translate().render()
    assert f"all self:this/Person|(self not in ({translation}))" in specification
