import pytest

from besser.BUML.metamodel.structural import (
    Class, Property, BinaryAssociation, Generalization, DomainModel, Multiplicity, Constraint,
)

# ---------------------------------------------------------------------------
# Fixture: genealogy domain model
# ---------------------------------------------------------------------------

@pytest.fixture
def genealogy_model():
    """Adam, Eve and their descendants: Person (abstract), Man, Woman, Adam and Eve, with parents (0..*)
    and a spouse (0..1), and OCL invariants on marriage and ancestry."""
    # Classes
    Adam = Class(name="Adam")
    Eve = Class(name="Eve")
    Man = Class(name="Man")
    Person = Class(name="Person", is_abstract=True)
    Woman = Class(name="Woman")

    # Relationships
    is_child_of: BinaryAssociation = BinaryAssociation(
        name="is_child_of",
        ends={
            Property(name="children", type=Person, multiplicity=Multiplicity(0, 9999), is_navigable=False),
            Property(name="parents", type=Person, multiplicity=Multiplicity(0, 9999))
        }
    )
    is_coupled_with: BinaryAssociation = BinaryAssociation(
        name="is_coupled_with",
        ends={
            Property(name="spouse_of", type=Person, multiplicity=Multiplicity(0, 1), is_navigable=False),
            Property(name="spouse", type=Person, multiplicity=Multiplicity(0, 1))
        }
    )

    # Generalizations
    gen_Adam_Man = Generalization(general=Man, specific=Adam)
    gen_Eve_Woman = Generalization(general=Woman, specific=Eve)
    gen_Man_Person = Generalization(general=Person, specific=Man)
    gen_Woman_Person = Generalization(general=Person, specific=Woman)

    # OCL Constraints
    one_adam: Constraint = Constraint(
        name="one_adam",
        context=Adam,
        expression="context Adam inv one_adam: Adam::allInstances()->size() = 1",
        language="OCL"
    )
    adam_has_no_parents: Constraint = Constraint(
        name="adam_has_no_parents",
        context=Adam,
        expression="context Adam inv adam_has_no_parents: self.parents->isEmpty()",
        language="OCL"
    )
    adam_married_to_eve: Constraint = Constraint(
        name="adam_married_to_eve",
        context=Adam,
        expression="context Adam inv adam_married_to_eve: self.spouse.oclIsTypeOf(Eve) and self.spouse->size() = 1",
        language="OCL"
    )
    eve_married_to_adam: Constraint = Constraint(
        name="eve_married_to_adam",
        context=Eve,
        expression="context Eve inv eve_married_to_adam: self.spouse.oclIsTypeOf(Adam) and self.spouse->size() = 1",
        language="OCL"
    )
    one_eve: Constraint = Constraint(
        name="one_eve",
        context=Eve,
        expression="context Eve inv one_eve: Eve::allInstances()->size() = 1",
        language="OCL"
    )
    eve_has_no_parents: Constraint = Constraint(
        name="eve_has_no_parents",
        context=Eve,
        expression="context Eve inv eve_has_no_parents: self.parents->isEmpty()",
        language="OCL"
    )
    man_married_to_a_woman: Constraint = Constraint(
        name="man_married_to_a_woman",
        context=Man,
        expression="context Man inv man_married_to_a_woman: self.spouse.oclIsKindOf(Woman)",
        language="OCL"
    )
    two_parents_are_a_woman_and_a_man: Constraint = Constraint(
        name="two_parents_are_a_woman_and_a_man",
        context=Person,
        expression="context Person inv two_parents_are_a_woman_and_a_man: self.parents->size() = 2 implies "
                   "self.parents->exists(p, q: Person | p.oclIsKindOf(Woman) and q.oclIsKindOf(Man))",
        language="OCL"
    )
    not_married_to_a_parent: Constraint = Constraint(
        name="not_married_to_a_parent",
        context=Person,
        expression="context Person inv not_married_to_a_parent: self.parents->intersection(self.spouse)->isEmpty()",
        language="OCL"
    )
    spouses_share_no_parent: Constraint = Constraint(
        name="spouses_share_no_parent",
        context=Person,
        expression="context Person inv spouses_share_no_parent: "
                   "self.spouse.parents->intersection(self.parents)->isEmpty()",
        language="OCL"
    )
    not_own_ancestor: Constraint = Constraint(
        name="not_own_ancestor",
        context=Person,
        expression="context Person inv not_own_ancestor: self.parents->closure(parents)->excludes(self)",
        language="OCL"
    )
    marriage_is_symmetric: Constraint = Constraint(
        name="marriage_is_symmetric",
        context=Person,
        expression="context Person inv marriage_is_symmetric: "
                   "(self.spouse <> null) implies (self = self.spouse.spouse)",
        language="OCL"
    )
    two_parents_except_adam_and_eve: Constraint = Constraint(
        name="two_parents_except_adam_and_eve",
        context=Person,
        expression="context Person inv two_parents_except_adam_and_eve: "
                   "not (self.oclIsTypeOf(Adam) or self.oclIsTypeOf(Eve)) implies (self.parents->size() = 2)",
        language="OCL"
    )
    woman_married_to_a_man: Constraint = Constraint(
        name="woman_married_to_a_man",
        context=Woman,
        expression="context Woman inv woman_married_to_a_man: self.spouse.oclIsKindOf(Man)",
        language="OCL"
    )

    # Domain Model
    model = DomainModel(
        name="Genealogy",
        types={Adam, Eve, Man, Person, Woman},
        associations={is_child_of, is_coupled_with},
        constraints={
            one_adam, adam_has_no_parents, adam_married_to_eve, one_eve, eve_has_no_parents, eve_married_to_adam,
            man_married_to_a_woman, woman_married_to_a_man, two_parents_are_a_woman_and_a_man, not_married_to_a_parent,
            spouses_share_no_parent, not_own_ancestor, marriage_is_symmetric, two_parents_except_adam_and_eve,
        },
        generalizations={gen_Adam_Man, gen_Eve_Woman, gen_Man_Person, gen_Woman_Person},
    )
    return model
