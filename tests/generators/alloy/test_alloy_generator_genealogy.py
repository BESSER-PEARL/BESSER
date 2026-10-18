import os
import pytest
from tests.generators.alloy.BUML_genealogy_model import domain_model
from besser.generators.alloy.alloy_generator_new import AlloyGeneratorNew
from tests.generators.alloy.utils_alloy_translation import AlloyTestSpec


@pytest.fixture
def spec(tmpdir):
    tmpdir = tmpdir.mkdir("output")
    generator = AlloyGeneratorNew(model=domain_model, output_dir=str(tmpdir))
    generator.generate()
    als_file = generator.get_specification_path()
    assert os.path.isfile(als_file), f"Alloy specification not generated at {als_file}"
    # No Int, String, Date or Boolean types: no library module is generated
    assert sorted(os.listdir(str(tmpdir))) == ["model.als"]
    with open(als_file, "r", encoding="utf-8") as f:
        content = f.read()
    return AlloyTestSpec(content)


def test_signatures(spec):
    assert "abstract sig Person" in spec.sigs
    assert "sig Man extends Person" in spec.sigs
    assert "sig Woman extends Person" in spec.sigs
    assert "sig Adam extends Man" in spec.sigs
    assert "sig Eve extends Woman" in spec.sigs


def test_signature_fields(spec):
    # Both associations are reflexive over Person: only the navigable ends
    # (parents, spouse) become fields, each declared exactly once
    assert len(spec.fields["Person"]) == 2
    assert "Person_parents: set Person" in spec.fields["Person"]
    assert "Person_spouse: lone Person" in spec.fields["Person"]

    # Subclasses inherit the relations without redeclaring them
    assert len(spec.fields["Man"]) == 0
    assert len(spec.fields["Woman"]) == 0
    assert len(spec.fields["Adam"]) == 0
    assert len(spec.fields["Eve"]) == 0


def test_facts(spec):
    assert len(spec.facts) == 18

    # person_1 (1..1) and person (1..1) are not navigable, so there is no `one` field enforcing
    # them: the bounds are checked navigating the opposite field backwards
    assert "all a: Person | #(Person_parents.a) >= 1" in spec.facts
    assert "all a: Person | #(Person_parents.a) <= 1" in spec.facts
    assert "all a: Person | #(Person_spouse.a) >= 1" in spec.facts
    assert "all a: Person | #(Person_spouse.a) <= 1" in spec.facts
    # Person -> Person, spouse (0..1)
    assert "all a: Person | #(a.Person_spouse) <= 1" in spec.facts
    # Unidirectional associations: no inverse facts
    assert not any("~" in f for f in spec.facts)

    # OCL: Adam::allInstances()->size() = 1
    assert "all self:this/Adam|(#(Adam) = 1)" in spec.facts
    # OCL: Eve::allInstances()->size() = 1
    assert "all self:this/Eve|(#(Eve) = 1)" in spec.facts
    # OCL: Adam has no parents (self.parents->isEmpty())
    assert "all self:this/Adam|(no self.Person_parents)" in spec.facts
    # OCL: Eve has no parents (self.parents->isEmpty())
    assert "all self:this/Eve|(no self.Person_parents)" in spec.facts
    # OCL: Adam's spouse is Eve (self.spouse.oclIsTypeOf(Eve) and self.spouse->size()=1)
    assert "all self:this/Adam|((self.Person_spouse in Eve) and (#(self.Person_spouse) = 1))" in spec.facts
    # OCL: Eve's spouse is Adam (self.spouse.oclIsTypeOf(Adam) and self.spouse->size()=1)
    assert "all self:this/Eve|((self.Person_spouse in Adam) and (#(self.Person_spouse) = 1))" in spec.facts
    # OCL: a man's spouse is a woman (self.spouse.oclIsKindOf(Woman))
    assert "all self:this/Man|(self.Person_spouse in Woman)" in spec.facts
    # OCL: a woman's spouse is a man (self.spouse.oclIsKindOf(Man))
    assert "all self:this/Woman|(self.Person_spouse in Man)" in spec.facts
    # OCL: two parents are a woman and a man (exists with two iterator variables)
    assert "all self:this/Person|((#(self.Person_parents) = 2) => " \
        "(some p, q : self.Person_parents | ((p in Woman) and (q in Man))))" in spec.facts
    # OCL: nobody is married to a parent (self.parents->intersection(self.spouse)->isEmpty())
    assert "all self:this/Person|(no (self.Person_parents & self.Person_spouse))" in spec.facts
    # OCL: spouses share no parent (self.spouse.parents->intersection(self.parents)->isEmpty())
    assert "all self:this/Person|(no (self.Person_spouse.Person_parents & self.Person_parents))" in spec.facts
    # OCL: marriage is symmetric ((self.spouse<>null) implies (self = self.spouse.spouse))
    assert "all self:this/Person|((some self.Person_spouse) => " \
        "(self = self.Person_spouse.Person_spouse))" in spec.facts
    # OCL: everyone except Adam and Eve has two parents
    assert "all self:this/Person|((not ((self in Adam) or (self in Eve))) => " \
        "(#(self.Person_parents) = 2))" in spec.facts


def test_run_command(spec):
    assert spec.run_pred == "instance_model"
    assert "5 Person" in spec.run_scopes
    assert "5 Man" in spec.run_scopes
    assert "5 Woman" in spec.run_scopes
    assert "5 Adam" in spec.run_scopes
    assert "5 Eve" in spec.run_scopes
