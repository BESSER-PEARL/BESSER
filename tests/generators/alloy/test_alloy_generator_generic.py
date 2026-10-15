import os
import re

import pytest

from tests.generators.alloy.BUML_generic_model import domain_model
from besser.generators.alloy.alloy_generator import AlloyGenerator
from tests.generators.alloy.utils_alloy_re_parser import AlloySpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """Generates the Alloy specification of the generic model once per module.

    Returns ``(output_dir, model.als content)``.
    """
    output_dir = tmp_path_factory.mktemp("output")
    generator = AlloyGenerator(model=domain_model, output_dir=str(output_dir))
    generator.generate()
    als_file = generator.get_specification_path()
    assert os.path.isfile(als_file), f"Alloy specification not generated at {als_file}"
    with open(als_file, "r", encoding="utf-8") as f:
        content = f.read()
    return str(output_dir), content


@pytest.fixture(scope="module")
def raw(generated):
    _, content = generated
    return content


@pytest.fixture(scope="module")
def spec(raw):
    return AlloySpec(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Association ends are stored in a set, so the order in which the generator
# visits them is not fixed: the quantified variable (a/b, x/y) and the side
# an inverse fact starts from may change between runs.

_QUANTIFIED_VARS = ("a", "b", "x", "y")


def _has_fact(spec, template):
    """True if some fact matches ``template`` with any quantified variable ``{v}``."""
    return any(template.format(v=v) in spec.facts for v in _QUANTIFIED_VARS)


def _has_inverse_fact(spec, rel1, rel2):
    """True if the facts state that ``rel1`` and ``rel2`` are inverse relations."""
    return f"{rel1} = ~{rel2}" in spec.facts or f"{rel2} = ~{rel1}" in spec.facts


def _facts_mentioning(spec, name):
    return {f for f in spec.facts if name in f}


# ---------------------------------------------------------------------------
# Signatures and inheritance
# ---------------------------------------------------------------------------

def test_signatures(spec):
    assert "abstract sig ClaseA0" in spec.headers
    assert "sig ClaseA1 extends ClaseA0" in spec.headers or \
        "abstract sig ClaseA1 extends ClaseA0" in spec.headers
    assert "sig ClaseA extends ClaseA1" in spec.headers
    assert "sig ClaseB" in spec.headers
    assert "sig ClaseD" in spec.headers
    # The association class is rendered as a plain signature
    assert "sig ClaseC" in spec.headers


def test_inheritance_chain(spec):
    assert spec.parents["ClaseA0"] is None
    assert spec.parents["ClaseA1"] == "ClaseA0"
    assert spec.parents["ClaseA"] == "ClaseA1"
    assert spec.subsigs_of("ClaseA0") == {"ClaseA1"}
    assert spec.subsigs_of("ClaseA1") == {"ClaseA"}


@pytest.mark.xfail(
    strict=True,
    reason="The template drops 'abstract' on classes that have a parent",
)
def test_abstract_subclass_keeps_abstract_keyword(spec):
    assert "abstract sig ClaseA1 extends ClaseA0" in spec.headers


def test_inherited_attributes_are_not_redeclared(spec):
    # ida0 / ida1 live only in the class that declares them
    assert spec.fields["ClaseA0"] >= {"ClaseA0_ida0: Int"}
    assert spec.fields["ClaseA1"] == {"ClaseA1_ida1: Int"}
    all_fields = set().union(*spec.fields.values())
    assert not any(f.startswith(("ClaseA1_ida0", "ClaseA_ida0", "ClaseA_ida1")) for f in all_fields)


# ---------------------------------------------------------------------------
# Signature fields (attributes and navigable association ends)
# ---------------------------------------------------------------------------

def test_signature_fields(spec):
    assert spec.fields["ClaseA0"] == {
        "ClaseA0_ida0: Int",
        "ClaseA0_d: one ClaseD",
    }

    assert spec.fields["ClaseA1"] == {"ClaseA1_ida1: Int"}

    assert spec.fields["ClaseA"] == {
        "ClaseA_at1: Int",
        "ClaseA_at2: Str",
        "ClaseA_atr3: Int",
        "ClaseA_at4: Int",
        "ClaseA_at5: Date",
        "ClaseA_clasec: set ClaseC",
    }

    assert spec.fields["ClaseB"] == {
        "ClaseB_bt1: Int",
        "ClaseB_bt2: Str",
        "ClaseB_btr3: Int",
        "ClaseB_bt4: Date",
        "ClaseB_clasec: set ClaseC",
    }

    assert spec.fields["ClaseD"] == {
        "ClaseD_dt1: Int",
        "ClaseD_dt2: Int",
        "ClaseD_rolb: lone ClaseB",
        "ClaseD_reld2: set ClaseB",
        "ClaseD_a0s: set ClaseA0",
    }


def test_primitive_type_mapping(spec):
    # FloatType -> Int, DateType -> Date, StringType -> Str
    assert "ClaseA_atr3: Int" in spec.fields["ClaseA"]
    assert "ClaseB_btr3: Int" in spec.fields["ClaseB"]
    assert "ClaseA_at5: Date" in spec.fields["ClaseA"]
    assert "ClaseB_bt4: Date" in spec.fields["ClaseB"]
    assert "ClaseA_at2: Str" in spec.fields["ClaseA"]
    assert "ClaseB_bt2: Str" in spec.fields["ClaseB"]


def test_non_navigable_ends_have_no_field(spec):
    # D_B_1.reld and D_B_2.reld1 are not navigable from ClaseB
    all_fields = set().union(*spec.fields.values())
    assert not any(f.startswith(("ClaseB_reld", "ClaseB_reld1")) for f in all_fields)


# ---------------------------------------------------------------------------
# Multiplicities
# ---------------------------------------------------------------------------

def test_multiplicity_zero_to_one(spec):
    # D_B_1: ClaseD -> ClaseB (0..1): 'lone' field plus an upper-bound fact
    assert "ClaseD_rolb: lone ClaseB" in spec.fields["ClaseD"]
    assert _has_fact(spec, "all {v}: ClaseD | #({v}.ClaseD_rolb) <= 1")
    assert not _has_fact(spec, "all {v}: ClaseD | #({v}.ClaseD_rolb) >= 1")


def test_multiplicity_one_to_many(spec):
    # D_B_2: ClaseD -> ClaseB (1..*): only the lower bound is enforced
    assert "ClaseD_reld2: set ClaseB" in spec.fields["ClaseD"]
    assert _has_fact(spec, "all {v}: ClaseD | #({v}.ClaseD_reld2) >= 1")
    assert not any("#" in f and "ClaseD_reld2) <=" in f for f in spec.facts)


def test_multiplicity_exactly_two(spec):
    # D_A0: ClaseD -> ClaseA0 (2..2): lower and upper bound facts
    assert "ClaseD_a0s: set ClaseA0" in spec.fields["ClaseD"]
    assert _has_fact(spec, "all {v}: ClaseD | #({v}.ClaseD_a0s) >= 2")
    assert _has_fact(spec, "all {v}: ClaseD | #({v}.ClaseD_a0s) <= 2")


def test_multiplicity_one_to_one_has_no_cardinality_facts(spec):
    # 1..1 ends are expressed with 'one' in the field, never with facts
    assert "ClaseA0_d: one ClaseD" in spec.fields["ClaseA0"]
    assert not any("#" in f and "ClaseA0_d" in f for f in spec.facts)
    assert not _facts_mentioning(spec, "ClaseB_reld")


def test_bidirectional_association_inverse_fact(spec):
    # D_A0 is navigable in both directions
    assert _has_inverse_fact(spec, "ClaseD_a0s", "ClaseA0_d")


def test_unidirectional_associations_have_no_inverse_fact(spec):
    assert not any("~" in f and "ClaseD_rolb" in f for f in spec.facts)
    assert not any("~" in f and "ClaseD_reld2" in f for f in spec.facts)


# ---------------------------------------------------------------------------
# Association class ClaseC over ab: ClaseA (2..2) <-> ClaseB (3..5)
# ---------------------------------------------------------------------------

def test_association_class_links_both_classes(spec):
    # Each ClaseC link points to exactly one ClaseA and one ClaseB
    assert spec.fields["ClaseC"] == {
        "ClaseC_ctr1: Int",
        "ClaseC_ctr2: Str",
        "ClaseC_aes: one ClaseA",
        "ClaseC_bes: one ClaseB",
    }
    # ... and both ends navigate back to their links
    assert "ClaseA_clasec: set ClaseC" in spec.fields["ClaseA"]
    assert "ClaseB_clasec: set ClaseC" in spec.fields["ClaseB"]


def test_association_class_inverse_facts(spec):
    assert _has_inverse_fact(spec, "ClaseC_aes", "ClaseA_clasec")
    assert _has_inverse_fact(spec, "ClaseC_bes", "ClaseB_clasec")


def test_association_is_reified_through_the_association_class(spec):
    # ClaseA and ClaseB are not connected directly, only through ClaseC
    assert not any(f.startswith("ClaseA_bes") for f in spec.fields["ClaseA"])
    assert not any(f.startswith("ClaseB_aes") for f in spec.fields["ClaseB"])


def test_association_class_navigation_functions(raw):
    assert re.search(
        r"fun bes\[(\w) : ClaseA\] : set ClaseB \{ \1\.~ClaseC_aes\.ClaseC_bes \}", raw
    )
    assert re.search(
        r"fun aes\[(\w) : ClaseB\] : set ClaseA \{ \1\.~ClaseC_bes\.ClaseC_aes \}", raw
    )


def test_association_class_unique_link_per_pair(spec, raw):
    assert "fact UniqueLinkPerPair_ClaseC {" in raw
    assert (
        "all disj c1, c2 : ClaseC | not (c1.ClaseC_aes = c2.ClaseC_aes and c1.ClaseC_bes = c2.ClaseC_bes)"
        in spec.facts
        or "all disj c1, c2 : ClaseC | not (c1.ClaseC_bes = c2.ClaseC_bes and c1.ClaseC_aes = c2.ClaseC_aes)"
        in spec.facts
    )


def test_association_class_multiplicities(spec, raw):
    # Each ClaseA is linked to 3..5 ClaseB, each ClaseB to exactly 2 ClaseA
    assert "fact multiplicity_Bes {" in raw
    assert "fact multiplicity_Aes {" in raw
    assert _has_fact(spec, "all {v} : ClaseA | #bes[{v}] >= 3 and #bes[{v}] =< 5")
    assert _has_fact(spec, "all {v} : ClaseB | #aes[{v}] = 2")


# ---------------------------------------------------------------------------
# OCL constraints
# ---------------------------------------------------------------------------

def test_every_ocl_constraint_is_translated(raw):
    assert raw.count("all self:this/ClaseA|") == 4
    assert raw.count("all self:this/ClaseB|") == 1
    assert raw.count("all self:this/ClaseD|") == 3


def test_ocl_implies(raw):
    # ClaseA_inv_1_1: self.at1>0 implies self.at4<0
    assert "all self:this/ClaseA|((self.ClaseA_at1 > 0) => (self.ClaseA_at4 < 0))" in raw


def test_ocl_implies_false(raw):
    # ClaseA_inv_2_1: self.at1<0 implies false
    assert "all self:this/ClaseA|((self.ClaseA_at1 < 0) => not isTrue[True])" in raw


def test_ocl_if_then_else(raw):
    # ClaseA_inv_3_1: if self.at1>0 then self.at4>self.at1 else self.at4<=0 endif
    assert (
        "all self:this/ClaseA|(((self.ClaseA_at1 > 0)) implies ((self.ClaseA_at4 > self.ClaseA_at1))) "
        "and ((not ((self.ClaseA_at1 > 0))) implies ((self.ClaseA_at4 <= 0)))"
    ) in raw


def test_ocl_xor(raw):
    # ClaseD_inv_4_1: self.dt1=2 xor self.dt2='pepe'  ->  (p and not q) or (not p and q)
    assert (
        "all self:this/ClaseD|((((self.ClaseD_dt1 = 2)) and not ((self.ClaseD_dt2 = Str0))) "
        "or (not ((self.ClaseD_dt1 = 2)) and ((self.ClaseD_dt2 = Str0))))"
    ) in raw


def test_ocl_string_literal_is_declared(generated):
    # 'pepe' becomes the Str0 signature in strings.als
    output_dir, _ = generated
    with open(os.path.join(output_dir, "strings.als"), "r", encoding="utf-8") as f:
        strings_mod = f.read()
    assert "one sig Str0 extends Str {}" in strings_mod
    for i, ch in enumerate("pepe"):
        assert f"data[{i}] = {ch}" in strings_mod


def test_ocl_abs(raw):
    # ClaseB_inv_5_1: self.bt1=self.bt2.abs()
    assert "all self:this/ClaseB|(self.ClaseB_bt1 = abs[self.ClaseB_bt2])" in raw
    assert "fun abs[x: Int]: Int {" in raw


def test_ocl_size(raw):
    # ClaseD_inv_6_1: self.reld2->size()>3
    assert "all self:this/ClaseD|(#(self.ClaseD_reld2) > 3)" in raw


def test_ocl_allinstances_select_size(raw):
    # ClaseA_inv_7_1: ClaseA.allInstances()->select(a | a.at1>0)->size()>=1
    assert (
        "fun select0[suva:set univ]: set univ "
        "{{ ___x___:univ | ___x___ in suva and (___x___.ClaseA_at1 > 0) }}"
    ) in raw
    assert "all self:this/ClaseA|(#({select0[ClaseA]}) >= 1)" in raw


def test_ocl_allinstances_forall(raw):
    # ClaseD_inv_8_1: ClaseD.allInstances()->forAll(d | d.dt1>0 and d.dt2>0)
    assert (
        "all self:this/ClaseD|all d : ClaseD | ((d.ClaseD_dt1 > 0) and (d.ClaseD_dt2 > 0))"
    ) in raw


# ---------------------------------------------------------------------------
# Modules, predicate and run command
# ---------------------------------------------------------------------------

def test_opened_modules(generated, raw):
    output_dir, _ = generated
    for module in ("open util/boolean", "open util/integer", "open util/ordering[Date]",
                   "open date", "open utils", "open strings"):
        assert module in raw
    for als in ("model.als", "utils.als", "date.als", "strings.als"):
        assert os.path.isfile(os.path.join(output_dir, als))


def test_instance_model_predicate(raw):
    for sig in ("ClaseA0", "ClaseA1", "ClaseA", "ClaseB", "ClaseC", "ClaseD", "Date"):
        assert re.search(rf"^\s*some {sig}$", raw, re.MULTILINE)


def test_run_command(spec):
    assert spec.run_pred == "instance_model"
    for sig in ("ClaseA0", "ClaseA1", "ClaseA", "ClaseB", "ClaseC", "ClaseD", "Date"):
        assert f"5 {sig}" in spec.run_scopes
    assert "5 Int" in spec.run_scopes
    assert "5 Str" in spec.run_scopes
    assert "5 seq" in spec.run_scopes
