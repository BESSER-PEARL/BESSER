"""Tests for AlloyGenerator.
Model under test (Team / Player):
    Team    (name: str)
    Player  (name: str, age: int)
    Team 1 ---PlaysFor--- 3..4 Player   (team / players)
"""
import glob
import os
import re
import sys
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    Constraint,
    DateTimeType,
    DateType,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    Generalization,
    IntegerType,
    Multiplicity,
    Property,
    StringType,
    TimeDeltaType,
    TimeType,
)
from besser.generators.alloy_generator.alloy_generator import AlloyGenerator
from besser.generators.alloy_generator.utils_alloy import (
    build_consistency_rule,
    sanitize_alloy_name,
)
from besser.generators.alloy_generator.step_3_alloy_to_buml import (
    AlloyToBesserConverter,
)
from besser.generators.alloy_generator.translate_ocl_alloy import (
    EnumReferenceError,
    TranslatorState,
    encode_date,
    generate_dates_and_order,
    is_date,
    ocl_to_alloy,
    parse_ocl_date,
    random_date,
)

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def team_player_model():
    """A minimal two-class model: Team 1 -- 3..4 Player.

    Structure:
        Team    (name: str)
        Player  (name: str, age: int)
        Team 1 ---PlaysFor--- 3..4 Player  (team / players)
    """
    Team = Class(name="Team")
    Player = Class(name="Player")

    Team_name = Property(name="name", type=StringType)
    Team.attributes = {Team_name}

    Player_name = Property(name="name", type=StringType)
    Player_age = Property(name="age", type=IntegerType)
    Player.attributes = {Player_name, Player_age}

    plays_for = BinaryAssociation(
        name="PlaysFor",
        ends={
            Property(name="players", type=Player, multiplicity=Multiplicity(3, 4)),
            Property(name="team", type=Team, multiplicity=Multiplicity(1, 1)),
        },
    )

    model = DomainModel(
        name="TeamModel",
        types={Team, Player},
        associations={plays_for},
    )
    return model


def _generated_als_path(output_dir):
    """Locate the single .als file produced in output_dir.
    """
    matches = glob.glob(os.path.join(output_dir, "model.als"))
    assert len(matches) == 1, (
        f"Expected exactly one model.als file in {output_dir}, found {matches}"
    )
    return matches[0]

# ---------------------------------------------------------------------------
# Generation tests
# ---------------------------------------------------------------------------

def test_generator_creates_als_file(team_player_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=team_player_model, output_dir=str(output_dir))

    generator.generate()

    als_file = _generated_als_path(str(output_dir))
    assert os.path.isfile(als_file)


def test_signatures_and_attributes_present(team_player_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=team_player_model, output_dir=str(output_dir))
    generator.generate()

    with open(_generated_als_path(str(output_dir)), "r", encoding="utf-8") as f:
        spec = f.read()

    # Class signatures
    assert "sig Team" in spec
    assert "sig Player" in spec

    # Scalar attributes show up under their owning class.
    assert "Team_name: str" in spec
    assert "Player_name: str" in spec
    assert "Player_age: Int" in spec

    # "int" itself never gets its own basic sig (it's mapped to Alloy's
    assert "open util/integer" in spec
    assert "sig str {}" in spec


def test_one_to_one_end_renders_as_alloy_one_keyword(team_player_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=team_player_model, output_dir=str(output_dir))
    generator.generate()

    with open(_generated_als_path(str(output_dir)), "r", encoding="utf-8") as f:
        spec = f.read()

    assert "Player_team: one Team" in spec


def test_three_to_four_end_renders_as_alloy_set_keyword(team_player_model, tmpdir):
    """The Player side has multiplicity 3..4 (max > 1)"""
    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=team_player_model, output_dir=str(output_dir))
    generator.generate()

    with open(_generated_als_path(str(output_dir)), "r", encoding="utf-8") as f:
        spec = f.read()

    assert "Team_players: set Player" in spec


def test_one_to_one_end_has_no_cardinality_facts(team_player_model, tmpdir):
    """The Team side (1..1) is the implicit default multiplicity, so
    build_consistency_rule must NOT emit min/max cardinality facts for it.

     """
    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=team_player_model, output_dir=str(output_dir))
    generator.generate()

    with open(_generated_als_path(str(output_dir)), "r", encoding="utf-8") as f:
        spec = f.read()

    # No fact constrains the cardinality of Player's "team" navigation
    assert "#(a.Player_team)" not in spec
    assert "#(b.Player_team)" not in spec


def test_three_to_four_end_has_lower_and_upper_bound_facts(team_player_model, tmpdir):
    """The Player side (3..4) must produce both a lower-bound (>=3) and an
    upper-bound (<=4) cardinality fact over Team's navigation to players.

    """
    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=team_player_model, output_dir=str(output_dir))
    generator.generate()

    with open(_generated_als_path(str(output_dir)), "r", encoding="utf-8") as f:
        spec = f.read()

    assert re.search(r"#\([ab]\.Team_players\)\s*>=\s*3", spec), spec
    assert re.search(r"#\([ab]\.Team_players\)\s*<=\s*4", spec), spec


def test_generic_instance_model_predicate_and_run(team_player_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=team_player_model, output_dir=str(output_dir))
    generator.generate()

    with open(_generated_als_path(str(output_dir)), "r", encoding="utf-8") as f:
        spec = f.read()
    # Normalize the template's uneven whitespace so substrings match reliably.
    spec = re.sub(r"\s+", " ", spec)

    assert "pred instance_model[]{" in spec
    assert "some Team" in spec
    assert "some Player" in spec
    assert "run instance_model for" in spec
    # The template emits a per-signature scope (at most 5 atoms per sig), not
    # the Alloy ``exactly`` keyword.
    assert "5 Team" in spec
    assert "5 Player" in spec


def test_ocl_constraint_is_translated_to_alloy_fact(tmpdir):
    Team = Class(name="Team")
    Player = Class(name="Player")
    Team.attributes = {Property(name="name", type=StringType)}
    Player.attributes = {
        Property(name="name", type=StringType),
        Property(name="age", type=IntegerType),
    }
    plays_for = BinaryAssociation(
        name="PlaysFor",
        ends={
            Property(name="players", type=Player, multiplicity=Multiplicity(3, 4)),
            Property(name="team", type=Team, multiplicity=Multiplicity(1, 1)),
        },
    )
    age_positive = Constraint(
        name="AgePositive",
        context=Player,
        expression="context Player inv AgePositive: self.age > 0",
        language="OCL",
    )
    model = DomainModel(
        name="TeamModel",
        types={Team, Player},
        associations={plays_for},
        constraints={age_positive},
    )

    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=model, output_dir=str(output_dir))
    generator.generate()

    with open(_generated_als_path(str(output_dir)), "r", encoding="utf-8") as f:
        spec = f.read()

    # Real ocl_to_alloy output: fact{ all self:this/Player|(self.Player_age > 0)}
    assert "this/Player" in spec
    assert "Player_age > 0" in spec
    assert "fact{" in spec


def test_ocl_not_implies_with_oclistypeof_and_arrow_size(tmpdir):
    """A constraint mixing a unary ``not``, ``oclIsTypeOf``, ``implies`` and
    a collection ``->size()`` must parse and translate:
        context Person inv : not (self.oclIsTypeOf(Adam)) implies (self.parents->size()>0)

    Regression test: the recursive-descent parser in translate_ocl_alloy used
    to raise ``ValueError: Unexpected token ... ('operator', 'not')`` because
    unary operators were never handled.
    """
    Person = Class(name="Person")
    Adam = Class(name="Adam")
    Person.attributes = {Property(name="name", type=StringType)}
    Adam.attributes = {Property(name="name", type=StringType)}
    parents = BinaryAssociation(
        name="Parents",
        ends={
            Property(name="parents", type=Person, multiplicity=Multiplicity(0, "*")),
            Property(name="children", type=Person, multiplicity=Multiplicity(0, "*")),
        },
    )
    genealogy = Generalization(general=Person, specific=Adam)
    inv = Constraint(
        name="AdamHasParents",
        context=Person,
        expression=(
            "context Person inv :  not (self.oclIsTypeOf(Adam)) "
            "implies (self.parents->size()>0)"
        ),
        language="OCL",
    )
    model = DomainModel(
        name="Genealogy",
        types={Person, Adam},
        associations={parents},
        generalizations={genealogy},
        constraints={inv},
    )

    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=model, output_dir=str(output_dir))
    generator.generate()

    with open(_generated_als_path(str(output_dir)), "r", encoding="utf-8") as f:
        spec = f.read()

    assert "all self:this/Person" in spec
    assert "self in Adam" in spec
    assert "=>" in spec
    assert "#(self.Person_parents) > 0" in spec


def test_bidirectional_navigation_consistency_fact(team_player_model, tmpdir):
    """Both ends are navigable by default, so the generator must assert that
    the two navigation relations are inverses of one another.

    """
    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=team_player_model, output_dir=str(output_dir))
    generator.generate()

    with open(_generated_als_path(str(output_dir)), "r", encoding="utf-8") as f:
        spec = f.read()

    assert "Team_players= ~Player_team" in spec or "Player_team= ~Team_players" in spec


def test_non_navigable_end_is_omitted_entirely(tmpdir):
    """If one end is not navigable, alloy_model.j2 must omit that
    relation line from the owning sig altogether (not just skip its
    cardinality/inverse facts)."""
    Team = Class(name="Team")
    Player = Class(name="Player")
    Team.attributes = {Property(name="name", type=StringType)}
    Player.attributes = {Property(name="name", type=StringType)}

    owns = BinaryAssociation(
        name="Owns",
        ends={
            Property(name="players", type=Player, multiplicity=Multiplicity(3, 4),
                      is_navigable=True),
            Property(name="team", type=Team, multiplicity=Multiplicity(1, 1),
                      is_navigable=False),
        },
    )
    model = DomainModel(name="OwnsModel", types={Team, Player}, associations={owns})

    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=model, output_dir=str(output_dir))
    generator.generate()

    with open(_generated_als_path(str(output_dir)), "r", encoding="utf-8") as f:
        spec = f.read()

    # The non-navigable "team" end never appears as a Player field
    assert "Player_team" not in spec
    # No inverse-relation fact either, since one side isn't navigable
    assert "~" not in spec
    # The navigable side is still present
    assert "Team_players: set Player" in spec


def test_unidirectional_three_to_star_side_is_enforced_via_inverse(tmpdir):
    """Regression: unidirectional A→B with 3..* on the A side and 0..* on the
    B side.  The B→A direction is not navigable, so the 3..* fact must be
    expressed through the inverse of the navigable ``A_bs`` field.  Otherwise
    Alloy freely produces B atoms with fewer than 3 A's."""
    A = Class(name="A")
    B = Class(name="B")
    A.attributes = {Property(name="name", type=StringType)}
    B.attributes = {Property(name="name", type=StringType)}

    rel = BinaryAssociation(
        name="Rel",
        ends={
            # 3..* del lado de A (source, not navigable in a unidirectional A→B)
            Property(name="as", type=A, multiplicity=Multiplicity(3, "*"),
                      is_navigable=False),
            # 0..* del lado de B (target, navigable)
            Property(name="bs", type=B, multiplicity=Multiplicity(0, "*"),
                      is_navigable=True),
        },
    )
    model = DomainModel(name="ABModel", types={A, B}, associations={rel})

    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=model, output_dir=str(output_dir))
    generator.generate()

    with open(_generated_als_path(str(output_dir)), "r", encoding="utf-8") as f:
        spec = f.read()

    # No navigable B→A field exists…
    assert "B_as" not in spec
    # …but every B is still forced to link at least 3 A's via the inverse.
    # (The quantified variable name depends on the association-end unpack order.)
    assert re.search(r"#\(A_bs\.[ab]\)>=3", spec), spec
    # And the 0..* B side must not emit spurious min/max facts.
    assert "#(a.A_bs)>=" not in spec
    assert "#(a.A_bs)<=" not in spec
    assert re.search(r"#\(A_bs\.[ab]\)<=", spec) is None, spec


# ---------------------------------------------------------------------------
# allInstances() OCL constraints
# ---------------------------------------------------------------------------

def _allinstances_model(expression: str, context_name: str = "Employee") -> DomainModel:
    """Build a Company model (Employee / Department) whose single OCL
    constraint uses ``<Class>.allInstances()``.

    Args:
        expression:    OCL body placed after ``context <Context> inv``.
        context_name:  Context class of the invariant ("Employee" or "Department").
    """
    Employee = Class(name="Employee")
    Employee.attributes = {
        Property(name="name", type=StringType),
        Property(name="age", type=IntegerType),
    }
    Department = Class(name="Department")
    Department.attributes = {Property(name="name", type=StringType)}
    context = Employee if context_name == "Employee" else Department
    inv = Constraint(
        name="AllInstancesInv",
        context=context,
        expression=f"context {context_name} inv AllInstancesInv: {expression}",
        language="OCL",
    )
    return DomainModel(
        name="CompanyModel",
        types={Employee, Department},
        constraints={inv},
    )


def _generate_allinstances_spec(expression, tmpdir, context_name="Employee"):
    """Run AlloyGenerator on an allInstances model and return the .als text."""
    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(
        model=_allinstances_model(expression, context_name),
        output_dir=str(output_dir),
    )
    generator.generate()
    with open(_generated_als_path(str(output_dir)), "r", encoding="utf-8") as f:
        return f.read()


def test_allinstances_forall_over_context_type(tmpdir):
    """Class.allInstances()->forAll(v | ...) on the context class must become
    an Alloy quantification over the class signature, with attribute names
    prefixed by their owning class."""
    spec = _generate_allinstances_spec(
        "Employee.allInstances()->forAll(e | e.age > 16)",
        tmpdir,
    )
    assert "all self:this/Employee" in spec
    assert "all e : Employee" in spec
    assert "e.Employee_age > 16" in spec


def test_allinstances_exists_over_context_type(tmpdir):
    """Class.allInstances()->exists(v | ...) on the context class must become
    an Alloy ``some`` quantification over the class signature."""
    spec = _generate_allinstances_spec(
        "Employee.allInstances()->exists(e | e.age > 16)",
        tmpdir,
    )
    assert "all self:this/Employee" in spec
    assert "some e : Employee" in spec
    assert "e.Employee_age > 16" in spec


def test_allinstances_size_over_context_type(tmpdir):
    """Class.allInstances()->size() must translate to the Alloy cardinality of
    the class signature (``#(Class)``)."""
    spec = _generate_allinstances_spec(
        "Employee.allInstances()->size() = 3",
        tmpdir,
    )
    assert "all self:this/Employee" in spec
    assert "#(Employee) = 3" in spec


def test_allinstances_over_other_class_than_context(tmpdir):
    """allInstances() on a class different from the constraint's context must
    quantify over the target class and still prefix its attribute names."""
    spec = _generate_allinstances_spec(
        "Employee.allInstances()->forAll(e | e.age > 16)",
        tmpdir,
        context_name="Department",
    )
    assert "all self:this/Department" in spec
    assert "all e : Employee" in spec
    assert "e.Employee_age > 16" in spec


# ---------------------------------------------------------------------------
# Enumeration reference validation (OCL constraints against enum literals)
# ---------------------------------------------------------------------------

def _enum_model(literal_name: str, ocl_literal: str, ocl_enum: str = "TCategory"):
    """Build a Person model with an enum-typed ``category`` attribute and one
    OCL invariant comparing it against ``<ocl_enum>::<ocl_literal>``.
    """
    TCategory = Enumeration(
        name="TCategory",
        literals={EnumerationLiteral(name=literal_name)},
    )
    Person = Class(name="Person")
    Person.attributes = {
        Property(name="name", type=StringType),
        Property(name="category", type=TCategory),
    }
    inv = Constraint(
        name="CatConstraint",
        context=Person,
        expression=f"context Person inv CatConstraint: self.category = {ocl_enum}::{ocl_literal}",
        language="OCL",
    )
    return DomainModel(
        name="EnumModel",
        types={Person, TCategory},
        constraints={inv},
    )


def test_valid_enum_literal_reference_generates_ok(tmpdir):
    """An OCL constraint referencing an existing literal must generate
    successfully and reference the matching Alloy enum signature."""
    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(
        model=_enum_model("JUNIOR", "JUNIOR"),
        output_dir=str(output_dir),
    )
    generator.generate()

    with open(_generated_als_path(str(output_dir)), "r", encoding="utf-8") as f:
        spec = f.read()
    assert "one sig ENUM_TCategory_JUNIOR extends TCategory{}" in spec
    assert "ENUM_TCategory_JUNIOR" in spec


def test_missing_enum_literal_reference_raises_clear_error(tmpdir):
    """An OCL constraint referencing a literal that no longer exists in the
    enumeration must raise a clear EnumReferenceError instead of silently
    generating Alloy that references an undefined signature."""
    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(
        model=_enum_model("JUNIOR", "JUNIO"),
        output_dir=str(output_dir),
    )

    with pytest.raises(EnumReferenceError) as excinfo:
        generator.generate()

    message = str(excinfo.value)
    assert "TCategory" in message
    assert "JUNIO" in message
    assert "JUNIOR" in message


def test_unknown_enum_type_reference_raises_clear_error(tmpdir):
    """An OCL constraint referencing an enumeration type that does not exist
    in the model must raise a clear EnumReferenceError."""
    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(
        model=_enum_model("JUNIOR", "JUNIOR", ocl_enum="NoSuchEnum"),
        output_dir=str(output_dir),
    )

    with pytest.raises(EnumReferenceError) as excinfo:
        generator.generate()

    assert "NoSuchEnum" in str(excinfo.value)


def test_ocl_to_alloy_validates_enum_references():
    """ocl_to_alloy rejects enum references that do not match the model."""
    data = {"Person": ["name:str", "category:TCategory"]}
    inherits_from = {"Person": "_"}
    enums = {"TCategory": {"JUNIOR"}}
    estado = TranslatorState()

    with pytest.raises(EnumReferenceError) as excinfo:
        ocl_to_alloy(
            inherits_from, data,
            "self.category = TCategory::JUNIO",
            "Person", estado, enums,
        )
    assert "JUNIO" in str(excinfo.value)
    assert "JUNIOR" in str(excinfo.value)


def test_ocl_to_alloy_skips_validation_when_enums_is_none():
    """Without an ``enums`` map the translation stays backward compatible
    and does not raise."""
    data = {"Person": ["name:str", "category:TCategory"]}
    inherits_from = {"Person": "_"}
    estado = TranslatorState()

    result = ocl_to_alloy(
        inherits_from, data,
        "self.category = TCategory::JUNIO",
        "Person", estado, None,
    )
    assert "ENUM_TCategory_JUNIO" in result


def test_ocl_to_alloy_translates_set_union():
    """``->union(...)`` must translate to Alloy set union (``+``), not the
    bogus ``.union`` method fallback."""
    data = {"Person": ["spouse:Person", "parents:Person"]}
    inherits_from = {"Person": "_"}
    estado = TranslatorState()

    result = ocl_to_alloy(
        inherits_from, data,
        "self.spouse.parents->union(self.parents)",
        "Person", estado, None,
    )
    assert "self.Person_spouse.Person_parents + self.Person_parents" in result
    assert ".union" not in result


def test_ocl_to_alloy_translates_set_intersection():
    """``->intersection(...)`` must translate to Alloy set intersection
    (``&``), not the bogus ``.intersection`` method fallback."""
    data = {"Person": ["spouse:Person", "parents:Person"]}
    inherits_from = {"Person": "_"}
    estado = TranslatorState()

    result = ocl_to_alloy(
        inherits_from, data,
        "self.spouse.parents->intersection(self.parents)",
        "Person", estado, None,
    )
    assert "self.Person_spouse.Person_parents & self.Person_parents" in result
    assert ".intersection" not in result


# ---------------------------------------------------------------------------
# Date literals and date attributes (OCL constraints)
# ---------------------------------------------------------------------------

def _date_person_model(expressions, with_birth_attr=True) -> DomainModel:
    """Build a Person model whose single OCL constraint uses date literals.

    Args:
        expressions:       OCL bodies placed after ``context Person inv <name>:``.
        with_birth_attr:   Whether ``Person`` gets a ``birthDate: date`` attribute.
    """
    attrs = {Property(name="name", type=StringType)}
    if with_birth_attr:
        attrs.add(Property(name="birthDate", type=DateType))
    Person = Class(name="Person")
    Person.attributes = attrs
    constraints = {
        Constraint(
            name=f"DateConstraint{i}",
            context=Person,
            expression=f"context Person inv DateConstraint{i}: {expr}",
            language="OCL",
        )
        for i, expr in enumerate(expressions)
    }
    return DomainModel(
        name="DatePersonModel",
        types={Person},
        constraints=constraints,
    )


def _generate_date_spec(model, tmpdir) -> str:
    """Run AlloyGenerator on a date model and return the .als text."""
    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=model, output_dir=str(output_dir))
    generator.generate()
    with open(_generated_als_path(str(output_dir)), "r", encoding="utf-8") as f:
        return f.read()


def test_date_attribute_renders_with_ordering_sig(tmpdir):
    """A date attribute plus a date literal must yield ``sig date {}``,
    ``open util/ordering[date]`` and a ``date``-typed attribute."""
    spec = _generate_date_spec(
        _date_person_model(["self.birthDate >= '01-01-2000'"]),
        tmpdir,
    )
    assert "open util/ordering[date]" in spec
    assert "sig date {}" in spec
    assert "Person_birthDate: date" in spec
    assert "one sig d01012000 extends date{}" in spec
    assert "I16" not in spec


def test_date_ocl_equality_translates_to_one_sig(tmpdir):
    """Date equality must render as Alloy ``=`` between the attribute and the
    emitted ``one sig``, without any I16 machinery."""
    spec = _generate_date_spec(
        _date_person_model(["self.birthDate = '01-01-2000'"]),
        tmpdir,
    )
    assert "one sig d01012000 extends date{}" in spec
    assert "(self.Person_birthDate = d01012000)" in spec


@pytest.mark.parametrize(
    "operator,expected",
    [
        (">", "(gt[self.Person_birthDate,d01012000])"),
        (">=", "(gte[self.Person_birthDate,d01012000])"),
        ("<", "(lt[self.Person_birthDate,d01012000])"),
        ("<=", "(lte[self.Person_birthDate,d01012000])"),
        ("<>", "(self.Person_birthDate != d01012000)"),
    ],
)
def test_date_comparison_operators(operator, expected, tmpdir):
    """Ordered comparisons between a date attribute and a date literal must use
    util/ordering predicates (gt/gte/lt/lte), and inequality (``<>``) must
    render as ``!=``."""
    spec = _generate_date_spec(
        _date_person_model([f"self.birthDate {operator} '01-01-2000'"]),
        tmpdir,
    )
    assert "one sig d01012000 extends date{}" in spec
    assert expected in spec


def test_date_order_fact_is_emitted(tmpdir):
    """With two date literals, ``fact Order`` pins the util/ordering chain:
    ``d01012000 = first`` and ``d03152021 = last`` (sorted ascending)."""
    output_dir = tmpdir.mkdir("output")
    model = _date_person_model([
        "self.birthDate > '15-03-2021'",
        "self.birthDate <= '01-01-2000'",
    ])
    generator = AlloyGenerator(model=model, output_dir=str(output_dir), scope=2)
    generator.generate()
    with open(_generated_als_path(str(output_dir)), "r", encoding="utf-8") as f:
        spec = f.read()
    assert "one sig d01012000 extends date{}" in spec
    assert "one sig d03152021 extends date{}" in spec
    assert "fact Order {" in spec
    assert "d01012000 = first" in spec
    assert "d03152021 = last" in spec


def test_date_value_deduped_across_constraints(tmpdir):
    """The same date literal in two constraints must declare a single sig."""
    spec = _generate_date_spec(
        _date_person_model([
            "self.birthDate = '01-01-2000'",
            "self.birthDate <> '01-01-2000'",
        ]),
        tmpdir,
    )
    assert spec.count("one sig d01012000 extends date{}") == 1


def test_date_attribute_without_date_literals_opens_ordering(tmpdir):
    """A date attribute with no OCL date literal must declare ``sig date {}`` and
    open ``util/ordering[date]``: the generator opens the ordering module whenever
    the model has any date-typed attribute or date literal."""
    spec = _generate_date_spec(_date_person_model([]), tmpdir)
    assert "sig date {}" in spec
    assert "open util/ordering[date]" in spec
    assert "Person_birthDate: date" in spec


def test_date_literal_without_date_attribute_emits_ordering_sig(tmpdir):
    """An OCL constraint comparing two date literals (no date attribute at all)
    must still declare the ordering sig."""
    spec = _generate_date_spec(
        _date_person_model(
            ["'01-01-2000' < '15-03-2021'"],
            with_birth_attr=False,
        ),
        tmpdir,
    )
    assert "open util/ordering[date]" in spec
    assert "sig date {}" in spec
    assert "one sig d01012000 extends date{}" in spec
    assert "one sig d03152021 extends date{}" in spec
    assert "(lt[d01012000,d03152021])" in spec


def test_datetime_time_timedelta_attributes_map_to_date(tmpdir):
    """DateTimeType/TimeType/TimeDeltaType attributes must render as Alloy
    ``date`` fields instead of dead ``sig datetime {}``/``sig time {}``/``sig
    timedelta {}`` signatures that can never be populated."""
    Event = Class(name="Event")
    Event.attributes = {
        Property(name="name", type=StringType),
        Property(name="happensAt", type=DateTimeType),
        Property(name="startsAt", type=TimeType),
        Property(name="duration", type=TimeDeltaType),
    }
    model = DomainModel(name="EventModel", types={Event})

    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=model, output_dir=str(output_dir))
    generator.generate()

    with open(_generated_als_path(str(output_dir)), "r", encoding="utf-8") as f:
        spec = f.read()

    assert "Event_happensAt: date" in spec
    assert "Event_startsAt: date" in spec
    assert "Event_duration: date" in spec
    assert "sig datetime {}" not in spec
    assert "sig time {}" not in spec
    assert "sig timedelta {}" not in spec


def test_datetime_attribute_vs_date_literal_uses_ordering(tmpdir):
    """A DateTimeType attribute compared with a date literal must produce a
    valid util/ordering predicate against the literal's one sig (previously it
    produced ``gte[datetime,date]`` — an Alloy type error)."""
    Event = Class(name="Event")
    Event.attributes = {
        Property(name="name", type=StringType),
        Property(name="happensAt", type=DateTimeType),
    }
    inv = Constraint(
        name="FutureEvent",
        context=Event,
        expression="context Event inv FutureEvent: self.happensAt >= '2024-01-01'",
        language="OCL",
    )
    model = DomainModel(name="EventModel", types={Event}, constraints={inv})

    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=model, output_dir=str(output_dir))
    generator.generate()

    with open(_generated_als_path(str(output_dir)), "r", encoding="utf-8") as f:
        spec = f.read()

    assert "Event_happensAt: date" in spec
    assert "one sig d01012024 extends date{}" in spec
    assert "(gte[self.Event_happensAt,d01012024])" in spec
    assert "open util/ordering[date]" in spec


def test_date_attribute_vs_date_attribute_uses_ordering(tmpdir):
    """Comparing two date-typed attributes must use the util/ordering predicate
    (``gt``) and open ``util/ordering[date]`` even when no date literal appears
    anywhere (previously it fell back to Alloy set superset comparison)."""
    Patient = Class(name="Patient")
    Record = Class(name="Record")
    Patient.attributes = {Property(name="birthDate", type=DateType)}
    Record.attributes = {Property(name="createdDate", type=DateType)}
    has_record = BinaryAssociation(
        name="HasRecord",
        ends={
            Property(name="records", type=Record, multiplicity=Multiplicity(0, "*")),
            Property(name="patient", type=Patient, multiplicity=Multiplicity(1, 1)),
        },
    )
    inv = Constraint(
        name="AfterBirth",
        context=Patient,
        expression="context Patient inv AfterBirth: self.records.createdDate > self.birthDate",
        language="OCL",
    )
    model = DomainModel(
        name="PatientModel",
        types={Patient, Record},
        associations={has_record},
        constraints={inv},
    )

    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=model, output_dir=str(output_dir))
    generator.generate()

    with open(_generated_als_path(str(output_dir)), "r", encoding="utf-8") as f:
        spec = f.read()

    assert "open util/ordering[date]" in spec
    assert "(gt[self.Patient_records.Record_createdDate,self.Patient_birthDate])" in spec
    assert "one sig d" in spec
    assert "fact Order {" in spec


def test_string_literal_with_date_substring_is_not_treated_as_date(tmpdir):
    """A string literal that merely contains a date-like substring must not be
    routed through the date machinery.  Regression: it used to ``sys.exit(1)``
    inside ``parse_date`` and kill the whole generation."""
    Ticket = Class(name="Ticket")
    Ticket.attributes = {Property(name="code", type=StringType)}
    inv = Constraint(
        name="HasCode",
        context=Ticket,
        expression="context Ticket inv HasCode: self.code = 'fecha 2024-01-01 x'",
        language="OCL",
    )
    model = DomainModel(name="TicketModel", types={Ticket}, constraints={inv})

    output_dir = tmpdir.mkdir("output")
    generator = AlloyGenerator(model=model, output_dir=str(output_dir))
    generator.generate()

    with open(_generated_als_path(str(output_dir)), "r", encoding="utf-8") as f:
        spec = f.read()

    assert "one sig d01012024 extends date{}" not in spec
    assert "open util/ordering[date]" not in spec


def test_isdate_only_accepts_whole_date_literals():
    """is_date must only match literals whose entire content is a date."""
    assert is_date("'2024-01-01'") == "d01012024"
    assert is_date('"2024-01-01"') == "d01012024"
    assert is_date("'01-01-2000'") == "d01012000"
    assert is_date("'13-10-1977'") == "d10131977"
    assert is_date("'2024-13-45'") is None
    assert is_date("'2024'") is None
    assert is_date("'fecha 2024-01-01 x'") is None
    assert is_date("self.Patient_birthDate") is None


def test_encode_and_parse_ocl_date_roundtrip():
    """encode_date/parse_ocl_date must round-trip in MMDDYYYY format."""
    expected = date(1977, 10, 13)
    assert encode_date(expected) == "d10131977"
    assert parse_ocl_date("d10131977") == expected
    assert encode_date(date(2000, 1, 1)) == "d01012000"
    assert parse_ocl_date(encode_date(date(2021, 3, 15))) == date(2021, 3, 15)


def test_random_date_within_bounds():
    """random_date must return a date inside the inclusive [start, end] range."""
    start = date(2020, 1, 1)
    end = date(2020, 1, 31)
    for _ in range(50):
        d = random_date(start, end)
        assert start <= d <= end


def test_generate_dates_and_order():
    """generate_dates_and_order fills up to scope, emits one sigs only for new
    dates and a fact Order with the dates sorted ascending."""
    existing = ["d01012000"]  # 2000-01-01
    result = generate_dates_and_order(
        ocl_dates=existing,
        scope=3,
        start=date(2001, 1, 1),
        end=date(2001, 1, 5),
    )

    sigs = re.findall(r"one sig (d\d{8}) extends date \{\}", result)
    assert len(sigs) == 2  # only NEW dates emit a one sig
    assert "d01012000" not in sigs
    for sig in sigs:
        assert date(2001, 1, 1) <= parse_ocl_date(sig) <= date(2001, 1, 5)

    assert "fact Order {" in result
    ordered = sorted(existing + sigs, key=parse_ocl_date)
    assert ordered[0] == "d01012000"
    assert f"{ordered[0]} = first" in result
    for i in range(len(ordered) - 1):
        assert f"{ordered[i]}.next = {ordered[i + 1]}" in result
    assert f"{ordered[-1]} = last" in result


# ---------------------------------------------------------------------------
# Alloy XML instance -> object diagram (AlloyToBesserConverter)
# ---------------------------------------------------------------------------

_DATE_INSTANCE_XML = """\
<alloy builddate="2025-01-01T00:00:00.000Z">
<instance bitwidth="4" maxseq="4" command="Run instance_model for 5 Person, 5 str, 5 date">
<sig label="this/str" ID="4" parentID="2">
   <atom label="str$0"/>
</sig>
<sig label="this/d01012000" ID="5" parentID="6" one="yes">
   <atom label="d01012000$0"/>
</sig>
<sig label="this/date" ID="6" parentID="2">
   <atom label="date$2"/>
   <atom label="date$1"/>
   <atom label="date$0"/>
</sig>
<sig label="this/Person" ID="7" parentID="2">
   <atom label="Person$0"/>
</sig>
<field label="Person_name" ID="8" parentID="7">
   <tuple> <atom label="Person$0"/> <atom label="str$0"/> </tuple>
</field>
<field label="Person_birthDate" ID="9" parentID="7">
   <tuple> <atom label="Person$0"/> <atom label="d01012000$0"/> </tuple>
</field>
<sig label="ordering/Ord" ID="13" parentID="2" one="yes" private="yes">
   <atom label="ordering/Ord$0"/>
</sig>
<field label="First" ID="14" parentID="13" private="yes">
   <tuple> <atom label="ordering/Ord$0"/> <atom label="d01012000$0"/> </tuple>
</field>
<sig label="univ" ID="2" builtin="yes">
</sig>
</instance>
</alloy>
"""


def _date_instance_converter(tmp_path) -> AlloyToBesserConverter:
    xml_file = tmp_path / "instance.xml"
    xml_file.write_text(_DATE_INSTANCE_XML, encoding="utf-8")
    converter = AlloyToBesserConverter(str(xml_file))
    converter.parse_xml()
    return converter


def test_date_atom_is_rendered_as_attribute(tmp_path):
    """A date literal atom (dMMDDYYYY) must appear as a primitive attribute
    value, not as an object reference."""
    code = _date_instance_converter(tmp_path).generate_object_diagram_code()
    assert "'birthDate': \"01-01-2000\"" in code


def test_date_and_ordering_sigs_are_not_rendered_as_objects(tmp_path):
    """Spurious date / one-sig / ordering atoms must not become objects or
    relations in the generated object diagram."""
    code = _date_instance_converter(tmp_path).generate_object_diagram_code()
    assert "d01012000_0_obj" not in code
    assert "date_0_obj" not in code
    assert "Ord" not in code
    assert "setattr(person_0_obj, 'birthDate'" not in code
    assert "person_0_obj = Person(" in code


def test_free_date_atom_rendered_with_atom_label(tmp_path):
    """A free atom of the ``date`` sig (e.g. ``date$0``) — produced by Alloy
    when the date attribute has no OCL literal — must be shown with the atom
    label Alloy found, not as an object reference."""
    xml = _DATE_INSTANCE_XML.replace(
        '<tuple> <atom label="Person$0"/> <atom label="d01012000$0"/> </tuple>',
        '<tuple> <atom label="Person$0"/> <atom label="date$0"/> </tuple>',
    )
    xml_file = tmp_path / "instance.xml"
    xml_file.write_text(xml, encoding="utf-8")
    converter = AlloyToBesserConverter(str(xml_file))
    converter.parse_xml()
    code = converter.generate_object_diagram_code()
    assert "'birthDate': \"date$0\"" in code
    assert "setattr(person_0_obj, 'birthDate'" not in code


def test_two_associations_between_same_classes_are_both_kept(tmp_path):
    """Regression: two distinct associations between the same two classes must
    both survive the XML -> BUML object code conversion.  Previously the
    symmetric-pair dedup keyed on the unordered object pair dropped the second
    association entirely."""
    xml = """\
<alloy builddate="2026-01-01T00:00:00.000Z">
<instance bitwidth="4" maxseq="4" command="Run instance_model for 5 A, 5 B">
<sig label="this/A" ID="1" parentID="9">
   <atom label="A$0"/>
</sig>
<field label="A_bs1" ID="2" parentID="1">
   <tuple> <atom label="A$0"/> <atom label="B$0"/> </tuple>
</field>
<field label="A_bs2" ID="3" parentID="1">
   <tuple> <atom label="A$0"/> <atom label="B$0"/> </tuple>
</field>
<sig label="this/B" ID="4" parentID="9">
   <atom label="B$0"/>
</sig>
<field label="B_as1" ID="5" parentID="4">
   <tuple> <atom label="B$0"/> <atom label="A$0"/> </tuple>
</field>
<field label="B_as2" ID="6" parentID="4">
   <tuple> <atom label="B$0"/> <atom label="A$0"/> </tuple>
</field>
<sig label="univ" ID="9" builtin="yes">
</sig>
</instance>
</alloy>
"""
    xml_file = tmp_path / "instance.xml"
    xml_file.write_text(xml, encoding="utf-8")
    converter = AlloyToBesserConverter(str(xml_file))
    converter.parse_xml()
    code = converter.generate_object_diagram_code()

    assert "setattr(a_0_obj, 'bs1', b_0_obj)" in code
    assert "setattr(a_0_obj, 'bs2', b_0_obj)" in code


def test_full_model_two_associations_between_same_classes_are_both_kept(tmp_path):
    """Regression for the EducationSystem case: two A--B associations (relAB1,
    relAB2) must both produce a link even when the Alloy instance also contains
    other classes, attribute fields, and inherited attributes.  The pair-finding
    logic must never pair the two halves of *different* associations, otherwise
    both A--B links collapse into a single one."""
    xml = """\
<alloy builddate="2026-01-01T00:00:00.000Z">
<instance bitwidth="4" maxseq="4" command="Run instance_model for 5 A, 5 B, 5 Subject, 5 Student, 5 Teacher">
<sig label="this/Teacher" ID="24" parentID="27">
   <atom label="Researcher$0"/>
</sig>
<field label="Teacher_subject" ID="25" parentID="24">
   <tuple> <atom label="Researcher$0"/> <atom label="Subject$0"/> </tuple>
</field>
<sig label="this/Student" ID="26" parentID="27">
   <atom label="Student$0"/>
</sig>
<field label="Student_subject" ID="28" parentID="26">
   <tuple> <atom label="Student$0"/> <atom label="Subject$0"/> </tuple>
</field>
<sig label="this/Person" ID="27" parentID="9">
   <atom label="Student$0"/>
   <atom label="Researcher$0"/>
</sig>
<field label="Person_name" ID="29" parentID="27">
   <tuple> <atom label="Student$0"/> <atom label="Alice$0"/> </tuple>
   <tuple> <atom label="Researcher$0"/> <atom label="Bob$0"/> </tuple>
</field>
<sig label="this/Subject" ID="19" parentID="9">
   <atom label="Subject$0"/>
</sig>
<field label="Subject_name" ID="20" parentID="19">
   <tuple> <atom label="Subject$0"/> <atom label="Math$0"/> </tuple>
</field>
<field label="Subject_teacher" ID="21" parentID="19">
   <tuple> <atom label="Subject$0"/> <atom label="Researcher$0"/> </tuple>
</field>
<field label="Subject_student" ID="22" parentID="19">
   <tuple> <atom label="Subject$0"/> <atom label="Student$0"/> </tuple>
</field>
<sig label="this/A" ID="14" parentID="9">
   <atom label="A$0"/>
</sig>
<field label="A_idA" ID="15" parentID="14">
   <tuple> <atom label="A$0"/> <atom label="str$0"/> </tuple>
</field>
<field label="A_bs1" ID="16" parentID="14">
   <tuple> <atom label="A$0"/> <atom label="B$0"/> </tuple>
</field>
<field label="A_bs2" ID="17" parentID="14">
   <tuple> <atom label="A$0"/> <atom label="B$0"/> </tuple>
</field>
<sig label="this/B" ID="11" parentID="9">
   <atom label="B$0"/>
</sig>
<field label="B_idB" ID="12" parentID="11">
   <tuple> <atom label="B$0"/> <atom label="str$0"/> </tuple>
</field>
<field label="B_as1" ID="13" parentID="11">
   <tuple> <atom label="B$0"/> <atom label="A$0"/> </tuple>
</field>
<field label="B_as2" ID="18" parentID="11">
   <tuple> <atom label="B$0"/> <atom label="A$0"/> </tuple>
</field>
<sig label="univ" ID="9" builtin="yes">
</sig>
</instance>
</alloy>
"""
    xml_file = tmp_path / "instance.xml"
    xml_file.write_text(xml, encoding="utf-8")
    converter = AlloyToBesserConverter(str(xml_file))
    converter.parse_xml()
    code = converter.generate_object_diagram_code()

    # Exactly one link per association between A and B (either side may be kept).
    ab_links = [
        ln for ln in code.splitlines()
        if ln.startswith("setattr(") and "a_0_obj" in ln and "b_0_obj" in ln
    ]
    assert len(ab_links) == 2, code
    assert any("'bs1'" in ln or "'as1'" in ln for ln in ab_links), code
    assert any("'bs2'" in ln or "'as2'" in ln for ln in ab_links), code

    # Unrelated associations between the other classes survive untouched: the
    # Subject object is linked once to the Teacher and once to the Student.
    subject_links = [
        ln for ln in code.splitlines()
        if ln.startswith("setattr(") and "subject_0_obj" in ln
    ]
    assert len(subject_links) == 2, code
    assert any("teacher_0_obj" in ln for ln in subject_links), code
    assert any("student_0_obj" in ln for ln in subject_links), code


def test_concrete_subclass_inherits_ancestor_attributes_without_abstract_object(tmp_path):
     """A concrete subclass atom must become only one object of the subclass,
     carrying both its own attributes and the inherited ones from abstract
     ancestors."""
     xml = """\
<alloy builddate="2026-01-01T00:00:00.000Z">
<instance bitwidth="4" maxseq="4" command="Run instance_model for 5 Person, 5 Student">
<sig label="this/date" ID="1" parentID="9">
    <atom label="d01012000$0"/>
</sig>
<sig label="this/Person" ID="2" parentID="9" abstract="yes">
    <atom label="Student$0"/>
</sig>
<field label="Person_name" ID="3" parentID="2">
    <tuple> <atom label="Student$0"/> <atom label="Alice$0"/> </tuple>
</field>
<field label="Person_birth_date" ID="4" parentID="2">
    <tuple> <atom label="Student$0"/> <atom label="d01012000$0"/> </tuple>
</field>
<sig label="this/Student" ID="5" parentID="2">
    <atom label="Student$0"/>
</sig>
<field label="Student_student_id" ID="6" parentID="5">
    <tuple> <atom label="Student$0"/> <atom label="S123$0"/> </tuple>
</field>
<sig label="univ" ID="9" builtin="yes">
</sig>
</instance>
</alloy>
"""
     xml_file = tmp_path / "instance.xml"
     xml_file.write_text(xml, encoding="utf-8")
     converter = AlloyToBesserConverter(str(xml_file))
     converter.parse_xml()

     code = converter.generate_object_diagram_code()

     assert "person_0_obj = Person(" not in code
     # The single Student atom becomes one object whose instance name is
     # derived from the atom label (Student$0 -> Student_0).
     assert 'student_0_obj = Student("Student_0")' in code
     assert "student_0_obj = Student(" in code
     assert "'name': \"Alice\"" in code
     assert "'birth_date': \"01-01-2000\"" in code
     assert "'student_id': \"S123\"" in code


def test_two_self_associations_are_both_kept_once_each(tmp_path):
     """Two different bidirectional self-associations must each yield one link.
     The two halves of the same association should collapse to one assignment,
     but separate associations must remain separate."""
     xml = """\
<alloy builddate="2026-01-01T00:00:00.000Z">
<instance bitwidth="4" maxseq="4" command="Run instance_model for 5 Node">
<sig label="this/Node" ID="1" parentID="9">
    <atom label="Node$0"/>
    <atom label="Node$1"/>
</sig>
<field label="Node_next" ID="2" parentID="1">
    <tuple> <atom label="Node$0"/> <atom label="Node$1"/> </tuple>
</field>
<field label="Node_previous" ID="3" parentID="1">
    <tuple> <atom label="Node$1"/> <atom label="Node$0"/> </tuple>
</field>
<field label="Node_peer" ID="4" parentID="1">
    <tuple> <atom label="Node$0"/> <atom label="Node$1"/> </tuple>
</field>
<field label="Node_peer_of" ID="5" parentID="1">
    <tuple> <atom label="Node$1"/> <atom label="Node$0"/> </tuple>
</field>
<sig label="univ" ID="9" builtin="yes">
</sig>
</instance>
</alloy>
"""
     xml_file = tmp_path / "instance.xml"
     xml_file.write_text(xml, encoding="utf-8")
     converter = AlloyToBesserConverter(str(xml_file))
     converter.parse_xml()

     code = converter.generate_object_diagram_code()

     assert code.count("setattr(node_0_obj, 'next', node_1_obj)") == 1
     assert code.count("setattr(node_0_obj, 'peer', node_1_obj)") == 1
     assert "setattr(node_1_obj, 'previous', node_0_obj)" not in code
     assert "setattr(node_1_obj, 'peer_of', node_0_obj)" not in code


def test_multiple_targets_for_same_self_association_emit_single_setattr(tmp_path):
     """A single object with two outgoing self-links on the same association
     end must be emitted as one set assignment so neither link is overwritten."""
     xml = """\
<alloy builddate="2026-01-01T00:00:00.000Z">
<instance bitwidth="4" maxseq="4" command="Run instance_model for 5 Person">
<sig label="this/Person" ID="1" parentID="9">
    <atom label="Person$0"/>
    <atom label="Person$1"/>
    <atom label="Person$2"/>
</sig>
<field label="Person_friends" ID="2" parentID="1">
    <tuple> <atom label="Person$0"/> <atom label="Person$1"/> </tuple>
    <tuple> <atom label="Person$0"/> <atom label="Person$2"/> </tuple>
</field>
<sig label="univ" ID="9" builtin="yes">
</sig>
</instance>
</alloy>
"""
     xml_file = tmp_path / "instance.xml"
     xml_file.write_text(xml, encoding="utf-8")
     converter = AlloyToBesserConverter(str(xml_file))
     converter.parse_xml()

     code = converter.generate_object_diagram_code()

     assert "setattr(person_0_obj, 'friends', {person_1_obj, person_2_obj})" in code
     assert code.count("setattr(person_0_obj, 'friends',") == 1


# ---------------------------------------------------------------------------
# Unit tests for pure helper functions (no model / templates required)
# ---------------------------------------------------------------------------

class TestSanitizeAlloyName:

    def test_keeps_valid_identifier_unchanged(self):
        assert sanitize_alloy_name("Player") == "Player"

    def test_strips_invalid_characters(self):
        assert sanitize_alloy_name("Player-Name!") == "PlayerName"

    def test_prefixes_leading_digit(self):
        assert sanitize_alloy_name("1Player") == "_1Player"

    def test_handles_accented_and_special_chars(self):
        # Non-ASCII letters aren't valid Alloy identifier chars either
        result = sanitize_alloy_name("Jugador_Núm")
        assert re.fullmatch(r"[A-Za-z0-9_]+", result)

    def test_empty_or_fully_invalid_name_falls_back(self):
        assert sanitize_alloy_name("!!!") == "_unnamed"
        assert sanitize_alloy_name("") == "_unnamed"


class TestBuildConsistencyRule:

    def test_one_to_one_both_navigable_no_cardinality_facts(self):
        rule = build_consistency_rule(
            "Team", "players", [1, 1], "Player", "team", [1, 1],
            arrow_a_b=True, arrow_b_a=True,
        )
        assert "#(" not in rule

    def test_min_and_max_bounds_emitted_for_3_4(self):
        rule = build_consistency_rule(
            "Team", "players", [3, 4], "Player", "team", [1, 1],
            arrow_a_b=True, arrow_b_a=True,
        )
        assert "#(a.Team_players)>=3" in rule
        assert "#(a.Team_players)<=4" in rule

    def test_unbounded_max_is_not_emitted(self):
        # UNLIMITED_MAX_MULTIPLICITY is 9999; anything below it is a real bound.
        rule = build_consistency_rule(
            "Team", "players", [3, 9999], "Player", "team", [1, 1],
            arrow_a_b=True, arrow_b_a=True,
        )
        assert "#(a.Team_players)>=3" in rule
        assert "<=" not in rule

    def test_non_navigable_direction_uses_inverse_field_for_its_facts(self):
        """When A→B is not navigable but B→A is, the B-side (3..4) facts must
        still be enforced by navigating the inverse of the ``Player_team``
        field, instead of the missing ``Team_players`` navigation."""
        rule = build_consistency_rule(
            "Team", "players", [3, 4], "Player", "team", [1, 1],
            arrow_a_b=False, arrow_b_a=True,
        )
        assert "Team_players" not in rule
        assert "#(Player_team.a)>=3" in rule
        assert "#(Player_team.a)<=4" in rule

    def test_non_navigable_b_to_a_side_fact_uses_inverse_of_a_to_b_field(self):
        """Regression: unidirectional A→B (3..* on the A side, 0..* on the B
        side).  The A-side multiplicity must still force every B to link at
        least 3 A's, expressed through the reverse of the ``A_bs`` field."""
        rule = build_consistency_rule(
            "A", "bs", [0, 9999], "B", "as", [3, 9999],
            arrow_a_b=True, arrow_b_a=False,
        )
        assert "#(A_bs.b)>=3" in rule
        assert "b.B_as" not in rule
