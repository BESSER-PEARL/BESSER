"""
Tests for the OCL -> Pydantic validator emission of the Pydantic generator.

These tests cover the translation rules implemented in
``besser/generators/pydantic_classes/ocl_utils.py`` and the corresponding
template branches:

* ``matches('<regex>')`` becomes a ``re.match`` call (OCL ``matches`` is not a
  Python string method, so emitting it verbatim produced non-working code);
* error messages are emitted through ``repr()`` so a regex containing quotes or
  backslashes can never break the generated file's syntax;
* constraints over two or more properties of the same class become a
  ``@model_validator(mode='after')`` instead of being dropped;
* constraints over collections/relationships leave a NOTE comment behind
  instead of being dropped silently or emitting broken code;
* whatever happens, the generated ``pydantic_classes.py`` compiles.
"""

import datetime
import importlib.util
import os
import py_compile
import sys
import uuid

import pytest

from besser.BUML.metamodel.structural import (
    Class, Constraint, DomainModel, Property,
    BinaryAssociation, Multiplicity,
    DateType, IntegerType,
)
from besser.generators.pydantic_classes import PydanticGenerator
from besser.generators.pydantic_classes.ocl_utils import (
    build_constraints_map, get_constraints_for_class, parse_ocl_constraint,
)


# ============================================================================
# Helpers and fixtures
# ============================================================================

def make_constraint(context_class, expression, name="constraint"):
    """Build an OCL Constraint bound to ``context_class``."""
    return Constraint(
        name=name,
        context=context_class,
        expression=expression,
        language="OCL",
    )


def parse_single(domain_model, context_class, expression, name="constraint"):
    """Parse one OCL expression and return the resulting validator descriptor."""
    constraint = make_constraint(context_class, expression, name)
    domain_model.constraints = {constraint}
    return parse_ocl_constraint(constraint, domain_model)


def generate(domain_model, output_dir, backend=True):
    """Run the Pydantic generator and return (path, source)."""
    PydanticGenerator(
        model=domain_model, backend=backend, output_dir=str(output_dir)
    ).generate()
    path = os.path.join(str(output_dir), "pydantic_classes.py")
    with open(path, "r", encoding="utf-8") as handle:
        return path, handle.read()


def load_generated(path):
    """Import the generated module so its validators can be exercised."""
    module_name = f"pydantic_classes_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


@pytest.fixture
def booking_class():
    """A Booking class with two date attributes, for multi-property constraints."""
    return Class(name="Booking", attributes={
        Property(name="check_in", type=DateType),
        Property(name="check_out", type=DateType),
        Property(name="nights", type=IntegerType),
    })


@pytest.fixture
def room_class():
    """A Room class, used as the far end of a Booking association."""
    return Class(name="Room", attributes={
        Property(name="number", type=IntegerType),
        Property(name="max_people", type=IntegerType),
    })


@pytest.fixture
def booking_model(booking_class, room_class):
    """A Booking/Room model with a 1:N association (booking -> rooms)."""
    association = BinaryAssociation(
        name="booking_room",
        ends={
            Property(name="booking", type=booking_class, multiplicity=Multiplicity(1, 1)),
            Property(name="rooms", type=room_class, multiplicity=Multiplicity(0, 9999)),
        }
    )
    return DomainModel(
        name="BookingModel",
        types={booking_class, room_class},
        associations={association},
    )


# ============================================================================
# matches() translation
# ============================================================================

class TestMatchesTranslation:
    """``self.<prop>.matches('<regex>')`` must become a real Python expression."""

    def test_matches_becomes_re_match(self, player_class, player_team_domain_model):
        result = parse_single(
            player_team_domain_model, player_class,
            "context Player inv ValidName: self.name.matches('^[A-Z][a-z]+$')",
            name="ValidName",
        )

        assert result is not None
        assert not result.get('skipped')
        assert result['property'] == 'name'
        assert result['python_expression'] == "re.match(r'^[A-Z][a-z]+$', v) is not None"
        assert result['uses_re'] is True
        # The OCL operation must not leak into the generated Python.
        assert '.matches(' not in result['python_expression']

    def test_ocl_backslash_escapes_are_resolved(self, player_class, player_team_domain_model):
        """``\\\\.`` in an OCL string literal is a single backslash in the regex."""
        result = parse_single(
            player_team_domain_model, player_class,
            r"context Player inv ValidEmail: self.name.matches('^[a-z]+@[a-z]+\\.[a-z]{2,}$')",
            name="ValidEmail",
        )

        assert result['python_expression'] == (
            r"re.match(r'^[a-z]+@[a-z]+\.[a-z]{2,}$', v) is not None"
        )

    def test_matches_message_is_readable(self, player_class, player_team_domain_model):
        result = parse_single(
            player_team_domain_model, player_class,
            "context Player inv ValidName: self.name.matches('^[A-Z][a-z]+$')",
            name="ValidName",
        )

        assert result['message'] == "name must match '^[A-Z][a-z]+$'"
        # message_repr must evaluate back to the message, unchanged.
        assert eval(result['message_repr']) == result['message']  # noqa: S307

    def test_generated_regex_validator_works(self, player_class, player_team_domain_model, tmpdir):
        player_team_domain_model.constraints = {
            make_constraint(
                player_class,
                "context Player inv ValidName: self.name.matches('^[A-Z][a-z]+$')",
                name="ValidName",
            )
        }
        path, source = generate(player_team_domain_model, tmpdir.mkdir("output"))

        assert source.startswith("import re\n")
        py_compile.compile(path, doraise=True)

        module = load_generated(path)
        player_create = module.PlayerCreate

        valid = player_create(age=30, name="Alice", salary=1.0, active=True, jerseyNumber=7)
        assert valid.name == "Alice"

        with pytest.raises(Exception):
            player_create(age=30, name="alice1", salary=1.0, active=True, jerseyNumber=7)

    def test_no_re_import_when_unused(self, player_class, player_team_domain_model, tmpdir):
        player_team_domain_model.constraints = {
            make_constraint(
                player_class, "context Player inv MinAge: self.age > 10", name="MinAge"
            )
        }
        _, source = generate(player_team_domain_model, tmpdir.mkdir("output"))

        assert "import re" not in source
        assert "model_validator" not in source


# ============================================================================
# Message quoting
# ============================================================================

class TestMessageQuoting:
    """Messages are emitted as Python literals, never interpolated into quotes."""

    def test_regex_with_single_quote_does_not_break_the_file(
        self, player_class, player_team_domain_model, tmpdir
    ):
        # An OCL string literal escapes an inner quote as \' ; the regex allows
        # apostrophes in names ("O'Brien").
        expression = (
            r"context Player inv NameWithApostrophe: self.name.matches('^[A-Za-z\']+$')"
        )
        result = parse_single(
            player_team_domain_model, player_class, expression, name="NameWithApostrophe"
        )

        assert result['python_expression'] == "re.match(r\"^[A-Za-z']+$\", v) is not None"
        assert result['message'] == "name must match '^[A-Za-z']+$'"
        assert eval(result['message_repr']) == result['message']  # noqa: S307

        player_team_domain_model.constraints = {
            make_constraint(player_class, expression, name="NameWithApostrophe")
        }
        path, source = generate(player_team_domain_model, tmpdir.mkdir("output"))

        py_compile.compile(path, doraise=True)
        assert "raise ValueError(\"name must match '^[A-Za-z']+$'\")" in source

        module = load_generated(path)
        player_create = module.PlayerCreate
        assert player_create(
            age=30, name="O'Brien", salary=1.0, active=True, jerseyNumber=7
        ).name == "O'Brien"
        with pytest.raises(Exception):
            player_create(age=30, name="O Brien 3", salary=1.0, active=True, jerseyNumber=7)

    def test_simple_operator_message_is_a_literal(self, player_class, player_team_domain_model, tmpdir):
        player_team_domain_model.constraints = {
            make_constraint(
                player_class, "context Player inv MinAge: self.age > 10", name="MinAge"
            )
        }
        path, source = generate(player_team_domain_model, tmpdir.mkdir("output"))

        py_compile.compile(path, doraise=True)
        assert "raise ValueError('age must be > 10')" in source

    def test_string_value_with_quote_is_escaped(self, player_class, player_team_domain_model, tmpdir):
        player_team_domain_model.constraints = {
            make_constraint(
                player_class,
                r"context Player inv NotUnknown: self.name <> 'O\'Brien'",
                name="NotUnknown",
            )
        }
        path, _ = generate(player_team_domain_model, tmpdir.mkdir("output"))
        py_compile.compile(path, doraise=True)


# ============================================================================
# Model-level (multi-property) validators
# ============================================================================

class TestModelValidator:
    """Constraints over 2+ properties become model validators, not dropped."""

    def test_two_properties_produce_a_model_expression(self, booking_class, booking_model):
        result = parse_single(
            booking_model, booking_class,
            "context Booking inv CheckInBeforeCheckOut: self.check_in <= self.check_out",
            name="CheckInBeforeCheckOut",
        )

        assert result is not None
        assert not result.get('skipped')
        assert result['model_expression'] == "self.check_in <= self.check_out"
        assert result['validator_name'] == "CheckInBeforeCheckOut"
        assert result['properties'] == ['check_in', 'check_out']
        assert eval(result['message_repr']) == result['message']  # noqa: S307

    def test_ocl_equality_operators_are_translated(self, booking_class, booking_model):
        result = parse_single(
            booking_model, booking_class,
            "context Booking inv Differ: self.check_in <> self.check_out",
            name="Differ",
        )
        assert result['model_expression'] == "self.check_in != self.check_out"

        result = parse_single(
            booking_model, booking_class,
            "context Booking inv Same: self.check_in = self.check_out",
            name="Same",
        )
        assert result['model_expression'] == "self.check_in == self.check_out"

    def test_generated_model_validator_enforces_the_constraint(
        self, booking_class, booking_model, tmpdir
    ):
        booking_model.constraints = {
            make_constraint(
                booking_class,
                "context Booking inv CheckInBeforeCheckOut: self.check_in <= self.check_out",
                name="CheckInBeforeCheckOut",
            )
        }
        path, source = generate(booking_model, tmpdir.mkdir("output"))

        py_compile.compile(path, doraise=True)
        assert "from pydantic import BaseModel, field_validator, model_validator" in source
        assert "@model_validator(mode='after')" in source
        assert "def validate_CheckInBeforeCheckOut(self):" in source
        assert "if not (self.check_in <= self.check_out):" in source

        module = load_generated(path)
        booking_create = module.BookingCreate

        valid = booking_create(
            check_in=datetime.date(2025, 5, 1),
            check_out=datetime.date(2025, 5, 10),
            nights=9,
        )
        assert valid.check_in < valid.check_out

        with pytest.raises(Exception):
            booking_create(
                check_in=datetime.date(2025, 5, 10),
                check_out=datetime.date(2025, 5, 1),
                nights=9,
            )

    def test_constraint_name_becomes_a_valid_identifier(self, booking_class, booking_model, tmpdir):
        booking_model.constraints = {
            make_constraint(
                booking_class,
                "context Booking inv CheckRange: self.check_in <= self.check_out",
                name="Booking.check_range",
            )
        }
        constraints = build_constraints_map(
            booking_model, include_model_level=True
        )["Booking"]
        assert constraints[0]['validator_name'] == "Booking_check_range"

        path, source = generate(booking_model, tmpdir.mkdir("output"))
        py_compile.compile(path, doraise=True)
        assert "def validate_Booking_check_range(self):" in source

    def test_duplicate_constraint_names_get_unique_validators(self, booking_class, booking_model):
        # The DomainModel setter rejects duplicate names, so exercise the
        # de-duplication of get_constraints_for_class on a raw constraint set.
        constraints = {
            make_constraint(
                booking_class,
                "context Booking inv Range: self.check_in <= self.check_out",
                name="Range",
            ),
            make_constraint(
                booking_class,
                "context Booking inv Range: self.check_in <> self.check_out",
                name="Range",
            ),
        }
        parsed = get_constraints_for_class(
            constraints, "Booking", booking_model, include_model_level=True
        )
        names = [c['validator_name'] for c in parsed if 'validator_name' in c]

        assert len(names) == 2
        assert len(names) == len(set(names))


# ============================================================================
# Skipped (untranslatable) constraints
# ============================================================================

class TestSkippedConstraints:
    """Collection/relationship constraints leave a trace instead of vanishing."""

    @pytest.mark.parametrize("expression", [
        "context Booking inv Capacity: self.rooms->size() <= 5",
        "context Booking inv Capacity: self.rooms->collect(r | r.max_people)->sum() > 0",
        "context Booking inv Capacity: self.rooms->forAll(r | r.max_people > 0)",
        "context Booking inv Capacity: self.rooms->isEmpty()",
    ], ids=["size", "collect-sum", "forAll", "isEmpty"])
    def test_collection_constraints_are_marked_skipped(
        self, booking_class, booking_model, expression
    ):
        result = parse_single(booking_model, booking_class, expression, name="Capacity")
        assert result == {'skipped': True, 'constraint_name': 'Capacity'} or result['skipped'] is True

    def test_skip_emits_a_note_comment(self, booking_class, booking_model, tmpdir):
        booking_model.constraints = {
            make_constraint(
                booking_class,
                "context Booking inv NumberOfGuestsDoesNotExceedRoomCapacity: "
                "self.nights->size() <= self.rooms->collect(r | r.max_people)->sum()",
                name="NumberOfGuestsDoesNotExceedRoomCapacity",
            )
        }
        path, source = generate(booking_model, tmpdir.mkdir("output"))

        py_compile.compile(path, doraise=True)
        assert (
            "# NOTE: OCL constraint 'NumberOfGuestsDoesNotExceedRoomCapacity' involves "
            "collections/relationships and is not enforced by this Create model."
        ) in source
        # Nothing was emitted that could reference the collection at runtime.
        assert "->" not in source
        assert "field_validator('nights')" not in source

    def test_map_defaults_to_field_validators_only(self, booking_class, booking_model):
        """Callers that only render per-field validators (Django) opt out by default."""
        booking_model.constraints = {
            make_constraint(
                booking_class,
                "context Booking inv Range: self.check_in <= self.check_out",
                name="Range",
            ),
            make_constraint(
                booking_class,
                "context Booking inv Capacity: self.rooms->size() <= 5",
                name="Capacity",
            ),
            make_constraint(
                booking_class,
                "context Booking inv MinNights: self.nights >= 1",
                name="MinNights",
            ),
        }

        default_map = build_constraints_map(booking_model)
        assert [c['constraint_name'] for c in default_map["Booking"]] == ["MinNights"]

        full_map = build_constraints_map(
            booking_model, include_model_level=True, include_skipped=True
        )
        assert [c['constraint_name'] for c in full_map["Booking"]] == [
            "Capacity", "MinNights", "Range"
        ]

    def test_untranslatable_expression_is_skipped_not_broken(
        self, player_class, player_team_domain_model
    ):
        result = parse_single(
            player_team_domain_model, player_class,
            "context Player inv Weird: self.name.toUpper() = 'A'",
            name="Weird",
        )
        assert result['skipped'] is True

    def test_unknown_free_variable_is_skipped(self, player_class, player_team_domain_model):
        result = parse_single(
            player_team_domain_model, player_class,
            "context Player inv Limit: self.age > MAX_AGE",
            name="Limit",
        )
        assert result['skipped'] is True


# ============================================================================
# The generated file always compiles
# ============================================================================

class TestGeneratedFileAlwaysCompiles:
    """Every emitted expression is compile-checked before it reaches the template."""

    def test_all_four_constraint_shapes_in_one_model(self, booking_class, room_class, booking_model, tmpdir):
        booking_model.constraints = {
            make_constraint(
                booking_class,
                "context Booking inv CheckInBeforeCheckOut: self.check_in <= self.check_out",
                name="CheckInBeforeCheckOut",
            ),
            make_constraint(
                booking_class,
                "context Booking inv MinNights: self.nights >= 1",
                name="MinNights",
            ),
            make_constraint(
                booking_class,
                "context Booking inv Capacity: self.rooms->collect(r | r.max_people)->sum() > 0",
                name="Capacity",
            ),
            make_constraint(
                room_class,
                r"context Room inv ValidNumber: self.number.matches('^[0-9]{1,4}$')",
                name="ValidNumber",
            ),
        }
        path, source = generate(booking_model, tmpdir.mkdir("output"))

        py_compile.compile(path, doraise=True)
        assert "@model_validator(mode='after')" in source
        assert "re.match(r'^[0-9]{1,4}$', v) is not None" in source
        assert "if not (v >= 1):" in source
        assert "# NOTE: OCL constraint 'Capacity'" in source

    @pytest.mark.parametrize("expression", [
        "context Player inv A: self.name.matches('it''s broken')",
        "context Player inv B: self.age > (10",
        "context Player inv C: self.age ??? 10",
        "context Player inv D: self.name.matches('\\\\')",
        "context Player inv E: self.age > 10 and",
    ], ids=["unbalanced-quotes", "unbalanced-paren", "garbage-operator",
            "trailing-backslash-regex", "dangling-and"])
    def test_malformed_expressions_never_emit_broken_code(
        self, player_class, player_team_domain_model, expression, tmpdir
    ):
        player_team_domain_model.constraints = {
            make_constraint(player_class, expression, name="Malformed")
        }
        path, _ = generate(player_team_domain_model, tmpdir.mkdir("output"))
        py_compile.compile(path, doraise=True)
