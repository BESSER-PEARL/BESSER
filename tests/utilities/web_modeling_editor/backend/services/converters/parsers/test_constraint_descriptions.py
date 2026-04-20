"""
Tests for natural-language descriptions on OCL Constraints.

Covers Issue #499: when an OCL constraint is invalid during validation,
end users should see a plain-language explanation alongside the raw OCL
expression. This file exercises the four touch points that thread the
``description`` field end-to-end:

  1. ``Constraint`` metamodel field
  2. ``process_ocl_constraints`` (JSON -> BUML) — both the dedicated
     ``description`` parameter and the inline ``--`` OCL comment fallback
  3. ``check_ocl_constraint`` validator output
  4. ``buml_to_json`` class-diagram converter round-trip
"""

import pytest

from besser.BUML.metamodel.structural import (
    Class,
    Constraint,
    DomainModel,
    IntegerType,
    Property,
)
from besser.utilities.web_modeling_editor.backend.services.converters.parsers.ocl_parser import (
    process_ocl_constraints,
)
from besser.utilities.web_modeling_editor.backend.services.validators.ocl_checker import (
    check_ocl_constraint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def professor_domain_model():
    """A trivial single-class domain model used by the OCL parser tests."""
    professor = Class(name="Professor")
    professor.attributes = {Property(name="age", type=IntegerType)}
    return DomainModel(name="University", types={professor})


# ===========================================================================
# 1. Constraint metamodel
# ===========================================================================

class TestConstraintDescriptionField:
    """The ``description`` field on the Constraint metamodel class."""

    def _make_class(self):
        return Class(name="Professor")

    def test_default_description_is_none(self):
        c = Constraint(
            name="c1",
            context=self._make_class(),
            expression="context Professor inv: self.age > 25",
            language="OCL",
        )
        assert c.description is None

    def test_description_can_be_set_via_constructor(self):
        c = Constraint(
            name="c1",
            context=self._make_class(),
            expression="context Professor inv: self.age > 25",
            language="OCL",
            description="Professors must be over 25",
        )
        assert c.description == "Professors must be over 25"

    def test_description_setter_round_trip(self):
        c = Constraint(
            name="c1",
            context=self._make_class(),
            expression="context Professor inv: self.age > 25",
            language="OCL",
        )
        c.description = "Updated reason"
        assert c.description == "Updated reason"

    def test_description_setter_rejects_non_string(self):
        c = Constraint(
            name="c1",
            context=self._make_class(),
            expression="context Professor inv: self.age > 25",
            language="OCL",
        )
        with pytest.raises(TypeError):
            c.description = 123

    def test_repr_includes_description(self):
        c = Constraint(
            name="c1",
            context=self._make_class(),
            expression="context Professor inv: self.age > 25",
            language="OCL",
            description="Realistic age",
        )
        assert "Realistic age" in repr(c)


# ===========================================================================
# 2. JSON -> BUML parser (process_ocl_constraints)
# ===========================================================================

class TestProcessOclConstraintsDescription:
    """``process_ocl_constraints`` should attach descriptions to Constraint objects."""

    def test_default_description_applied_to_all_blocks(self, professor_domain_model):
        ocl = "context Professor inv: self.age > 25"
        routing, warnings = process_ocl_constraints(
            ocl,
            professor_domain_model,
            counter=0,
            default_description="Be sure to choose a realistic age",
        )
        assert warnings == []
        assert len(routing) == 1
        _kind, constraint, _cls, _method = routing[0]
        assert constraint.description == "Be sure to choose a realistic age"

    def test_inline_comment_takes_precedence_over_default(self, professor_domain_model):
        ocl = "context Professor inv: self.age > 25 -- Inline wins"
        routing, _ = process_ocl_constraints(
            ocl,
            professor_domain_model,
            counter=0,
            default_description="Default loses",
        )
        assert len(routing) == 1
        _kind, constraint, _cls, _method = routing[0]
        assert constraint.description == "Inline wins"
        # The comment is stripped from the expression so the OCL parser
        # never has to deal with it.
        assert "--" not in constraint.expression
        assert "Inline wins" not in constraint.expression

    def test_no_description_yields_none(self, professor_domain_model):
        ocl = "context Professor inv: self.age > 25"
        routing, _ = process_ocl_constraints(
            ocl, professor_domain_model, counter=0
        )
        assert len(routing) == 1
        _kind, constraint, _cls, _method = routing[0]
        assert constraint.description is None

    def test_inline_comment_per_constraint_in_multi_block(self):
        """Per-constraint inline comments are attributed to the right Constraint."""
        prof = Class(name="Professor")
        prof.attributes = {Property(name="age", type=IntegerType)}
        student = Class(name="Student")
        student.attributes = {Property(name="gpa", type=IntegerType)}
        dm = DomainModel(name="Uni", types={prof, student})

        ocl = (
            "context Professor inv: self.age > 25 -- Profs must be experienced\n"
            "context Student inv: self.gpa >= 0 -- GPA cannot be negative\n"
        )
        routing, warnings = process_ocl_constraints(ocl, dm, counter=0)
        assert warnings == []
        # Order is the order of appearance in the input
        by_context = {c.context.name: c for (_kind, c, _cls, _method) in routing}
        assert by_context["Professor"].description == "Profs must be experienced"
        assert by_context["Student"].description == "GPA cannot be negative"


# ===========================================================================
# 3. Validator output (check_ocl_constraint)
# ===========================================================================

class TestCheckOclConstraintIncludesDescription:
    """``check_ocl_constraint`` must surface descriptions in result strings."""

    def test_syntax_only_pass_message_includes_description(self, professor_domain_model):
        c = Constraint(
            name="c1",
            context=next(iter(professor_domain_model.types)),
            expression="context Professor inv: self.age > 25",
            language="OCL",
            description="Realistic age",
        )
        professor_domain_model.constraints = {c}

        result = check_ocl_constraint(professor_domain_model)
        assert result["success"] is True
        joined = " ".join(result["valid_constraints"])
        assert "Realistic age" in joined

    def test_syntax_error_message_includes_description(self, professor_domain_model):
        bad = Constraint(
            name="c_bad",
            context=next(iter(professor_domain_model.types)),
            expression="this is not valid OCL @@@",
            language="OCL",
            description="Friendly hint to the user",
        )
        professor_domain_model.constraints = {bad}

        result = check_ocl_constraint(professor_domain_model)
        assert result["success"] is False
        joined = " ".join(result["invalid_constraints"])
        assert "Friendly hint to the user" in joined

    def test_no_description_means_no_suffix_change(self, professor_domain_model):
        c = Constraint(
            name="c1",
            context=next(iter(professor_domain_model.types)),
            expression="context Professor inv: self.age > 25",
            language="OCL",
        )
        professor_domain_model.constraints = {c}

        result = check_ocl_constraint(professor_domain_model)
        assert result["success"] is True
        # The legacy success format must not be polluted by a stray separator
        # when no description is provided.
        for entry in result["valid_constraints"]:
            assert " — " not in entry  # em-dash separator is description-only


# ===========================================================================
# 4. BUML -> JSON round-trip
# ===========================================================================

class TestBumlToJsonRoundTripDescription:
    """The class-diagram converter should emit the description field."""

    def test_description_appears_in_json_output(self, professor_domain_model):
        from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
            class_buml_to_json,
        )

        c = Constraint(
            name="c1",
            context=next(iter(professor_domain_model.types)),
            expression="context Professor inv: self.age > 25",
            language="OCL",
            description="Realistic age",
        )
        professor_domain_model.constraints = {c}

        json_output = class_buml_to_json(professor_domain_model)
        elements = json_output.get("elements", {})
        constraint_elements = [
            e for e in elements.values() if e.get("type") == "ClassOCLConstraint"
        ]
        assert constraint_elements, "expected at least one ClassOCLConstraint element"
        assert constraint_elements[0].get("description") == "Realistic age"
