"""
Tests for OCL constraint support in the Django generator.

DjangoGenerator reuses the same OCL parser as PydanticGenerator
(besser.generators.pydantic_classes.ocl_utils.build_constraints_map) and emits
a clean() method per constrained class instead of Pydantic's per-field
@field_validator, since Django models validate via clean()/full_clean()
rather than at construction time.

Following the precedent in test_django_self_assoc.py, these tests render
models.py.j2 in-memory and assert on the generated code (string content +
successful compile()) rather than instantiating real Django model instances,
which would require registering a throwaway app with Django's app registry.
"""
import re

from besser.BUML.metamodel.structural import (
    DomainModel,
    Class,
    Property,
    Constraint,
    IntegerType,
    StringType,
)
from besser.generators.django import DjangoGenerator
from besser.generators.pydantic_classes.ocl_utils import build_constraints_map


def _render_models(model: DomainModel) -> str:
    """Render models.py.j2 for `model` and return the generated code string."""
    gen = DjangoGenerator(
        model=model,
        project_name="test_project",
        app_name="test_app",
        gui_model=None,
        containerization=False,
    )
    constraints_map = build_constraints_map(model)
    template = gen.env.get_template("models.py.j2")
    return template.render(
        model=model,
        sort_by_timestamp=lambda items: sorted(items, key=lambda a: a.name),
        one_to_one={},
        many_to_many={},
        fkeys={},
        constraints_map=constraints_map,
    )


class TestSimpleConstraint:
    """A single comparison constraint (property op value)."""

    def _model(self):
        age_attr = Property(name="age", type=IntegerType)
        player = Class(name="Player", attributes={age_attr})
        domain_model = DomainModel(
            name="M",
            types={player},
            constraints={
                Constraint(
                    name="min_age",
                    context=player,
                    expression="context Player inv: self.age > 10",
                    language="OCL",
                )
            },
        )
        return domain_model

    def test_generates_clean_method(self):
        output = _render_models(self._model())
        assert "def clean(self):" in output
        assert "super().clean()" in output
        assert "from django.core.exceptions import ValidationError" in output

    def test_condition_matches_ocl_operator(self):
        output = _render_models(self._model())
        assert "if not (self.age > 10):" in output

    def test_compiles_as_valid_python(self):
        output = _render_models(self._model())
        compile(output, "<generated>", "exec")


class TestCompoundConstraint:
    """A compound constraint (property op value and property op value)."""

    def _model(self):
        age_attr = Property(name="age", type=IntegerType)
        match_class = Class(name="Match", attributes={age_attr})
        domain_model = DomainModel(
            name="M",
            types={match_class},
            constraints={
                Constraint(
                    name="range_check",
                    context=match_class,
                    expression="context Match inv: self.age > 10 and self.age < 65",
                    language="OCL",
                )
            },
        )
        return domain_model

    def test_generates_python_expression_branch(self):
        output = _render_models(self._model())
        assert "v = self.age" in output
        assert "if not (v > 10 and v < 65):" in output

    def test_compiles_as_valid_python(self):
        output = _render_models(self._model())
        compile(output, "<generated>", "exec")


class TestMultipleConstraintsSameProperty:
    """Two constraints on the same property must both be checked and both
    accumulate into the same errors[property] list, not overwrite each other."""

    def _model(self):
        age_attr = Property(name="age", type=IntegerType)
        player = Class(name="Player", attributes={age_attr})
        domain_model = DomainModel(
            name="M",
            types={player},
            constraints={
                Constraint(
                    name="min_age",
                    context=player,
                    expression="context Player inv: self.age > 10",
                    language="OCL",
                ),
                Constraint(
                    name="max_age",
                    context=player,
                    expression="context Player inv: self.age < 65",
                    language="OCL",
                ),
            },
        )
        return domain_model

    def test_both_conditions_present(self):
        output = _render_models(self._model())
        assert "if not (self.age > 10):" in output
        assert "if not (self.age < 65):" in output

    def test_uses_setdefault_to_accumulate_errors(self):
        output = _render_models(self._model())
        assert output.count("errors.setdefault('age', [])") == 2

    def test_compiles_as_valid_python(self):
        output = _render_models(self._model())
        compile(output, "<generated>", "exec")


class TestNoConstraints:
    """Classes/models without OCL constraints must not get a clean() method
    or the ValidationError import (avoids unused-import lint noise)."""

    def test_no_clean_method_without_constraints(self):
        plain = Class(name="Plain", attributes={Property(name="x", type=IntegerType)})
        domain_model = DomainModel(name="M", types={plain}, constraints=set())
        output = _render_models(domain_model)
        assert "def clean(self):" not in output
        assert "ValidationError" not in output

    def test_compiles_as_valid_python(self):
        plain = Class(name="Plain", attributes={Property(name="x", type=IntegerType)})
        domain_model = DomainModel(name="M", types={plain}, constraints=set())
        output = _render_models(domain_model)
        compile(output, "<generated>", "exec")


class TestConstraintOnlyAffectsTargetClass:
    """A constraint on one class must not add a clean() method to a sibling
    class in the same domain model."""

    def test_unconstrained_sibling_has_no_clean_method(self):
        age_attr = Property(name="age", type=IntegerType)
        player = Class(name="Player", attributes={age_attr})
        team = Class(name="Team", attributes={Property(name="name", type=StringType)})
        domain_model = DomainModel(
            name="M",
            types={player, team},
            constraints={
                Constraint(
                    name="min_age",
                    context=player,
                    expression="context Player inv: self.age > 10",
                    language="OCL",
                )
            },
        )
        output = _render_models(domain_model)

        # Class order isn't guaranteed, so split on both markers and match by name.
        blocks = re.split(r"(?=^class \w+\()", output, flags=re.MULTILINE)
        player_block = next(b for b in blocks if b.startswith("class Player("))
        team_block = next(b for b in blocks if b.startswith("class Team("))

        assert "def clean(self):" in player_block
        assert "def clean(self):" not in team_block
