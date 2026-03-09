"""
Tests for Django generator self-association support.

Self-associations (reflexive associations) occur when a class has a relationship
with itself. For example, an Employee class where one employee manages other
employees, or a Category class with parent-child hierarchy.

These tests verify:
1. The association classification logic correctly categorizes self-associations
   into ForeignKey, ManyToMany, or OneToOne buckets.
2. The models.py.j2 template renders correct Django model fields for
   self-referential relationships.
"""
import os
import pytest
from jinja2 import Environment, FileSystemLoader

from besser.BUML.metamodel.structural import (
    DomainModel,
    Class,
    Property,
    BinaryAssociation,
    Multiplicity,
)
from besser.generators.django import DjangoGenerator
from besser.utilities import sort_by_timestamp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_generator(model: DomainModel) -> DjangoGenerator:
    """Create a DjangoGenerator with only a DomainModel (no GUI model)."""
    return DjangoGenerator(
        model=model,
        project_name="test_project",
        app_name="test_app",
        gui_model=None,
        containerization=False,
    )


def _classify_associations(gen: DjangoGenerator) -> None:
    """
    Run the association classification logic from generate_models() without
    writing any files.  This populates gen.one_to_one, gen.fkeys, and
    gen.many_to_many.
    """
    for association in gen.model.associations:
        ends = list(association.ends)

        # One-to-one
        if ends[0].multiplicity.max == 1 and ends[1].multiplicity.max == 1:
            if ends[1].is_navigable and not ends[0].is_navigable:
                gen.one_to_one[association.name] = ends[0].type.name
            elif not ends[1].is_navigable and ends[0].is_navigable:
                gen.one_to_one[association.name] = ends[1].type.name
            elif ends[1].multiplicity.min == 0:
                gen.one_to_one[association.name] = ends[1].type.name
            else:
                gen.one_to_one[association.name] = ends[0].type.name

        # Foreign Keys
        elif ends[0].multiplicity.max > 1 and ends[1].multiplicity.max <= 1:
            gen.fkeys[association.name] = ends[0].type.name

        elif ends[0].multiplicity.max <= 1 and ends[1].multiplicity.max > 1:
            gen.fkeys[association.name] = ends[1].type.name

        # Many to many
        elif ends[0].multiplicity.max > 1 and ends[1].multiplicity.max > 1:
            if ends[1].is_navigable and not ends[0].is_navigable:
                gen.many_to_many[association.name] = ends[0].type.name
            elif not ends[1].is_navigable and ends[0].is_navigable:
                gen.many_to_many[association.name] = ends[1].type.name
            elif ends[0].multiplicity.min >= 1:
                gen.many_to_many[association.name] = ends[1].type.name
            else:
                gen.many_to_many[association.name] = ends[0].type.name


def _render_models_template(
    model: DomainModel,
    one_to_one: dict,
    many_to_many: dict,
    fkeys: dict,
) -> str:
    """Render models.py.j2 in-memory and return the generated code string."""
    templates_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..",
        "besser", "generators", "django", "templates",
    )
    templates_path = os.path.normpath(templates_path)

    env = Environment(loader=FileSystemLoader(templates_path))
    env.tests["is_primitive_data_type"] = DjangoGenerator.is_primitive_data_type
    template = env.get_template("models.py.j2")
    return template.render(
        model=model,
        sort_by_timestamp=sort_by_timestamp,
        one_to_one=one_to_one,
        many_to_many=many_to_many,
        fkeys=fkeys,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def employee_fk_model():
    """
    Self-referential ForeignKey: Employee.manager -> Employee (many-to-one).

    An employee has at most one manager, but a manager can manage many employees.
    """
    employee_name = Property(name="name", type="str")
    employee = Class(name="Employee", attributes=[employee_name])

    # end pointing to the "one" side (manager): max=1
    manager_end = Property(
        name="manager",
        type=employee,
        multiplicity=Multiplicity(0, 1),
    )
    # end pointing to the "many" side (subordinates): max=*
    subordinates_end = Property(
        name="subordinates",
        type=employee,
        multiplicity=Multiplicity(0, "*"),
    )
    assoc = BinaryAssociation(
        name="employee_manages",
        ends={manager_end, subordinates_end},
    )

    domain_model = DomainModel(
        name="employee_model",
        types={employee},
        associations={assoc},
    )
    return domain_model, employee, assoc


@pytest.fixture
def category_m2m_model():
    """
    Self-referential ManyToMany: Category.related_categories <-> Category.

    A category can be related to many other categories, and vice versa.
    """
    category_title = Property(name="title", type="str")
    category = Class(name="Category", attributes=[category_title])

    end_a = Property(
        name="related_to",
        type=category,
        multiplicity=Multiplicity(0, "*"),
    )
    end_b = Property(
        name="related_from",
        type=category,
        multiplicity=Multiplicity(0, "*"),
    )
    assoc = BinaryAssociation(
        name="category_related",
        ends={end_a, end_b},
    )

    domain_model = DomainModel(
        name="category_model",
        types={category},
        associations={assoc},
    )
    return domain_model, category, assoc


@pytest.fixture
def node_one_to_one_model():
    """
    Self-referential OneToOne: Node.next <-> Node.prev.

    A linked-list style relationship: each node has at most one next node.
    """
    node_value = Property(name="value", type="int")
    node = Class(name="Node", attributes=[node_value])

    next_end = Property(
        name="next_node",
        type=node,
        multiplicity=Multiplicity(0, 1),
    )
    prev_end = Property(
        name="prev_node",
        type=node,
        multiplicity=Multiplicity(0, 1),
    )
    assoc = BinaryAssociation(
        name="node_link",
        ends={next_end, prev_end},
    )

    domain_model = DomainModel(
        name="node_model",
        types={node},
        associations={assoc},
    )
    return domain_model, node, assoc


# ---------------------------------------------------------------------------
# Tests: Association classification
# ---------------------------------------------------------------------------

class TestSelfAssociationClassification:
    """Verify that the classification loop correctly buckets self-associations."""

    def test_self_fk_classified(self, employee_fk_model):
        """A many-to-one self-association should land in the fkeys dict."""
        model, employee, assoc = employee_fk_model
        gen = _make_generator(model)
        _classify_associations(gen)

        assert assoc.name in gen.fkeys, (
            f"Expected self-referential FK association '{assoc.name}' in fkeys, "
            f"got fkeys={gen.fkeys}"
        )
        # The value should be the Employee class name (the side that "owns" the FK)
        assert gen.fkeys[assoc.name] == "Employee"
        # Should NOT appear in the other buckets
        assert assoc.name not in gen.one_to_one
        assert assoc.name not in gen.many_to_many

    def test_self_m2m_classified(self, category_m2m_model):
        """A many-to-many self-association should land in the many_to_many dict."""
        model, category, assoc = category_m2m_model
        gen = _make_generator(model)
        _classify_associations(gen)

        assert assoc.name in gen.many_to_many, (
            f"Expected self-referential M2M association '{assoc.name}' in many_to_many, "
            f"got many_to_many={gen.many_to_many}"
        )
        assert gen.many_to_many[assoc.name] == "Category"
        assert assoc.name not in gen.one_to_one
        assert assoc.name not in gen.fkeys

    def test_self_one_to_one_classified(self, node_one_to_one_model):
        """A one-to-one self-association should land in the one_to_one dict."""
        model, node, assoc = node_one_to_one_model
        gen = _make_generator(model)
        _classify_associations(gen)

        assert assoc.name in gen.one_to_one, (
            f"Expected self-referential O2O association '{assoc.name}' in one_to_one, "
            f"got one_to_one={gen.one_to_one}"
        )
        assert gen.one_to_one[assoc.name] == "Node"
        assert assoc.name not in gen.fkeys
        assert assoc.name not in gen.many_to_many


# ---------------------------------------------------------------------------
# Tests: Template rendering for self-associations
# ---------------------------------------------------------------------------

class TestSelfAssociationTemplateRendering:
    """Verify that models.py.j2 produces correct Django fields for self-associations."""

    def test_self_fk_renders_foreign_key(self, employee_fk_model):
        """Self-referential FK should render as models.ForeignKey('Employee', ...)."""
        model, employee, assoc = employee_fk_model
        gen = _make_generator(model)
        _classify_associations(gen)

        output = _render_models_template(model, gen.one_to_one, gen.many_to_many, gen.fkeys)

        assert "class Employee(models.Model):" in output, (
            "Expected Employee model class definition in output"
        )
        assert "models.ForeignKey(" in output, (
            "Expected a ForeignKey field for the self-referential many-to-one association"
        )
        # The FK should reference 'Employee' (the same class)
        assert "'Employee'" in output, (
            "Expected the ForeignKey to reference 'Employee' (self-referential)"
        )

    def test_self_m2m_renders_many_to_many(self, category_m2m_model):
        """Self-referential M2M should render as models.ManyToManyField('Category', ...)."""
        model, category, assoc = category_m2m_model
        gen = _make_generator(model)
        _classify_associations(gen)

        output = _render_models_template(model, gen.one_to_one, gen.many_to_many, gen.fkeys)

        assert "class Category(models.Model):" in output, (
            "Expected Category model class definition in output"
        )
        assert "models.ManyToManyField(" in output, (
            "Expected a ManyToManyField for the self-referential many-to-many association"
        )
        assert "'Category'" in output, (
            "Expected the ManyToManyField to reference 'Category' (self-referential)"
        )

    def test_self_one_to_one_renders_field(self, node_one_to_one_model):
        """Self-referential O2O should render as models.OneToOneField('Node', ...)."""
        model, node, assoc = node_one_to_one_model
        gen = _make_generator(model)
        _classify_associations(gen)

        output = _render_models_template(model, gen.one_to_one, gen.many_to_many, gen.fkeys)

        assert "class Node(models.Model):" in output, (
            "Expected Node model class definition in output"
        )
        assert "models.OneToOneField(" in output, (
            "Expected a OneToOneField for the self-referential one-to-one association"
        )
        assert "'Node'" in output, (
            "Expected the OneToOneField to reference 'Node' (self-referential)"
        )

    def test_self_fk_has_related_name(self, employee_fk_model):
        """Self-referential FK should have a related_name to avoid Django clashes."""
        model, employee, assoc = employee_fk_model
        gen = _make_generator(model)
        _classify_associations(gen)

        output = _render_models_template(model, gen.one_to_one, gen.many_to_many, gen.fkeys)

        assert "related_name=" in output, (
            "Expected a related_name argument on the self-referential ForeignKey "
            "to avoid reverse accessor clashes"
        )

    @pytest.mark.xfail(
        reason=(
            "Known bug: models.py.j2 template iterates association_ends() which "
            "returns BOTH ends for self-associations (since both point to the same "
            "class). The template condition 'class_obj.name == fkeys.get(end.owner.name)' "
            "matches for both ends, producing a duplicate field. The template or "
            "classification logic needs to track which specific end owns the field."
        ),
        strict=True,
    )
    def test_self_fk_no_duplicate_fields(self, employee_fk_model):
        """Self-referential FK should produce exactly one ForeignKey field, not two."""
        model, employee, assoc = employee_fk_model
        gen = _make_generator(model)
        _classify_associations(gen)

        output = _render_models_template(model, gen.one_to_one, gen.many_to_many, gen.fkeys)

        fk_count = output.count("models.ForeignKey(")
        assert fk_count == 1, (
            f"Expected exactly 1 ForeignKey for the self-referential association, "
            f"but found {fk_count}. This may indicate the template renders the field "
            f"for both ends instead of just the owning side."
        )

    @pytest.mark.xfail(
        reason=(
            "Known bug: models.py.j2 template iterates association_ends() which "
            "returns BOTH ends for self-associations. The template condition "
            "'class_obj.name == many_to_many.get(end.owner.name)' matches for both "
            "ends, producing a duplicate ManyToManyField."
        ),
        strict=True,
    )
    def test_self_m2m_no_duplicate_fields(self, category_m2m_model):
        """Self-referential M2M should produce exactly one ManyToManyField, not two."""
        model, category, assoc = category_m2m_model
        gen = _make_generator(model)
        _classify_associations(gen)

        output = _render_models_template(model, gen.one_to_one, gen.many_to_many, gen.fkeys)

        m2m_count = output.count("models.ManyToManyField(")
        assert m2m_count == 1, (
            f"Expected exactly 1 ManyToManyField for the self-referential association, "
            f"but found {m2m_count}. This may indicate the template renders the field "
            f"for both ends instead of just the owning side."
        )

    @pytest.mark.xfail(
        reason=(
            "Known bug: models.py.j2 template iterates association_ends() which "
            "returns BOTH ends for self-associations. The template condition "
            "'class_obj.name == one_to_one.get(end.owner.name)' matches for both "
            "ends, producing a duplicate OneToOneField."
        ),
        strict=True,
    )
    def test_self_one_to_one_no_duplicate_fields(self, node_one_to_one_model):
        """Self-referential O2O should produce exactly one OneToOneField, not two."""
        model, node, assoc = node_one_to_one_model
        gen = _make_generator(model)
        _classify_associations(gen)

        output = _render_models_template(model, gen.one_to_one, gen.many_to_many, gen.fkeys)

        o2o_count = output.count("models.OneToOneField(")
        assert o2o_count == 1, (
            f"Expected exactly 1 OneToOneField for the self-referential association, "
            f"but found {o2o_count}. This may indicate the template renders the field "
            f"for both ends instead of just the owning side."
        )
