"""The validator must not report a verdict for ``->closure`` (BESSER#635).

``bocl`` 1.0.1, the evaluator behind ``/validate-diagram``, has no
``closure`` support: it reports ``Evaluates to: True`` (or ``()``) whatever
the object diagram, or fails with a Python ``invalid syntax`` error. The
checker reports such invariants as syntactically valid and not evaluated,
with a diagram warning, like the other constructs the evaluator cannot handle.
"""

import pytest

from besser.BUML.metamodel.object import (
    AttributeLink, DataValue, Link, LinkEnd, Object, ObjectModel,
)
from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, Constraint, DomainModel, Multiplicity, Property,
    StringType,
)
from besser.utilities.web_modeling_editor.backend.services.validators.ocl_checker import (
    check_ocl_constraint,
)


@pytest.fixture()
def cyclic_genealogy():
    """Two persons that are each other's parent: every closure check fails."""
    name = Property("name", StringType)
    person = Class("Person", attributes={name})
    children = Property("children", person, multiplicity=Multiplicity(0, "*"))
    parents = Property("parents", person, multiplicity=Multiplicity(0, "*"))
    assoc = BinaryAssociation("is_child_of", ends={children, parents})
    model = DomainModel("Genealogy", types={person}, associations={assoc})

    def person_obj(n):
        return Object(n, person, slots=[
            AttributeLink(attribute=name, value=DataValue(classifier=StringType, value=n)),
        ])

    a, b = person_obj("a"), person_obj("b")
    Link("l1", assoc, [LinkEnd("c1", children, a), LinkEnd("p1", parents, b)])
    Link("l2", assoc, [LinkEnd("c2", children, b), LinkEnd("p2", parents, a)])
    return model, person, ObjectModel("people", objects={a, b})


@pytest.mark.parametrize("expression", [
    # bocl: "Evaluates to: True" although every closure has size >= 1.
    "context Person inv no_ancestor: self->closure(parents)->size() < 1",
    # bocl: "Evaluates to: ()".
    "context Person inv nested: Person.allInstances()->forAll(p | p.parents->closure(parents)->excludes(p))",
    # bocl: "invalid syntax (<string>, line 0)".
    "context Person inv all: Person::allInstances()->closure(parents)->isEmpty()",
])
def test_closure_is_reported_as_not_evaluated(cyclic_genealogy, expression):
    model, person, objects = cyclic_genealogy
    model.constraints = {Constraint("c", person, expression, "OCL")}

    result = check_ocl_constraint(model, objects)

    assert result["invalid_constraints"] == []
    (message,) = result["valid_constraints"]
    assert "Syntax valid; not evaluated against the object diagram" in message
    assert "the 'closure' iterator" in message
    assert "Evaluates to" not in message
    (warning,) = result["warnings"]
    assert "the 'closure' iterator" in warning


def test_closure_syntax_error_is_still_reported(cyclic_genealogy):
    model, person, objects = cyclic_genealogy
    model.constraints = {Constraint(
        "c", person, "context Person inv bad: self->closure(parents", "OCL",
    )}

    result = check_ocl_constraint(model, objects)

    assert result["valid_constraints"] == []
    assert len(result["invalid_constraints"]) == 1
    assert result["warnings"] == []
