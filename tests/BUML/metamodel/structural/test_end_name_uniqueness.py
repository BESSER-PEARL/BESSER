"""Association end-name uniqueness across a class hierarchy.

A class may reach at most one association end of a given name through
``Class.all_association_ends()`` -- its own ends and every ancestor's. The rule
is enforced on construction (``Association.ends`` and both ``Generalization``
setters) and reported by ``DomainModel.validate()``; all three go through the
same check, so the order a model is built in does not change whether it is
accepted. An end reached along two inheritance paths (a diamond) is one end,
not two.
"""

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    Generalization,
    Multiplicity,
    Property,
)


def _link(source: Class, target: Class, role: str, reverse: str) -> BinaryAssociation:
    """``source --role--> target``: the ``role`` end is reached from ``source``."""
    return BinaryAssociation(name=f"{source.name}_{role}", ends={
        Property(name=reverse, type=source, multiplicity=Multiplicity(0, "*")),
        Property(name=role, type=target, multiplicity=Multiplicity(0, "*")),
    })


def _role_count(cls: Class, name: str) -> int:
    return [end.name for end in cls.all_association_ends()].count(name)


# --- Diamonds: one end, two paths -------------------------------------------


def _diamond():
    top, left, right, bottom, part = (Class(name=n) for n in ("Top", "Left", "Right", "Bottom", "Part"))
    return top, left, right, bottom, part


def test_diamond_closed_after_the_association_is_accepted():
    """``Bottom`` reaches ``Top.parts`` through ``Left`` and through ``Right``.

    Closing the diamond used to raise: the generalization setter compared names,
    so the end ``Bottom`` already inherited looked like a second ``parts``.
    """
    top, left, right, bottom, part = _diamond()
    association = _link(top, part, "parts", "whole")
    generalizations = {
        Generalization(general=top, specific=left),
        Generalization(general=top, specific=right),
        Generalization(general=left, specific=bottom),
        Generalization(general=right, specific=bottom),
    }

    assert _role_count(bottom, "parts") == 1
    model = DomainModel(name="M", types={top, left, right, bottom, part},
                        associations={association}, generalizations=generalizations)
    assert model.validate(raise_exception=False)["errors"] == []


def test_diamond_closed_before_the_association_is_accepted():
    top, left, right, bottom, part = _diamond()
    Generalization(general=top, specific=left)
    Generalization(general=top, specific=right)
    Generalization(general=left, specific=bottom)
    Generalization(general=right, specific=bottom)

    _link(top, part, "parts", "whole")

    assert _role_count(bottom, "parts") == 1


def test_closing_a_diamond_through_the_general_setter_is_accepted():
    """The ``general`` setter runs the same check as the ``specific`` one."""
    top, left, right, bottom, part = _diamond()
    _link(top, part, "parts", "whole")
    Generalization(general=top, specific=left)
    Generalization(general=top, specific=right)
    Generalization(general=left, specific=bottom)
    closing = Generalization(general=Class(name="Placeholder"), specific=bottom)

    closing.general = right

    assert right in bottom.parents()
    assert _role_count(bottom, "parts") == 1


# --- Two different ends of one name are still rejected ----------------------


def _two_parents():
    """``Child`` specialises ``A`` and ``B``, which each declare their own ``hasPart``."""
    a, b, child, part = (Class(name=n) for n in ("A", "B", "Child", "Part"))
    return a, b, child, part


def test_a_second_parent_end_added_after_the_generalizations_is_rejected():
    """``B.hasPart`` would give ``Child`` a second ``hasPart``, inherited from ``A``.

    This used to be accepted: the association check only looked at the
    descendants' *own* ends, not what they inherit from their other parents.
    """
    a, b, child, part = _two_parents()
    Generalization(general=a, specific=child)
    Generalization(general=b, specific=child)
    _link(a, part, "hasPart", "a")

    with pytest.raises(ValueError, match="'Child' cannot have two association ends with the same name: 'hasPart'"):
        _link(b, part, "hasPart", "b")
    assert _role_count(child, "hasPart") == 1


def test_a_second_parent_added_after_the_associations_is_rejected():
    a, b, child, part = _two_parents()
    _link(a, part, "hasPart", "a")
    _link(b, part, "hasPart", "b")
    Generalization(general=a, specific=child)

    with pytest.raises(ValueError, match="'Child' cannot have two association ends with the same name: 'hasPart'"):
        Generalization(general=b, specific=child)


def test_a_generalization_that_gives_a_grandchild_two_ends_is_rejected():
    """The clash is on a descendant of the new specific class, via its other parent.

    ``Middle`` itself never reaches two ``hasPart`` ends; ``Child`` (below it,
    and also below ``A``) would. Only the descendants' own ends used to be checked.
    """
    a, b, child, part = _two_parents()
    middle = Class(name="Middle")
    _link(a, part, "hasPart", "a")
    _link(b, part, "hasPart", "b")
    Generalization(general=a, specific=child)
    Generalization(general=middle, specific=child)

    with pytest.raises(ValueError, match="'Child' cannot have two association ends with the same name: 'hasPart'"):
        Generalization(general=b, specific=middle)


def test_validate_reports_two_inherited_ends_of_one_name():
    """``validate()`` applies the same rule to a model a rename made invalid."""
    a, b, child, part = _two_parents()
    first = _link(a, part, "hasPart", "a")
    second = _link(b, part, "partOf", "b")
    generalizations = {Generalization(general=a, specific=child), Generalization(general=b, specific=child)}
    next(e for e in second.ends if e.name == "partOf").name = "hasPart"   # bypasses the setters
    model = DomainModel(name="M", types={a, b, child, part},
                        associations={first, second}, generalizations=generalizations)

    errors = model.validate(raise_exception=False)["errors"]

    assert errors == ["Class 'Child' has more than one association end named 'hasPart'."]


# --- Unchanged behaviour ------------------------------------------------------


def test_two_ends_of_one_name_on_the_same_class_are_rejected():
    owner, a, b = Class(name="Owner"), Class(name="A"), Class(name="B")
    _link(owner, a, "item", "owner")

    with pytest.raises(ValueError, match="'Owner' cannot have two association ends with the same name: 'item'"):
        _link(owner, b, "item", "owner_b")


def test_an_end_a_subclass_already_declares_is_rejected_on_the_superclass():
    parent, child, a, b = Class(name="Parent"), Class(name="Child"), Class(name="A"), Class(name="B")
    Generalization(general=parent, specific=child)
    _link(child, a, "item", "child")

    with pytest.raises(ValueError, match="'Child' cannot have two association ends with the same name: 'item'"):
        _link(parent, b, "item", "parent")
