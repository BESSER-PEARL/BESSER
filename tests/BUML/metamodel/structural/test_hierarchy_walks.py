"""The class-hierarchy walks: deep chains, diamonds and cycles.

``all_parents`` / ``all_specializations`` and the ``DomainModel`` walks built on
them (``classes_sorted_by_inheritance``, the circular-inheritance check of
``validate()``) are iterative, so an inheritance chain deeper than Python's
recursion limit (about 1000 frames) is walked like any other.
"""

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    Generalization,
    Property,
)

DEPTH = 3000


@pytest.fixture(scope="module")
def chain():
    """``C0000 <- C0001 <- … <- C2999``, with an association end declared on the root."""
    classes = [Class(name=f"C{i:04d}") for i in range(DEPTH)]
    part = Class(name="Part")
    association = BinaryAssociation(name="root_parts", ends={
        Property(name="root", type=classes[0]), Property(name="parts", type=part)})
    generalizations = [
        Generalization(general=classes[i], specific=classes[i + 1]) for i in range(DEPTH - 1)
    ]
    model = DomainModel(name="Deep", types=set(classes) | {part},
                        associations={association}, generalizations=set(generalizations))
    return classes, model


def test_all_parents_walks_a_chain_deeper_than_the_recursion_limit(chain):
    classes, _ = chain
    assert classes[-1].all_parents() == set(classes[:-1])
    assert classes[0].all_parents() == set()


def test_all_specializations_walks_a_chain_deeper_than_the_recursion_limit(chain):
    classes, _ = chain
    assert classes[0].all_specializations() == set(classes[1:])
    assert classes[-1].all_specializations() == set()


def test_inherited_features_reach_the_bottom_of_a_deep_chain(chain):
    classes, _ = chain
    assert [end.name for end in classes[-1].all_association_ends()] == ["parts"]


def test_classes_sorted_by_inheritance_handles_a_deep_chain(chain):
    classes, model = chain
    ordered = [c for c in model.classes_sorted_by_inheritance() if c.name != "Part"]
    assert ordered == classes


def test_validate_handles_a_deep_chain(chain):
    _, model = chain
    assert model.validate(raise_exception=False)["errors"] == []


def test_walks_visit_a_diamond_ancestor_once():
    top, left, right, bottom = (Class(name=n) for n in ("Top", "Left", "Right", "Bottom"))
    for general, specific in ((top, left), (top, right), (left, bottom), (right, bottom)):
        Generalization(general=general, specific=specific)

    assert bottom.all_parents() == {left, right, top}
    assert top.all_specializations() == {left, right, bottom}


def test_walks_terminate_on_an_inheritance_cycle():
    """The setters do not forbid a cycle (re-pointing ``general`` can close one);
    ``validate()`` reports it. Walking one must still terminate -- it used to
    recurse until RecursionError -- and a class is never its own ancestor."""
    a, b, c = (Class(name=n) for n in "ABC")
    generalizations = {
        Generalization(general=a, specific=b),
        Generalization(general=b, specific=c),
    }
    closing = Generalization(general=Class(name="Placeholder"), specific=a)
    closing.general = c
    generalizations.add(closing)

    assert a.all_parents() == {b, c}
    assert a.all_specializations() == {b, c}
    model = DomainModel(name="Cyclic", types={a, b, c}, generalizations=generalizations)
    assert "Circular inheritance detected involving class 'A'." in model.validate(raise_exception=False)["errors"]


def test_classes_sorted_by_inheritance_keeps_its_order():
    """Parents before children; the iterative walk visits in the order the
    recursive one did (reverse post-order over name-sorted classes)."""
    root, mid_b, mid_a, leaf, other = (Class(name=n) for n in ("Root", "MidB", "MidA", "Leaf", "Other"))
    generalizations = {
        Generalization(general=root, specific=mid_a),
        Generalization(general=root, specific=mid_b),
        Generalization(general=mid_b, specific=leaf),
    }
    model = DomainModel(name="M", types={root, mid_a, mid_b, leaf, other}, generalizations=generalizations)

    assert [c.name for c in model.classes_sorted_by_inheritance()] == ["Root", "Other", "MidB", "MidA", "Leaf"]
