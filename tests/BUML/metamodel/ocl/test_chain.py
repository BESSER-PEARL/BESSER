"""Tests for besser.BUML.metamodel.ocl.chain."""

from besser.BUML.metamodel.ocl import (
    is_chain_from_self, walk_chain_from_self,
    chain_min_multiplicity, chain_max_multiplicity,
)
from besser.BUML.metamodel.ocl.ocl import (
    PropertyCallExpression, VariableExp, IntegerLiteralExpression,
)
from besser.BUML.metamodel.structural import (
    Property, IntegerType, Multiplicity,
)


def _chain(*props_with_source):
    """Build self.r1.r2.... from a sequence of (name, mult) tuples."""
    cur = VariableExp("self", None)
    for name, mult in props_with_source:
        prop = Property(name, IntegerType, multiplicity=mult)
        pce = PropertyCallExpression(name, prop)
        pce.source = cur
        cur = pce
    return cur


def test_is_chain_from_self_recognises_simple_chain():
    chain = _chain(("r1", Multiplicity(1, 1)))
    assert is_chain_from_self(chain)


def test_is_chain_from_self_recognises_long_chain():
    chain = _chain(
        ("r1", Multiplicity(1, 1)),
        ("r2", Multiplicity(0, "*")),
        ("r3", Multiplicity(1, 1)),
    )
    assert is_chain_from_self(chain)


def test_is_chain_from_self_rejects_bare_property():
    # A bare Property leaked from an unwrapped visitor — not recognised as a chain.
    bare = Property("r1", IntegerType, multiplicity=Multiplicity(1, 1))
    assert not is_chain_from_self(bare)


def test_is_chain_from_self_rejects_non_self_root():
    # `e.r1` where `e` is an iterator variable is a chain but not from self.
    cur = VariableExp("e", None)
    prop = Property("r1", IntegerType, multiplicity=Multiplicity(1, 1))
    pce = PropertyCallExpression("r1", prop)
    pce.source = cur
    assert not is_chain_from_self(pce)


def test_is_chain_from_self_rejects_non_chain():
    assert not is_chain_from_self(IntegerLiteralExpression("NP", 1))
    assert not is_chain_from_self(VariableExp("self", None))


def test_walk_chain_returns_property_list_in_order():
    chain = _chain(
        ("r1", Multiplicity(1, 1)),
        ("r2", Multiplicity(0, 5)),
        ("r3", Multiplicity(2, 2)),
    )
    props = walk_chain_from_self(chain)
    assert [p.name for p in props] == ["r1", "r2", "r3"]


def test_walk_chain_returns_none_for_non_chain():
    assert walk_chain_from_self(IntegerLiteralExpression("NP", 1)) is None


def test_chain_min_multiplicity_min_across_steps():
    # mins: 1, 0, 1 → min = 0
    chain = _chain(
        ("r1", Multiplicity(1, 1)),
        ("r2", Multiplicity(0, 5)),
        ("r3", Multiplicity(1, 1)),
    )
    assert chain_min_multiplicity(chain) == 0


def test_chain_min_multiplicity_all_at_least_one_signals_non_empty():
    chain = _chain(
        ("r1", Multiplicity(1, 1)),
        ("r2", Multiplicity(2, 5)),
        ("r3", Multiplicity(1, 1)),
    )
    assert chain_min_multiplicity(chain) == 1


def test_chain_max_multiplicity_max_across_steps():
    # maxes: 1, 5, 1 → max = 5
    chain = _chain(
        ("r1", Multiplicity(1, 1)),
        ("r2", Multiplicity(0, 5)),
        ("r3", Multiplicity(1, 1)),
    )
    assert chain_max_multiplicity(chain) == 5


def test_chain_max_multiplicity_all_at_most_one_signals_singleton():
    chain = _chain(
        ("r1", Multiplicity(1, 1)),
        ("r2", Multiplicity(0, 1)),
    )
    assert chain_max_multiplicity(chain) == 1


def test_chain_multiplicity_returns_none_for_non_chain():
    assert chain_min_multiplicity(IntegerLiteralExpression("NP", 1)) is None
    assert chain_max_multiplicity(IntegerLiteralExpression("NP", 1)) is None
