from types import SimpleNamespace

from besser.utilities.utils import sort_by_timestamp


def test_timestamp_ties_are_broken_by_name():
    # Built in both insertion orders so set iteration order can't decide the result.
    names = ["zeta", "alpha", "mid"]
    for order in (names, list(reversed(names))):
        objs = [SimpleNamespace(name=n, timestamp=1) for n in order]
        assert [o.name for o in sort_by_timestamp(objs)] == ["alpha", "mid", "zeta"]


def test_timestamp_still_takes_precedence():
    objs = [SimpleNamespace(name="a", timestamp=2), SimpleNamespace(name="b", timestamp=1)]
    assert [o.name for o in sort_by_timestamp(objs)] == ["b", "a"]
