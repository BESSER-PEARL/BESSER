"""``merge_trees``: a merge where each file changed on one side at most needs no pygit2."""
import importlib.util
import itertools

import pytest

from besser.utilities import three_way_merge
from besser.utilities.three_way_merge import MergeUnavailableError, merge_trees

# Each side of one file: missing, the base text, or one of two other texts.
VERSIONS = [None, b"base\n", b"one\n", b"two\n"]


def _tree(content):
    return {} if content is None else {"f.txt": content}


def _one_sided_cases():
    for base, ours, theirs in itertools.product(VERSIONS, repeat=3):
        if base == ours or base == theirs or ours == theirs:
            yield base, ours, theirs


@pytest.mark.parametrize("base, ours, theirs", list(_one_sided_cases()))
def test_without_pygit2_a_one_sided_merge_takes_the_changed_side(monkeypatch, base, ours, theirs):
    monkeypatch.setattr(three_way_merge, "pygit2", None)

    merged = merge_trees(_tree(base), _tree(ours), _tree(theirs))

    expected = theirs if ours == base else ours
    assert merged.clean == _tree(expected) and merged.conflicts == {}


@pytest.mark.skipif(importlib.util.find_spec("pygit2") is None, reason="compares with pygit2's merge")
@pytest.mark.parametrize("base, ours, theirs", list(_one_sided_cases()))
def test_a_one_sided_merge_is_the_one_libgit2_makes(monkeypatch, base, ours, theirs):
    with_libgit2 = three_way_merge._changed_on_both_sides
    monkeypatch.setattr(three_way_merge, "_changed_on_both_sides", lambda *trees: with_libgit2(*trees) or ["all"])
    libgit2 = merge_trees(_tree(base), _tree(ours), _tree(theirs))
    monkeypatch.undo()

    assert merge_trees(_tree(base), _tree(ours), _tree(theirs)) == libgit2


def test_without_pygit2_files_changed_on_both_sides_are_named(monkeypatch):
    monkeypatch.setattr(three_way_merge, "pygit2", None)
    base = {"a.txt": b"a\n", "b.txt": b"b\n"}

    with pytest.raises(MergeUnavailableError) as raised:
        merge_trees(base, {"a.txt": b"mine\n", "b.txt": b"b\n"}, {"a.txt": b"model\n", "b.txt": b"new\n"})

    assert raised.value.files == ["a.txt"]
