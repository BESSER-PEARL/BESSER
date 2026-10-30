"""The three-way merge of two versions of a project's diagrams.json."""

import pytest

from besser.utilities.web_modeling_editor.backend.services.deployment.model_merge import (
    MergeConflict,
    merge_documents,
)


def test_elements_added_on_both_sides_are_all_kept():
    base = {"elements": {"a": {"name": "A"}}}
    ours = {"elements": {"a": {"name": "A"}, "b": {"name": "B"}}}
    theirs = {"elements": {"a": {"name": "A"}, "c": {"name": "C"}}}

    assert merge_documents(base, ours, theirs) == {"elements": {"a": {"name": "A"}, "c": {"name": "C"},
                                                                "b": {"name": "B"}}}


def test_react_flow_node_lists_merge_by_id():
    """The React Flow format keeps nodes in a list; each node has an id."""
    base = {"nodes": [{"id": "a", "data": {"name": "A"}}]}
    ours = {"nodes": [{"id": "a", "data": {"name": "A2"}}, {"id": "b", "data": {"name": "B"}}]}
    theirs = {"nodes": [{"id": "a", "data": {"name": "A"}}, {"id": "c", "data": {"name": "C"}}]}

    merged = merge_documents(base, ours, theirs)

    assert merged["nodes"] == [{"id": "a", "data": {"name": "A2"}}, {"id": "c", "data": {"name": "C"}},
                               {"id": "b", "data": {"name": "B"}}]


def test_attribute_id_lists_merge_as_sets():
    base = {"attributes": ["x"]}
    ours = {"attributes": ["x", "ours"]}
    theirs = {"attributes": ["theirs"]}  # they removed x

    assert merge_documents(base, ours, theirs) == {"attributes": ["theirs", "ours"]}


def test_the_same_name_changed_twice_is_a_conflict():
    with pytest.raises(MergeConflict) as conflict:
        merge_documents({"a": {"name": "A"}}, {"a": {"name": "B"}}, {"a": {"name": "C"}})
    assert conflict.value.path == "a.name"


def test_a_change_to_something_the_other_side_deleted_is_a_conflict():
    with pytest.raises(MergeConflict):
        merge_documents({"a": {"name": "A"}}, {"a": {"name": "B"}}, {})


def test_layout_and_timestamps_keep_ours():
    base = {"a": {"bounds": {"x": 0}, "name": "A"}, "exportedAt": "1"}
    ours = {"a": {"bounds": {"x": 10}, "name": "A"}, "exportedAt": "2"}
    theirs = {"a": {"bounds": {"x": 20}, "name": "A"}, "exportedAt": "3"}

    assert merge_documents(base, ours, theirs) == ours


def test_a_model_the_other_side_deleted_cannot_be_merged():
    with pytest.raises(MergeConflict):
        merge_documents({"a": 1}, {"a": 2}, None)
