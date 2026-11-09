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


def _opened_by(person: str, classes: dict, **view) -> dict:
    """A model as saved by someone who opened the repository: opening mints a project id and createdAt."""
    return {
        "editorVersion": view.get("editor", "1.0.0"),
        "project": {
            "id": f"project-of-{person}",
            "createdAt": f"2026-11-05T09:00:00Z-{person}",
            "name": "Grand Hotel",
            "currentDiagramType": view.get("tab", "ClassDiagram"),
            "currentDiagramIndices": {"ClassDiagram": view.get("index", 0)},
            "diagrams": {"ClassDiagram": [{"id": "d1", "model": {"elements": classes}}]},
        },
    }


def test_two_people_who_each_opened_the_repo_can_merge():
    # Live 2026-11-05: Ana and Ben both opened the repo; Ben's save failed with CONFLICT at project.id.
    repo = {
        "project": {
            "id": "project-in-repo",
            "createdAt": "2026-10-31T09:00:00Z",
            "name": "Grand Hotel",
            "currentDiagramType": "ClassDiagram",
            "currentDiagramIndices": {"ClassDiagram": 0},
            "diagrams": {"ClassDiagram": [{"id": "d1", "model": {"elements": {"a": {"name": "Room"}}}}]},
        },
    }
    ana = _opened_by("ana", {"a": {"name": "Room"}, "b": {"name": "Amenity"}})
    ben = _opened_by("ben", {"a": {"name": "Room"}, "c": {"name": "Review"}})

    merged = merge_documents(repo, ben, ana)

    assert merged["project"]["id"] == "project-of-ana"
    assert merged["project"]["createdAt"] == "2026-11-05T09:00:00Z-ana"
    assert set(merged["project"]["diagrams"]["ClassDiagram"][0]["model"]["elements"]) == {"a", "b", "c"}


def test_what_each_person_was_looking_at_is_no_conflict():
    repo = _opened_by("repo", {})
    ana = _opened_by("ana", {}, tab="GUINoCodeDiagram", editor="1.0.1")
    ben = _opened_by("ben", {}, tab="AgentDiagram", editor="1.0.2")

    merged = merge_documents(repo, ben, ana)

    assert merged["project"]["currentDiagramType"] == "AgentDiagram"
    assert merged["editorVersion"] == "1.0.2"


def test_a_bare_project_keeps_the_repos_identity_too():
    repo = {"id": "r", "createdAt": "0", "name": "A"}
    ours = {"id": "o", "createdAt": "1", "name": "A"}
    theirs = {"id": "t", "createdAt": "2", "name": "A"}

    assert merge_documents(repo, ours, theirs) == {"id": "t", "createdAt": "2", "name": "A"}


def test_a_save_over_an_unchanged_repo_id_writes_its_own():
    # Only both sides changing it needs a rule; the web_editor_github docs say so.
    assert merge_documents({"project": {"id": "r"}}, {"project": {"id": "o"}}, {"project": {"id": "r"}}) == {
        "project": {"id": "o"},
    }


def test_a_created_at_elsewhere_in_the_model_still_conflicts():
    with pytest.raises(MergeConflict) as conflict:
        merge_documents({"a": {"createdAt": "0"}}, {"a": {"createdAt": "1"}}, {"a": {"createdAt": "2"}})
    assert conflict.value.path == "a.createdAt"


def test_only_the_project_id_is_exempt_not_a_diagrams_or_an_elements():
    with pytest.raises(MergeConflict) as conflict:
        merge_documents({"elements": {"a": {"id": "x"}}}, {"elements": {"a": {"id": "y"}}},
                        {"elements": {"a": {"id": "z"}}})
    assert conflict.value.path == "elements.a.id"


def test_a_project_renamed_twice_is_still_a_conflict():
    with pytest.raises(MergeConflict) as conflict:
        merge_documents({"project": {"name": "A"}}, {"project": {"name": "B"}}, {"project": {"name": "C"}})
    assert conflict.value.path == "project.name"


def test_two_peoples_assistant_conversations_are_no_conflict():
    def saved(turns):
        return {"project": {"name": "Shop", "assistantConversation": {"turns": turns, "updatedAt": "t"}}}

    repo = saved([{"role": "user", "content": "A shop"}])
    ana = saved([{"role": "user", "content": "A shop"}, {"role": "user", "content": "Add Order"}])
    ben = saved([{"role": "user", "content": "A shop"}, {"role": "user", "content": "Add Invoice"}])

    merged = merge_documents(repo, ben, ana)

    assert merged["project"]["assistantConversation"] == ben["project"]["assistantConversation"]
