"""A ``{{N.id}}`` reference to a create response that wraps the row resolves through it.

Runs r11-web1 t4, r11-p11 t6, r11-p21 t10, r9-p21 t4: the generated create answers
``{"todolist": {"id": 1, ...}, "items_ids": []}``, the scenario wrote ``{{0.id}}``
and lost a turn to "did you mean 'todolist.id'?". When exactly one dict wrapper
holds the path (the condition the hint already used), the reference now resolves
through it and the response carries a note naming the real path. ``expected_fields``
stay strict, and no guess is made between two holders, through a list, or for a
field no wrapper holds.
"""
from __future__ import annotations

import pytest

from besser.spec_driven_agent.validation import api_probe


def test_a_reference_resolves_through_the_one_wrapper_and_says_so():
    notes: list[str] = []
    resolved = api_probe._resolve_references(
        {"list": "{{0.id}}", "path": "/todolist/{{0.id}}/"},
        [{"todolist": {"id": 7, "name": "Home"}, "items_ids": []}], notes=notes)
    assert resolved == {"list": 7, "path": "/todolist/7/"}
    assert notes == ["`{{0.id}}` resolved as `{{0.todolist.id}}`"] * 2


def test_an_embedded_path_reference_resolves_too():
    notes: list[str] = []
    assert api_probe._resolve_references(
        "/todoitem/{{0.id}}/", [{"todoitem": {"id": 3}}], in_path=True, notes=notes) == "/todoitem/3/"
    assert notes == ["`{{0.id}}` resolved as `{{0.todoitem.id}}`"]


@pytest.mark.parametrize("body", [
    {"meta": {"id": "req-7"}, "data": {"id": 1}},        # two wrappers hold it
    {"page": {"items": [{"id": 1}]}},                    # only a list below
    {"items": [{"id": 1}], "total": 1},                  # a list wrapper
    {"todolist": {"name": "Home"}, "items_ids": []},     # genuinely missing
    [{"id": 1}],                                         # a list body
])
def test_no_guess_where_the_hint_would_not_guess(body):
    with pytest.raises(ValueError, match="was not present"):
        api_probe._resolve_references("{{0.id}}", [body], notes=[])


def test_an_existing_path_needs_no_note():
    notes: list[str] = []
    assert api_probe._resolve_references("{{0.todolist.id}}", [{"todolist": {"id": 7}}], notes=notes) == 7
    assert notes == []


pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def test_a_scenario_passes_with_a_note_and_expected_fields_stay_strict(tmp_path):
    from tests.spec_driven_agent.test_constructibility_probe import _scaffold

    workspace = _scaffold(tmp_path)
    room = {"roomNumber": 101, "status": "AVAILABLE"}
    report = api_probe.probe_api_scenario(workspace, [
        {"method": "POST", "path": "/room/", "json": room},
        {"method": "GET", "path": "/room/{{0.id}}/", "expected_fields": {"room.roomNumber": 101}},
    ])
    assert report["status"] == "passed", report
    assert report["responses"][1]["notes"] == ["`{{0.id}}` resolved as `{{0.room.id}}`"]

    strict = api_probe.probe_api_scenario(workspace, [
        {"method": "POST", "path": "/room/", "json": room, "expected_fields": {"id": 1}},
    ])
    assert strict["status"] == "failed"
    assert "did you mean 'room.id'?" in strict["assertion_failures"][0]["message"]
