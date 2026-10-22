"""POST /spec-driven/preview reports how a modify's model relates to its base app.

The frontend calls it before a modify run and asks "update the existing app /
start fresh" when the model changed a lot or is another app (#659). No key and
no LLM call: the comparison reads the base run's recipe.
"""
import asyncio
import json
import time

import pytest

from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    SMART_RUN_REGISTRY,
    SmartRunEntry,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_model_assembly import (
    CLASS_DIAGRAM_MODEL,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_smart_preview import (
    _preview,
    _project,
)

BASE_ID = "b" * 32


def _classes(attributes: dict) -> dict:
    return {"classes": {
        cls: {"attributes": {a: {"type": "str", "optional": False} for a in attrs}}
        for cls, attrs in attributes.items()
    }}


@pytest.fixture
def base_run(tmp_path):
    def _register(user_model: dict) -> None:
        (tmp_path / ".besser_recipe.json").write_text(
            json.dumps({"model": {"user_model": user_model}}), encoding="utf-8")
        asyncio.run(SMART_RUN_REGISTRY.put(BASE_ID, SmartRunEntry(
            file_path=str(tmp_path / "out.zip"), file_name="out.zip", is_zip=True,
            temp_dir=str(tmp_path), created_at=time.time(),
        )))
    yield _register
    asyncio.run(SMART_RUN_REGISTRY.pop(BASE_ID))


def _body(mode="modify"):
    return {
        "project": _project({"ClassDiagram": [{"id": "cd1", "title": "Library", "model": CLASS_DIAGRAM_MODEL}]}),
        "instructions": "Update the app",
        "mode": mode,
        "base_run_id": BASE_ID,
    }


def test_small_change_lists_it(base_run):
    base_run(_classes({"Author": ["name"], "Book": []}))
    change = _preview(_body()).json()["model_change"]
    assert change["verdict"] == "small" and change["added"] == ["Book.title"]
    assert change["changes"] == ["Book.title: str added"] and change["renamed"] == []


def test_renames_are_reported_structured(base_run):
    # Same name ignoring case pairs the elements, so it reads as a rename.
    base_run(_classes({"author": ["Name"], "Book": ["title"]}))
    change = _preview(_body()).json()["model_change"]
    assert change["renamed"] == [{"from": "author", "to": "Author"},
                                 {"from": "author.Name", "to": "Author.name"}]
    assert change["removed"] == [] and change["added"] == []
    assert change["changes"] == ["class author renamed to Author",
                                 "author.Name renamed to Author.name"]


def test_unrelated_base_model(base_run):
    base_run(_classes({"Guest": ["email"], "Room": ["number"]}))
    change = _preview(_body()).json()["model_change"]
    assert change["verdict"] == "unrelated"
    assert set(change["removed"]) == {"Guest", "Guest.email", "Room", "Room.number"}
    assert "class Guest removed" in change["changes"]


def test_unknown_base_or_generate_mode_reports_none(base_run):
    assert _preview(_body()).json()["model_change"] is None  # base run not registered
    base_run(_classes({"Guest": ["email"]}))
    body = _body(mode="generate")
    del body["base_run_id"]
    resp = _preview(body)
    assert resp.status_code == 200 and resp.json()["model_change"] is None
