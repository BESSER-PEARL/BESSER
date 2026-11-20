"""POST /spec-driven/preview reports how a modify's model relates to its base app.

The frontend calls it before a modify run and asks "update the existing app /
start fresh" when the model changed a lot or is another app (#659). No key and
no LLM call: the comparison reads the base run's recipe.
"""
import asyncio
import json
import os
import time

import pytest

from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    SmartRunEntry,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_model_assembly import (
    CLASS_DIAGRAM_MODEL,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_smart_preview import (
    _preview,
    _project,
)
from tests.utilities.web_modeling_editor.backend._runtime import current_runtime

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
        asyncio.run(current_runtime().registry.put(BASE_ID, SmartRunEntry(
            file_path=str(tmp_path / "out.zip"), file_name="out.zip", is_zip=True,
            temp_dir=str(tmp_path), created_at=time.time(),
        )))
    yield _register
    asyncio.run(current_runtime().registry.pop(BASE_ID))


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


def test_preview_says_whether_the_base_can_be_edited(base_run, tmp_path):
    assert _preview(_body()).json()["base_available"] is False  # never registered
    base_run(_classes({"Author": ["name"], "Book": []}))
    assert _preview(_body()).json()["base_available"] is True  # its workspace is there
    body = _body(mode="generate")
    del body["base_run_id"]
    assert _preview(body).json()["base_available"] is None


def test_an_expired_workspace_with_its_download_is_still_a_base(tmp_path):
    import zipfile

    from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import run_download_dir

    workspace = str(tmp_path / f"besser_llm_{BASE_ID}_x")  # swept
    download = run_download_dir(workspace)
    os.makedirs(download)
    artifact = os.path.join(download, "artifact")
    with zipfile.ZipFile(artifact, "w") as zf:
        zf.writestr(".besser_recipe.json",
                    json.dumps({"model": {"user_model": _classes({"Author": ["name"], "Book": []})}}))
        zf.writestr("app/main.py", "")
    entry = SmartRunEntry(file_path=artifact, file_name="out.zip", is_zip=True, temp_dir=workspace,
                          created_at=time.time(), packaged=True)
    asyncio.run(current_runtime().registry.put(BASE_ID, entry))
    try:
        payload = _preview(_body()).json()
        assert payload["base_available"] is True
        assert payload["model_change"]["added"] == ["Book.title"]
        # An uploaded ZIP is still a base, but its run records are not read.
        entry.packaged = False
        asyncio.run(current_runtime().registry.put(BASE_ID, entry))
        payload = _preview(_body()).json()
        assert payload["base_available"] is True and payload["model_change"] is None
        os.remove(artifact)  # the download is gone too
        payload = _preview(_body()).json()
        assert payload["base_available"] is False and payload["model_change"] is None
    finally:
        asyncio.run(current_runtime().registry.pop(BASE_ID))


def test_an_expired_result_without_a_zip_download_is_no_base(tmp_path):
    from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import run_download_dir

    for name, is_zip in (("single", False), ("lost", True)):
        workspace = str(tmp_path / f"besser_llm_{BASE_ID}_{name}")  # swept
        os.makedirs(run_download_dir(workspace))
        artifact = os.path.join(run_download_dir(workspace), "artifact")
        if not is_zip:  # a run whose output was one file
            with open(artifact, "w", encoding="utf-8") as fh:
                fh.write("print('one file')\n")
        # else: the download could not be kept, so there is no artifact
        asyncio.run(current_runtime().registry.put(BASE_ID, SmartRunEntry(
            file_path=artifact, file_name="out", is_zip=is_zip, temp_dir=workspace, created_at=time.time(),
            packaged=is_zip)))
        try:
            assert _preview(_body()).json()["base_available"] is False, name
        finally:
            asyncio.run(current_runtime().registry.pop(BASE_ID))


def test_preview_work_runs_off_the_event_loop(base_run, monkeypatch):
    """Assembly, planning and the base comparison can take seconds on a large
    project; on the loop they stall every SSE stream and /cancel meanwhile."""
    import threading

    from besser.utilities.web_modeling_editor.backend.routers import spec_driven_router

    loop_thread = threading.current_thread()  # asyncio.run drives the app here
    threads: dict[str, threading.Thread] = {}

    def recording(name):
        original = getattr(spec_driven_router, name)

        def wrapper(*args, **kwargs):
            threads[name] = threading.current_thread()
            return original(*args, **kwargs)
        monkeypatch.setattr(spec_driven_router, name, wrapper)

    for name in ("assemble_models_from_project", "build_preview", "compare_with_base"):
        recording(name)
    base_run(_classes({"Book": ["title"]}))

    assert _preview(_body()).status_code == 200
    assert set(threads) == {"assemble_models_from_project", "build_preview", "compare_with_base"}
    assert all(thread is not loop_thread for thread in threads.values()), threads
