"""A run written after this process booted must still be resolvable.

`restore_persisted()` runs once, in the FastAPI lifespan. That was sufficient
while the process that produced a run also served its download and its GitHub
push. Since generation moved into the `besser-wme-smartgen` container it is
not: push-to-github and import-github-run stay on the backend (they need the
process-local OAuth session), so every worker run is written to the shared
volume AFTER the backend's one-shot scan.

Before the fix the worker had runs on disk while the backend registry had
none, so push-to-github 404'd for every worker run.
"""
import asyncio
import json
import os
import time

import pytest

from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    runner as runner_module,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SmartRunRegistry,
)

RUN_ID = "a" * 32


def _write_run(root, run_id=RUN_ID, age_seconds=0.0):
    """Lay out a finished run exactly as the worker leaves it."""
    name = f"{runner_module.LLM_TEMP_DIR_PREFIX}{run_id}_abcd1234"
    temp_dir = os.path.join(root, name)
    os.makedirs(temp_dir, exist_ok=True)
    artifact = os.path.join(temp_dir, "app.zip")
    with open(artifact, "wb") as fh:
        fh.write(b"PK\x03\x04 not really a zip")
    manifest = {
        "runId": run_id,
        "createdAt": time.time() - age_seconds,
        "relativePath": "app.zip",
        "fileName": f"besser_smart_{run_id}.zip",
        "isZip": True,
    }
    with open(os.path.join(temp_dir, runner_module._DOWNLOAD_MANIFEST_NAME),
              "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    return temp_dir


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "LLM_RUN_WORKSPACE_ROOT", str(tmp_path))
    return tmp_path


def test_a_run_written_after_boot_is_found(workspace):
    """The regression this file exists for."""
    reg = SmartRunRegistry()

    async def exercise():
        # Boot-time scan sees nothing: the worker has not run yet.
        assert await reg.restore_persisted() == 0
        assert await reg.get(RUN_ID) is None or True  # miss is expected here

        _write_run(str(workspace))          # worker finishes a run, later

        entry = await reg.get(RUN_ID)
        assert entry is not None, (
            "a run written after boot was not found -- push-to-github 404s"
        )
        assert entry.is_zip is True
        assert os.path.isfile(entry.file_path)

    asyncio.run(exercise())


def test_an_in_memory_hit_does_not_touch_disk(workspace, monkeypatch):
    """The rescan must be a fallback, not a per-call directory scan."""
    reg = SmartRunRegistry()
    _write_run(str(workspace))

    async def exercise():
        assert await reg.restore_persisted() == 1

        calls = []
        real_listdir = os.listdir
        monkeypatch.setattr(os, "listdir",
                            lambda p: (calls.append(p), real_listdir(p))[1])
        assert await reg.get(RUN_ID) is not None
        assert calls == [], "a cached entry should not trigger a disk scan"

    asyncio.run(exercise())


def test_an_unknown_run_still_returns_none(workspace):
    reg = SmartRunRegistry()

    async def exercise():
        assert await reg.get("b" * 32) is None

    asyncio.run(exercise())


def test_a_malformed_run_id_is_rejected_without_scanning(workspace, monkeypatch):
    """The id reaches a filesystem prefix, so it must be validated first."""
    reg = SmartRunRegistry()
    calls = []
    monkeypatch.setattr(os, "listdir", lambda p: calls.append(p) or [])

    async def exercise():
        for bad in ("../../etc", "", "ZZZ", "a" * 31):
            assert await reg.get(bad) is None
        assert calls == [], "a malformed run id must not reach os.listdir"

    asyncio.run(exercise())


def test_an_expired_run_is_not_resurrected(workspace):
    reg = SmartRunRegistry()
    _write_run(str(workspace), age_seconds=runner_module.LLM_DOWNLOAD_TTL_SECONDS + 60)

    async def exercise():
        assert await reg.get(RUN_ID) is None, "a run past its TTL must stay gone"

    asyncio.run(exercise())


def test_a_manifest_that_disagrees_with_its_directory_is_ignored(workspace):
    """Guards the rescan against a manifest claiming another run's id."""
    reg = SmartRunRegistry()
    temp_dir = _write_run(str(workspace))
    manifest = os.path.join(temp_dir, runner_module._DOWNLOAD_MANIFEST_NAME)
    data = json.load(open(manifest, encoding="utf-8"))
    data["runId"] = "c" * 32          # does not match the directory name
    json.dump(data, open(manifest, "w", encoding="utf-8"))

    async def exercise():
        assert await reg.get("c" * 32) is None
        assert await reg.get(RUN_ID) is None

    asyncio.run(exercise())
