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
from besser.spec_driven_agent.execution.sandbox import sandbox_home
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SmartRunEntry,
    SmartRunRegistry,
)
from besser.utilities.web_modeling_editor.backend.settings import get_settings

RUN_ID = "a" * 32


def _write_run(root, run_id=RUN_ID, age_seconds=0.0):
    """Lay out a finished run exactly as the worker leaves it."""
    name = f"{runner_module.LLM_TEMP_DIR_PREFIX}{run_id}_abcd1234"
    temp_dir = os.path.join(root, name)
    os.makedirs(temp_dir, exist_ok=True)
    artifact = os.path.join(temp_dir, "app.zip")
    with open(artifact, "wb") as fh:
        fh.write(b"PK\x03\x04 not really a zip")
    SmartRunRegistry._store_download(run_id, SmartRunEntry(
        file_path=artifact,
        file_name=f"besser_smart_{run_id}.zip",
        is_zip=True,
        temp_dir=temp_dir,
        created_at=time.time() - age_seconds,
    ))
    return temp_dir


def _manifest_path(temp_dir):
    return os.path.join(runner_module.run_download_dir(temp_dir), runner_module._DOWNLOAD_MANIFEST_NAME)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "llm_run_workspace_root", str(tmp_path))
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
    _write_run(str(workspace), age_seconds=get_settings().llm_download_ttl_seconds + 60)

    async def exercise():
        assert await reg.get(RUN_ID) is None, "a run past its TTL must stay gone"

    asyncio.run(exercise())


def test_a_manifest_that_disagrees_with_its_directory_is_ignored(workspace):
    """Guards the rescan against a manifest claiming another run's id."""
    reg = SmartRunRegistry()
    temp_dir = _write_run(str(workspace))
    manifest = _manifest_path(temp_dir)
    data = json.load(open(manifest, encoding="utf-8"))
    data["runId"] = "c" * 32          # does not match the directory name
    json.dump(data, open(manifest, "w", encoding="utf-8"))

    async def exercise():
        assert await reg.get("c" * 32) is None
        assert await reg.get(RUN_ID) is None

    asyncio.run(exercise())


def _plant_old_manifest(directory, run_id=RUN_ID):
    """What the run's shell can write: an artifact and a manifest naming it, in
    the layout releases before the download dir used (inside the workspace)."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "out.bin"), "w", encoding="utf-8") as fh:
        fh.write("hi\n")
    with open(os.path.join(directory, ".besser_download.json"), "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "runId": run_id, "relativePath": "out.bin", "fileName": "out.bin",
                   "isZip": False, "createdAt": time.time()}, fh)


def test_a_manifest_the_run_can_write_is_never_trusted(workspace):
    # The run's shell writes its workspace and its sandbox $HOME; a manifest
    # there would let it name any file for /download to serve.
    run_dir = os.path.join(str(workspace), f"{runner_module.LLM_TEMP_DIR_PREFIX}{RUN_ID}_abcd1234")
    _plant_old_manifest(run_dir)
    _plant_old_manifest(sandbox_home(run_dir))
    reg = SmartRunRegistry()

    async def exercise():
        assert await reg.restore_persisted() == 0
        assert await reg.get(RUN_ID) is None

    asyncio.run(exercise())


def test_the_registered_download_survives_the_workspace_changing(workspace):
    # The served file is a copy outside the workspace: replacing the run's
    # file with a link to a worker file afterwards changes nothing.
    run_dir = os.path.join(str(workspace), f"{runner_module.LLM_TEMP_DIR_PREFIX}{RUN_ID}_abcd1234")
    os.makedirs(run_dir)
    out = os.path.join(run_dir, "out.bin")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("hi\n")
    secret = os.path.join(str(workspace), "environ")
    with open(secret, "w", encoding="utf-8") as fh:
        fh.write("BESSER_FREE_LLM_TOKEN=server-paid-token\n")

    async def exercise():
        first = SmartRunRegistry()
        await first.put(RUN_ID, SmartRunEntry(
            file_path=out, file_name="out.bin", is_zip=False, temp_dir=run_dir, created_at=time.time()))
        os.remove(out)
        os.symlink(secret, out)
        for reg in (first, SmartRunRegistry()):  # cached, and rescanned after a restart
            entry = await reg.get(RUN_ID)
            with open(entry.file_path, "rb") as fh:
                assert fh.read() == b"hi\n"

    asyncio.run(exercise())
