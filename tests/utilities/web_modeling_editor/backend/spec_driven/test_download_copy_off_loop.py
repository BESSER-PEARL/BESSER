"""Keeping a run's download copies the file off the event loop.

``SmartRunRegistry.put`` is awaited on the event loop and copied the artifact
(tens of MB) synchronously, stalling every other request and SSE stream for the
duration of the copy.
"""

from __future__ import annotations

import asyncio
import threading
import time

from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SmartRunEntry,
    SmartRunRegistry,
)


def test_put_copies_the_download_off_the_event_loop(monkeypatch, tmp_path):
    copy_threads: list[int] = []

    def store_download(run_id: str, entry: SmartRunEntry) -> SmartRunEntry:
        copy_threads.append(threading.get_ident())
        return entry

    monkeypatch.setattr(SmartRunRegistry, "_store_download", staticmethod(store_download))
    registry = SmartRunRegistry()
    run_id = "d" * 32
    entry = SmartRunEntry(
        file_path=str(tmp_path / "app.zip"),
        file_name="app.zip",
        is_zip=True,
        temp_dir=str(tmp_path),
        created_at=time.time(),
    )

    async def exercise() -> int:
        await registry.put(run_id, entry)
        return threading.get_ident()

    loop_thread = asyncio.run(exercise())

    assert len(copy_threads) == 1
    assert copy_threads[0] != loop_thread, "the copy ran on the event loop thread"
    assert asyncio.run(registry.get(run_id)) is not None
