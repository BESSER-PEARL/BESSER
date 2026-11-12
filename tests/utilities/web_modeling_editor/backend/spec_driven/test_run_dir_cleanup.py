"""Deleting a run also deletes the sandbox $HOME beside it (cargo target, npm
and pip caches), which otherwise waits for the 24 h sweep on a small disk."""
import os

from besser.spec_driven_agent.execution.sandbox import sandbox_home
from besser.utilities.web_modeling_editor.backend.services.spec_driven import runner


def test_run_dir_removal_takes_the_sandbox_home(tmp_path):
    run_dir = tmp_path / "besser_llm_abc_1"
    run_dir.mkdir()
    home = sandbox_home(str(run_dir))
    import os
    os.makedirs(os.path.join(home, ".cache"))
    runner._remove_run_dir(str(run_dir))
    assert not run_dir.exists()
    assert not os.path.exists(home)


def test_registry_sweeper_removes_run_dirs_off_the_event_loop(tmp_path, monkeypatch):
    """Removing a large workspace or closing a busy shell session takes
    seconds; on the loop it stalls every SSE stream meanwhile."""
    import asyncio
    import threading
    import time

    loop_thread = threading.current_thread()  # asyncio.run drives the sweeper here
    removed_on: list[threading.Thread] = []
    swept = threading.Event()

    def recording_remove(path):
        removed_on.append(threading.current_thread())
        swept.set()

    monkeypatch.setattr(runner, "_remove_run_dir", recording_remove)

    async def exercise():
        registry = runner.SmartRunRegistry()
        await registry.put("e" * 32, runner.SmartRunEntry(
            file_path=str(tmp_path / "out.zip"), file_name="out.zip", is_zip=True,
            temp_dir=str(tmp_path), created_at=time.time() - 3600,
        ))
        task = asyncio.create_task(registry.periodic_sweep(ttl_seconds=1, interval_seconds=0))
        while not swept.is_set():
            await asyncio.sleep(0.01)
        task.cancel()

    asyncio.run(asyncio.wait_for(exercise(), 5))
    assert removed_on and removed_on[0] is not loop_thread


def test_resume_never_takes_the_sandbox_home_for_the_workspace(tmp_path, monkeypatch):
    # The sandbox $HOME shares the run dir's name prefix and the run's shell
    # writes it: a checkpoint planted there must not make it the resumed run.
    from besser.spec_driven_agent.state.checkpoint import CHECKPOINT_FILENAME
    from besser.utilities.web_modeling_editor.backend.settings import get_settings

    monkeypatch.setattr(get_settings(), "llm_run_workspace_root", str(tmp_path))
    run_id = "ef" * 16
    run_dir = tmp_path / f"besser_llm_{run_id}_pg2wfopn"
    run_dir.mkdir()
    for planted in (sandbox_home(str(run_dir)), runner.run_download_dir(str(run_dir))):
        os.makedirs(planted)
        with open(os.path.join(planted, CHECKPOINT_FILENAME), "w", encoding="utf-8") as fh:
            fh.write("{}")

    assert runner._locate_run_temp_dir(run_id) is None
    (run_dir / CHECKPOINT_FILENAME).write_text("{}", encoding="utf-8")
    assert runner._locate_run_temp_dir(run_id) == str(run_dir)
