"""Deleting a run also deletes the sandbox $HOME beside it (cargo target, npm
and pip caches), which otherwise waits for the 24 h sweep on a small disk."""
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
