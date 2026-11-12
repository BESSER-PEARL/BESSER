"""Symlinks in a run workspace never reach a download, a modify seed, or a push.

A workspace is written by model-authored code, so ``ln -s /proc/self/environ
leak.txt`` (or a link to any out-of-tree file or directory) must be dropped,
not read through, on every path that copies the tree out.
"""

import asyncio
import contextlib
import json
import os
import pathlib
import zipfile

import pytest

from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SmartGenerationRunner,
    _seed_workspace_from_base,
)
from besser.utilities.workspace_fs import copytree_plain, is_plain_entry
from tests.utilities.web_modeling_editor.backend.spec_driven import (
    test_push_smart_to_github as push_tests,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import (
    _build_request,
    _clear_registry,
)
from tests.utilities.web_modeling_editor.backend._runtime import current_runtime

SECRET = "TOP-SECRET-OUTSIDE-THE-TREE"


@pytest.fixture(autouse=True)
def reset_registry():
    asyncio.run(_clear_registry())
    yield
    asyncio.run(_clear_registry())


def _symlink(target, link, *, is_dir=False):
    try:
        os.symlink(str(target), str(link), target_is_directory=is_dir)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"cannot create symlinks here: {exc}")


def _outside(tmp_path):
    """An out-of-tree file and directory the workspace links point at."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text(SECRET, encoding="utf-8")
    return outside


def _add_links(workspace, outside):
    _symlink(outside / "secret.txt", workspace / "leak.txt")
    _symlink(outside, workspace / "linked_dir", is_dir=True)


def _all_files(root):
    found = []
    for base, _dirs, names in os.walk(root):
        for name in names:
            found.append(os.path.relpath(os.path.join(base, name), root).replace("\\", "/"))
    return sorted(found)


def test_is_plain_entry(tmp_path):
    outside = _outside(tmp_path)
    _symlink(outside / "secret.txt", tmp_path / "link")
    assert is_plain_entry(str(outside / "secret.txt"))
    assert is_plain_entry(str(outside))
    assert not is_plain_entry(str(tmp_path / "link"))
    assert not is_plain_entry(str(tmp_path / "missing"))


def test_copytree_no_links_drops_links_and_applies_patterns(tmp_path):
    outside = _outside(tmp_path)
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (src / "app.db").write_text("db", encoding="utf-8")
    _add_links(src, outside)
    _symlink(outside / "secret.txt", src / "pkg" / "nested_leak.txt")
    dst = tmp_path / "dst"
    dst.mkdir()

    copytree_plain(str(src), str(dst), ["*.db"], dirs_exist_ok=True)

    assert _all_files(dst) == ["pkg/mod.py"]


def test_download_zip_skips_symlinked_file_and_dir(tmp_path):
    outside = _outside(tmp_path)
    result_path = tmp_path / "result"
    result_path.mkdir()
    (result_path / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (result_path / "app.py").write_text("app = 1\n", encoding="utf-8")
    _add_links(result_path, outside)

    runner = SmartGenerationRunner(_build_request(), runtime=current_runtime())
    runner.temp_dir = str(tmp_path)
    done, entry = runner._package_result(str(result_path))

    assert entry.is_zip
    with zipfile.ZipFile(entry.file_path) as archive:
        names = archive.namelist()
        trace = f".agent-trace/{done.runId}.json"
        assert sorted(names) == [trace, ".besser_attribution.json", "BESSER_GENERATION.md", "app.py", "main.py"]
        # Linked files are not attributed either.
        assert {f["path"] for f in json.loads(archive.read(trace))["files"]} == {"app.py", "main.py"}
        assert all(SECRET not in archive.read(n).decode("utf-8") for n in names)


def test_attribution_writes_never_follow_a_planted_link(tmp_path):
    # The harness writes the attribution state and the Agent Trace record
    # unsandboxed; a link the model planted at either must not redirect it.
    outside = _outside(tmp_path)
    result_path = tmp_path / "result"
    result_path.mkdir()
    (result_path / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (result_path / "app.py").write_text("app = 1\n", encoding="utf-8")
    _symlink(outside / "secret.txt", result_path / ".besser_attribution.json.tmp")
    _symlink(outside, result_path / ".agent-trace", is_dir=True)

    runner = SmartGenerationRunner(_build_request(), runtime=current_runtime())
    runner.temp_dir = str(tmp_path)
    done, entry = runner._package_result(str(result_path))

    assert (outside / "secret.txt").read_text(encoding="utf-8") == SECRET
    assert sorted(os.listdir(outside)) == ["secret.txt"]
    trace_dir = result_path / ".agent-trace"
    assert not trace_dir.is_symlink() and (trace_dir / f"{done.runId}.json").is_file()
    with zipfile.ZipFile(entry.file_path) as archive:
        assert f".agent-trace/{done.runId}.json" in archive.namelist()


def test_packaging_never_writes_or_reads_through_a_planted_link(tmp_path):
    # The download manifest's temp file, the result ZIP and the recipe sit at
    # names the model can predict; a link it planted at any of them must not
    # make the unsandboxed harness overwrite (or read) the link's target.
    outside = _outside(tmp_path)
    victims = {}
    for name in ("manifest_tmp", "zip", "recipe"):
        victims[name] = outside / f"{name}.txt"
        victims[name].write_text(json.dumps({"other_run_secret": SECRET}), encoding="utf-8")
    workspace = tmp_path / "besser_llm_ws"
    workspace.mkdir()
    (workspace / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (workspace / "app.py").write_text("app = 1\n", encoding="utf-8")
    runner = SmartGenerationRunner(_build_request(), runtime=current_runtime())
    _symlink(victims["manifest_tmp"], workspace / ".besser_download.json.tmp")
    _symlink(victims["zip"], workspace / f"besser_smart_{runner.run_id}.zip")
    _symlink(victims["recipe"], workspace / ".besser_recipe.json")

    runner.temp_dir = str(workspace)
    done, entry = runner._package_result(str(workspace))
    asyncio.run(current_runtime().registry.put(runner.run_id, entry))

    assert "other_run_secret" not in done.recipe
    for victim in victims.values():
        assert json.loads(victim.read_text(encoding="utf-8")) == {"other_run_secret": SECRET}
    with zipfile.ZipFile(entry.file_path) as archive:
        assert sorted(n for n in archive.namelist() if not n.startswith((".", "BESSER"))) == ["app.py", "main.py"]


def test_a_file_swapped_for_a_link_after_the_walk_never_reaches_the_zip(tmp_path, monkeypatch):
    # The run's shell outlives Phase 3, so a background job can swap a file
    # for a link between the walk that listed it and the ZIP writer reading it.
    outside = _outside(tmp_path)
    workspace = tmp_path / "besser_llm_ws"
    workspace.mkdir()
    (workspace / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (workspace / "app.py").write_text("app = 1\n", encoding="utf-8")
    from besser.utilities import workspace_fs
    from besser.utilities.web_modeling_editor.backend.services.spec_driven import runner as runner_module
    real_check = workspace_fs._refuse_unless_plain_file

    def check_then_swap(path, root):
        real_check(path, root)
        if path.endswith("app.py"):
            os.remove(path)
            _symlink(outside / "secret.txt", path)

    def swap_while_zipping(*args, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(workspace_fs, "_refuse_unless_plain_file", check_then_swap)
            return real_zip(*args, **kwargs)

    real_zip = runner_module._zip_plain_file
    monkeypatch.setattr(runner_module, "_zip_plain_file", swap_while_zipping)
    runner = SmartGenerationRunner(_build_request(), runtime=current_runtime())
    runner.temp_dir = str(workspace)
    _done, entry = runner._package_result(str(workspace))

    with zipfile.ZipFile(entry.file_path) as archive:
        assert "app.py" not in archive.namelist()
        assert all(SECRET not in archive.read(n).decode("utf-8") for n in archive.namelist())


def test_seed_and_push_copy_never_follow_a_link_swapped_in_after_the_check(tmp_path, monkeypatch):
    from besser.utilities import workspace_fs

    outside = _outside(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("print('hi')\n", encoding="utf-8")
    real_check = workspace_fs.is_plain_entry

    def check_then_swap(path):
        plain = real_check(path)
        if path.endswith("main.py") and plain:
            os.remove(path)
            _symlink(outside / "secret.txt", path)
        return plain

    monkeypatch.setattr(workspace_fs, "is_plain_entry", check_then_swap)
    dst = tmp_path / "dst"
    with contextlib.suppress(OSError):  # shutil.Error: the swapped file is refused
        copytree_plain(str(src), str(dst))

    assert not (dst / "main.py").exists() or SECRET not in (dst / "main.py").read_text(encoding="utf-8")


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="POSIX link races")
def test_a_copy_never_sets_mode_or_times_through_a_link_swapped_in_after_the_write(tmp_path, monkeypatch):
    from besser.utilities import workspace_fs

    victim = tmp_path / "victim.txt"
    victim.write_text("server file\n", encoding="utf-8")
    victim.chmod(0o600)
    victim_mtime = victim.stat().st_mtime_ns
    src = tmp_path / "src.txt"
    src.write_text("payload\n", encoding="utf-8")
    src.chmod(0o777)
    os.utime(src, ns=(0, 0))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    dst = workspace / "copy.txt"
    real_copy = workspace_fs.shutil.copyfileobj

    def copy_then_swap(source, target):
        real_copy(source, target)
        # The run's background process swaps the fresh copy for a link.
        os.remove(dst)
        _symlink(victim, dst)

    monkeypatch.setattr(workspace_fs.shutil, "copyfileobj", copy_then_swap)
    with contextlib.suppress(OSError):
        workspace_fs.copy_plain_file(str(src), str(dst), dst_root=str(workspace))

    assert victim.stat().st_mode & 0o777 == 0o600
    assert victim.stat().st_mtime_ns == victim_mtime


def test_a_linked_attribution_state_is_replaced_not_written_through(tmp_path):
    from besser.spec_driven_agent.state import agent_trace as at

    outside = _outside(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "main.py").write_text("x = 1\n", encoding="utf-8")
    _symlink(outside / "secret.txt", workspace / at.ATTRIBUTION_FILENAME)

    with pytest.raises(at.AttributionStateError):       # never read through
        at.AttributionState.load(str(workspace))
    at.record_generated_files(str(workspace), ["main.py"], "backend")
    at.AttributionState().save(str(workspace))

    assert (outside / "secret.txt").read_text(encoding="utf-8") == SECRET
    assert not (workspace / at.ATTRIBUTION_FILENAME).is_symlink()


def test_single_file_download_is_never_a_symlink(tmp_path):
    outside = _outside(tmp_path)
    result_path = tmp_path / "result"
    result_path.mkdir()
    (result_path / "main.py").write_text("print('hi')\n", encoding="utf-8")
    _symlink(outside / "secret.txt", result_path / "leak.txt")

    runner = SmartGenerationRunner(_build_request(), runtime=current_runtime())
    runner.temp_dir = str(tmp_path)
    _done, entry = runner._package_result(str(result_path))

    assert not entry.is_zip
    assert entry.file_name == "main.py"


def test_modify_seed_skips_symlinks(tmp_path):
    outside = _outside(tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    (base / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (base / ".besser_recipe.json").write_text("{}", encoding="utf-8")
    _add_links(base, outside)
    dest = tmp_path / "dest"
    dest.mkdir()

    _seed_workspace_from_base(str(base), str(dest))

    assert _all_files(dest) == [".besser_recipe.json", "main.py"]


def test_push_to_github_skips_symlinks(tmp_path, monkeypatch):
    outside = _outside(tmp_path)
    run_id = "e" * 32
    workspace = push_tests._seed_run(run_id, with_secret_env=False)
    _add_links(pathlib.Path(workspace), outside)
    # The recipe is copied separately; a linked recipe must not be read through.
    os.remove(os.path.join(workspace, ".besser_recipe.json"))
    _symlink(outside / "secret.txt", os.path.join(workspace, ".besser_recipe.json"))
    fake = push_tests._FakeGitHubService()
    push_tests._install(monkeypatch, fake)

    r = push_tests._post(
        {
            "run_id": run_id,
            "projectExport": push_tests._project_export(),
            "deploy_config": {"repo_name": "my-generated-app"},
        },
        headers={"X-GitHub-Session": "sess"},
    )

    assert r.status_code == 200, r.text
    pushed = fake.push["files"]
    assert "main.py" in pushed
    assert "leak.txt" not in pushed
    assert ".besser_recipe.json" not in pushed
    assert not any(p.startswith("linked_dir/") for p in pushed)


def test_a_linked_carried_file_never_reaches_the_zip(tmp_path):
    """BESSER's own records ride in the ZIP (#659); one planted as a link must not leak its target."""
    outside = _outside(tmp_path)
    result_path = tmp_path / "result"
    result_path.mkdir()
    (result_path / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (result_path / "app.py").write_text("app = 1\n", encoding="utf-8")
    _symlink(outside / "secret.txt", result_path / ".besser_generator_output.json")

    runner = SmartGenerationRunner(_build_request(), runtime=current_runtime())
    runner.temp_dir = str(tmp_path)
    _, entry = runner._package_result(str(result_path))

    with zipfile.ZipFile(entry.file_path) as archive:
        assert ".besser_generator_output.json" not in archive.namelist()
        assert all(SECRET not in archive.read(n).decode("utf-8") for n in archive.namelist())
