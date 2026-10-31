"""Symlinks in a run workspace never reach a download, a modify seed, or a push.

A workspace is written by model-authored code, so ``ln -s /proc/self/environ
leak.txt`` (or a link to any out-of-tree file or directory) must be dropped,
not read through, on every path that copies the tree out.
"""

import asyncio
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

    runner = SmartGenerationRunner(_build_request())
    runner.temp_dir = str(tmp_path)
    done, entry = runner._package_result(str(result_path))

    assert entry.is_zip
    with zipfile.ZipFile(entry.file_path) as archive:
        names = archive.namelist()
        trace = f".agent-trace/{done.runId}.json"
        assert sorted(names) == [trace, "BESSER_GENERATION.md", "app.py", "main.py"]
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

    runner = SmartGenerationRunner(_build_request())
    runner.temp_dir = str(tmp_path)
    done, entry = runner._package_result(str(result_path))

    assert (outside / "secret.txt").read_text(encoding="utf-8") == SECRET
    assert sorted(os.listdir(outside)) == ["secret.txt"]
    trace_dir = result_path / ".agent-trace"
    assert not trace_dir.is_symlink() and (trace_dir / f"{done.runId}.json").is_file()
    with zipfile.ZipFile(entry.file_path) as archive:
        assert f".agent-trace/{done.runId}.json" in archive.namelist()


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

    runner = SmartGenerationRunner(_build_request())
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
