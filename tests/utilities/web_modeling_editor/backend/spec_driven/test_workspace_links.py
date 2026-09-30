"""Symlinks in a run workspace never reach a download, a modify seed, or a push.

A workspace is written by model-authored code, so ``ln -s /proc/self/environ
leak.txt`` (or a link to any out-of-tree file or directory) must be dropped,
not read through, on every path that copies the tree out.
"""

import asyncio
import os
import pathlib
import zipfile

import pytest

from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SmartGenerationRunner,
    _seed_workspace_from_base,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.workspace_tree import (
    copytree_no_links,
    is_plain_entry,
)
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

    copytree_no_links(str(src), str(dst), ["*.db"])

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
    _done, entry = runner._package_result(str(result_path))

    assert entry.is_zip
    with zipfile.ZipFile(entry.file_path) as archive:
        names = archive.namelist()
        assert sorted(names) == ["BESSER_GENERATION.md", "app.py", "main.py"]
        assert all(SECRET not in archive.read(n).decode("utf-8") for n in names)


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
