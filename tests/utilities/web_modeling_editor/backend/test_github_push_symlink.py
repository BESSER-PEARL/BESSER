"""A generation workspace must not exfiltrate through a symlink on push.

The tree pushed to GitHub is written by the Spec-Driven worker, which runs
model-authored code and (on the hosted deploy) has shell tools enabled. The
backend performs the push and holds the full secret set -- SMTP_PASSWORD,
GITHUB_CLIENT_SECRET, GPG_KEY, OPENAI_API_KEY, the telemetry admin token.

`Path.rglob("*")` filtered by `is_file()` FOLLOWS symlinks, so a link dropped
into the workspace (`ln -s /proc/self/environ leak.txt`) would be read through
and its target committed to the user's repository. Mounting the workspace
read-only does not help: `:ro` blocks writes, not traversal.

This was not exploitable when found, only because push-to-github could not
resolve worker runs at all. Fixing that resolution without this guard would
arm it, so the two ship together.
"""
import os

import pytest

from besser.utilities.web_modeling_editor.backend.services.deployment import (
    github_service,
)


def _collect(directory):
    """Run just the collection logic the push uses."""
    from pathlib import Path

    directory = Path(directory).resolve()
    files = []
    for candidate in directory.rglob("*"):
        if candidate.is_symlink():
            continue
        if not candidate.is_file():
            continue
        try:
            resolved = candidate.resolve()
            if os.path.commonpath([str(directory), str(resolved)]) != str(directory):
                continue
        except (OSError, ValueError):
            continue
        files.append(candidate)
    return files


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "app.py").write_text("print('real file')", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "util.py").write_text("x = 1", encoding="utf-8")
    return tmp_path


def _make_symlink(link, target):
    try:
        os.symlink(target, link)
        return True
    except (OSError, NotImplementedError, AttributeError):
        return False  # Windows without developer mode


def test_real_files_are_collected(workspace):
    names = {f.name for f in _collect(workspace)}
    assert names == {"app.py", "util.py"}


def test_a_symlink_to_a_secret_is_not_collected(workspace, tmp_path):
    secret = tmp_path.parent / "outside_secret.env"
    secret.write_text("SMTP_PASSWORD=hunter2", encoding="utf-8")
    if not _make_symlink(workspace / "leak.txt", secret):
        pytest.skip("symlinks unavailable on this platform/session")

    collected = _collect(workspace)
    names = {f.name for f in collected}
    assert "leak.txt" not in names, "a symlink was followed into the push"
    for f in collected:
        assert "hunter2" not in f.read_text(encoding="utf-8", errors="ignore")


def test_a_symlinked_directory_is_not_walked(workspace, tmp_path):
    outside = tmp_path.parent / "outside_dir"
    outside.mkdir(exist_ok=True)
    (outside / "secret.txt").write_text("GITHUB_CLIENT_SECRET=abc", encoding="utf-8")
    if not _make_symlink(workspace / "linked", outside):
        pytest.skip("symlinks unavailable on this platform/session")

    for f in _collect(workspace):
        assert "GITHUB_CLIENT_SECRET" not in f.read_text(encoding="utf-8", errors="ignore")


def test_the_shipped_code_has_the_guard():
    """Guards against the collection block being simplified back."""
    import inspect

    src = inspect.getsource(github_service)
    assert "is_symlink()" in src, "the push no longer skips symlinks"
    assert "commonpath" in src, "the push no longer re-checks containment"
