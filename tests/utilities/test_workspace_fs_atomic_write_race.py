"""``write_atomic_plain_bytes`` must not write through a parent swapped for a link after its check.

The temp file and the rename went by path, so a directory replaced by a link between
``_refuse_link_parents`` and the write put the file outside the workspace. Where the
platform opens beneath a directory descriptor (Linux), the swap is now refused.
"""

import os

import pytest

from besser.utilities import workspace_fs

pytestmark = pytest.mark.skipif(not workspace_fs._BENEATH, reason="no O_NOFOLLOW / dir_fd opens here")


def test_a_parent_swapped_for_a_link_after_the_check_is_refused(tmp_path, monkeypatch):
    workspace, outside = tmp_path / "ws", tmp_path / "outside"
    (workspace / "sub").mkdir(parents=True)
    outside.mkdir()
    original = workspace_fs._refuse_link_parents

    def checked_then_swapped(path, root):
        checked = original(path, root)
        os.rename(workspace / "sub", workspace / "sub.real")
        os.symlink(outside, workspace / "sub")
        return checked

    monkeypatch.setattr(workspace_fs, "_refuse_link_parents", checked_then_swapped)

    with pytest.raises(OSError):
        workspace_fs.write_atomic_plain_bytes(workspace / "sub" / "x.json", b"{}", root=workspace)

    assert os.listdir(outside) == []


def test_a_plain_write_still_lands_with_its_mode(tmp_path):
    (tmp_path / "sub").mkdir()
    target = tmp_path / "sub" / "x.json"
    target.write_bytes(b"old")

    workspace_fs.write_atomic_plain_bytes(target, b"new", root=tmp_path, mode=0o644)

    assert target.read_bytes() == b"new"
    assert (target.stat().st_mode & 0o777) == 0o644
    assert os.listdir(tmp_path / "sub") == ["x.json"]
