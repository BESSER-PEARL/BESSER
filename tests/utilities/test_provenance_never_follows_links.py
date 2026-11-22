"""BESSER_GENERATION.md is written into run workspaces, which generated code
can write to; a link planted at that name must not redirect the write."""
import os

import pytest

from besser.utilities.provenance import PROVENANCE_FILENAME, write_generation_provenance


def test_planted_link_is_not_written_through(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("canary", encoding="utf-8")
    workspace = tmp_path / "run"
    workspace.mkdir()
    try:
        os.symlink(outside, workspace / PROVENANCE_FILENAME)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    write_generation_provenance(str(workspace), "django")
    assert outside.read_text(encoding="utf-8") == "canary"
    written = workspace / PROVENANCE_FILENAME
    assert not written.is_symlink()
    assert "BESSER" in written.read_text(encoding="utf-8")
