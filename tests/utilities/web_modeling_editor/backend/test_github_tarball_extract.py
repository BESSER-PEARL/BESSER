"""GitHub repo import must not let a small archive fill the disk.

`download_repo_tarball` caps the *compressed* stream at 100 MB, but that
says nothing about expansion -- 100 MB of gzip holds on the order of
100 GB. The extracted tree is then copied again as a run seed and again
into the packaging zip, so the damage is roughly tripled. The extraction
scan therefore also totals the declared uncompressed sizes and counts the
members it keeps, and aborts before `extractall` writes anything.

These tests build the bomb from member metadata and inject a modest limit:
writing a real 100 GB archive would cause the exact outage being prevented.
"""
import io
import os
import tarfile

import pytest

from besser.utilities.web_modeling_editor.backend.services.deployment.github_service import (
    GitHubService,
)


ROOT = "besser-pearl-my-app-abc1234"


def _tarball(entries, *, root=ROOT):
    """Build an in-memory gzip tarball shaped like a GitHub tarball.

    ``entries`` maps a repo-relative path to its bytes. Everything is
    nested under a single ``{owner}-{repo}-{sha}/`` wrapper directory,
    which extraction is expected to strip.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=1) as tar:
        wrapper = tarfile.TarInfo(root)
        wrapper.type = tarfile.DIRTYPE
        tar.addfile(wrapper)
        for path, data in entries.items():
            info = tarfile.TarInfo(f"{root}/{path}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


def _extract(buf, dest, **caps):
    with tarfile.open(fileobj=buf, mode="r:*") as tar:
        GitHubService._extract_tarball_stripping_root(tar, str(dest), **caps)


def _tree(dest):
    return sorted(
        os.path.relpath(os.path.join(base, name), dest).replace("\\", "/")
        for base, _dirs, files in os.walk(dest)
        for name in files
    )


class TestUncompressedSizeCap:
    def test_expansion_past_the_cap_is_refused(self, tmp_path):
        # 3 MB of declared content against a 1 MB uncompressed ceiling.
        buf = _tarball({f"data/blob{i}.bin": b"\0" * (1024 * 1024) for i in range(3)})
        dest = tmp_path / "out"
        dest.mkdir()

        with pytest.raises(ValueError) as exc:
            _extract(buf, dest, max_extracted_bytes=1024 * 1024)

        message = str(exc.value)
        assert "1 MB" in message                      # names the limit
        assert "blob" in message                      # names what tripped it
        assert "import again" in message              # tells the user what to do
        assert "assert" not in message.lower()

    def test_nothing_is_written_when_the_cap_trips(self, tmp_path):
        # The abort must happen during the member scan, not mid-extractall:
        # a partially written tree is still disk we cannot get back.
        buf = _tarball(
            {
                "keep.txt": b"a" * (600 * 1024),
                "data/blob.bin": b"\0" * (600 * 1024),
            }
        )
        dest = tmp_path / "out"
        dest.mkdir()

        with pytest.raises(ValueError):
            _extract(buf, dest, max_extracted_bytes=1024 * 1024)

        assert _tree(dest) == []

    def test_archive_under_the_cap_still_extracts(self, tmp_path):
        buf = _tarball({"small.bin": b"\0" * 1024})
        dest = tmp_path / "out"
        dest.mkdir()

        _extract(buf, dest, max_extracted_bytes=1024 * 1024)

        assert _tree(dest) == ["small.bin"]


class TestMemberCountCap:
    def test_member_explosion_is_refused(self, tmp_path):
        # Empty files weigh nothing against the byte cap but still cost an
        # inode each, and every copy step walks them.
        buf = _tarball({f"f{i}.txt": b"" for i in range(50)})
        dest = tmp_path / "out"
        dest.mkdir()

        with pytest.raises(ValueError) as exc:
            _extract(buf, dest, max_members=10)

        message = str(exc.value)
        assert "10" in message
        assert "import again" in message
        assert _tree(dest) == []

    def test_member_count_under_the_cap_still_extracts(self, tmp_path):
        buf = _tarball({f"f{i}.txt": b"x" for i in range(5)})
        dest = tmp_path / "out"
        dest.mkdir()

        _extract(buf, dest, max_members=10)

        assert len(_tree(dest)) == 5


class TestLegitimateImportIsUnchanged:
    def test_normal_project_extracts_with_the_root_stripped(self, tmp_path):
        buf = _tarball(
            {
                "README.md": b"# My App\n",
                "backend/main_api.py": b"print('hi')\n",
                "frontend/src/App.tsx": b"export default App;\n",
                "buml/diagrams.json": b'{"id": "p1"}',
            }
        )
        dest = tmp_path / "out"
        dest.mkdir()

        _extract(buf, dest)

        assert _tree(dest) == [
            "README.md",
            "backend/main_api.py",
            "buml/diagrams.json",
            "frontend/src/App.tsx",
        ]
        # The wrapper directory is gone, not nested one level down.
        assert not (dest / ROOT).exists()
        assert (dest / "buml" / "diagrams.json").read_bytes() == b'{"id": "p1"}'

    def test_caps_do_not_contradict_the_download_cap(self):
        # A ceiling below the compressed cap would reject archives the
        # download already accepted.
        assert (
            GitHubService._DEFAULT_MAX_EXTRACTED_BYTES
            > GitHubService._DEFAULT_MAX_ARCHIVE_BYTES
        )
        assert GitHubService._DEFAULT_MAX_ARCHIVE_MEMBERS > 0


class TestPreExistingGuardsStillHold:
    """Regression cover for the guards already in the loop being edited."""

    def test_path_traversal_is_rejected(self, tmp_path):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=1) as tar:
            info = tarfile.TarInfo(f"{ROOT}/../escaped.txt")
            info.size = 3
            tar.addfile(info, io.BytesIO(b"bad"))
        buf.seek(0)
        dest = tmp_path / "out"
        dest.mkdir()

        with pytest.raises(ValueError, match="Unsafe path"):
            _extract(buf, dest)

        assert not (tmp_path / "escaped.txt").exists()

    def test_symlink_members_are_skipped(self, tmp_path):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=1) as tar:
            real = tarfile.TarInfo(f"{ROOT}/app.py")
            real.size = 2
            tar.addfile(real, io.BytesIO(b"x\n"))
            link = tarfile.TarInfo(f"{ROOT}/leak.txt")
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            tar.addfile(link)
        buf.seek(0)
        dest = tmp_path / "out"
        dest.mkdir()

        _extract(buf, dest)

        assert _tree(dest) == ["app.py"]
