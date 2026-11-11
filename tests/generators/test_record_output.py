"""``record_output`` and the previews: the record without a merge, and runs that write nothing.

Recording needs no pygit2, so only the tests that merge skip without it.
"""
import importlib.util
from pathlib import Path

import pytest

from besser.generators import GeneratorInterface, regeneration
from besser.generators.regeneration import (
    CHANGED,
    NEW,
    RECORD_FILENAME,
    UNCHANGED,
    NotRecordableError,
    RegenerationError,
    UnrecordedOutputError,
    preview_output,
    preview_regeneration,
    record_output,
)
from besser.utilities import three_way_merge

needs_pygit2 = pytest.mark.skipif(importlib.util.find_spec("pygit2") is None, reason="the merge needs pygit2")

APP = b"def a():\n    return 1\n"
LOGO = b"\x89PNG\r\n\x1a\n\x00\x00"


class FilesGenerator(GeneratorInterface):
    """Writes the files it is given."""

    regenerable = True

    def __init__(self, files, output_dir=None, refusal=None):
        super().__init__(None, output_dir)
        self.files = files
        self.refusal = refusal

    def generate(self):
        for rel, content in self.files.items():
            path = Path(self.build_generation_path(rel))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    def _regeneration_refusal(self):
        return self.refusal


class UnstableGenerator(FilesGenerator):
    regenerable = False


def _tree(folder: Path) -> dict[str, bytes]:
    return {path.relative_to(folder).as_posix(): path.read_bytes() for path in folder.rglob("*") if path.is_file()}


@pytest.fixture
def without_pygit2(monkeypatch):
    monkeypatch.setattr(three_way_merge, "pygit2", None)
    monkeypatch.setattr(regeneration, "merge_available", lambda: False)


def test_it_writes_the_output_and_its_record_without_pygit2(tmp_path, without_pygit2):
    out = tmp_path / "out"

    written = record_output(FilesGenerator({"app.txt": APP, "static/logo.png": LOGO}, str(out)))

    assert written == ["app.txt", "static/logo.png"]
    assert (out / "app.txt").read_bytes() == APP and (out / "static" / "logo.png").read_bytes() == LOGO
    assert (out / RECORD_FILENAME).is_file()


def test_the_record_is_the_same_bytes_for_the_same_output(tmp_path, without_pygit2):
    record_output(FilesGenerator({"app.txt": APP}, str(tmp_path / "a")))
    record_output(FilesGenerator({"app.txt": APP}, str(tmp_path / "b")))

    assert (tmp_path / "a" / RECORD_FILENAME).read_bytes() == (tmp_path / "b" / RECORD_FILENAME).read_bytes()


@needs_pygit2
def test_the_record_is_the_one_a_first_regenerate_writes(tmp_path):
    files = {"app.txt": APP, "static/logo.png": LOGO}
    record_output(FilesGenerator(files, str(tmp_path / "recorded")))
    FilesGenerator(files, str(tmp_path / "regenerated")).regenerate()

    assert _tree(tmp_path / "recorded") == _tree(tmp_path / "regenerated")


@needs_pygit2
def test_a_recorded_folder_regenerates_without_adopting(tmp_path):
    out = tmp_path / "out"
    record_output(FilesGenerator({"app.txt": APP}, str(out)))
    (out / "app.txt").write_bytes(b"# mine\n" + APP)

    report = FilesGenerator({"app.txt": APP + b"\n\ndef b():\n    return 2\n"}, str(out)).regenerate()

    assert report.merged == ["app.txt"]
    assert (out / "app.txt").read_bytes().startswith(b"# mine\n")


def test_it_overwrites_generated_files_and_leaves_the_others(tmp_path, without_pygit2):
    out = tmp_path / "out"
    out.mkdir()
    (out / "app.txt").write_bytes(b"old\n")
    (out / "notes.txt").write_bytes(b"mine\n")

    record_output(FilesGenerator({"app.txt": APP}, str(out)))

    assert (out / "app.txt").read_bytes() == APP and (out / "notes.txt").read_bytes() == b"mine\n"


@pytest.mark.parametrize("generator", [
    UnstableGenerator({"app.txt": APP}),
    FilesGenerator({"app.txt": APP}, refusal="it pushes an image"),
], ids=["not regenerable", "refused"])
def test_a_generator_that_cannot_take_a_record_is_refused_before_it_runs(tmp_path, generator):
    generator.output_dir = str(tmp_path / "out")

    with pytest.raises(NotRecordableError):
        record_output(generator)
    assert not (tmp_path / "out").exists()


def test_a_spec_driven_app_takes_no_record(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / ".besser_attribution.json").write_text("{}", encoding="utf-8")

    with pytest.raises(NotRecordableError, match="Spec-Driven"):
        record_output(FilesGenerator({"app.txt": APP}, str(out)))
    assert not (out / "app.txt").exists()


def test_a_link_where_a_file_goes_fails_before_anything_is_written(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "app.txt").symlink_to(tmp_path / "elsewhere.txt")

    with pytest.raises(RegenerationError, match="Cannot write app.txt"):
        record_output(FilesGenerator({"app.txt": APP, "b.txt": APP}, str(out)))
    assert not (tmp_path / "elsewhere.txt").exists() and not (out / "b.txt").exists()


def test_preview_tells_new_changed_and_unchanged_and_writes_nothing(tmp_path, without_pygit2):
    out = tmp_path / "out"
    record_output(FilesGenerator({"same.txt": APP, "edited.txt": APP}, str(out)))
    (out / "edited.txt").write_bytes(b"mine\n")
    before = _tree(out)

    plan = preview_output(FilesGenerator({"same.txt": APP, "edited.txt": APP, "added.txt": APP}, str(out)), record=True)

    assert plan == {RECORD_FILENAME: CHANGED, "added.txt": NEW, "edited.txt": CHANGED, "same.txt": UNCHANGED}
    assert _tree(out) == before


def test_preview_without_a_record_lists_only_the_output(tmp_path):
    plan = preview_output(UnstableGenerator({"app.txt": APP}, str(tmp_path / "out")))

    assert plan == {"app.txt": NEW} and not (tmp_path / "out").exists()


def test_preview_refuses_a_generator_that_acts_outside_its_folder(tmp_path):
    generator = FilesGenerator({"app.txt": APP}, str(tmp_path / "out"), refusal="it pushes an image")

    with pytest.raises(RegenerationError, match="cannot preview its output: it pushes an image"):
        preview_output(generator)


@needs_pygit2
def test_preview_regeneration_merges_into_a_copy(tmp_path):
    out = tmp_path / "out"
    record_output(FilesGenerator({"app.txt": APP}, str(out)))
    (out / "app.txt").write_bytes(b"# mine\n" + APP)
    before = _tree(out)
    generator = FilesGenerator({"app.txt": APP + b"\n\ndef b():\n    return 2\n"}, str(out))

    report = preview_regeneration(generator)

    assert report.merged == ["app.txt"] and generator.output_dir == str(out)
    assert _tree(out) == before


@needs_pygit2
def test_preview_regeneration_refusals_name_the_folder_not_the_copy(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "app.txt").write_bytes(APP)

    with pytest.raises(UnrecordedOutputError) as refusal:
        preview_regeneration(FilesGenerator({"app.txt": APP}, str(out)))
    assert str(refusal.value).startswith(f"{out} holds generated files (app.txt)")
    assert refusal.value.folder == str(out) and refusal.value.files == ["app.txt"]


@pytest.mark.parametrize("output", ["out", "out/app"], ids=["a file", "under a file"])
def test_recording_into_a_file_is_refused(tmp_path, output):
    (tmp_path / "out").write_bytes(b"mine\n")

    with pytest.raises(RegenerationError, match="folder"):
        record_output(FilesGenerator({"app.txt": APP}, str(tmp_path / output)))
    assert (tmp_path / "out").read_bytes() == b"mine\n"
