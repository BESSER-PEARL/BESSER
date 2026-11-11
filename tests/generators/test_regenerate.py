"""``generator.regenerate()``: a model change reaches a generated folder without losing edits.

Every edit is either kept or flagged; ``generate()`` itself is unchanged.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("pygit2")

from besser.BUML.metamodel.structural import Class, DomainModel, IntegerType, Property, StringType
from besser.generators import GeneratorInterface
from besser.generators import regeneration as regeneration_module
from besser.generators.java_classes import JavaGenerator
from besser.generators.python_classes import PythonGenerator
from besser.generators.regeneration import RECORD_FILENAME, RegenerationError
from besser.generators.supabase import SupabaseGenerator
from besser.utilities import three_way_merge, workspace_fs


class FilesGenerator(GeneratorInterface):
    """Writes the files it is given, so each merge case is exact."""

    regenerable = True

    def __init__(self, files, output_dir=None):
        super().__init__(None, output_dir)
        self.files = files

    def generate(self):
        for rel, content in self.files.items():
            path = Path(self.build_generation_path(rel))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)


APP = b"def a():\n    return 1\n\n\ndef b():\n    return 2\n\n\ndef c():\n    return 3\n"


def regen(folder, files, **kwargs):
    return FilesGenerator(files, output_dir=str(folder)).regenerate(**kwargs)


def read(folder, rel="app.txt"):
    return (folder / rel).read_bytes()


def edit(folder, old, new, rel="app.txt"):
    path = folder / rel
    content = path.read_bytes()
    assert old in content
    path.write_bytes(content.replace(old, new))


@pytest.fixture
def folder(tmp_path):
    out = tmp_path / "out"
    regen(out, {"app.txt": APP})
    return out


def test_the_first_run_writes_the_output_and_its_record(tmp_path):
    out = tmp_path / "out"
    report = regen(out, {"app.txt": APP, "pkg/mod.txt": b"x = 1\n"})
    assert read(out) == APP
    assert sorted(report.written) == ["app.txt", "pkg/mod.txt"]
    assert (out / RECORD_FILENAME).is_file()


def test_generate_is_unchanged_and_writes_no_record(tmp_path):
    out = tmp_path / "out"
    FilesGenerator({"app.txt": APP}, output_dir=str(out)).generate()
    assert not (out / RECORD_FILENAME).exists()


def test_an_edit_survives_a_model_change_elsewhere_in_the_file(folder):
    edit(folder, b"return 1\n", b"return 1  # mine\n")
    report = regen(folder, {"app.txt": APP.replace(b"return 3", b"return 33")})
    assert read(folder) == APP.replace(b"return 1\n", b"return 1  # mine\n").replace(b"return 3", b"return 33")
    assert report.merged == ["app.txt"] and not report.conflicted


def test_a_file_nobody_edited_takes_the_new_output(folder):
    report = regen(folder, {"app.txt": APP.replace(b"return 1", b"return 11")})
    assert report.written == ["app.txt"] and not report.merged
    assert b"return 11" in read(folder)


def test_a_run_with_no_model_change_touches_nothing(folder):
    edit(folder, b"return 1\n", b"return 1  # mine\n")
    report = regen(folder, {"app.txt": APP})
    assert not (report.written or report.merged or report.conflicted or report.deleted)
    assert b"# mine" in read(folder)


def test_the_users_own_files_are_never_touched(folder):
    (folder / "mine.txt").write_bytes(b"mine = 1\n")
    regen(folder, {"app.txt": APP.replace(b"return 1", b"return 11")})
    assert read(folder, "mine.txt") == b"mine = 1\n"


def test_a_clashing_edit_gets_markers_and_is_reported(folder):
    edit(folder, b"return 1\n", b"return 100\n")
    report = regen(folder, {"app.txt": APP.replace(b"return 1\n", b"return 11\n")})
    assert report.conflicted == ["app.txt"]
    assert read(folder) == (b"def a():\n"
                            b"<<<<<<< your edit\n    return 100\n=======\n    return 11\n"
                            b">>>>>>> generated from your model\n"
                            b"\n\ndef b():\n    return 2\n\n\ndef c():\n    return 3\n")


def test_the_next_run_refuses_until_the_conflict_is_resolved(folder):
    edit(folder, b"return 1\n", b"return 100\n")
    regen(folder, {"app.txt": APP.replace(b"return 1\n", b"return 11\n")})
    before = read(folder)

    with pytest.raises(regeneration_module.ConflictsPendingError, match="app.txt") as refusal:
        regen(folder, {"app.txt": APP.replace(b"return 1\n", b"return 12\n")})
    assert refusal.value.folder == str(folder) and refusal.value.files == ["app.txt"]
    assert read(folder) == before

    (folder / "app.txt").write_bytes(APP.replace(b"return 1\n", b"return 100\n"))
    report = regen(folder, {"app.txt": APP.replace(b"return 1\n", b"return 11\n").replace(b"return 3", b"return 33")})
    assert not report.conflicted
    assert b"return 100" in read(folder) and b"return 33" in read(folder)


def test_an_unedited_file_the_model_dropped_is_deleted(tmp_path):
    out = tmp_path / "out"
    regen(out, {"app.txt": APP, "old.txt": b"old = 1\n"})
    report = regen(out, {"app.txt": APP})
    assert report.deleted == ["old.txt"] and not (out / "old.txt").exists()


def test_an_edited_file_the_model_dropped_is_kept_and_reported(tmp_path):
    out = tmp_path / "out"
    regen(out, {"app.txt": APP, "old.txt": b"old = 1\n"})
    edit(out, b"old = 1", b"old = 2  # mine", rel="old.txt")
    report = regen(out, {"app.txt": APP})
    assert report.kept_removed_from_model == ["old.txt"] and b"# mine" in read(out, "old.txt")


def test_a_deleted_file_the_model_changed_is_reported_not_restored(tmp_path):
    out = tmp_path / "out"
    regen(out, {"app.txt": APP, "b.txt": b"b = 1\n"})
    (out / "b.txt").unlink()
    report = regen(out, {"app.txt": APP, "b.txt": b"b = 2\n"})
    assert report.not_restored == ["b.txt"] and not (out / "b.txt").exists()


def test_a_deleted_file_the_model_left_alone_stays_deleted(tmp_path):
    out = tmp_path / "out"
    regen(out, {"app.txt": APP, "b.txt": b"b = 1\n"})
    (out / "b.txt").unlink()
    report = regen(out, {"app.txt": APP, "b.txt": b"b = 1\n"})
    assert not (out / "b.txt").exists() and not report.written


def test_a_file_the_user_created_where_the_model_adds_one_is_flagged(folder):
    (folder / "new.txt").write_bytes(b"mine = 1\n")
    report = regen(folder, {"app.txt": APP, "new.txt": b"generated = 1\n"})
    assert report.conflicted == ["new.txt"]
    assert b"mine = 1" in read(folder, "new.txt") and b"generated = 1" in read(folder, "new.txt")


def test_a_folder_generated_before_is_refused_then_adopted(tmp_path):
    out = tmp_path / "out"
    FilesGenerator({"app.txt": APP}, output_dir=str(out)).generate()
    edit(out, b"return 1\n", b"return 1  # mine\n")

    with pytest.raises(RegenerationError, match="adopt"):
        regen(out, {"app.txt": APP.replace(b"return 3", b"return 33")})
    assert b"return 33" not in read(out)

    before = read(out)
    report = regen(out, {"app.txt": APP}, adopt=True)  # the model those files came from
    assert read(out) == before and not report.written

    regen(out, {"app.txt": APP.replace(b"return 3", b"return 33")})
    assert b"# mine" in read(out) and b"return 33" in read(out)


def test_adopting_a_folder_that_has_a_record_is_refused(folder):
    with pytest.raises(regeneration_module.AlreadyRecordedError, match="already") as refusal:
        regen(folder, {"app.txt": APP}, adopt=True)
    assert refusal.value.folder == str(folder)


def test_the_refusal_of_an_unrecorded_folder_names_its_generated_files(tmp_path):
    out = tmp_path / "out"
    files = {f"m{index}.py": APP for index in range(5)}
    FilesGenerator(files, output_dir=str(out)).generate()

    with pytest.raises(regeneration_module.UnrecordedOutputError) as refusal:
        regen(out, files)
    assert refusal.value.folder == str(out) and refusal.value.files == sorted(files)
    assert refusal.value.named_files() == "m0.py, m1.py, m2.py, ..."


@pytest.mark.parametrize("output", ["out", "out/app"], ids=["a file", "under a file"])
def test_an_output_folder_that_is_a_file_is_refused(tmp_path, output):
    (tmp_path / "out").write_bytes(b"mine\n")

    with pytest.raises(RegenerationError, match="folder"):
        regen(tmp_path / output, {"app.txt": APP})
    assert (tmp_path / "out").read_bytes() == b"mine\n"


def test_windows_line_endings_are_kept_and_do_not_read_as_edits(folder):
    (folder / "app.txt").write_bytes(APP.replace(b"\n", b"\r\n"))
    report = regen(folder, {"app.txt": APP.replace(b"return 3", b"return 33")})
    assert report.written == ["app.txt"] and not report.merged
    assert read(folder) == APP.replace(b"return 3", b"return 33").replace(b"\n", b"\r\n")


def test_binary_output_is_written_once_and_never_merged(tmp_path):
    out = tmp_path / "out"
    assert regen(out, {"logo.png": b"\x89PNG\x00one"}).written == ["logo.png"]
    (out / "logo.png").write_bytes(b"\x89PNG\x00mine")
    report = regen(out, {"logo.png": b"\x89PNG\x00two"})
    assert read(out, "logo.png") == b"\x89PNG\x00mine" and report.not_merged == ["logo.png"]
    assert "logo.png" not in json.loads((out / RECORD_FILENAME).read_text(encoding="utf-8"))["files"]


def test_an_edit_in_another_encoding_that_clashes_is_kept_and_reported(folder):
    edited = APP.replace(b"return 1\n", b"return '\xe9'\n")
    (folder / "app.txt").write_bytes(edited)
    report = regen(folder, {"app.txt": APP.replace(b"return 1\n", b"return 11\n")})
    assert read(folder) == edited
    assert report.not_merged == ["app.txt"] and not report.conflicted


BAD_PATHS = ["../evil.txt", "/etc/evil.txt", "a/../../evil.txt", "C:/evil.txt", "a\\..\\..\\evil.py", "link/../evil.txt",
             "nul\x00.py", ".git/hooks/pre-commit", ".GIT/config", "pkg/.git/config"]


def _record(**changes):
    return json.dumps({"format": 1, "generator": "FilesGenerator", "binaries": [], "conflicted": [],
                       "files": {"app.txt": APP.decode("utf-8")}, **changes})


def test_a_complete_record_is_used(folder):
    (folder / RECORD_FILENAME).write_text(_record(), encoding="utf-8")
    edit(folder, b"return 1\n", b"return 1  # mine\n")
    report = regen(folder, {"app.txt": APP.replace(b"return 3", b"return 33")})
    assert report.merged == ["app.txt"]


@pytest.mark.parametrize("rel", BAD_PATHS)
def test_a_record_naming_a_path_it_must_not_touch_is_refused(folder, rel):
    (folder / RECORD_FILENAME).write_text(_record(files={"app.txt": APP.decode("utf-8"), rel: "x\n"}), encoding="utf-8")
    with pytest.raises(RegenerationError, match="record"):
        regen(folder, {"app.txt": APP})


@pytest.mark.parametrize("rel", BAD_PATHS)
def test_a_record_listing_such_a_path_as_conflicted_is_refused(folder, rel):
    (folder / RECORD_FILENAME).write_text(_record(conflicted=[rel]), encoding="utf-8")
    with pytest.raises(RegenerationError, match="record"):
        regen(folder, {"app.txt": APP})


@pytest.mark.parametrize("content", [
    "{not json", "[]", '{"format": 1, "files": {}}',
    _record(format=99), _record(files=3), _record(files=["app.txt"]), _record(files={"app.txt": 1}),
    _record(conflicted="app.txt"), _record(conflicted=[1]), _record(binaries="logo.png"), _record(binaries=[".git/x"]),
], ids=["not_json", "not_an_object", "missing_keys", "format", "files_number", "files_list", "file_not_text",
        "conflicted_string", "conflicted_not_paths", "binaries_string", "binaries_bad_path"])
def test_an_unusable_record_is_refused_not_overwritten(folder, content):
    (folder / RECORD_FILENAME).write_text(content, encoding="utf-8")
    edit(folder, b"return 1\n", b"return 1  # mine\n")
    with pytest.raises(RegenerationError, match="record"):
        regen(folder, {"app.txt": APP})
    assert b"# mine" in read(folder)


def test_without_pygit2_it_refuses_and_writes_nothing(folder, monkeypatch):
    monkeypatch.setattr(three_way_merge, "pygit2", None)
    edit(folder, b"return 1\n", b"return 1  # mine\n")
    with pytest.raises(RegenerationError, match=r"besser\[regen\]"):
        regen(folder, {"app.txt": APP.replace(b"return 3", b"return 33")})
    assert b"return 33" not in read(folder)


def test_a_generator_that_has_not_opted_in_is_refused(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(RegenerationError, match="SupabaseGenerator"):
        SupabaseGenerator(_shelf(), output_dir=str(out)).regenerate()
    assert not out.exists()


@pytest.mark.parametrize("name", [".besser_generator_output.json", ".besser_attribution.json"])
@pytest.mark.parametrize("adopt", [False, True])
def test_a_spec_driven_app_is_refused_so_its_authorship_stays_true(tmp_path, name, adopt):
    out = tmp_path / "out"
    out.mkdir()
    (out / "app.txt").write_bytes(APP)
    (out / name).write_text("{}", encoding="utf-8")
    with pytest.raises(RegenerationError, match="Spec-Driven"):
        regen(out, {"app.txt": APP.replace(b"return 3", b"return 33")}, adopt=adopt)
    assert read(out) == APP and not (out / RECORD_FILENAME).exists()


def test_the_generators_output_dir_is_restored(tmp_path):
    out = str(tmp_path / "out")
    generator = FilesGenerator({"app.txt": APP}, output_dir=out)
    generator.regenerate()
    assert generator.output_dir == out


def test_a_failing_generator_changes_nothing(folder):
    generator = FilesGenerator({"app.txt": b"half\n"}, output_dir=str(folder))
    real_generate = generator.generate

    def broken():
        real_generate()
        raise ValueError("boom")

    generator.generate = broken
    with pytest.raises(ValueError, match="boom"):
        generator.regenerate()
    assert read(folder) == APP


def test_merged_files_come_with_a_warning_to_run_the_tests(folder, caplog):
    edit(folder, b"return 1\n", b"return 1  # mine\n")
    with caplog.at_level("WARNING", logger=regeneration_module.__name__):
        regen(folder, {"app.txt": APP.replace(b"return 3", b"return 33")})
    assert "run your tests" in caplog.text


# -- review of the branch: an edit must survive a failure ---------------------------------


def _fail_replacing(monkeypatch, name):
    """``os.replace`` raises for one file, as on a full disk."""
    real = workspace_fs.os.replace

    def replace(src, dst, **kwargs):
        if str(dst).endswith(name):
            raise OSError(28, "No space left on device")
        return real(src, dst, **kwargs)

    monkeypatch.setattr(workspace_fs.os, "replace", replace)


def test_a_full_disk_does_not_truncate_an_edit(folder, monkeypatch):
    edit(folder, b"return 1\n", b"return 1  # mine\n")
    edited = read(folder)
    with monkeypatch.context() as patch:
        _fail_replacing(patch, "app.txt")
        with pytest.raises(OSError):
            regen(folder, {"app.txt": APP.replace(b"return 3", b"return 33")})
    assert read(folder) == edited
    assert not list(folder.glob(".besser_tmp_*"))

    report = regen(folder, {"app.txt": APP.replace(b"return 3", b"return 33")})
    assert report.merged == ["app.txt"] and b"# mine" in read(folder) and b"return 33" in read(folder)


def test_an_interrupted_first_run_resumes_without_adopting(tmp_path, monkeypatch):
    out = tmp_path / "out"
    files = {"a.txt": b"a = 1\n", "b.txt": b"b = 1\n"}
    with monkeypatch.context() as patch:
        _fail_replacing(patch, "b.txt")
        with pytest.raises(OSError):
            regen(out, files)
    assert read(out, "a.txt") == files["a.txt"] and not (out / "b.txt").exists()

    report = regen(out, files)
    assert read(out, "b.txt") == files["b.txt"] and not report.conflicted


def test_a_crash_after_a_conflict_cannot_nest_markers(tmp_path, monkeypatch):
    out = tmp_path / "out"
    regen(out, {"a.txt": APP, "z.txt": b"z = 1\n"})
    edit(out, b"return 1\n", b"return 100\n", rel="a.txt")
    changed = {"a.txt": APP.replace(b"return 1\n", b"return 11\n"), "z.txt": b"z = 2\n"}
    with monkeypatch.context() as patch:
        _fail_replacing(patch, "z.txt")
        with pytest.raises(OSError):
            regen(out, changed)
    assert read(out, "a.txt").count(b"<<<<<<< ") == 1

    with pytest.raises(RegenerationError, match="a.txt"):
        regen(out, changed)
    assert read(out, "a.txt").count(b"<<<<<<< ") == 1


def test_a_folder_holding_only_other_files_needs_no_adoption(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "README.md").write_bytes(b"# mine\n")
    (out / ".git").mkdir()
    report = regen(out, {"app.txt": APP})
    assert report.written == ["app.txt"] and read(out, "README.md") == b"# mine\n"


def test_adopting_says_which_files_differ_from_that_model(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "app.txt").write_bytes(APP + b"# mine\n")
    (out / "same.txt").write_bytes(b"s = 1\n")
    report = regen(out, {"app.txt": APP, "same.txt": b"s = 1\n", "absent.txt": b"x = 1\n"}, adopt=True)
    assert report.differs_from_model == ["absent.txt", "app.txt"]
    assert not (out / "absent.txt").exists()


def test_another_generators_folder_is_refused_not_deleted(folder):
    class OtherGenerator(FilesGenerator):
        pass

    with pytest.raises(RegenerationError, match="FilesGenerator"):
        OtherGenerator({"other.txt": b"o = 1\n"}, output_dir=str(folder)).regenerate()
    assert read(folder) == APP


def test_regenerating_needs_an_explicit_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    generator = FilesGenerator({"app.txt": APP})
    with pytest.raises(RegenerationError, match="output_dir"):
        generator.regenerate()
    assert not (tmp_path / "output").exists()


def test_a_class_renamed_in_case_only_keeps_its_edit(tmp_path):
    out = tmp_path / "out"
    regen(out, {"book.java": APP})
    edit(out, b"return 1\n", b"return 1  // mine\n", rel="book.java")

    report = regen(out, {"Book.java": APP.replace(b"return 3", b"return 33")})

    assert [p.name for p in out.iterdir() if p.suffix == ".java"] == ["Book.java"]
    content = read(out, "Book.java")
    assert b"// mine" in content and b"return 33" in content
    assert report.merged == ["Book.java"] and not (report.deleted or report.conflicted)


@pytest.mark.parametrize("edited", [APP.replace(b"return 1\n", b"return '\x00'\n"), APP.replace(b"\n", b"\n\x00")])
def test_a_clash_with_a_file_holding_nul_bytes_is_kept_and_reported(folder, edited):
    (folder / "app.txt").write_bytes(edited)
    report = regen(folder, {"app.txt": APP.replace(b"return 1\n", b"return 11\n")})
    assert read(folder) == edited and report.not_merged == ["app.txt"]


def test_a_folder_at_a_generated_path_fails_before_anything_is_written(tmp_path):
    out = tmp_path / "out"
    regen(out, {"a.txt": b"a = 1\n", "b.txt": b"b = 1\n"})
    (out / "b.txt").unlink()
    (out / "b.txt").mkdir()
    with pytest.raises(RegenerationError, match="b.txt"):
        regen(out, {"a.txt": b"a = 2\n", "b.txt": b"b = 2\n"})
    assert read(out, "a.txt") == b"a = 1\n"


def test_a_file_at_a_generated_parent_fails_before_anything_is_written(tmp_path):
    out = tmp_path / "out"
    regen(out, {"a.txt": b"a = 1\n"})
    (out / "pkg").write_bytes(b"mine\n")
    with pytest.raises(RegenerationError, match="pkg"):
        regen(out, {"a.txt": b"a = 2\n", "pkg/mod.txt": b"m = 1\n"})
    assert read(out, "a.txt") == b"a = 1\n" and read(out, "pkg") == b"mine\n"


@pytest.mark.skipif(os.name == "nt", reason="needs symlinks and POSIX modes")
def test_a_link_at_a_generated_path_is_refused(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"mine\n")
    out = tmp_path / "out"
    regen(out, {"a.txt": b"a = 1\n"})
    (out / "link.txt").symlink_to(outside)
    with pytest.raises(RegenerationError, match="link.txt"):
        regen(out, {"a.txt": b"a = 1\n", "link.txt": b"x = 1\n"})
    assert outside.read_bytes() == b"mine\n"


@pytest.mark.skipif(os.name == "nt", reason="needs POSIX modes")
def test_executable_bits_are_kept(tmp_path):
    out = tmp_path / "out"

    class ScriptGenerator(FilesGenerator):
        def generate(self):
            super().generate()
            os.chmod(self.build_generation_path("mvnw"), 0o755)

    ScriptGenerator({"mvnw": b"run\n"}, output_dir=str(out)).regenerate()
    assert os.access(out / "mvnw", os.X_OK)
    ScriptGenerator({"mvnw": b"run again\n"}, output_dir=str(out)).regenerate()
    assert os.access(out / "mvnw", os.X_OK)


def test_marker_text_inside_a_line_does_not_block_the_next_run(folder):
    edit(folder, b"return 1\n", b"return 100\n")
    regen(folder, {"app.txt": APP.replace(b"return 1\n", b"return 11\n")})
    (folder / "app.txt").write_bytes(APP + b"SAMPLE = 'x <<<<<<< y'\n")
    report = regen(folder, {"app.txt": APP.replace(b"return 2", b"return 22")})
    assert not report.conflicted and b"return 22" in read(folder)


def test_conflict_markers_say_which_side_is_which(folder):
    edit(folder, b"return 1\n", b"return 100\n")
    regen(folder, {"app.txt": APP.replace(b"return 1\n", b"return 11\n")})
    assert b"<<<<<<< your edit\n" in read(folder) and b">>>>>>> generated from your model\n" in read(folder)


def test_a_backend_that_pushes_an_image_is_refused(tmp_path):
    from besser.generators.backend import BackendGenerator

    config = tmp_path / "docker.ini"
    config.write_text("[DEFAULT]\n" + "".join(f"docker_{key} = x\n" for key in (
        "username", "password", "image_name", "repository", "tag", "port")), encoding="utf-8")
    generator = BackendGenerator(_shelf(), output_dir=str(tmp_path / "out"), docker_image=True,
                                 docker_config_path=str(config))
    with pytest.raises(RegenerationError, match="image"):
        generator.regenerate()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("options", [{"docker_image": True}, {"docker_config_path": "docker.ini"}])
def test_a_backend_that_pushes_nothing_regenerates(tmp_path, monkeypatch, options):
    from besser.generators.backend import BackendGenerator

    monkeypatch.chdir(tmp_path)
    out = str(tmp_path / "out")
    BackendGenerator(_shelf(), output_dir=out, **options).regenerate()
    report = BackendGenerator(_shelf(), output_dir=out, **options).regenerate()
    assert not (report.written or report.merged or report.conflicted)


# -- a merge must not cut a file short, and a record must not reach outside its folder ------


# libgit2's binary check stops at 8000 bytes, and pygit2 reads the merge result as a C string.
LONG = b"".join(b"line %d\n" % i for i in range(2000))


def test_a_clash_in_a_file_with_a_late_nul_byte_keeps_every_byte(tmp_path):
    out = tmp_path / "out"
    regen(out, {"app.txt": LONG + b"shared\nend\n"})
    edited = LONG + b"mine\nend\nbin\x00ary\nmy later edit\n"
    (out / "app.txt").write_bytes(edited)
    report = regen(out, {"app.txt": LONG + b"model\nend\n"})
    assert read(out) == edited
    assert report.not_merged == ["app.txt"] and not report.conflicted


def test_a_late_nul_byte_survives_a_clean_merge(tmp_path):
    out = tmp_path / "out"
    regen(out, {"app.txt": b"first\n" + LONG + b"end\n"})
    (out / "app.txt").write_bytes(b"first\n" + LONG + b"end\nbin\x00ary\nmine\n")
    report = regen(out, {"app.txt": b"model\n" + LONG + b"end\n"})
    assert read(out) == b"model\n" + LONG + b"end\nbin\x00ary\nmine\n"
    assert report.merged == ["app.txt"]


def test_control_characters_merge_like_any_other_text(tmp_path):
    out = tmp_path / "out"
    control = b"\x01" * 400 + b"\n"
    regen(out, {"app.txt": control + b"shared\n"})
    (out / "app.txt").write_bytes(control + b"mine\n")
    report = regen(out, {"app.txt": control + b"model\n"})
    assert report.conflicted == ["app.txt"]
    assert read(out) == control + b"<<<<<<< your edit\nmine\n=======\nmodel\n>>>>>>> generated from your model\n"


def test_emptying_a_file_that_clashes_still_blocks_the_next_run(folder):
    (folder / "app.txt").write_bytes(b"")
    changed = {"app.txt": APP.replace(b"return 1\n", b"return 11\n")}
    report = regen(folder, changed)
    assert report.conflicted == ["app.txt"] and read(folder).startswith(b"<<<<<<< your edit\n")
    with pytest.raises(RegenerationError, match="app.txt"):
        regen(folder, changed)


@pytest.fixture
def case_sensitive(tmp_path):
    probe = tmp_path / "CaseProbe"
    probe.touch()
    if (tmp_path / "caseprobe").exists():
        pytest.skip("needs a case-sensitive disk")
    probe.unlink()


@pytest.mark.skipif(os.name == "nt", reason="needs symlinks")
def test_a_case_rename_cannot_move_a_file_through_a_link(tmp_path, case_sensitive):
    outside = tmp_path / "outside"
    outside.mkdir()
    out = tmp_path / "out"
    regen(out, {"PKG/mod.txt": b"m = 1\n"})
    (out / "pkg").symlink_to(outside)
    with pytest.raises(RegenerationError, match="pkg"):
        regen(out, {"pkg/mod.txt": b"m = 1\n"})
    assert not list(outside.iterdir()) and read(out, "PKG/mod.txt") == b"m = 1\n"


def test_a_case_rename_never_overwrites_a_file_with_the_new_name(tmp_path, case_sensitive):
    out = tmp_path / "out"
    regen(out, {"book.java": APP})
    edit(out, b"return 1\n", b"return 1  // mine\n", rel="book.java")
    (out / "Book.java").write_bytes(b"my own class\n")
    regen(out, {"Book.java": APP})
    assert b"// mine" in read(out, "book.java") and read(out, "Book.java") == b"my own class\n"


def test_output_paths_differing_only_in_case_are_refused(tmp_path, case_sensitive):
    out = tmp_path / "out"
    with pytest.raises(RegenerationError, match="only in case"):
        regen(out, {"Book.java": APP, "book.java": APP})
    assert not (out / "Book.java").exists()


@pytest.mark.parametrize("line", [
    b"'x'; __import__('os').system('echo PWNED')",
    b"get_key()",
    b"'a' + get_key()",
    b"'x'  # note",
    b"'x\\'",
    b'"x"',
])
def test_django_does_not_carry_over_anything_but_a_plain_key(line):
    from besser.generators.django import DjangoGenerator

    settings = "s/s/settings.py"
    generated = {settings: b"A = 1\nSECRET_KEY = 'fresh'\nB = 2\n"}
    previous = {settings: b"SECRET_KEY = " + line + b"\n"}
    assert DjangoGenerator(_shelf(), "s", "a")._keep_generated_values(previous, generated) == generated


@pytest.mark.parametrize("ending", [b"\n", b"\r\n"])
def test_django_carries_a_plain_key_over(ending):
    from besser.generators.django import DjangoGenerator

    settings = "s/s/settings.py"
    previous = {settings: b"SECRET_KEY = 'django-insecure-!@#$%^&*(-_=+)'" + ending}
    generated = {settings: b"SECRET_KEY = 'fresh'\nB = 2\n"}
    kept = DjangoGenerator(_shelf(), "s", "a")._keep_generated_values(previous, generated)
    assert kept[settings] == b"SECRET_KEY = 'django-insecure-!@#$%^&*(-_=+)'\nB = 2\n"


def _web_app(tmp_path, agent_config):
    from besser.BUML.metamodel.state_machine.agent import Agent
    from besser.generators.web_app import WebAppGenerator
    from tests.generators.regenerable_builders import _editor_models

    agent = Agent("helper")
    agent.new_state("init", initial=True)
    return WebAppGenerator(*_editor_models(), output_dir=str(tmp_path / "out"), agent_model=agent,
                           agent_config=agent_config)


@pytest.mark.parametrize("config", [
    {"agentLanguage": "french"},
    {"presentation": {"agentLanguage": "french"}},
    {"personalizationMapping": [{"name": "default"}]},
])
def test_a_web_app_with_a_personalized_agent_is_refused(tmp_path, config, monkeypatch):
    from besser.generators.agents import agent_personalization

    def no_llm(*args, **kwargs):
        raise AssertionError("regenerate() reached the LLM")

    monkeypatch.setattr(agent_personalization, "call_openai_chat", no_llm)
    with pytest.raises(RegenerationError, match="personalized agent"):
        _web_app(tmp_path, config).regenerate()
    assert not (tmp_path / "out" / "agents").exists()


def test_a_web_app_with_a_plain_agent_regenerates(tmp_path):
    _web_app(tmp_path, {"platform": "websocket"}).regenerate()
    report = _web_app(tmp_path, {"platform": "websocket"}).regenerate()
    assert not (report.written or report.merged or report.conflicted)
    assert (tmp_path / "out" / "agents" / "helper" / "helper.py").is_file()


@pytest.mark.parametrize("dbms, url", [(None, "sqlite:///"), ("postgresql", "postgresql"), ("mysql", "mysql")])
def test_sqlalchemy_options_reach_a_regeneration(tmp_path, dbms, url):
    from besser.generators.sql_alchemy import SQLAlchemyGenerator

    out = tmp_path / "out"
    options = {"dbms": dbms} if dbms else {}
    SQLAlchemyGenerator(_shelf(), output_dir=str(out), **options).regenerate()
    assert f'"{url}' in read(out, "sql_alchemy.py").decode("utf-8")


def test_a_users_own_marker_lines_are_neither_relabelled_nor_blocking(folder):
    own = b"<<<<<<< HEAD\n>>>>>>> theirs\n"
    edit(folder, b"return 1\n", b"return 100\n")
    edit(folder, b"return 3\n", b"return 3\n" + own)
    regen(folder, {"app.txt": APP.replace(b"return 1\n", b"return 11\n")})
    assert own in read(folder) and read(folder).count(b"<<<<<<< your edit\n") == 1

    (folder / "app.txt").write_bytes(APP + own)
    report = regen(folder, {"app.txt": APP.replace(b"return 2", b"return 22")})
    assert own in read(folder) and b"return 22" in read(folder) and not report.conflicted


def test_a_binary_file_the_model_dropped_is_kept_and_reported(tmp_path, caplog):
    out = tmp_path / "out"
    regen(out, {"app.txt": APP, "logo.png": b"\x89PNG\x00one"})
    with caplog.at_level("WARNING", logger=regeneration_module.__name__):
        report = regen(out, {"app.txt": APP})
    assert read(out, "logo.png") == b"\x89PNG\x00one"
    assert report.kept_removed_from_model == ["logo.png"] and "logo.png" in caplog.text
    assert not regen(out, {"app.txt": APP}).kept_removed_from_model  # no longer recorded


def test_the_old_name_of_a_case_rename_on_a_case_sensitive_disk_is_reported(tmp_path, case_sensitive):
    out = tmp_path / "out"
    regen(out, {"book.java": APP})
    (out / "Book.java").write_bytes(APP)
    report = regen(out, {"Book.java": APP})
    assert report.kept_removed_from_model == ["book.java"] and (out / "book.java").is_file()


def test_a_full_disk_while_saving_the_record_keeps_the_old_one(folder, monkeypatch):
    before = (folder / RECORD_FILENAME).read_bytes()
    with monkeypatch.context() as patch:
        _fail_replacing(patch, RECORD_FILENAME)
        with pytest.raises(OSError):
            regen(folder, {"app.txt": APP.replace(b"return 3", b"return 33")})
    assert (folder / RECORD_FILENAME).read_bytes() == before


def test_a_failed_merge_is_reported_and_changes_no_file(folder, monkeypatch):
    def broken(*args):
        raise ValueError("boom")

    edit(folder, b"return 1\n", b"return 1  # mine\n")
    edited = read(folder)
    monkeypatch.setattr(regeneration_module, "merge_trees", broken)
    with pytest.raises(RegenerationError, match="merge failed.*boom"):
        regen(folder, {"app.txt": APP.replace(b"return 3", b"return 33")})
    assert read(folder) == edited


def test_a_conflict_in_a_crlf_file_keeps_crlf(folder):
    (folder / "app.txt").write_bytes(APP.replace(b"return 1\n", b"return 100\n").replace(b"\n", b"\r\n"))
    regen(folder, {"app.txt": APP.replace(b"return 1\n", b"return 11\n")})
    content = read(folder)
    assert b"<<<<<<< your edit\r\n" in content and content.count(b"\n") == content.count(b"\r\n")


def test_adopting_keeps_generated_values_from_the_folder(tmp_path):
    class KeyGenerator(FilesGenerator):
        def _keep_generated_values(self, previous, generated):
            return {**generated, "key": previous.get("key", generated["key"])}

    out = tmp_path / "out"
    out.mkdir()
    (out / "key").write_bytes(b"from the folder\n")
    report = KeyGenerator({"key": b"drawn this run\n"}, output_dir=str(out)).regenerate(adopt=True)
    assert not report.differs_from_model


def test_build_leftovers_are_not_output(tmp_path):
    out = tmp_path / "out"
    report = regen(out, {"app.txt": APP, ".git/config": b"x\n", "pkg/__pycache__/m.pyc": b"x\n",
                         "web/node_modules/a.js": b"x\n"})
    assert report.written == ["app.txt"]
    assert sorted(p.name for p in out.rglob("*") if p.is_file()) == [RECORD_FILENAME, "app.txt"]


@pytest.mark.skipif(os.name == "nt", reason="needs symlinks")
def test_a_record_that_is_a_link_is_refused(folder, tmp_path):
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_bytes((folder / RECORD_FILENAME).read_bytes())
    (folder / RECORD_FILENAME).unlink()
    (folder / RECORD_FILENAME).symlink_to(elsewhere)
    with pytest.raises(RegenerationError, match="not a regular file"):
        regen(folder, {"app.txt": APP})


@pytest.mark.skipif(os.name == "nt", reason="needs symlinks")
def test_a_link_in_the_scratch_output_is_not_read(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"secret\n")

    class LinkGenerator(FilesGenerator):
        def generate(self):
            super().generate()
            os.symlink(secret, self.build_generation_path("leak.txt"))

    out = tmp_path / "out"
    report = LinkGenerator({"app.txt": APP}, output_dir=str(out)).regenerate()
    assert report.written == ["app.txt"] and not (out / "leak.txt").exists()


@pytest.fixture
def junction(tmp_path):
    """``make(link, target)``, an NTFS junction."""
    if os.name != "nt":
        pytest.skip("needs NTFS")

    def make(link, target):
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)

    return make


def test_a_junction_at_a_generated_folder_is_refused(tmp_path, junction):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "mod.txt").write_bytes(b"mine\n")
    out = tmp_path / "out"
    regen(out, {"app.txt": APP})
    junction(out / "pkg", outside)
    with pytest.raises(RegenerationError, match="pkg"):
        regen(out, {"app.txt": APP, "pkg/mod.txt": b"m = 1\n"})
    assert (outside / "mod.txt").read_bytes() == b"mine\n"


def test_a_file_reached_through_a_junction_is_not_read_as_an_edit(tmp_path, junction):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "mod.txt").write_bytes(b"m = 1\n")
    out = tmp_path / "out"
    regen(out, {"pkg/mod.txt": b"m = 1\n"})
    shutil.rmtree(out / "pkg")
    junction(out / "pkg", outside)
    report = regen(out, {"app.txt": APP})
    assert (outside / "mod.txt").read_bytes() == b"m = 1\n" and "pkg/mod.txt" not in report.deleted


def test_merges_and_scratch_runs_leave_nothing_in_the_temp_folder(tmp_path, monkeypatch):
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    out = tmp_path / "out"
    regen(out, {"app.txt": APP})
    edit(out, b"return 1\n", b"return 100\n")
    regen(out, {"app.txt": APP.replace(b"return 1\n", b"return 11\n")})
    assert not list(temp.iterdir())


@pytest.mark.parametrize("arrange, change, expected", [
    ("edit", {"app.txt": APP.replace(b"return 1\n", b"return 11\n")}, "Conflicts to resolve"),
    ("delete", {"app.txt": APP.replace(b"return 1\n", b"return 11\n")}, "left deleted"),
    ("edit", {"other.txt": b"o = 1\n"}, "may hold your work"),
    ("binary", {"app.txt": APP.replace(b"return 1\n", b"return 11\n")}, "cannot be merged"),
], ids=["conflict", "not_restored", "kept", "not_merged"])
def test_what_needs_attention_is_logged(folder, caplog, arrange, change, expected):
    if arrange == "edit":
        edit(folder, b"return 1\n", b"return 100\n")
    elif arrange == "delete":
        (folder / "app.txt").unlink()
    else:
        (folder / "app.txt").write_bytes(APP.replace(b"return 1\n", b"return '\xe9'\n"))
    with caplog.at_level("WARNING", logger=regeneration_module.__name__):
        regen(folder, change)
    assert expected in caplog.text and "app.txt" in caplog.text


def test_adopting_a_different_model_is_logged(tmp_path, caplog):
    out = tmp_path / "out"
    out.mkdir()
    (out / "app.txt").write_bytes(b"something else\n")
    with caplog.at_level("WARNING", logger=regeneration_module.__name__):
        regen(out, {"app.txt": APP}, adopt=True)
    assert "read as edits from now on" in caplog.text


def test_a_nul_byte_in_a_path_is_simply_not_a_plain_file():
    assert not workspace_fs.is_plain_file("a\x00b")
    assert not workspace_fs.is_link("a\x00b")


@pytest.mark.parametrize("newline, expected", [("", b"a\nb\r\nc"), ("\n", b"a\nb\r\nc"), ("\r\n", b"a\r\nb\r\r\nc"),
                                               (None, f"a{os.linesep}b\r{os.linesep}c".encode())])
def test_atomic_text_writes_translate_line_endings_like_open(tmp_path, newline, expected):
    path = tmp_path / "f.txt"
    workspace_fs.write_atomic_plain(str(path), "a\nb\r\nc", newline=newline)
    assert path.read_bytes() == expected
    with open(tmp_path / "g.txt", "w", encoding="utf-8", newline=newline) as handle:
        handle.write("a\nb\r\nc")
    assert (tmp_path / "g.txt").read_bytes() == expected


# -- real generators ---------------------------------------------------------------------


def _shelf(isbn=False, author=False):
    attributes = {Property(name="title", type=StringType), Property(name="pages", type=IntegerType)}
    if isbn:
        attributes.add(Property(name="isbn", type=StringType))
    types = {Class(name="Book", attributes=attributes)}
    if author:
        types.add(Class(name="Author", attributes={Property(name="name", type=StringType)}))
    return DomainModel(name="Shelf", types=types)


def test_a_java_edit_survives_an_added_attribute_and_class(tmp_path):
    out = tmp_path / "java"
    JavaGenerator(_shelf(), output_dir=str(out)).regenerate()
    book = out / "Book.java"
    text = book.read_text(encoding="utf-8")
    anchor = "public int getPages() {"
    assert anchor in text
    book.write_text(text.replace(anchor, "public boolean isLong() { return pages > 300; }\n\n    " + anchor),
                    encoding="utf-8")

    report = JavaGenerator(_shelf(isbn=True, author=True), output_dir=str(out)).regenerate()

    text = book.read_text(encoding="utf-8")
    assert "isLong" in text and "isbn" in text and (out / "Author.java").is_file()
    assert report.merged == ["Book.java"]


def test_python_keeps_a_helper_through_a_model_change(tmp_path):
    out = tmp_path / "python"
    PythonGenerator(_shelf(), output_dir=str(out)).regenerate()
    classes = out / "classes.py"
    helper = "\n\ndef shelf_size(books):  # mine\n    return len(books)\n"
    classes.write_text(classes.read_text(encoding="utf-8") + helper, encoding="utf-8")

    report = PythonGenerator(_shelf(isbn=True), output_dir=str(out)).regenerate()

    text = classes.read_text(encoding="utf-8")
    assert "shelf_size" in text and "isbn" in text and not report.conflicted


@pytest.mark.skipif(importlib.util.find_spec("django") is None, reason="needs Django")
def test_django_keeps_edits_own_files_and_its_secret_key(tmp_path):
    """DjangoGenerator.generate() deletes the project folder; regenerate() must not."""
    from besser.generators.django import DjangoGenerator

    out = tmp_path / "django"

    def make(model):
        return DjangoGenerator(model, project_name="shelfsite", app_name="shelfapp", output_dir=str(out),
                               containerization=False)

    make(_shelf()).regenerate()
    settings = out / "shelfsite" / "shelfsite" / "settings.py"
    key = next(line for line in settings.read_text(encoding="utf-8").splitlines() if line.startswith("SECRET_KEY"))
    models = out / "shelfsite" / "shelfapp" / "models.py"
    models.write_text(models.read_text(encoding="utf-8") + "\n\n# mine\n", encoding="utf-8")
    (out / "shelfsite" / "shelfapp" / "services.py").write_text("# my service\n", encoding="utf-8")

    report = make(_shelf(isbn=True)).regenerate()

    assert "# mine" in models.read_text(encoding="utf-8") and "isbn" in models.read_text(encoding="utf-8")
    assert (out / "shelfsite" / "shelfapp" / "services.py").is_file()
    assert key in settings.read_text(encoding="utf-8")
    assert "shelfsite/shelfsite/settings.py" not in report.written
    assert key not in (out / RECORD_FILENAME).read_text(encoding="utf-8")

    moved = "SECRET_KEY = os.environ['DJANGO_KEY']"
    settings.write_text(settings.read_text(encoding="utf-8").replace(key, moved), encoding="utf-8")
    make(_shelf(isbn=True)).regenerate()
    assert moved in settings.read_text(encoding="utf-8")


@pytest.mark.skipif(importlib.util.find_spec("django") is None, reason="needs Django")
def test_django_runs_no_code_planted_in_the_recorded_key(tmp_path):
    from besser.generators.django import DjangoGenerator

    out = tmp_path / "django"

    def make():
        return DjangoGenerator(_shelf(), project_name="shelfsite", app_name="shelfapp", output_dir=str(out),
                               containerization=False)

    make().regenerate()
    name = "shelfsite/shelfsite/settings.py"
    record_path = out / RECORD_FILENAME
    record = json.loads(record_path.read_text(encoding="utf-8"))
    # An extra first line of the record: the file itself stays clean, so nothing looks edited.
    record["files"][name] = "SECRET_KEY = 'x'; __import__('os').system('echo PWNED')\n" + record["files"][name]
    record_path.write_text(json.dumps(record), encoding="utf-8")

    make().regenerate()

    assert "PWNED" not in (out / name).read_text(encoding="utf-8")


def test_the_merge_helper_marks_only_the_clash():
    merge = three_way_merge.merge_trees(
        {"a": b"1\n2\n3\n4\n5\n6\n7\n", "gone": b"x\n"},
        {"a": b"mine\n2\n3\n4\n5\n6\n7\n", "gone": b"edited\n"},
        {"a": b"new\n2\n3\n4\n5\n6\n7\n", "added": b"y\n"},
    )
    assert merge.clean == {"added": b"y\n"}
    assert merge.conflicts["gone"] is None
    conflict = merge.conflicts["a"]
    assert conflict.startswith(b"<<<<<<< ") and conflict.endswith(b"\n2\n3\n4\n5\n6\n7\n")
    assert b"mine\n=======\nnew\n>>>>>>> " in conflict
