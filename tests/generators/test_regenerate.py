"""``generator.regenerate()``: a model change reaches a generated folder without losing edits.

Every edit is either kept or flagged; ``generate()`` itself is unchanged.
"""
import json
import os
import shutil
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


def read(folder, rel="app.py"):
    return (folder / rel).read_bytes()


def edit(folder, old, new, rel="app.py"):
    path = folder / rel
    content = path.read_bytes()
    assert old in content
    path.write_bytes(content.replace(old, new))


@pytest.fixture
def folder(tmp_path):
    out = tmp_path / "out"
    regen(out, {"app.py": APP})
    return out


def test_the_first_run_writes_the_output_and_its_record(tmp_path):
    out = tmp_path / "out"
    report = regen(out, {"app.py": APP, "pkg/mod.py": b"x = 1\n"})
    assert read(out) == APP
    assert sorted(report.written) == ["app.py", "pkg/mod.py"]
    assert (out / RECORD_FILENAME).is_file()


def test_generate_is_unchanged_and_writes_no_record(tmp_path):
    out = tmp_path / "out"
    FilesGenerator({"app.py": APP}, output_dir=str(out)).generate()
    assert not (out / RECORD_FILENAME).exists()


def test_an_edit_survives_a_model_change_elsewhere_in_the_file(folder):
    edit(folder, b"return 1\n", b"return 1  # mine\n")
    report = regen(folder, {"app.py": APP.replace(b"return 3", b"return 33")})
    assert read(folder) == APP.replace(b"return 1\n", b"return 1  # mine\n").replace(b"return 3", b"return 33")
    assert report.merged == ["app.py"] and not report.conflicted


def test_a_file_nobody_edited_takes_the_new_output(folder):
    report = regen(folder, {"app.py": APP.replace(b"return 1", b"return 11")})
    assert report.written == ["app.py"] and not report.merged
    assert b"return 11" in read(folder)


def test_a_run_with_no_model_change_touches_nothing(folder):
    edit(folder, b"return 1\n", b"return 1  # mine\n")
    report = regen(folder, {"app.py": APP})
    assert not (report.written or report.merged or report.conflicted or report.deleted)
    assert b"# mine" in read(folder)


def test_the_users_own_files_are_never_touched(folder):
    (folder / "mine.py").write_bytes(b"mine = 1\n")
    regen(folder, {"app.py": APP.replace(b"return 1", b"return 11")})
    assert read(folder, "mine.py") == b"mine = 1\n"


def test_a_clashing_edit_gets_markers_and_is_reported(folder):
    edit(folder, b"return 1\n", b"return 100\n")
    report = regen(folder, {"app.py": APP.replace(b"return 1\n", b"return 11\n")})
    content = read(folder)
    assert report.conflicted == ["app.py"]
    assert b"<<<<<<< " in content and b"return 100" in content and b"return 11" in content
    assert b"return 2" in content  # only the clash is marked, not the whole file


def test_the_next_run_refuses_until_the_conflict_is_resolved(folder):
    edit(folder, b"return 1\n", b"return 100\n")
    regen(folder, {"app.py": APP.replace(b"return 1\n", b"return 11\n")})
    before = read(folder)

    with pytest.raises(RegenerationError, match="app.py"):
        regen(folder, {"app.py": APP.replace(b"return 1\n", b"return 12\n")})
    assert read(folder) == before

    (folder / "app.py").write_bytes(APP.replace(b"return 1\n", b"return 100\n"))
    report = regen(folder, {"app.py": APP.replace(b"return 1\n", b"return 11\n").replace(b"return 3", b"return 33")})
    assert not report.conflicted
    assert b"return 100" in read(folder) and b"return 33" in read(folder)


def test_an_unedited_file_the_model_dropped_is_deleted(tmp_path):
    out = tmp_path / "out"
    regen(out, {"app.py": APP, "old.py": b"old = 1\n"})
    report = regen(out, {"app.py": APP})
    assert report.deleted == ["old.py"] and not (out / "old.py").exists()


def test_an_edited_file_the_model_dropped_is_kept_and_reported(tmp_path):
    out = tmp_path / "out"
    regen(out, {"app.py": APP, "old.py": b"old = 1\n"})
    edit(out, b"old = 1", b"old = 2  # mine", rel="old.py")
    report = regen(out, {"app.py": APP})
    assert report.kept_removed_from_model == ["old.py"] and b"# mine" in read(out, "old.py")


def test_a_deleted_file_the_model_changed_is_reported_not_restored(tmp_path):
    out = tmp_path / "out"
    regen(out, {"app.py": APP, "b.py": b"b = 1\n"})
    (out / "b.py").unlink()
    report = regen(out, {"app.py": APP, "b.py": b"b = 2\n"})
    assert report.not_restored == ["b.py"] and not (out / "b.py").exists()


def test_a_deleted_file_the_model_left_alone_stays_deleted(tmp_path):
    out = tmp_path / "out"
    regen(out, {"app.py": APP, "b.py": b"b = 1\n"})
    (out / "b.py").unlink()
    report = regen(out, {"app.py": APP, "b.py": b"b = 1\n"})
    assert not (out / "b.py").exists() and not report.written


def test_a_file_the_user_created_where_the_model_adds_one_is_flagged(folder):
    (folder / "new.py").write_bytes(b"mine = 1\n")
    report = regen(folder, {"app.py": APP, "new.py": b"generated = 1\n"})
    assert report.conflicted == ["new.py"]
    assert b"mine = 1" in read(folder, "new.py") and b"generated = 1" in read(folder, "new.py")


def test_a_folder_generated_before_is_refused_then_adopted(tmp_path):
    out = tmp_path / "out"
    FilesGenerator({"app.py": APP}, output_dir=str(out)).generate()
    edit(out, b"return 1\n", b"return 1  # mine\n")

    with pytest.raises(RegenerationError, match="adopt"):
        regen(out, {"app.py": APP.replace(b"return 3", b"return 33")})
    assert b"return 33" not in read(out)

    before = read(out)
    report = regen(out, {"app.py": APP}, adopt=True)  # the model those files came from
    assert read(out) == before and report.adopted and not report.written

    regen(out, {"app.py": APP.replace(b"return 3", b"return 33")})
    assert b"# mine" in read(out) and b"return 33" in read(out)


def test_adopting_a_folder_that_has_a_record_is_refused(folder):
    with pytest.raises(RegenerationError, match="already"):
        regen(folder, {"app.py": APP}, adopt=True)


def test_windows_line_endings_are_kept_and_do_not_read_as_edits(folder):
    (folder / "app.py").write_bytes(APP.replace(b"\n", b"\r\n"))
    report = regen(folder, {"app.py": APP.replace(b"return 3", b"return 33")})
    assert report.written == ["app.py"] and not report.merged
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
    (folder / "app.py").write_bytes(edited)
    report = regen(folder, {"app.py": APP.replace(b"return 1\n", b"return 11\n")})
    assert read(folder) == edited
    assert report.not_merged == ["app.py"] and not report.conflicted


@pytest.mark.parametrize("rel", ["../evil.py", "/etc/evil.py", "a/../../evil.py", "C:/evil.py", "a\\..\\..\\evil.py"])
def test_a_record_naming_a_path_outside_the_folder_is_refused(folder, rel):
    record = folder / RECORD_FILENAME
    data = json.loads(record.read_text(encoding="utf-8"))
    data["files"][rel] = "x\n"
    record.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RegenerationError, match="record"):
        regen(folder, {"app.py": APP})


@pytest.mark.parametrize("content", ["{not json", "[]", '{"format": 99, "files": {}}', '{"format": 1, "files": 3}'])
def test_an_unusable_record_is_refused_not_overwritten(folder, content):
    (folder / RECORD_FILENAME).write_text(content, encoding="utf-8")
    edit(folder, b"return 1\n", b"return 1  # mine\n")
    with pytest.raises(RegenerationError, match="record"):
        regen(folder, {"app.py": APP})
    assert b"# mine" in read(folder)


def test_without_pygit2_it_refuses_and_writes_nothing(folder, monkeypatch):
    monkeypatch.setattr(three_way_merge, "pygit2", None)
    edit(folder, b"return 1\n", b"return 1  # mine\n")
    with pytest.raises(RegenerationError, match=r"besser\[regen\]"):
        regen(folder, {"app.py": APP.replace(b"return 3", b"return 33")})
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
    (out / "app.py").write_bytes(APP)
    (out / name).write_text("{}", encoding="utf-8")
    with pytest.raises(RegenerationError, match="Spec-Driven"):
        regen(out, {"app.py": APP.replace(b"return 3", b"return 33")}, adopt=adopt)
    assert read(out) == APP and not (out / RECORD_FILENAME).exists()


def test_the_generators_output_dir_is_restored(tmp_path):
    out = str(tmp_path / "out")
    generator = FilesGenerator({"app.py": APP}, output_dir=out)
    generator.regenerate()
    assert generator.output_dir == out


def test_a_failing_generator_changes_nothing(folder):
    generator = FilesGenerator({"app.py": b"half\n"}, output_dir=str(folder))
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
        regen(folder, {"app.py": APP.replace(b"return 3", b"return 33")})
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
        _fail_replacing(patch, "app.py")
        with pytest.raises(OSError):
            regen(folder, {"app.py": APP.replace(b"return 3", b"return 33")})
    assert read(folder) == edited
    assert not list(folder.glob(".besser_tmp_*"))

    report = regen(folder, {"app.py": APP.replace(b"return 3", b"return 33")})
    assert report.merged == ["app.py"] and b"# mine" in read(folder) and b"return 33" in read(folder)


def test_an_interrupted_first_run_resumes_without_adopting(tmp_path, monkeypatch):
    out = tmp_path / "out"
    files = {"a.py": b"a = 1\n", "b.py": b"b = 1\n"}
    with monkeypatch.context() as patch:
        _fail_replacing(patch, "b.py")
        with pytest.raises(OSError):
            regen(out, files)
    assert read(out, "a.py") == files["a.py"] and not (out / "b.py").exists()

    report = regen(out, files)
    assert read(out, "b.py") == files["b.py"] and not report.conflicted


def test_a_crash_after_a_conflict_cannot_nest_markers(tmp_path, monkeypatch):
    out = tmp_path / "out"
    regen(out, {"a.py": APP, "z.py": b"z = 1\n"})
    edit(out, b"return 1\n", b"return 100\n", rel="a.py")
    changed = {"a.py": APP.replace(b"return 1\n", b"return 11\n"), "z.py": b"z = 2\n"}
    with monkeypatch.context() as patch:
        _fail_replacing(patch, "z.py")
        with pytest.raises(OSError):
            regen(out, changed)
    assert read(out, "a.py").count(b"<<<<<<< ") == 1

    with pytest.raises(RegenerationError, match="a.py"):
        regen(out, changed)
    assert read(out, "a.py").count(b"<<<<<<< ") == 1


def test_a_folder_holding_only_other_files_needs_no_adoption(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "README.md").write_bytes(b"# mine\n")
    (out / ".git").mkdir()
    report = regen(out, {"app.py": APP})
    assert report.written == ["app.py"] and read(out, "README.md") == b"# mine\n"


def test_adopting_says_which_files_differ_from_that_model(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "app.py").write_bytes(APP + b"# mine\n")
    (out / "same.py").write_bytes(b"s = 1\n")
    report = regen(out, {"app.py": APP, "same.py": b"s = 1\n", "absent.py": b"x = 1\n"}, adopt=True)
    assert report.differs_from_model == ["absent.py", "app.py"]
    assert not (out / "absent.py").exists()


def test_another_generators_folder_is_refused_not_deleted(folder):
    class OtherGenerator(FilesGenerator):
        pass

    with pytest.raises(RegenerationError, match="FilesGenerator"):
        OtherGenerator({"other.py": b"o = 1\n"}, output_dir=str(folder)).regenerate()
    assert read(folder) == APP


def test_regenerating_needs_an_explicit_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    generator = FilesGenerator({"app.py": APP})
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
    (folder / "app.py").write_bytes(edited)
    report = regen(folder, {"app.py": APP.replace(b"return 1\n", b"return 11\n")})
    assert read(folder) == edited and report.not_merged == ["app.py"]


def test_a_folder_at_a_generated_path_fails_before_anything_is_written(tmp_path):
    out = tmp_path / "out"
    regen(out, {"a.py": b"a = 1\n", "b.py": b"b = 1\n"})
    (out / "b.py").unlink()
    (out / "b.py").mkdir()
    with pytest.raises(RegenerationError, match="b.py"):
        regen(out, {"a.py": b"a = 2\n", "b.py": b"b = 2\n"})
    assert read(out, "a.py") == b"a = 1\n"


def test_a_file_at_a_generated_parent_fails_before_anything_is_written(tmp_path):
    out = tmp_path / "out"
    regen(out, {"a.py": b"a = 1\n"})
    (out / "pkg").write_bytes(b"mine\n")
    with pytest.raises(RegenerationError, match="pkg"):
        regen(out, {"a.py": b"a = 2\n", "pkg/mod.py": b"m = 1\n"})
    assert read(out, "a.py") == b"a = 1\n" and read(out, "pkg") == b"mine\n"


@pytest.mark.skipif(os.name == "nt", reason="needs symlinks and POSIX modes")
def test_a_link_at_a_generated_path_is_refused(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"mine\n")
    out = tmp_path / "out"
    regen(out, {"a.py": b"a = 1\n"})
    (out / "link.py").symlink_to(outside)
    with pytest.raises(RegenerationError, match="link.py"):
        regen(out, {"a.py": b"a = 1\n", "link.py": b"x = 1\n"})
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
    regen(folder, {"app.py": APP.replace(b"return 1\n", b"return 11\n")})
    (folder / "app.py").write_bytes(APP + b"SAMPLE = 'x <<<<<<< y'\n")
    report = regen(folder, {"app.py": APP.replace(b"return 2", b"return 22")})
    assert not report.conflicted and b"return 22" in read(folder)


def test_conflict_markers_say_which_side_is_which(folder):
    edit(folder, b"return 1\n", b"return 100\n")
    regen(folder, {"app.py": APP.replace(b"return 1\n", b"return 11\n")})
    assert b"<<<<<<< your edit\n" in read(folder) and b">>>>>>> generated from your model\n" in read(folder)


def test_a_backend_that_pushes_an_image_is_refused(tmp_path):
    from besser.generators.backend import BackendGenerator

    generator = BackendGenerator(_shelf(), output_dir=str(tmp_path / "out"), docker_image=True)
    with pytest.raises(RegenerationError, match="image"):
        generator.regenerate()


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
    classes.write_text(classes.read_text(encoding="utf-8") + "\n\ndef shelf_size(books):  # mine\n    return len(books)\n",
                       encoding="utf-8")

    report = PythonGenerator(_shelf(isbn=True), output_dir=str(out)).regenerate()

    text = classes.read_text(encoding="utf-8")
    assert "shelf_size" in text and "isbn" in text and not report.conflicted


@pytest.mark.skipif(shutil.which("django-admin") is None, reason="needs django-admin")
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


def test_the_merge_helper_marks_only_the_clash():
    merge = three_way_merge.merge_trees(
        {"a": b"1\n2\n3\n4\n5\n6\n7\n", "gone": b"x\n"},
        {"a": b"mine\n2\n3\n4\n5\n6\n7\n", "gone": b"edited\n"},
        {"a": b"new\n2\n3\n4\n5\n6\n7\n", "added": b"y\n"},
    )
    assert merge.clean == {"added": b"y\n"}
    assert merge.conflicts["gone"] is None
    assert b"<<<<<<< " in merge.conflicts["a"] and b"7\n" in merge.conflicts["a"]
