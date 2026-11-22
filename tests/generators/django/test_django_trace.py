"""DjangoGenerator records which lines of the files it renders come from which model element."""

import importlib.util
import os
import re

import pytest

from besser.BUML.metamodel.structural import PrimitiveDataType
from besser.BUML.metamodel.traceability.traceability import model_elements
from besser.generators.django import DjangoGenerator
from tests.generators.regenerable_builders import library, library_gui

pytestmark = pytest.mark.skipif(importlib.util.find_spec("django") is None, reason="Django is not installed")

APP = "shelfsite/shelfapp"


def _models(with_ids: bool):
    model = library()
    gui = library_gui(model)
    if with_ids:
        # The primitive types are shared by every model: they keep no id.
        elements = [e for e in (*model_elements(model), *model_elements(gui)) if not isinstance(e, PrimitiveDataType)]
        for count, element in enumerate(elements):
            element.id = element.id or f"el-{count}"
    return model, gui


def _generate(tmp_path, name, with_ids):
    model, gui = _models(with_ids)
    out = tmp_path / name
    generator = DjangoGenerator(model, project_name="shelfsite", app_name="shelfapp", gui_model=gui,
                                output_dir=str(out))
    generator.generate()
    return generator, out


def _files(folder) -> dict:
    files = {}
    for root, _, names in os.walk(folder):
        for name in names:
            if name.endswith(".pyc"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as f:
                files[os.path.relpath(path, folder).replace(os.sep, "/")] = f.read()
    # startproject draws a random SECRET_KEY.
    settings = "shelfsite/shelfsite/settings.py"
    files[settings] = re.sub(r"^SECRET_KEY = .*$", "SECRET_KEY = X", files[settings], flags=re.M)
    return files


@pytest.fixture(scope="module")
def traced(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("django_trace")
    generator, out = _generate(tmp_path, "traced", with_ids=True)
    return generator, _files(out), tmp_path


def _regions(generator, path, label):
    return [r for r in generator.traced_files[path].regions if r.rule.endswith(f"#{label}")]


def _text(files, path, region):
    return "\n".join(files[path].split("\n")[region.start_line - 1:region.end_line])


def test_models_py_has_regions_for_enumerations_classes_attributes_methods_and_associations(traced):
    generator, files, _ = traced
    path = f"{APP}/models.py"
    kinds = {r.rule.split("#")[1] for r in generator.traced_files[path].regions}
    assert kinds == {"enumeration", "literal", "class", "attribute", "method", "foreign_key", "many_to_many"}
    for region in generator.traced_files[path].regions:
        if region.element_type != "BinaryAssociation":  # a field is named after the association's end
            assert region.element_name in _text(files, path, region), region
    assert "locatedIn = models.ForeignKey(" in _text(files, path, _regions(generator, path, "foreign_key")[0])
    assert "writtenBy = models.ManyToManyField(" in _text(files, path, _regions(generator, path, "many_to_many")[0])
    book = next(r for r in _regions(generator, path, "class") if r.element_name == "Book")
    assert files[path].split("\n")[book.start_line - 1] == "class Book(models.Model):"
    # Members sit inside their class and name it as owner.
    for region in generator.traced_files[path].regions:
        if region.rule.endswith(("#attribute", "#method")) and region.owner.element_name == "Book":
            assert book.start_line < region.start_line <= book.end_line
    assert {r.element_name for r in _regions(generator, path, "method")} == {"summary", "blurb"}
    assert {r.element_name for r in _regions(generator, path, "foreign_key")} == {"lib_book_assoc"}
    assert {r.element_name for r in _regions(generator, path, "many_to_many")} == {"book_author"}


def test_forms_views_urls_and_admin_have_regions_per_class(traced):
    generator, files, _ = traced
    for path, label, header in ((f"{APP}/forms.py", "form", "class BookForm("),
                                (f"{APP}/views.py", "create_view", "def create_book("),
                                (f"{APP}/views.py", "list_view", "def book_list("),
                                (f"{APP}/urls.py", "routes", "# Book-related routes"),
                                (f"{APP}/admin.py", "class", "# Admin actions for Book")):
        region = next(r for r in _regions(generator, path, label) if r.element_name == "Book")
        assert files[path].split("\n")[region.start_line - 1].strip().startswith(header), (path, label)
    assert {r.element_name for r in _regions(generator, f"{APP}/forms.py", "association_field")} == {
        "lib_book_assoc", "book_author"}
    assert {r.element_name for r in _regions(generator, f"{APP}/admin.py", "method")} == {"summary", "blurb"}


def test_html_pages_are_generated_from_their_class_and_screen(traced):
    generator, _, _ = traced
    for page, template in (("book.html", "basePageFile"), ("book_list.html", "list_page"),
                           ("book_form.html", "form_page")):
        sources = {(s.rule, s.element.element_name) for s in generator.traced_files[f"{APP}/templates/{page}"].sources}
        assert sources == {(f"{template}.py.j2#class", "Book"), (f"{template}.py.j2#screen", "BookScreen")}
    home = generator.traced_files[f"{APP}/templates/home.html"].sources
    assert [(s.rule, s.element.element_name) for s in home] == [("home_page.py.j2#screen", "Home")]


def test_every_traced_path_is_a_file_written_by_the_generator(traced):
    generator, files, _ = traced
    assert set(generator.traced_files) <= set(files)
    # Files made by django-admin are not rendered by the generator: they stay file level.
    assert "shelfsite/manage.py" not in generator.traced_files


def test_tracing_does_not_change_the_generated_project(traced, tmp_path):
    _, with_ids, _ = traced
    untraced_generator, out = _generate(tmp_path, "untraced", with_ids=False)
    plain = _files(out)
    assert with_ids == plain
    assert not any("\x02" in text or "\x03" in text for text in with_ids.values())
    # Elements without ids render untraced.
    assert all(not t.regions and not t.sources for t in untraced_generator.traced_files.values())
