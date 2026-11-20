"""FlutterGenerator records which lines of sql_helper.dart and main.dart come from which model element."""
import os

import pytest

from besser.BUML.metamodel.gui import DataList, DataSourceElement, GUIModel, Module, Screen
from besser.BUML.metamodel.structural import Class, DomainModel, IntegerType, Property, StringType
from besser.generators.flutter import FlutterGenerator


def _models(with_ids: bool, rich: bool = True):
    """``rich`` has two classes and attributes; the generator iterates sets of them in an order that changes
    between runs, so a comparison of two generations uses the single class and attribute of ``rich=False``."""
    title, pages = Property(name="title", type=StringType), Property(name="pages", type=IntegerType)
    book = Class(name="Book", attributes={title, pages} if rich else {title})
    name = Property(name="name", type=StringType)
    author = Class(name="Author", attributes={name})
    domain = DomainModel(name="Shelf", types={book, author} if rich else {book})
    source = DataSourceElement(name="BookSource", dataSourceClass=book, fields={title, pages} if rich else {title})
    books = DataList(name="BookList", description="", list_sources={source})
    home = Screen(name="Home", description="", x_dpi="", y_dpi="", screen_size="Small", view_elements=set(),
                  is_main_page=True)
    listing = Screen(name="BookScreen", description="", x_dpi="", y_dpi="", screen_size="Small",
                     view_elements={books})
    module = Module(name="Main", screens={home, listing})
    gui = GUIModel(name="shelf", package="p", versionCode="1", versionName="1", description="",
                   screenCompatibility=True, modules={module})
    if with_ids:
        for ident, element in (("class-book", book), ("class-author", author), ("attr-title", title),
                               ("attr-pages", pages), ("attr-name", name), ("screen-home", home),
                               ("screen-books", listing)):
            element.id = ident
    return domain, gui, module, home


def _generate(tmp_path, folder, with_ids, rich=True):
    domain, gui, module, home = _models(with_ids, rich)
    generator = FlutterGenerator(domain, gui, home, module, output_dir=str(tmp_path / folder))
    generator.generate()
    files = {}
    for name in os.listdir(tmp_path / folder):
        with open(tmp_path / folder / name, encoding="utf-8") as f:
            files[name] = f.read()
    return generator, files


@pytest.fixture(scope="module")
def traced(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("flutter_trace")
    generator, files = _generate(tmp_path, "traced", with_ids=True)
    return generator, files, tmp_path


def _text(files, path, region):
    return "\n".join(files[path].split("\n")[region.start_line - 1:region.end_line])


def test_traces_of_the_three_files_are_merged_under_their_paths(traced):
    generator, files, _ = traced
    assert set(generator.traced_files) == set(files) == {"sql_helper.dart", "main.dart", "pubspec.yaml"}


def test_sql_helper_has_regions_for_tables_columns_and_create_statements(traced):
    generator, files, _ = traced
    regions = generator.traced_files["sql_helper.dart"].regions
    by_rule = {}
    for region in regions:
        by_rule.setdefault(region.rule.split("#")[1], {})[region.element_name] = region
    assert set(by_rule) == {"table_name", "column_name", "create_table"}
    assert _text(files, "sql_helper.dart", by_rule["table_name"]["Book"]) == "static const String tableBooks = 'books';"
    column = by_rule["column_name"]["title"]
    assert _text(files, "sql_helper.dart", column) == "static const String columnTitle = 'title';"
    assert (column.element_id, column.owner.element_id) == ("attr-title", "class-book")
    create = _text(files, "sql_helper.dart", by_rule["create_table"]["Book"])
    assert create.startswith("    await database.execute(") and "CREATE TABLE $tableBooks(" in create
    assert "$tableAuthors" not in create


def test_main_dart_has_regions_for_the_main_screen_and_each_other_screen(traced):
    generator, files, _ = traced
    regions = {r.element_name: r for r in generator.traced_files["main.dart"].regions}
    assert set(regions) == {"Home", "BookScreen"}
    home = _text(files, "main.dart", regions["Home"])
    assert home.startswith("class MyHomePage extends StatefulWidget") and "BookScreen" not in home
    listing = _text(files, "main.dart", regions["BookScreen"])
    assert listing.startswith("//for BookScreen screen") and "class _BookScreenState" in listing
    assert regions["Home"].end_line < regions["BookScreen"].start_line
    assert regions["Home"].rule == "flutterCodeGeneratorMainFile.py.j2#main_screen"


def test_pubspec_and_unlisted_files_stay_file_level(traced):
    generator, _, _ = traced
    assert not generator.traced_files["pubspec.yaml"].regions and not generator.traced_files["pubspec.yaml"].sources


def test_tracing_does_not_change_the_generated_code(tmp_path):
    traced_generator, with_ids = _generate(tmp_path, "traced", with_ids=True, rich=False)
    plain_generator, plain = _generate(tmp_path, "plain", with_ids=False, rich=False)
    assert any(t.regions for t in traced_generator.traced_files.values())
    assert with_ids == plain
    assert not any("\x02" in text or "\x03" in text for text in with_ids.values())
    # Elements without ids render untraced.
    assert not any(t.regions or t.sources for t in plain_generator.traced_files.values())
