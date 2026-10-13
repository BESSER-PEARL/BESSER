"""The GUI model's stylesheet must reach the generated React app.

Measured on three AI designs: the app had 0 of the design's CSS rules, no
class used in a page had a rule, and data widgets dropped their class names.
"""
import os

from besser.BUML.metamodel.gui import DataBinding, GUIModel, Module, Screen
from besser.BUML.metamodel.gui.dashboard import FieldColumn, Table
from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
from besser.generators.react import ReactGenerator

STYLESHEET = (
    ".app-btn{background:var(--ds-primary)}\n"
    ":root{--ds-primary:#15293d}\n"
    "@media (max-width:850px){.app-kpis{grid-template-columns:1fr}}\n"
)


def _generate(tmp_path, stylesheet):
    title = Property(name="title", type=StringType)
    book = Class(name="Book", attributes={title})
    domain = DomainModel(name="Library", types={book})
    table = Table(
        name="books",
        label="Books",
        columns=[FieldColumn(label="Title", field=title)],
        data_binding=DataBinding(domain_concept=book),
        css_classes=["app-table", "has-data-binding"],
    )
    screen = Screen(name="Books", description="Books", view_elements={table}, is_main_page=True)
    gui = GUIModel(name="LibraryGUI", package="", versionCode="1", versionName="1.0",
                   modules={Module(name="M", screens={screen})}, description="",
                   stylesheet=stylesheet)
    ReactGenerator(model=domain, gui_model=gui, output_dir=str(tmp_path)).generate()
    return os.path.join(str(tmp_path), "src")


def _read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


def test_the_stylesheet_is_written_and_imported(tmp_path):
    src = _generate(tmp_path, STYLESHEET)

    design_css = _read(src, "design.css")
    assert ".app-btn{" in design_css
    assert "--ds-primary" in design_css
    assert "@media (max-width:850px)" in design_css

    index = _read(src, "index.tsx")
    assert "import './design.css';" in index
    # After App so the design wins ties with component CSS pulled in by App.
    assert index.index("import App from './App';") < index.index("import './design.css';")


def test_no_stylesheet_means_no_design_css(tmp_path):
    src = _generate(tmp_path, "")

    assert not os.path.exists(os.path.join(src, "design.css"))
    assert "design.css" not in _read(src, "index.tsx")


def test_a_bound_table_keeps_its_class_names(tmp_path):
    src = _generate(tmp_path, STYLESHEET)

    page = _read(src, "pages", "Books.tsx")
    table_tag = page[page.index("<TableBlock"):]
    table_tag = table_tag[:table_tag.index("/>")]
    assert 'className="app-table has-data-binding"' in table_tag

    # ... and the runtime components forward it to the rendered root element.
    assert "className={className}" in _read(src, "components", "runtime", "TableBlock.tsx")
    table_component = _read(src, "components", "table", "TableComponent.tsx")
    assert "`table-wrapper ${className}`" in table_component
