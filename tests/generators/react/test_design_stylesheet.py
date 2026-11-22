"""The GUI model's stylesheet must reach the generated React app.

Measured on three AI designs: the app had 0 of the design's CSS rules, no
class used in a page had a rule, and data widgets dropped their class names.
"""
import os

from besser.BUML.metamodel.gui import DataBinding, GUIModel, Module, Screen
from besser.BUML.metamodel.gui.dashboard import FieldColumn, Table
from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
import besser.generators.react as react_package
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
    assert "`table-wrapper bsr-runtime ${className}`" in table_component


def test_browser_defaults_the_reset_removes_are_restored_below_the_design(tmp_path):
    """Tailwind's reset gives headings the body's size and weight and strips a
    plain button; a design that sets neither expects the browser's defaults."""
    src = _generate(tmp_path, STYLESHEET)
    design_css = _read(src, "design.css")

    restore = design_css[design_css.index("@layer base {"):design_css.index(".app-btn{")]
    headings = restore[restore.index(".app-main :is(h1, h2, h3, h4, h5, h6):not(.bsr-runtime *) {"):]
    assert "font-size: revert;" in headings and "font-weight: revert;" in headings
    button = restore[restore.index(".app-main button:not(.bsr-runtime *) {"):]
    assert "background-color: revert;" in button and "border: revert;" in button
    # Pages render inside .app-main; the reset the block answers is imported first.
    assert '<main className="app-main">' in _read(src, "App.tsx")
    assert '@import "tailwindcss";' in _read(src, "index.css")
    index = _read(src, "index.tsx")
    assert index.index("import './index.css';") < index.index("import './design.css';")


def test_the_restored_defaults_leave_runtime_components_and_text_colour_alone(tmp_path):
    """Tables, charts, maps, the agent widget and method dialogs style their own
    headings and buttons and let text colour inherit: reverting them made table
    and chart titles large and bold and the table's buttons black or grey."""
    src = _generate(tmp_path, STYLESHEET)
    design_css = _read(src, "design.css")

    restore = design_css[design_css.index("@layer base {"):design_css.index(".app-btn{")]
    assert "color: revert" not in restore.replace("background-color: revert", "")
    assert restore.count(":not(.bsr-runtime *)") == 2
    for parts in (("table", "TableComponent.tsx"), ("charts", "BarChartComponent.tsx"),
                  ("charts", "LineChartComponent.tsx"), ("charts", "PieChartComponent.tsx"),
                  ("charts", "RadarChartComponent.tsx"), ("charts", "RadialBarChartComponent.tsx"),
                  ("MethodButton.tsx",)):
        assert "bsr-runtime" in _read(src, "components", *parts), parts
    # Emitted only when a page uses them
    components = os.path.join(os.path.dirname(react_package.__file__), "templates", "src", "components")
    for parts in (("runtime", "MapBlock.tsx"), ("AgentComponent.tsx.j2",)):
        assert "bsr-runtime" in _read(components, *parts), parts
