"""A styled element's inline style leaves out the chart palette when it is the default.

Every ``Color`` carries ``color_palette="default"``, so each styled element of a
page got ``"--chart-color-palette": "default"`` in its ``style={{...}}``: a key
only a pie chart reads, where "default" is also what it falls back to. It is
kept where it decides something: a named palette, or "default" beside a bar
colour, which the pie chart would use instead.
"""
from besser.BUML.metamodel.gui import GUIModel, Module, Screen
from besser.BUML.metamodel.gui.graphical_ui import Text
from besser.BUML.metamodel.gui.style import Color, Styling
from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
from besser.generators.react import ReactGenerator

PALETTE = "--chart-color-palette"


def _generator(tmp_path, styling=None):
    note = Class(name="Note", attributes={Property(name="body", type=StringType)})
    text = Text(name="headline", content="Notes", styling=styling)
    screen = Screen(name="Home", description="", view_elements={text}, is_main_page=True)
    gui = GUIModel(name="GUI", package="app", versionCode="1", versionName="1.0", description="",
                   modules={Module(name="Main", screens={screen})})
    return ReactGenerator(model=DomainModel(name="Notes", types={note}), gui_model=gui, output_dir=str(tmp_path))


def test_the_default_palette_is_left_out(tmp_path):
    generator = _generator(tmp_path)

    style = generator._style_from_styling(Styling(color=Color(background_color="#fde68a")))

    assert style == {"background": "#fde68a"}


def test_a_palette_that_decides_something_is_kept(tmp_path):
    generator = _generator(tmp_path)

    named = generator._style_from_styling(Styling(color=Color(color_palette="vivid")))
    beside_a_bar_colour = generator._style_from_styling(Styling(color=Color(bar_color="#ef4444")))

    assert named[PALETTE] == "vivid"
    assert beside_a_bar_colour[PALETTE] == "default"


def test_a_generated_page_carries_no_default_palette(tmp_path):
    _generator(tmp_path, Styling(color=Color(text_color="#1e3a8a"))).generate()
    page = (tmp_path / "src" / "pages" / "Home.tsx").read_text(encoding="utf-8")

    assert '"#1e3a8a"' in page
    assert PALETTE not in page
