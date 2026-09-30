"""The GUI design's stylesheet must survive GUI JSON -> B-UML -> JSON/code.

The editor saves an AI-authored design's own rules (``.app-*``), compound
``.ds-*`` rules and the ``:root`` design tokens with an EMPTY ``selectors``
list -- the selector lives in ``selectorsAdd`` -- and ``@media``/state rules
next to them. The processor used to keep only rules with a non-empty
``selectors`` list, flattened into per-element styles, so measured AI designs
with 139/207/150 rules reached the generated app with 0 CSS rules and none of
the 123 class names used in pages had a rule. Every rule not tied to one
element id now lives verbatim in ``GUIModel.stylesheet``.
"""

import os

from besser.BUML.metamodel.gui import GUIModel
from besser.utilities.buml_code_builder.gui_model_builder import gui_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.gui_diagram_converter import (
    _serialize_gui_model,
    gui_buml_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.gui_processors import (
    process_gui_diagram,
)

# Rule shapes copied from an editor-saved AI design.
DESIGN_STYLES = [
    {"selectors": [], "selectorsAdd": ".app-btn", "style": {"background": "var(--ds-primary)", "border-radius": "8px"}},
    {"selectors": [], "selectorsAdd": ":root", "style": {"--ds-primary": "#15293d", "--ds-radius": "14px"}},
    {"selectors": [], "selectorsAdd": ".app-kpis", "style": {"grid-template-columns": "repeat(2,1fr)"},
     "mediaText": "(max-width:850px)", "atRuleType": "media"},
    {"selectors": [], "selectorsAdd": ".app-card h2,.app-card h3", "style": {"margin": "0"}},
    {"selectors": ["ds-btn"], "style": {"padding": "8px 16px"}},
    {"selectors": [], "selectorsAdd": ".ds-btn:hover", "style": {"transform": "translateY(-1px)"}},
    {"selectors": ["ds-grid-2"], "style": {"grid-template-columns": "1fr"},
     "mediaText": "(max-width: 768px)", "atRuleType": "media"},
    {"selectors": ["#hero"], "style": {"color": "#ffffff"}},
]


def _gui_json(styles=None):
    return {
        "title": "Shop",
        "styles": DESIGN_STYLES if styles is None else styles,
        "pages": [{
            "id": "home",
            "name": "Home",
            "frames": [{"component": {"type": "wrapper", "components": [
                {"tagName": "div", "attributes": {"id": "hero"}, "classes": ["ds-btn", "app-btn"],
                 "components": [{"type": "textnode", "content": "Start"}]},
                {"tagName": "div", "attributes": {"id": "plain"}, "classes": [{"name": "ds-btn"}]},
            ]}}],
        }],
    }


def _elements(gui_model):
    screen = next(iter(next(iter(gui_model.modules)).screens))
    return {element.name: element for element in screen.view_elements}


def test_gui_model_stylesheet_defaults_to_empty():
    gui = GUIModel(name="G", package="", versionCode="1", versionName="1", modules=set(), description="")
    assert gui.stylesheet == ""
    gui.stylesheet = None
    assert gui.stylesheet == ""


def test_rules_with_empty_selectors_reach_the_stylesheet():
    stylesheet = process_gui_diagram(_gui_json(), {}, None).stylesheet

    assert ".app-btn{background:var(--ds-primary);border-radius:8px}" in stylesheet
    assert ":root{--ds-primary:#15293d;--ds-radius:14px}" in stylesheet
    assert "@media (max-width:850px){.app-kpis{grid-template-columns:repeat(2,1fr)}}" in stylesheet
    assert ".app-card h2,.app-card h3{margin:0}" in stylesheet
    assert ".ds-btn:hover{transform:translateY(-1px)}" in stylesheet


def test_class_and_media_rules_on_selectors_keep_order():
    stylesheet = process_gui_diagram(_gui_json(), {}, None).stylesheet

    # A media rule used to overwrite the base rule of the same class.
    assert ".ds-btn{padding:8px 16px}" in stylesheet
    assert "@media (max-width: 768px){.ds-grid-2{grid-template-columns:1fr}}" in stylesheet
    order = [".app-btn{", ":root{", "@media (max-width:850px)", ".ds-btn{", ".ds-btn:hover{", "@media (max-width: 768px)"]
    positions = [stylesheet.index(marker) for marker in order]
    assert positions == sorted(positions)


def test_element_id_rules_stay_per_element():
    gui_model = process_gui_diagram(_gui_json(), {}, None)
    elements = _elements(gui_model)

    assert "#hero" not in gui_model.stylesheet
    assert elements["hero"].styling.color.text_color == "#ffffff"
    assert elements["hero"].css_classes == ["ds-btn", "app-btn"]
    # A class rule is not copied inline: an inline style would beat the
    # stylesheet's :hover / @media rules (and a second class on the element).
    assert elements["plain"].styling is None
    assert elements["plain"].css_classes == ["ds-btn"]


def test_id_only_gui_has_no_stylesheet():
    """A Basic CRUD GUI (id rules only) is unchanged: no stylesheet."""
    assert process_gui_diagram(_gui_json([DESIGN_STYLES[-1]]), {}, None).stylesheet == ""


def test_stylesheet_survives_buml_to_json():
    gui_model = process_gui_diagram(_gui_json(), {}, None)

    editor_json = _serialize_gui_model(gui_model)
    styles = editor_json["styles"]

    # Plain class selectors go back as GrapesJS selectors, the rest as selectorsAdd.
    assert {"selectors": ["ds-btn"], "style": {"padding": "8px 16px"}} in styles
    assert any(rule.get("selectorsAdd") == ":root" for rule in styles)
    assert any(rule.get("mediaText") == "(max-width:850px)" and rule.get("atRuleType") == "media"
               for rule in styles)
    assert process_gui_diagram(editor_json, {}, None).stylesheet == gui_model.stylesheet


def test_stylesheet_survives_the_buml_code_builder(tmp_path):
    gui_model = process_gui_diagram(_gui_json(), {}, None)
    # Characters that need escaping in the exported Python source.
    gui_model.stylesheet += '.md\\:flex::after{content:"*"}\n'

    code_path = os.path.join(str(tmp_path), "gui.py")
    gui_model_to_code(gui_model, code_path)
    with open(code_path, encoding="utf-8") as f:
        code = f.read()
    assert "stylesheet=(" in code

    editor_json = gui_buml_to_json(code)
    assert process_gui_diagram(editor_json, {}, None).stylesheet == gui_model.stylesheet


def test_builder_omits_an_empty_stylesheet(tmp_path):
    gui_model = process_gui_diagram(_gui_json([DESIGN_STYLES[-1]]), {}, None)
    code_path = os.path.join(str(tmp_path), "gui.py")
    gui_model_to_code(gui_model, code_path)
    with open(code_path, encoding="utf-8") as f:
        assert "stylesheet" not in f.read()
