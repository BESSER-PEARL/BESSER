"""An input inside its ``<label>`` is kept without unbinding its form.

A ``<label>`` around an input used to be read as text, which dropped the input.
Keeping it must not stop the form from binding to its class when that input
(a consent checkbox, a "remember me" switch) edits no attribute.
"""
from besser.BUML.metamodel.gui import Form, InputFieldType

from .utils_gui_binding_tests import class_diagram, import_gui


def _text(content):
    return {"type": "textnode", "content": content}


def _label(*components, **attributes):
    return {"tagName": "label", "attributes": attributes,
            "components": [_text(c) if isinstance(c, str) else c for c in components]}


def _input(**attributes):
    return {"tagName": "input", "attributes": attributes}


def _form(*components):
    return {"tagName": "form", "components": [
        _label("Title", **{"for": "t"}), _input(id="t", name="title"), *components,
        {"tagName": "button", "attributes": {"type": "submit"}, "components": [_text("Save")]},
    ]}


AGREE = _label(_input(type="checkbox", name="agree", required=""), " I agree to the ",
               {"tagName": "a", "attributes": {"href": "/terms"}, "components": [_text("terms")]})


def _classes(*names):
    """One class per name, each with a ``title`` and an optional ``done``."""
    class_json = None
    for name in names:
        one = class_diagram(name, [("title", "str"), ("done", "bool")])
        one["elements"][f"attr-{name}-done"]["isOptional"] = True
        if class_json is None:
            class_json = one
        else:
            class_json["elements"].update(one["elements"])
    return class_json


def _form_of(gui):
    form, = [e for m in gui.modules for s in m.screens for e in s.view_elements if isinstance(e, Form)]
    return form


def _bound_class(form):
    return getattr(getattr(form, "data_binding", None), "domain_concept", None) and form.data_binding.domain_concept.name


def test_a_consent_checkbox_in_its_label_does_not_unbind_the_form():
    form = _form_of(import_gui([_form(AGREE)], _classes("Task")))

    assert _bound_class(form) == "Task"
    agree, = [i for i in form.inputFields if i.field_type == InputFieldType.Checkbox]
    assert agree.label == "I agree to the terms" and agree.required is True
    assert agree.data_binding is None


def test_unnamed_radios_and_a_switch_in_labels_do_not_unbind_the_form():
    form = _form_of(import_gui([_form(
        _label(_input(type="checkbox", name="remember"), " Remember me"),
        _label(_input(type="radio", value="a"), " A"), _label(_input(type="radio", value="b"), " B"),
    )], _classes("Task")))

    assert _bound_class(form) == "Task"


def test_a_labelled_input_naming_an_attribute_still_chooses_the_class():
    form = _form_of(import_gui([_form(_label(_input(type="checkbox", name="done"), " Done"))],
                               _classes("Task", "Note")))
    # title and done name both classes: still ambiguous, so not bound
    assert _bound_class(form) is None

    classes = _classes("Task", "Note")
    del classes["elements"]["attr-Note-done"]
    classes["elements"]["cls-Note"]["attributes"].remove("attr-Note-done")
    form = _form_of(import_gui([_form(_label(_input(type="checkbox", name="done"), " Done"))], classes))
    assert _bound_class(form) == "Task"


def test_a_labelled_input_naming_nothing_does_not_pick_a_class():
    form = _form_of(import_gui([_form(AGREE)], _classes("Task", "Note")))

    assert _bound_class(form) is None  # title alone names both classes


def test_a_labelled_input_naming_another_class_does_not_unbind_the_form():
    classes = _classes("Task", "Note")
    for attr in ("title", "done"):
        del classes["elements"][f"attr-Note-{attr}"]
    classes["elements"]["cls-Note"]["attributes"] = []
    pinned = class_diagram("Note", [("pinned", "bool")])["elements"]
    classes["elements"].update({k: v for k, v in pinned.items() if k != "cls-Note"})
    classes["elements"]["cls-Note"]["attributes"] = ["attr-Note-pinned"]

    form = _form_of(import_gui([_form(_label(_input(type="checkbox", name="pinned"), " Pinned"))], classes))
    # Read without the labelled input, as before such inputs were kept: Task
    assert _bound_class(form) == "Task"


def test_a_consent_checkbox_in_a_text_block_does_not_unbind_the_form():
    """A GrapesJS text block (``type: text`` on a div) around an input is read like a label."""
    block = {"type": "text", "tagName": "div", "components": [
        _input(type="checkbox", name="agree"), _text(" I accept the terms")]}
    form = _form_of(import_gui([_form(block)], _classes("Task")))

    assert _bound_class(form) == "Task"
