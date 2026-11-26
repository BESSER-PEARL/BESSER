"""A radio group in a bound form renders one radio per option and posts the chosen one.

The page rendered one empty ``<div role="radiogroup">`` per authored radio, so the
attribute could never be set and a required one made every create fail.
"""
import json
import re

CLASSES = {"Task": [("title", "str"), ("priority", "str")]}


def _radio(value, text, **extra):
    return {"tagName": "label", "components": [
        {"tagName": "input", "attributes": {"type": "radio", "name": "priority", "value": value, **extra}},
        {"type": "textnode", "content": f" {text}"},
    ]}


def _task_form():
    return {"tagName": "form", "attributes": {"id": "task-form"}, "components": [
        {"tagName": "input", "attributes": {"id": "title", "name": "title", "placeholder": "Title"}},
        {"tagName": "label", "components": [{"type": "textnode", "content": "Priority"}]},
        _radio("low", "Low"), _radio("normal", "Normal", required=""), _radio("high", "High"),
        {"tagName": "button", "attributes": {"type": "submit"}, "components": [{"type": "textnode", "content": "Add"}]},
    ]}


def test_the_page_renders_one_radio_per_option_under_one_name(build_app, jsx):
    page = build_app(CLASSES, {"Tasks": [_task_form()]}).page("Tasks")

    radios = re.findall(r'<input type="radio"[^>]*/>', page)
    assert [re.search(r'value=("[^"]*")', r).group(1) for r in radios] == ['"low"', '"normal"', '"high"']
    assert {re.search(r'name=("[^"]*")', r).group(1) for r in radios} == {'"priority"'}
    assert all("required" in r for r in radios)
    assert '{"Low"}</label>' in page and '{"High"}</label>' in page
    assert page.count('role="radiogroup"') == 1
    assert '{"Priority"}<span' in page  # the group's caption, marked required

    block = jsx(page, "FormBlock")
    fields = json.JSONDecoder().raw_decode(block[block.index("fields={") + len("fields={"):])[0]
    assert {"name": "priority", "field": "priority", "type": "str"} in fields
    assert len(fields) == 2
