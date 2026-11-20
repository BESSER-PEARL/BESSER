"""An authored ``<select>`` in a bound form renders its options, with one blank choice."""
import re

CLASSES = {"Task": [("title", "str"), ("priority", "str")]}


def _option(text, value):
    return {"tagName": "option", "attributes": {"value": value}, "components": [{"type": "textnode", "content": text}]}


def test_the_page_renders_the_options_of_the_select(build_app, jsx):
    form = {"tagName": "form", "attributes": {"id": "task-form"}, "components": [
        {"tagName": "input", "attributes": {"id": "title", "name": "title", "placeholder": "Title"}},
        {"tagName": "select", "attributes": {"id": "priority", "name": "priority"}, "components": [
            _option("Choose a priority", ""), _option("Low", "low"), _option("High", "high")]},
        {"tagName": "button", "attributes": {"type": "submit"}, "components": [{"type": "textnode", "content": "Add"}]},
    ]}
    page = build_app(CLASSES, {"Tasks": [form]}).page("Tasks")

    select = page[page.index("<select"):page.index("</select>")]
    assert re.findall(r'<option value=("[^"]*")', select) == ['""', '"low"', '"high"']
    assert '{"Low"}</option>' in select and '{"High"}</option>' in select
    assert '{"name": "priority", "field": "priority", "type": "str"}' in jsx(page, "FormBlock")
