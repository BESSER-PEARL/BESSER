"""A consent checkbox inside its label keeps the form posting, and is not posted itself."""
import json

CLASSES = {"Task": [("title", "str"), ("done", "bool", {"isOptional": True})]}


def test_a_bound_form_with_a_consent_checkbox_posts_only_its_attributes(build_app, jsx):
    form = {"tagName": "form", "components": [
        {"tagName": "input", "attributes": {"id": "title", "name": "title", "placeholder": "Title"}},
        {"tagName": "label", "components": [
            {"tagName": "input", "attributes": {"type": "checkbox", "name": "agree", "required": ""}},
            {"type": "textnode", "content": " I agree to the terms"}]},
        {"tagName": "button", "attributes": {"type": "submit"}, "components": [{"type": "textnode", "content": "Save"}]},
    ]}
    page = build_app(CLASSES, {"Tasks": [form]}).page("Tasks")

    block = jsx(page, "FormBlock")
    assert 'endpoint="/task/"' in block
    fields = json.JSONDecoder().raw_decode(block[block.index("fields={") + len("fields={"):])[0]
    assert [f["field"] for f in fields] == ["title"]  # the checkbox is checked, never sent
    assert "required={true}" in jsx(page, "input", 'type="checkbox"')
