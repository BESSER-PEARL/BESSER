"""A method's parameter dialog names each parameter like a person would.

The dialog labelled a parameter with its identifier and its type
("min_pages (int)"), and its selects said "-- Select min_pages --".
"""

CLASSES = {"Order": [("reference", "str")]}


def test_parameters_are_shown_by_a_readable_label_without_their_type(build_app):
    app = build_app(CLASSES, {"Orders": [
        {"type": "table", "attributes": {"id": "orders", "data-source": "cls-Order"}}]})
    button = app.file("components", "MethodButton.tsx")

    label = button[button.index("{parameters.map(param => ("):]
    label = label[label.index("<label"):label.index("</label>")]
    assert "{humanize(param.name)}" in label
    assert "param.type" not in label and "{param.name}" not in label
    assert "-- Select {param.name} --" not in button
    assert button.count("-- Select {humanize(param.name)} --") == 2

    # One label rule for the table dialog and the method dialog.
    shared = app.file("components", "runtime", "humanize.ts")
    table = app.file("components", "table", "TableComponent.tsx")
    assert "export const humanize = (value: string): string =>" in shared
    assert 'import { humanize } from "./runtime/humanize";' in button
    assert 'import { humanize } from "../runtime/humanize";' in table
    assert "const humanize = (" not in table
