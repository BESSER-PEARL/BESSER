"""A table's lookup column finds its association whatever names it.

A column that named the related class instead of the association end was
dropped when two associations of the table's class led to that class (a
flight's origin and destination airports), so the column vanished from the
generated table.
"""
import json

from besser.BUML.metamodel.gui.dashboard import LookupColumn, Table

CLASSES = {"Flight": [("code", "str")], "Airport": [("name", "str")], "Airline": [("label", "str")]}
ASSOCIATIONS = [
    ("departs", ("Flight", "departures", "0..*"), ("Airport", "origin", "1")),
    ("arrives", ("Flight", "arrivals", "0..*"), ("Airport", "destination", "1")),
    ("operates", ("Flight", "flights", "0..*"), ("Airline", "airline", "1")),
]


def _table(*columns):
    return {"type": "table", "attributes": {
        "id": "flights", "chart-title": "Flights", "data-source": "cls-Flight",
        "columns": json.dumps([{"field": "code", "label": "Code", "columnType": "field"}, *columns]),
    }}


def _lookup(field, label, **extra):
    return {"field": field, "label": label, "columnType": "lookup", "lookupField": "name", **extra}


def _lookup_paths(gui_model):
    for module in gui_model.modules:
        for screen in module.screens:
            for element in screen.view_elements:
                if isinstance(element, Table):
                    return {c.label: c.path.name for c in element.columns if isinstance(c, LookupColumn)}
    return {}


def test_a_column_naming_a_class_reached_twice_is_kept(build_app, caplog):
    table = _table(_lookup("Airport", "Airport", lookupEntity="cls-Airport"))
    with caplog.at_level("WARNING"):
        app = build_app(CLASSES, {"Flights": [table]}, ASSOCIATIONS)

    # The first end by name, every time.
    assert _lookup_paths(app.gui_model) == {"Airport": "destination"}
    assert '"path": "destination"' in app.page("Flights")
    warnings = [record.getMessage() for record in caplog.records if "Lookup column" in record.getMessage()]
    assert len(warnings) == 1
    assert "reaches through 2 associations" in warnings[0] and "'destination'" in warnings[0]


def test_a_column_naming_the_association_end_selects_it(build_app, caplog):
    table = _table(_lookup("origin", "From"), _lookup("Airport", "To", path="destination"),
                   _lookup("Airline", "Airline", lookupField="label"))
    with caplog.at_level("WARNING"):
        app = build_app(CLASSES, {"Flights": [table]}, ASSOCIATIONS)

    assert _lookup_paths(app.gui_model) == {"From": "origin", "To": "destination", "Airline": "airline"}
    assert not [record for record in caplog.records if "Lookup column" in record.getMessage()]


def _lookup_fields(gui_model):
    for module in gui_model.modules:
        for screen in module.screens:
            for element in screen.view_elements:
                if isinstance(element, Table):
                    return {c.label: c.field.name for c in element.columns if isinstance(c, LookupColumn)}
    return {}


def test_a_column_without_a_lookup_field_shows_a_readable_attribute_not_the_generated_id(build_app):
    """An ``id`` flagged as external id is the server's generated key; the column shows the name."""
    classes = {**CLASSES, "Airline": [("id", "int", {"isExternalId": True}), ("code", "int"), ("label", "str")]}
    column = {"field": "airline", "label": "Airline", "columnType": "lookup"}
    app = build_app(classes, {"Flights": [_table(column)]}, ASSOCIATIONS)

    assert _lookup_fields(app.gui_model) == {"Airline": "label"}


def test_a_column_naming_a_class_reached_twice_takes_the_end_its_label_names(build_app, caplog):
    table = _table(_lookup("Airport", "Origin", lookupEntity="cls-Airport"),
                   _lookup("Airport", "Destination airport", lookupEntity="cls-Airport"))
    with caplog.at_level("WARNING"):
        app = build_app(CLASSES, {"Flights": [table]}, ASSOCIATIONS)

    assert _lookup_paths(app.gui_model) == {"Origin": "origin", "Destination airport": "destination"}
    assert not [record for record in caplog.records if "Lookup column" in record.getMessage()]
