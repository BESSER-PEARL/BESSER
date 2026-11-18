"""A bound component's filter compares strings as strings.

`Date.parse` accepts many ordinary strings (a name such as "Phase 1" parses to a
date), so a filter `name == 'Phase 1'` also matched a record named "Step 1".
Only ISO 8601 dates and date-times, which the backend emits, are compared as dates.
"""

CLASSES = {"Task": [("name", "str"), ("due", "date")]}
PAGES = {"Tasks": [
    {"type": "metric-card", "attributes": {"data-source": "cls-Task", "aggregation": "count"}},
]}


def test_filters_compare_dates_only_for_iso_strings(build_app):
    source = build_app(CLASSES, PAGES).file("components", "runtime", "bindingData.ts")

    assert "const ISO_DATE_PATTERN = " in source
    assert "return parseIsoDate(trimmed) ?? trimmed.toLowerCase();" in source
    assert "return parseIsoDate(trimmed);" in source
    # No comparison parses an arbitrary string as a date.
    assert source.count("Date.parse(") == 1
