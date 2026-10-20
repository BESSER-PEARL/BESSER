"""Element timestamps survive the class-diagram converters (GitHub issue #152).

``class_buml_to_json`` writes an optional ISO 8601 timestamp on every element,
relationship, association end and on the diagram model itself, when that
timestamp is explicit; ``process_class_diagram`` reads it back, and falls back
to the creation time when it is absent or malformed. Defaulted timestamps are
never written, so timestamp-less input converts identically.

v4 locations: ``model.timestamp``, ``node.data.timestamp`` (classes,
enumerations, OCL constraint nodes), ``timestamp`` on attribute / method /
literal rows, ``edge.data.timestamp`` (associations, generalizations) and
``edge.data.sourceTimestamp`` / ``targetTimestamp`` (association ends).
"""
import copy
from datetime import datetime, timedelta, timezone

import pytest

from besser.BUML.metamodel.structural import AssociationClass
from besser.BUML.metamodel.structural import structural
from besser.utilities.buml_code_builder.domain_model_builder import domain_model_to_code
from besser.utilities.utils import parse_iso_timestamp, timestamp_to_iso
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
    class_buml_to_json,
    parse_buml_content,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)
from tests.utilities.buml_code_builder.test_domain_model_builder_timestamps import (
    _build_model,
    _timestamps,
)

def _iso(second: int) -> str:
    return f"2024-05-06T07:08:{second:02d}.000042"


def _node(node_id, name, second, stereotype=None, attributes=(), methods=()):
    return {"id": node_id, "type": "class", "position": {"x": 0, "y": 0},
            "width": 160, "height": 100,
            "data": {"name": name, "stereotype": stereotype, "attributes": list(attributes),
                     "methods": list(methods), "timestamp": _iso(second)}}


def _row(row_id, name, second, attribute_type="str"):
    return {"id": row_id, "name": name, "visibility": "public",
            "attributeType": attribute_type, "timestamp": _iso(second)}


def _association(edge_id, name, source, target, ends, second):
    (source_role, source_mult, source_second), (target_role, target_mult, target_second) = ends
    return {"id": edge_id, "type": "ClassBidirectional", "source": source, "target": target,
            "data": {"name": name, "timestamp": _iso(second),
                     "sourceRole": source_role, "sourceMultiplicity": source_mult,
                     "sourceTimestamp": _iso(source_second),
                     "targetRole": target_role, "targetMultiplicity": target_mult,
                     "targetTimestamp": _iso(target_second), "points": []}}


def _diagram() -> dict:
    """A v4 class diagram whose every element carries a distinct timestamp."""
    renew = {"id": "renew", "name": "renew", "visibility": "public", "returnType": "str",
             "attributeType": "str", "timestamp": _iso(6),
             "parameters": [{"id": "renew-days", "name": "days", "parameterType": "int"}]}
    nodes = [
        _node("status", "Status", 1, "Enumeration",
              attributes=[_row("open", "OPEN", 2), _row("closed", "CLOSED", 3)]),
        _node("library", "Library", 4, attributes=[_row("lib-name", "name", 5)], methods=[renew]),
        _node("book", "Book", 7, "Abstract", attributes=[_row("book-status", "status", 8, "Status")]),
        _node("novel", "Novel", 9),
        _node("loan", "Loan", 10),
        {"id": "ocl", "type": "ClassOCLConstraint", "position": {"x": 0, "y": 0},
         "width": 160, "height": 100,
         "data": {"name": "", "timestamp": _iso(11),
                  "expression": "context Library inv named: self.name <> ''"}},
    ]
    edges = [
        _association("holds", "Library_Book", "library", "book",
                     (("library", "1", 13), ("books", "0..*", 14)), 12),
        _association("lends", "Library_Novel", "library", "novel",
                     (("lender", "1", 16), ("novels", "0..*", 17)), 15),
        {"id": "inherits", "type": "ClassInheritance", "source": "novel", "target": "book",
         "data": {"timestamp": _iso(18), "points": []}},
        {"id": "loan-link", "type": "ClassLinkRel", "source": "lends", "target": "loan",
         "data": {"points": []}},
    ]
    return {"title": "Lending", "model": {
        "version": "4.0.0", "type": "ClassDiagram", "timestamp": _iso(19),
        "nodes": nodes, "edges": edges,
    }}


def _node_by_id(diagram: dict, node_id: str) -> dict:
    return next(n for n in diagram["model"]["nodes"] if n["id"] == node_id)


def _json_timestamps(model_json: dict) -> dict:
    """Map a name-based key (node and row ids are regenerated) to each JSON timestamp."""
    names = {n["id"]: n["data"].get("name") for n in model_json["nodes"]}
    found = {("model",): model_json.get("timestamp")}
    for node in model_json["nodes"]:
        data = node["data"]
        if node["type"] == "ClassOCLConstraint":
            found[(node["type"], data["expression"])] = data.get("timestamp")
            continue
        found[(node["type"], data["name"])] = data.get("timestamp")
        for kind in ("attributes", "methods"):
            for row in data.get(kind) or []:
                found[(kind, data["name"], row["name"])] = row.get("timestamp")
    for edge in model_json["edges"]:
        data = edge["data"]
        if edge["type"] == "ClassInheritance":
            found[(edge["type"], names[edge["source"]])] = data.get("timestamp")
        elif data.get("name"):
            found[(edge["type"], data["name"])] = data.get("timestamp")
            for side in ("source", "target"):
                found[("end", data["name"], data[f"{side}Role"])] = data.get(f"{side}Timestamp")
    return found


def test_json_to_buml_reads_every_timestamp():
    model = process_class_diagram(_diagram())
    stamps = _timestamps(model)
    assert stamps[("model",)] == datetime.fromisoformat(_iso(19))
    assert stamps[("type", "Status")] == datetime.fromisoformat(_iso(1))
    assert stamps[("literal", "Status", "CLOSED")] == datetime.fromisoformat(_iso(3))
    assert stamps[("attribute", "Library", "name")] == datetime.fromisoformat(_iso(5))
    assert stamps[("method", "Library", "renew")] == datetime.fromisoformat(_iso(6))
    assert stamps[("type", "Book")] == datetime.fromisoformat(_iso(7))
    assert stamps[("end", "Library_Book", "books")] == datetime.fromisoformat(_iso(14))
    assert stamps[("association", "Library_Novel")] == datetime.fromisoformat(_iso(15))
    assert stamps[("generalization", "Novel")] == datetime.fromisoformat(_iso(18))
    assert stamps[("constraint", "named")] == datetime.fromisoformat(_iso(11))
    # The class promoted to an association class keeps its element's timestamp.
    loan = model.get_class_by_name("Loan")
    assert isinstance(loan, AssociationClass)
    assert loan.timestamp == datetime.fromisoformat(_iso(10))


def test_json_buml_json_preserves_timestamps():
    diagram = _diagram()
    exported = class_buml_to_json(process_class_diagram(diagram))
    stamps = _json_timestamps(exported)
    expected = _json_timestamps(diagram["model"])
    # The fixture uses the export's own naming (bare attribute and method
    # names), so every key lines up.
    assert stamps == expected
    assert all(value is not None for value in stamps.values())


def test_buml_json_buml_preserves_timestamps():
    model = _build_model()
    rebuilt = process_class_diagram({"title": model.name, "model": class_buml_to_json(model)})
    expected = _timestamps(model)
    # Parameters live inside the method signature string, which has no slot
    # for per-parameter metadata, so they are recreated with fresh timestamps.
    expected = {k: v for k, v in expected.items() if k[0] != "parameter"}
    actual = {k: v for k, v in _timestamps(rebuilt).items() if k[0] != "parameter"}
    assert actual == expected


def test_missing_timestamps_fall_back_to_creation_time():
    diagram = _without_timestamps(_diagram())
    before = datetime.now()
    model = process_class_diagram(diagram)
    assert all(stamp >= before for stamp in _timestamps(model).values())


@pytest.mark.parametrize("bad", ["not a date", "", "2024-13-45T99:99:99", 12345, None, ["x"]])
def test_malformed_timestamp_falls_back_to_creation_time(bad):
    diagram = copy.deepcopy(_diagram())
    _node_by_id(diagram, "library")["data"]["timestamp"] = bad
    before = datetime.now()
    model = process_class_diagram(diagram)
    assert model.get_class_by_name("Library").timestamp >= before
    # The other elements are unaffected.
    assert model.get_class_by_name("Novel").timestamp == datetime.fromisoformat(_iso(9))


def test_timezone_aware_timestamp_becomes_naive_utc():
    """Model timestamps are naive; an aware one would break sort_by_timestamp.
    It is normalised to UTC, never to the server's local time."""
    diagram = _diagram()
    aware = datetime(2024, 5, 6, 7, 8, 9, tzinfo=timezone(timedelta(hours=2)))
    _node_by_id(diagram, "library")["data"]["timestamp"] = aware.isoformat()
    model = process_class_diagram(diagram)
    stamp = model.get_class_by_name("Library").timestamp
    assert stamp.tzinfo is None
    assert stamp == datetime(2024, 5, 6, 5, 8, 9)


def test_aware_timestamp_in_uploaded_buml_becomes_naive_utc(tmp_path):
    """A B-UML file with an offset next to defaulted timestamps used to keep the
    aware value, so exporting it raised "can't compare offset-naive and
    offset-aware datetimes"."""
    source = (
        'A = Class(name="A", timestamp=datetime.datetime.fromisoformat("2024-01-01T00:00:00+02:00"))\n'
        'B = Class(name="B")\n'
        'domain_model = DomainModel(name="D", types={A, B}, associations=set(), generalizations=set())\n'
    )
    model = parse_buml_content(source)
    stamp = model.get_class_by_name("A").timestamp
    assert stamp.tzinfo is None
    assert stamp == datetime(2023, 12, 31, 22, 0, 0)
    domain_model_to_code(model, str(tmp_path / "model.py"))


@pytest.mark.skipif(not hasattr(__import__("time"), "tzset"), reason="needs time.tzset")
@pytest.mark.parametrize("server_tz", ["UTC", "America/New_York", "Asia/Tokyo"])
def test_aware_timestamp_does_not_depend_on_the_server_timezone(monkeypatch, server_tz):
    """The same JSON must give the same B-UML on every host."""
    import time

    monkeypatch.setenv("TZ", server_tz)
    time.tzset()
    try:
        assert parse_iso_timestamp("2024-05-06T07:08:09+02:00") == datetime(2024, 5, 6, 5, 8, 9)
        assert parse_iso_timestamp("2024-05-06T07:08:09Z") == datetime(2024, 5, 6, 7, 8, 9)
    finally:
        monkeypatch.undo()
        time.tzset()


def test_timestamp_helpers():
    stamp = datetime(2024, 1, 2, 3, 4, 5, 6)
    assert timestamp_to_iso(stamp) == "2024-01-02T03:04:05.000006"
    assert parse_iso_timestamp(timestamp_to_iso(stamp)) == stamp
    assert timestamp_to_iso(None) is None
    assert timestamp_to_iso("2024-01-02") is None
    assert parse_iso_timestamp(" 2024-01-02T03:04:05 ") == datetime(2024, 1, 2, 3, 4, 5)


def test_exported_model_and_elements_carry_iso_strings():
    exported = class_buml_to_json(_build_model())
    assert exported["timestamp"] == "2024-03-01T09:30:21.123456"
    classes = [n["data"] for n in exported["nodes"]
               if n["type"] == "class" and n["data"].get("stereotype") != "Enumeration"]
    assert classes and all(isinstance(c["timestamp"], str) for c in classes)
    constraint_boxes = [n["data"] for n in exported["nodes"] if n["type"] == "ClassOCLConstraint"]
    assert [c["timestamp"] for c in constraint_boxes] == ["2024-03-01T09:30:20.123456"]


def _without_timestamps(diagram: dict) -> dict:
    diagram = copy.deepcopy(diagram)
    diagram["model"].pop("timestamp", None)
    for node in diagram["model"]["nodes"]:
        node["data"].pop("timestamp", None)
        for row in node["data"].get("attributes", []) + node["data"].get("methods", []):
            row.pop("timestamp", None)
    for edge in diagram["model"]["edges"]:
        for key in ("timestamp", "sourceTimestamp", "targetTimestamp"):
            edge["data"].pop(key, None)
    return diagram


def _force_clock_forward(monkeypatch):
    """Make every later default timestamp differ from the earlier run's."""
    monkeypatch.setattr(structural, "_last_timestamp", datetime.now() + timedelta(days=365))


def test_timestampless_json_gives_identical_buml_code(tmp_path, monkeypatch):
    """Editor JSON usually has no timestamps; exporting it twice must give the
    same B-UML file even though the elements get different default timestamps."""
    diagram = _without_timestamps(_diagram())
    first, second = tmp_path / "first.py", tmp_path / "second.py"
    first_model = process_class_diagram(copy.deepcopy(diagram))
    domain_model_to_code(first_model, str(first))
    _force_clock_forward(monkeypatch)
    second_model = process_class_diagram(copy.deepcopy(diagram))
    domain_model_to_code(second_model, str(second))
    assert first_model.get_class_by_name("Library").timestamp != second_model.get_class_by_name("Library").timestamp
    code = first.read_text(encoding="utf-8")
    assert code == second.read_text(encoding="utf-8")
    assert "fromisoformat" not in code


def test_timestampless_json_gives_no_json_timestamps(monkeypatch):
    diagram = _without_timestamps(_diagram())
    first = class_buml_to_json(process_class_diagram(copy.deepcopy(diagram)))
    _force_clock_forward(monkeypatch)
    second = class_buml_to_json(process_class_diagram(copy.deepcopy(diagram)))
    assert set(_json_timestamps(first).values()) == {None}
    assert _json_timestamps(first) == _json_timestamps(second)


def test_explicit_json_timestamps_reach_buml_code_and_back(tmp_path, monkeypatch):
    """JSON -> BUML -> code -> BUML -> JSON keeps explicit timestamps."""
    path = tmp_path / "model.py"
    domain_model_to_code(process_class_diagram(_diagram()), str(path))
    code = path.read_text(encoding="utf-8")
    assert f'Library = Class(name="Library", timestamp=datetime.datetime.fromisoformat("{_iso(4)}"))' in code
    _force_clock_forward(monkeypatch)
    reexported = class_buml_to_json(parse_buml_content(code))
    assert _json_timestamps(reexported) == _json_timestamps(_diagram()["model"])
