"""Tests for besser.utilities.sdmx_to_buml (offline: the network is never used)."""

import socket

import pytest

from besser.BUML.metamodel.structural import Class, Enumeration
from besser.utilities.sdmx_to_buml import (
    SDMXClient, SDMXFetchError, SDMXTooLargeError, check_public_url, detect_roles, infer_prefix_parents,
    literal_name, parse_dataflow, parse_hierarchy, parse_structure, pascal, resolve_provider, sdmx_xml_to_buml,
    snake, structure_to_buml,
)
from tests.utilities.sdmx_to_buml.conftest import RECORDED, build_structure_xml, recorded
from tests.utilities.sdmx_to_buml.roundtrip import round_trip_differences


# --------------------------------------------------------------------------- parsing

def test_parse_reads_only_real_dimensions(structure_xml):
    dsd = parse_structure(structure_xml, "DF_TEST")
    assert [d.id for d in dsd.dimensions] == [
        "FREQ", "REPORTER", "PARTNER", "PRODUCT", "FLOW", "RANK", "SECTOR", "UNIT_MEASURE", "CLASS"]
    assert dsd.time_dimension.id == "TIME_PERIOD"
    assert [a.id for a in dsd.attributes] == ["OBS_FLAG"]
    assert dsd.measure.id == "OBS_VALUE"
    assert dsd.dataflow_name == "Test dataflow"  # the English name, not the French one
    assert dsd.codelist_for(dsd.dimensions[2]).codes["DE"].name == "Germany"
    assert dsd.dimensions[2].name == "Partner"  # from the concept scheme
    assert dsd.dimensions[1].codelist_source.agency == "TEST"
    assert dsd.constraints == {"FREQ": ["A", "M"], "UNIT_MEASURE": ["EUR"]}


def test_parse_refuses_doctype_and_garbage():
    with pytest.raises(ValueError, match="DOCTYPE"):
        parse_structure(build_structure_xml(doctype=True))
    with pytest.raises(ValueError, match="Not a valid SDMX-ML"):
        parse_structure(b"<html><body>Service unavailable")
    with pytest.raises(ValueError, match="No DataStructure"):
        parse_structure(b'<Structure xmlns="x"><Structures/></Structure>')


def test_parse_hierarchy():
    xml = b'''<m:Structure xmlns:m="m" xmlns:s="s"><s:HierarchicalCodelist id="HCL">
    <s:Hierarchy id="H"><s:HierarchicalCode id="a"><s:CodeID><Ref id="W"/></s:CodeID>
    <s:HierarchicalCode id="b"><s:CodeID><Ref id="EU"/></s:CodeID>
    <s:HierarchicalCode id="c"><s:CodeID><Ref id="DE"/></s:CodeID></s:HierarchicalCode></s:HierarchicalCode>
    </s:HierarchicalCode></s:Hierarchy></s:HierarchicalCodelist></m:Structure>'''
    assert parse_hierarchy(xml, "H") == {"W": None, "EU": "W", "DE": "EU"}


# --------------------------------------------------------------------------- names

def test_names_are_sanitised():
    assert pascal("REF_AREA") == "RefArea" and pascal("c_part") == "CPart" and pascal("2nd area") == "X2ndArea"
    assert pascal("DSD_INT_TRAD@DF_E4103") == "DsdIntTradDfE4103"
    assert snake("TIME_PERIOD") == "time_period" and snake("refArea") == "ref_area"
    assert snake("class") == "x_class" and snake("ID") == "x_id" and snake("1st") == "x_1st"
    assert literal_name("A", "Annual", set()) == "A"
    assert literal_name("1", "IMPORT", set()) == "IMPORT"          # numeric codes take their label
    assert literal_name("1", "IMPORT", {"IMPORT"}) == "C_1"
    assert literal_name("_T", "Total", set()) == "TOTAL"            # no leading underscore
    assert literal_name("def", "", set()) == "C_def"                 # never a Python keyword
    assert literal_name("A-1", "", {"C_A_1"}) == "C_A_1_2"


def test_prefix_hierarchy():
    parents = infer_prefix_parents(["TOTAL", "TO", "84", "8471", "847130", "84713000", "00", "0090"])
    assert parents == {"TOTAL": None, "TO": None, "84": "TOTAL", "8471": "84", "847130": "8471",
                       "84713000": "847130", "00": "TOTAL", "0090": "00"}


# --------------------------------------------------------------------------- roles and mapping

def test_roles(structure_xml):
    roles = detect_roles(parse_structure(structure_xml, "DF_TEST"))
    assert roles["FREQ"].role == "frequency" and roles["FREQ"].default_code == "A"
    assert roles["UNIT_MEASURE"].role == "unit"                     # a single code in the data
    assert roles["PARTNER"].role == "geo" and roles["PARTNER"].total_code == "WORLD"
    assert roles["PRODUCT"].role == "hierarchy" and roles["PRODUCT"].hierarchy == "prefix"
    assert roles["PRODUCT"].total_code == "TOTAL"
    assert roles["SECTOR"].hierarchy == "sdmx" and roles["SECTOR"].total_code == "_T"
    assert roles["RANK"].hierarchy is None                          # 1..60 is a ranking, not a nomenclature
    assert roles["FLOW"].role == "unit"                             # flows are never summed


def test_mapping_rules(structure_xml):
    model, mapping = structure_to_buml(parse_structure(structure_xml, "DF_TEST"), model_name="TestModel",
                                       class_names={"CL_AREA": "Country"})
    kinds = {d.id: (d.kind, d.type_name, d.hierarchy) for d in mapping.dimensions}
    assert kinds["FREQ"] == ("enum", "Freq", None)
    assert kinds["REPORTER"] == ("class", "Country", None)
    assert kinds["PARTNER"] == ("class", "Country", None)           # shared codelist -> one class
    assert kinds["PRODUCT"] == ("class", "Product", "prefix")
    assert kinds["RANK"] == ("class", "Rank", None)                 # more codes than enum_max
    assert kinds["SECTOR"] == ("class", "Sector", "sdmx")           # small but hierarchical -> a class
    assert kinds["CLASS"] == ("string", None, None)                 # not coded
    assert mapping.dimension("FLOW").literals == {"1": "IMPORT", "2": "EXPORT"}

    classes = {t.name: t for t in model.types if isinstance(t, Class)}
    enums = {t.name for t in model.types if isinstance(t, Enumeration)}
    assert set(classes) == {"Observation", "Country", "Product", "Rank", "Sector"}
    assert enums == {"Freq", "Flow", "UnitMeasure"}
    obs = {a.name: a for a in classes["Observation"].attributes}
    assert set(obs) == {"time_period", "obs_value", "freq", "flow", "unit_measure", "x_class", "obs_flag"}
    assert obs["obs_value"].is_optional and obs["obs_flag"].is_optional and not obs["time_period"].is_optional
    assert {a.name for a in classes["Product"].attributes} == {"code", "label", "level"}
    assert next(a for a in classes["Country"].attributes if a.name == "code").is_id
    ends = {(e.name, e.type.name) for a in model.associations for e in a.ends}
    assert {("parent", "Product"), ("children", "Product"), ("parent", "Sector")} <= ends
    assert {("observations_as_reporter", "Observation"), ("observations_as_partner", "Observation")} <= ends

    # The model documents its roles and sources.
    assert classes["Observation"].metadata.uri == "sdmx:dataflow:TEST:DF_TEST"
    assert "PARTNER (Partner): role=geo" in classes["Observation"].metadata.description
    assert classes["Product"].metadata.uri == "sdmx:codelist:TEST:CL_PRODUCT(1.0)"
    assert model.validate(raise_exception=False)["success"]


def test_enum_max_and_model_name(structure_xml):
    model = sdmx_xml_to_buml(structure_xml, "DF_TEST", enum_max=100)
    assert model.name == "DfTest"
    assert "Rank" in {t.name for t in model.types if isinstance(t, Enumeration)}


def test_a_dimension_named_like_the_fact_class_gets_another_name():
    xml = build_structure_xml().replace(b'"RANK"', b'"OBSERVATION"')
    model, _ = structure_to_buml(parse_structure(xml))
    assert {"Observation", "Observation2"} <= {t.name for t in model.types if isinstance(t, Class)}


@pytest.mark.parametrize("name", sorted(RECORDED))
def test_recorded_providers_import_validate_and_round_trip(name):
    dsd = parse_structure(recorded(name), RECORDED[name])
    if name.startswith("lustat"):
        dsd.hierarchies["C_PART"] = parse_hierarchy(recorded("lustat_HCL_E4103.xml.gz"), "H")
    model, mapping = structure_to_buml(dsd)
    assert model.validate(raise_exception=False)["success"]
    assert len(mapping.dimensions) == len(dsd.dimensions) >= 2
    assert round_trip_differences(model) == []


def test_recorded_lustat_partner_hierarchy():
    dsd = parse_structure(recorded("lustat_DSD_INT_TRAD_DF_E4103.xml.gz"), "DSD_INT_TRAD@DF_E4103")
    assert "C_PART" in dsd.hier_context()
    dsd.hierarchies["C_PART"] = parse_hierarchy(recorded("lustat_HCL_E4103.xml.gz"), "H")
    _, mapping = structure_to_buml(dsd)
    assert mapping.dimension("C_PART").hierarchy == "hcl" and mapping.dimension("C_PART").role == "geo"


def test_recorded_imf_codelists_come_from_the_concepts():
    """IMF dimensions have no local representation: the concept's core representation names the codelist."""
    dsd = parse_structure(recorded("imf_CPI.xml.gz"), "CPI")
    assert all(d.codelist_ref in dsd.codelists for d in dsd.dimensions)
    _, mapping = structure_to_buml(dsd)
    assert "string" not in {d.kind for d in mapping.dimensions}
    assert mapping.dimension("COUNTRY").role == "geo"


def test_round_trip_of_the_synthetic_model(structure_xml):
    model, _ = structure_to_buml(parse_structure(structure_xml, "DF_TEST"))
    assert round_trip_differences(model) == []


# --------------------------------------------------------------------------- references and URLs

def test_parse_dataflow_forms():
    assert parse_dataflow("EXR").id == "EXR" and parse_dataflow("EXR").agency is None
    ref = parse_dataflow("OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA,1.1")
    assert (ref.agency, ref.id, ref.version) == ("OECD.SDD.NAD", "DSD_NAMAIN1@DF_QNA", "1.1")
    ref = parse_dataflow("ECB:EXR(1.0)")
    assert (ref.agency, ref.id, ref.version) == ("ECB", "EXR", "1.0")
    assert parse_dataflow("EXR", agency="ECB", version="1.0").agency == "ECB"


@pytest.mark.parametrize("bad", ["", "../etc/passwd", "EXR/../x", "EXR?x=1", "A,B,C,D", "EXR#", "a b"])
def test_parse_dataflow_refuses_unsafe_ids(bad):
    with pytest.raises(ValueError):
        parse_dataflow(bad)


def test_resolve_provider():
    assert resolve_provider("ECB").key == "ecb"
    custom = resolve_provider("https://stats.example.org/rest/")
    assert custom.key == "custom" and custom.base_url == "https://stats.example.org/rest"
    with pytest.raises(ValueError, match="Unknown SDMX provider"):
        resolve_provider("nowhere")


def _resolve_to(monkeypatch, address):
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))])


@pytest.mark.parametrize("url, message", [
    ("http://stats.example.org/rest", "https"),
    ("https://user:pw@stats.example.org/rest", "Credentials"),
    ("https://stats.example.org:8443/rest", "port"),
])
def test_check_public_url_refuses_unsafe_urls(url, message):
    with pytest.raises(ValueError, match=message):
        check_public_url(url)


@pytest.mark.parametrize("address", ["127.0.0.1", "10.1.2.3", "192.168.0.1", "169.254.169.254", "::1", "fd00::1"])
def test_check_public_url_refuses_private_hosts(monkeypatch, address):
    _resolve_to(monkeypatch, address)
    with pytest.raises(ValueError, match="non-public"):
        check_public_url("https://stats.example.org/rest")


def test_check_public_url_accepts_public_hosts(monkeypatch):
    _resolve_to(monkeypatch, "93.184.216.34")
    check_public_url("https://stats.example.org/rest")


# --------------------------------------------------------------------------- the client, with a fake network

class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None, url="https://stats.example.org/rest/x"):
        self.status_code, self.body, self.headers, self.url = status, body, headers or {}, url
        self.is_redirect = status in (301, 302, 303, 307, 308)

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i:i + chunk_size]

    def close(self):
        pass


@pytest.fixture
def public_dns(monkeypatch):
    _resolve_to(monkeypatch, "93.184.216.34")


def test_safe_client_refuses_a_redirect_to_a_private_host(monkeypatch, public_dns):
    client = SDMXClient("https://stats.example.org/rest", safe=True, retries=1)
    monkeypatch.setattr(client.session, "get", lambda *a, **k: FakeResponse(302, headers={
        "Location": "https://internal.example.org/secret"}))
    hosts = {"stats.example.org": "93.184.216.34", "internal.example.org": "10.0.0.5"}
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (hosts[host], 443))])
    with pytest.raises(ValueError, match="non-public"):
        client.get("dataflow", "all", "X", "latest")


def test_client_stops_reading_past_max_bytes(monkeypatch, public_dns):
    client = SDMXClient("https://stats.example.org/rest", safe=True, max_bytes=1000, retries=1)
    monkeypatch.setattr(client.session, "get", lambda *a, **k: FakeResponse(body=b"x" * 5000))
    with pytest.raises(SDMXTooLargeError):
        client.get("dataflow", "all", "X", "latest")
    monkeypatch.setattr(client.session, "get", lambda *a, **k: FakeResponse(headers={"Content-Length": "999999"}))
    with pytest.raises(SDMXTooLargeError):
        client.get("dataflow", "all", "X", "latest")


def test_client_reports_http_errors_with_their_status(monkeypatch, public_dns):
    client = SDMXClient("https://stats.example.org/rest", safe=True, retries=1)
    monkeypatch.setattr(client.session, "get", lambda *a, **k: FakeResponse(404))
    with pytest.raises(SDMXFetchError) as info:
        client.fetch_structure("NOPE")
    assert info.value.status == 404 and "not found" in str(info.value)


def test_url_parts_are_quoted():
    client = SDMXClient("https://stats.example.org/rest")
    assert client.url("dataflow", "LU1", "DSD_X@DF_Y", "1.0") == \
        "https://stats.example.org/rest/dataflow/LU1/DSD_X@DF_Y/1.0"


def test_fetch_structure_falls_back_and_completes_codelists(monkeypatch, structure_xml):
    """references=descendants refused, references=all returns the dataflow only: the client follows the
    dataflow's structure reference, then fetches the codelist the structure did not include."""
    text = structure_xml.decode("utf-8")
    start, end = text.index('<s:DataStructures>'), text.index('</s:DataStructures>') + len('</s:DataStructures>')
    flow_only = (text[:text.index('<s:Codelists>')] + text[text.index('</s:Codelists>') + 14:start]
                 + text[end:]).encode()
    area_start = text.index('<s:Codelist id="CL_AREA"')
    area = text[area_start:text.index('</s:Codelist>', area_start) + len('</s:Codelist>')]
    dsd_without_area = text.replace(area, "").encode()
    codelist_message = f'<m:Structure {text[text.index("xmlns:m"):text.index("><m:Header")]}><m:Structures>' \
                       f'<s:Codelists>{area}</s:Codelists></m:Structures></m:Structure>'
    calls = []

    def fake_get(self, *path, params=None, max_bytes=None):
        calls.append((path[0], (params or {}).get("references")))
        if path[0] == "dataflow" and params["references"] == "descendants":
            raise SDMXFetchError("HTTP 400", 400)
        if path[0] == "dataflow":
            return flow_only
        if path[0] == "datastructure":
            return dsd_without_area
        if path[0] == "codelist":
            assert path[1:] == ("TEST", "CL_AREA", "1.0")
            return codelist_message.encode()
        raise SDMXFetchError("HTTP 404", 404)

    monkeypatch.setattr(SDMXClient, "get", fake_get)
    dsd = SDMXClient("https://stats.example.org/rest").fetch_structure("TEST,DF_TEST")
    assert dsd.dataflow_name == "Test dataflow"
    assert "CL_AREA" in dsd.codelists and not dsd.missing_codelists
    assert [c[0] for c in calls] == ["dataflow", "dataflow", "datastructure", "codelist"]
    model, _ = structure_to_buml(dsd)
    assert model.validate(raise_exception=False)["success"]


def test_fetch_structure_retries_without_the_version(monkeypatch, structure_xml):
    """SDMX 2.1 makes the version optional; some APIs answer 404 to ".../latest"."""
    paths = []

    def fake_get(self, *path, params=None, max_bytes=None):
        paths.append(path)
        if path[0] == "dataflow" and len(path) == 4:
            raise SDMXFetchError("HTTP 404", 404)
        if path[0] == "dataflow":
            return structure_xml
        raise SDMXFetchError("HTTP 404", 404)

    monkeypatch.setattr(SDMXClient, "get", fake_get)
    dsd = SDMXClient("https://stats.example.org/rest").fetch_structure("TEST,DF_TEST")
    assert dsd.dataflow_id == "DF_TEST"
    assert paths[:2] == [("dataflow", "TEST", "DF_TEST", "latest"), ("dataflow", "TEST", "DF_TEST")]
