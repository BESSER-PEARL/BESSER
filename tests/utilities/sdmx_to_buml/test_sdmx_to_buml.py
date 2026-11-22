"""Tests for besser.utilities.sdmx_to_buml (offline: the network is never used)."""

import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from besser.BUML.metamodel.structural import Class, Enumeration
from besser.utilities.sdmx_to_buml import (
    SDMXClient,
    SDMXFetchError,
    SDMXImportLimitError,
    SDMXTimeoutError,
    SDMXTooLargeError,
    parse_structure,
    sdmx_xml_to_buml,
    structure_to_buml,
)
from besser.utilities.sdmx_to_buml.builder import literal_name, pascal, snake
from besser.utilities.sdmx_to_buml.client import check_public_url, parse_dataflow, resolve_provider
from besser.utilities.sdmx_to_buml.roles import detect_roles, infer_prefix_parents
from besser.utilities.sdmx_to_buml.structure import parse_hierarchy
from tests.utilities.sdmx_to_buml.conftest import (
    PUBLIC_ADDRESS,
    RECORDED,
    build_structure_xml,
    recorded,
    resolve_to,
)
from tests.utilities.sdmx_to_buml.roundtrip import round_trip, round_trip_differences

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


@pytest.mark.xfail(strict=True, reason="class_buml_to_json writes a class description as its description and as a "
                                       "linked comment, and process_class_diagram appends the comment to it: a "
                                       "converter bug outside this importer, to fix separately")
def test_class_descriptions_survive_the_editor(structure_xml):
    model, _ = structure_to_buml(parse_structure(structure_xml, "DF_TEST"))
    before = {t.name: t.metadata.description for t in model.types if isinstance(t, Class)}
    after = {t.name: t.metadata.description for t in round_trip(model).types if isinstance(t, Class)}
    assert after == before


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


@pytest.mark.parametrize("address", ["127.0.0.1", "10.1.2.3", "192.168.0.1", "169.254.169.254", "::1", "fd00::1",
                                     "::ffff:127.0.0.1", "64:ff9b::7f00:1", "100.64.0.1"])
def test_check_public_url_refuses_private_hosts(monkeypatch, address):
    resolve_to(monkeypatch, address)
    with pytest.raises(ValueError, match="not a public address"):
        check_public_url("https://stats.example.org/rest")


def test_unresolvable_and_private_hosts_get_the_same_message(monkeypatch):
    """The error never tells which internal names exist."""
    resolve_to(monkeypatch, {"private.example.org": "10.0.0.5"})
    messages = set()
    for host in ("private.example.org", "missing.example.org"):
        with pytest.raises(ValueError) as info:
            check_public_url(f"https://{host}/rest")
        messages.add(str(info.value))
    assert len(messages) == 1


def test_check_public_url_accepts_public_hosts_and_returns_the_address(public_dns):
    assert check_public_url("https://stats.example.org/rest") == PUBLIC_ADDRESS


@pytest.mark.parametrize("url", [
    "https://127.0.0.1\\.x.evil.example/rest",   # urlsplit and urllib3 read different hosts
    "https://127.0.0.1\\@evil.example/rest",
    "https://evil.example\t@127.0.0.1/rest",
    "https://evil.example/re st",
    "https://evil.example/rest\x00",
])
def test_check_public_url_refuses_urls_two_parsers_could_read_differently(public_dns, url):
    with pytest.raises(ValueError, match="backslash, whitespace or a control character"):
        check_public_url(url)


# --------------------------------------------------------------------------- the client, with a fake network

class FakeRaw:
    """A response body read the way the client reads it: a socket read (``read1``) at a time."""

    def __init__(self, body: bytes, chunk: int = 4096):
        self.body, self.pos, self.chunk, self.connection = body, 0, chunk, None

    def read1(self, amt=65536, decode_content=True):
        out = self.body[self.pos:self.pos + min(amt, self.chunk)]
        self.pos += len(out)
        return out


class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code, self.headers, self.raw = status, headers or {}, FakeRaw(body)
        self.is_redirect = status in (301, 302, 303, 307, 308)

    def close(self):
        pass


class FakeNetwork(list):
    """What the client sent, and how; ``answer`` sets the response for every request."""

    def __init__(self):
        super().__init__()
        self.respond = lambda url: FakeResponse(body=b"<ok/>")

    def answer(self, respond):
        self.respond = respond


@pytest.fixture
def fake_network(monkeypatch, public_dns):
    network = FakeNetwork()

    def get(session, url, params=None, headers=None, verify=None, **kwargs):
        adapter = session.adapters.get("https://")
        network.append({"url": url, "headers": dict(headers or {}), "verify": verify,
                        "trust_env": session.trust_env, "pinned_host": getattr(adapter, "_host", None)})
        return network.respond(url)

    monkeypatch.setattr(requests.Session, "get", get)
    return network


def test_safe_client_connects_to_the_checked_address(fake_network, monkeypatch):
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/etc/ssl/bundle.pem")
    client = SDMXClient("https://stats.example.org/rest", safe=True, retries=1)
    assert client.get("dataflow", "all", "X", "latest") == b"<ok/>"
    request = fake_network[-1]
    assert request["url"] == f"https://{PUBLIC_ADDRESS}/rest/dataflow/all/X/latest"
    assert request["headers"]["Host"] == "stats.example.org"
    assert request["pinned_host"] == "stats.example.org"  # SNI and the certificate check use the name
    assert request["trust_env"] is False and request["verify"] == "/etc/ssl/bundle.pem"


def test_a_dns_answer_that_changes_after_the_check_is_not_followed(fake_network, monkeypatch):
    """DNS rebinding: the client connects to the address it checked and never resolves the name again."""
    answers = iter([PUBLIC_ADDRESS, "127.0.0.1"])
    lookups = []

    def rebinding(host, *args, **kwargs):
        lookups.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (next(answers), 443))]

    client = SDMXClient("https://stats.example.org/rest", safe=True, retries=1)
    monkeypatch.setattr(socket, "getaddrinfo", rebinding)
    client.get("dataflow", "all", "X", "latest")
    assert lookups == ["stats.example.org"]
    assert fake_network[-1]["url"].startswith(f"https://{PUBLIC_ADDRESS}/")


def test_safe_client_refuses_a_redirect_to_a_private_host(fake_network, monkeypatch):
    resolve_to(monkeypatch, {"stats.example.org": PUBLIC_ADDRESS, "internal.example.org": "10.0.0.5"})
    client = SDMXClient("https://stats.example.org/rest", safe=True, retries=1)
    fake_network.answer(lambda url: FakeResponse(302, headers={"Location": "https://internal.example.org/secret"}))
    with pytest.raises(ValueError, match="not a public address"):
        client.get("dataflow", "all", "X", "latest")
    assert len(fake_network) == 1  # the private host was never contacted


def test_safe_client_refuses_a_redirect_to_a_backslash_host(fake_network):
    client = SDMXClient("https://stats.example.org/rest", safe=True, retries=1)
    fake_network.answer(lambda url: FakeResponse(302, headers={"Location": "https://127.0.0.1\\.evil.example/x"}))
    with pytest.raises(ValueError, match="backslash"):
        client.get("dataflow", "all", "X", "latest")
    assert len(fake_network) == 1


def test_client_stops_reading_past_max_bytes(fake_network):
    client = SDMXClient("https://stats.example.org/rest", safe=True, max_bytes=1000, retries=1)
    fake_network.answer(lambda url: FakeResponse(body=b"x" * 5000))
    with pytest.raises(SDMXTooLargeError):
        client.get("dataflow", "all", "X", "latest")
    fake_network.answer(lambda url: FakeResponse(headers={"Content-Length": "999999"}))
    with pytest.raises(SDMXTooLargeError):
        client.get("dataflow", "all", "X", "latest")


def test_an_import_has_one_byte_and_request_budget(fake_network):
    client = SDMXClient("https://stats.example.org/rest", safe=True, max_bytes=1000, max_total_bytes=1500,
                        max_requests=3, retries=1)
    fake_network.answer(lambda url: FakeResponse(body=b"x" * 800))
    client._budget = client._new_budget()
    client.get("codelist", "A", "CL_1", "1.0")
    with pytest.raises(SDMXImportLimitError, match="in all"):
        client.get("codelist", "A", "CL_2", "1.0")
    fake_network.answer(lambda url: FakeResponse(body=b"x"))
    client._budget = client._new_budget()
    for _ in range(3):
        client.get("codelist", "A", "CL_1", "1.0")
    with pytest.raises(SDMXImportLimitError, match="requests"):
        client.get("codelist", "A", "CL_1", "1.0")


def test_an_import_limit_stops_the_import_instead_of_skipping_a_codelist(monkeypatch, structure_xml):
    def fake_get(self, *path, params=None, max_bytes=None):
        if path[0] == "dataflow":
            return structure_xml.replace(b'<s:Codelist id="CL_AREA"', b'<s:Codelist id="CL_GONE"')
        raise SDMXTimeoutError("took too long")

    monkeypatch.setattr(SDMXClient, "get", fake_get)
    with pytest.raises(SDMXTimeoutError):
        SDMXClient("https://stats.example.org/rest").fetch_structure("TEST,DF_TEST")


def test_client_reports_http_errors_with_their_status(fake_network):
    client = SDMXClient("https://stats.example.org/rest", safe=True, retries=1)
    fake_network.answer(lambda url: FakeResponse(404))
    with pytest.raises(SDMXFetchError) as info:
        client.fetch_structure("NOPE")
    assert info.value.status == 404 and "not found" in str(info.value)


def test_url_parts_are_quoted_and_checked():
    client = SDMXClient("https://stats.example.org/rest")
    assert client.url("dataflow", "LU1", "DSD_X@DF_Y", "1.0") == \
        "https://stats.example.org/rest/dataflow/LU1/DSD_X@DF_Y/1.0"
    for bad in ("..", "a/b", "x?y", ".hidden"):
        with pytest.raises(ValueError, match="path part"):
            client.url("codelist", bad, "CL", "1.0")


def test_a_provider_reference_with_dot_dot_is_skipped(monkeypatch, structure_xml, caplog):
    """Path parts that come from the provider's own message are checked like the user's."""
    hostile = structure_xml.replace(b'<s:Codelist id="CL_AREA"', b'<s:Codelist id="CL_GONE"').replace(
        b'<Ref id="CL_AREA" agencyID="TEST"', b'<Ref id="CL_AREA" agencyID=".."')
    sent = []

    def fake_get_once(self, url, *args):
        sent.append(url)
        return hostile

    monkeypatch.setattr(SDMXClient, "_get_once", fake_get_once)
    dsd = SDMXClient("https://stats.example.org/rest").fetch_structure("TEST,DF_TEST", with_constraint=False)
    assert "CL_AREA" in dsd.missing_codelists
    assert not any("/../" in url for url in sent)
    assert "unavailable" in caplog.text


def test_a_trickling_server_cannot_outlast_the_deadline():
    """A server sending one byte every 0.2 s is cut at the deadline, not after the whole body."""
    class Trickle(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "100000")
            self.end_headers()
            try:
                for _ in range(100000):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.2)
            except OSError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Trickle)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        # safe=False only to reach a local plain-HTTP server; the body is read by the same code in both modes.
        client = SDMXClient(f"http://127.0.0.1:{server.server_address[1]}", timeout=1.0, retries=1)
        started = time.monotonic()
        with pytest.raises(SDMXTimeoutError):
            client.get("x")
        assert time.monotonic() - started < 3
    finally:
        server.shutdown()


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


# --------------------------------------------------------------------------- review follow-ups

_PAD = b'<?xml version="1.0"?><!--' + b"x" * 70000 + b"-->"


@pytest.mark.parametrize("xml", [
    _PAD + b'<!DOCTYPE r [<!ENTITY x "INJECTED">]><r>&x;</r>',                       # after the first 64 KB
    _PAD + b'<!DOCTYPE r [<!ENTITY x SYSTEM "file:///etc/passwd">]><r>&x;</r>',
    '<?xml version="1.0" encoding="UTF-16"?><!DOCTYPE r [<!ENTITY x "X">]><r>&x;</r>'.encode("utf-16"),
], ids=["internal-entity-after-64KB", "external-entity-after-64KB", "utf-16"])
def test_a_doctype_anywhere_and_in_any_encoding_is_refused(xml):
    with pytest.raises(ValueError, match="DOCTYPE"):
        parse_structure(xml)


def test_an_explicit_dataflow_missing_from_the_message_is_an_error(structure_xml):
    with pytest.raises(ValueError, match="'DOES_NOT_EXIST' is not in the structure message"):
        parse_structure(structure_xml, "DOES_NOT_EXIST")
    assert parse_structure(structure_xml, "df_test").dataflow_id == "DF_TEST"  # ids match case-insensitively


def test_names_keep_their_letters_without_accents():
    assert pascal("Österreich") == "Osterreich" and snake("Größe") == "grosse"
    assert literal_name("AT", "Österreich", set()) == "AT"
    assert literal_name("_Z", "Größe über Ländern", set()) == "GROSSE_UBER_LANDERN"


def test_import_mapping_dimension_raises_key_error(structure_xml):
    _, mapping = structure_to_buml(parse_structure(structure_xml, "DF_TEST"))
    with pytest.raises(KeyError):
        mapping.dimension("NOPE")


def test_validation_warnings_are_logged(caplog):
    """The model is validated without raising on warnings; the warnings reach the log."""
    model, _ = structure_to_buml(parse_structure(recorded("ecb_EXR.xml.gz"), "EXR"))
    warnings = model.validate(raise_exception=False)["warnings"]
    assert warnings and all(w in caplog.text for w in warnings)


def test_fetch_structure_reads_the_hierarchy_the_annotations_name(monkeypatch):
    """.Stat Suite: the dataflow's HIER_CONTEXT annotation names a hierarchical codelist the client fetches."""
    asked = []

    def fake_get(self, *path, params=None, max_bytes=None):
        asked.append(path)
        if path[0] == "dataflow":
            return recorded("lustat_DSD_INT_TRAD_DF_E4103.xml.gz")
        if path[0] == "hierarchicalcodelist":
            return recorded("lustat_HCL_E4103.xml.gz")
        raise SDMXFetchError("HTTP 404", 404)

    monkeypatch.setattr(SDMXClient, "get", fake_get)
    dsd = SDMXClient("lustat").fetch_structure("DSD_INT_TRAD@DF_E4103")
    assert ("hierarchicalcodelist", "LU1", "HCL_E4103", "1.0") in asked
    assert dsd.hierarchies["C_PART"]
    _, mapping = structure_to_buml(dsd)
    assert mapping.dimension("C_PART").hierarchy == "hcl"


def test_fetch_structure_reads_the_content_constraint_when_the_message_has_none(monkeypatch, structure_xml):
    text = structure_xml.decode("utf-8")
    start, end = text.index("<s:Constraints>"), text.index("</s:Constraints>") + len("</s:Constraints>")
    without = (text[:start] + text[end:]).encode()
    head = text[:text.index("<m:Structures>") + len("<m:Structures>")]
    constraint = (head + text[start:end] + "</m:Structures></m:Structure>").encode()

    def fake_get(self, *path, params=None, max_bytes=None):
        if path[0] == "dataflow":
            return without
        if path[0] == "contentconstraint":
            assert path[1:] == ("TEST", "DF_TEST")
            return constraint
        raise SDMXFetchError("HTTP 404", 404)

    monkeypatch.setattr(SDMXClient, "get", fake_get)
    dsd = SDMXClient("https://stats.example.org/rest").fetch_structure("TEST,DF_TEST")
    assert dsd.constraints == {"FREQ": ["A", "M"], "UNIT_MEASURE": ["EUR"]}


@pytest.mark.parametrize("name", ["ecb_EXR.xml.gz", "lustat_DSD_INT_TRAD_DF_E4103.xml.gz"])
def test_generators_take_an_imported_model(name, tmp_path):
    """Offline check that the model generates: Python classes import, SQLAlchemy tables create in SQLite."""
    import importlib.util
    import sys

    from sqlalchemy import create_engine

    from besser.generators.python_classes import PythonGenerator
    from besser.generators.sql_alchemy import SQLAlchemyGenerator

    model, _ = structure_to_buml(parse_structure(recorded(name), RECORDED[name]))
    for generator, folder in ((PythonGenerator, "python"), (SQLAlchemyGenerator, "sqlalchemy")):
        out = tmp_path / folder
        out.mkdir()
        gen = generator(model, output_dir=str(out))
        gen.generate(dbms="sqlite") if generator is SQLAlchemyGenerator else gen.generate()
        source = next(out.glob("*.py"))
        module_name = f"sdmx_generated_{folder}_{name.split('.')[0]}"
        spec = importlib.util.spec_from_file_location(module_name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            if generator is SQLAlchemyGenerator:
                module.Base.metadata.create_all(create_engine("sqlite://"))
                assert "observation" in {t.lower() for t in module.Base.metadata.tables}
            else:
                assert hasattr(module, "Observation")
        finally:
            sys.modules.pop(module_name, None)
