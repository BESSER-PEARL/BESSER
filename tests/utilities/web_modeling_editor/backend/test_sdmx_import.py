"""Tests for the SDMX source of the "From data" import (/sdmx-to-domain-model, /sdmx-providers).

Offline: the SDMX client is replaced by a fake that returns a recorded structure.
"""

import asyncio
import socket

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.sdmx_to_buml import SDMXClient, SDMXFetchError, SDMXTooLargeError, parse_structure
from besser.utilities.web_modeling_editor.backend.backend import app
from tests.utilities.sdmx_to_buml.conftest import build_structure_xml

BASE_URL = "http://testserver"
URL = "/besser_api/sdmx-to-domain-model"


def _run(coro):
    return asyncio.run(coro)


async def _post(json):
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as ac:
        return await ac.post(URL, json=json)


async def _get(url):
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as ac:
        return await ac.get(url)


@pytest.fixture
def fake_fetch(monkeypatch):
    """Every fetch returns the synthetic structure; ``calls`` records what was asked."""
    calls = []

    def fetch(self, ref, **kwargs):
        calls.append((self.provider.key, self.provider.base_url, ref.agency, ref.id, ref.version, self.safe))
        return parse_structure(build_structure_xml(), "DF_TEST")

    monkeypatch.setattr(SDMXClient, "fetch_structure", fetch)
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))])
    return calls


def test_providers_are_listed():
    resp = _run(_get("/besser_api/sdmx-providers"))
    assert resp.status_code == 200
    data = resp.json()
    keys = {p["key"] for p in data["providers"]}
    assert {"eurostat", "ecb", "oecd", "lustat"} <= keys
    assert all(p["baseUrl"].startswith("https://") for p in data["providers"])
    assert data["customUrlAllowed"] is True


def test_import_from_a_preset_provider(fake_fetch):
    resp = _run(_post({"provider": "eurostat", "dataflow": "DF_TEST", "modelName": "my model"}))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["diagramType"] == "ClassDiagram" and data["model"]["type"] == "ClassDiagram"
    assert data["title"] == "MyModel"
    names = {e["name"] for e in data["model"]["elements"].values() if e.get("type") in ("Class", "Enumeration")}
    assert {"Observation", "Freq", "Product"} <= names
    assert fake_fetch == [("eurostat", "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1", None,
                           "DF_TEST", "latest", True)]


def test_import_from_a_custom_url_with_agency_and_version(fake_fetch):
    resp = _run(_post({"baseUrl": "https://stats.example.org/rest", "dataflow": "DF_TEST",
                       "agency": "TEST", "version": "1.0"}))
    assert resp.status_code == 200, resp.text
    assert fake_fetch[0][1:] == ("https://stats.example.org/rest", "TEST", "DF_TEST", "1.0", True)


@pytest.mark.parametrize("body, message", [
    ({"dataflow": "X"}, "either a provider or a base URL"),
    ({"provider": "ecb", "baseUrl": "https://stats.example.org", "dataflow": "X"}, "either a provider"),
    ({"provider": "nowhere", "dataflow": "X"}, "Unknown SDMX provider"),
    ({"baseUrl": "http://stats.example.org/rest", "dataflow": "X"}, "https"),
    ({"baseUrl": "file:///etc/passwd", "dataflow": "X"}, "https"),
    ({"provider": "ecb", "dataflow": "../../admin"}, "dataflow id"),
    ({"provider": "ecb", "dataflow": "EXR", "version": "1.0/../x"}, "version"),
])
def test_invalid_requests_are_refused(fake_fetch, body, message):
    resp = _run(_post(body))
    assert resp.status_code == 400, resp.text
    assert message in resp.json()["detail"]
    assert fake_fetch == []


def test_private_hosts_are_refused(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 443))])
    resp = _run(_post({"baseUrl": "https://metadata.example.org/rest", "dataflow": "X"}))
    assert resp.status_code == 400 and "non-public" in resp.json()["detail"]


def test_custom_urls_can_be_disabled(fake_fetch, monkeypatch):
    monkeypatch.setenv("BESSER_SDMX_ALLOW_CUSTOM_URL", "0")
    resp = _run(_post({"baseUrl": "https://stats.example.org/rest", "dataflow": "DF_TEST"}))
    assert resp.status_code == 400 and "disabled" in resp.json()["detail"]
    assert _run(_get("/besser_api/sdmx-providers")).json()["customUrlAllowed"] is False
    assert _run(_post({"provider": "ecb", "dataflow": "DF_TEST"})).status_code == 200


@pytest.mark.parametrize("error, status", [
    (SDMXFetchError("Dataflow ECB:NOPE not found at European Central Bank", 404), 404),
    (SDMXTooLargeError("The structure is larger than 64 MB"), 413),
    (SDMXFetchError("European Central Bank could not be reached: ConnectTimeout"), 502),
])
def test_provider_errors_map_to_http_statuses(monkeypatch, error, status):
    def fetch(self, ref, **kwargs):
        raise error

    monkeypatch.setattr(SDMXClient, "fetch_structure", fetch)
    resp = _run(_post({"provider": "ecb", "dataflow": "NOPE"}))
    assert resp.status_code == status
    assert resp.json()["detail"] == str(error)
