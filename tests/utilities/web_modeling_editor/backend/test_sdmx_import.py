"""Tests for the SDMX source of the "From data" import (/sdmx-to-domain-model, /sdmx-providers).

Offline: the SDMX client is replaced by a fake that returns a recorded structure.
"""

import asyncio
import os
import threading

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.sdmx_to_buml import (
    SDMXClient,
    SDMXFetchError,
    SDMXImportLimitError,
    SDMXTimeoutError,
    SDMXTooLargeError,
    parse_structure,
)
from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.constants import constants as backend_constants
from besser.utilities.web_modeling_editor.backend.services.reverse_engineering import sdmx_reverse
from tests.utilities.sdmx_to_buml.conftest import PUBLIC_ADDRESS, build_structure_xml, resolve_to

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
def public_dns(monkeypatch):
    """Every host resolves to a public address: no test reaches a real DNS server."""
    resolve_to(monkeypatch, PUBLIC_ADDRESS)


@pytest.fixture
def custom_urls(monkeypatch):
    """Custom SDMX URLs accepted (they are off by default, as on a hosted server)."""
    monkeypatch.setattr(backend_constants, "SDMX_ALLOW_CUSTOM_URL", True)


@pytest.fixture
def fake_fetch(monkeypatch, public_dns):
    """Every fetch returns the synthetic structure; ``calls`` records what was asked."""
    calls = []

    def fetch(self, ref, **kwargs):
        calls.append((self.provider.key, self.provider.base_url, ref.agency, ref.id, ref.version, self.safe))
        return parse_structure(build_structure_xml(), "DF_TEST")

    monkeypatch.setattr(SDMXClient, "fetch_structure", fetch)
    return calls


def test_providers_are_listed(monkeypatch):
    monkeypatch.delenv("BESSER_SDMX_ALLOW_CUSTOM_URL", raising=False)
    resp = _run(_get("/besser_api/sdmx-providers"))
    assert resp.status_code == 200
    data = resp.json()
    keys = {p["key"] for p in data["providers"]}
    assert {"eurostat", "ecb", "oecd", "lustat"} <= keys
    assert all(p["baseUrl"].startswith("https://") for p in data["providers"])
    assert data["customUrlAllowed"] is backend_constants.SDMX_ALLOW_CUSTOM_URL


def test_custom_urls_are_off_by_default():
    """Like BESSER_LLM_ALLOW_CUSTOM_BASE_URL: a hosted server does not fetch arbitrary URLs for anyone."""
    if "BESSER_SDMX_ALLOW_CUSTOM_URL" in os.environ:
        pytest.skip("BESSER_SDMX_ALLOW_CUSTOM_URL is set in this environment")
    assert backend_constants.SDMX_ALLOW_CUSTOM_URL is False


def test_import_from_a_preset_provider(fake_fetch):
    resp = _run(_post({"provider": "eurostat", "dataflow": "DF_TEST", "modelName": "my model"}))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["diagramType"] == "ClassDiagram" and data["model"]["type"] == "ClassDiagram"
    assert data["title"] == "MyModel"
    names = {n["data"]["name"] for n in data["model"]["nodes"] if n.get("type") == "class"}
    assert {"Observation", "Freq", "Product"} <= names
    assert fake_fetch == [("eurostat", "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1", None,
                           "DF_TEST", "latest", True)]


def test_import_from_a_custom_url_with_agency_and_version(fake_fetch, custom_urls):
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
    ({"baseUrl": "https://127.0.0.1\\.x.evil.example/rest", "dataflow": "X"}, "backslash"),
    ({"provider": "ecb", "dataflow": "../../admin"}, "dataflow id"),
    ({"provider": "ecb", "dataflow": "EXR", "version": "1.0/../x"}, "version"),
])
def test_invalid_requests_are_refused(fake_fetch, custom_urls, body, message):
    resp = _run(_post(body))
    assert resp.status_code == 400, resp.text
    assert message in resp.json()["detail"]
    assert fake_fetch == []


def test_private_hosts_are_refused(monkeypatch, custom_urls):
    resolve_to(monkeypatch, "169.254.169.254")
    resp = _run(_post({"baseUrl": "https://metadata.example.org/rest", "dataflow": "X"}))
    assert resp.status_code == 400 and "not a public address" in resp.json()["detail"]


def test_custom_urls_can_be_disabled(fake_fetch, monkeypatch):
    monkeypatch.setattr(backend_constants, "SDMX_ALLOW_CUSTOM_URL", False)
    resp = _run(_post({"baseUrl": "https://stats.example.org/rest", "dataflow": "DF_TEST"}))
    assert resp.status_code == 400 and "disabled" in resp.json()["detail"]
    assert _run(_get("/besser_api/sdmx-providers")).json()["customUrlAllowed"] is False
    assert _run(_post({"provider": "ecb", "dataflow": "DF_TEST"})).status_code == 200


@pytest.mark.parametrize("error, status", [
    (SDMXFetchError("Dataflow ECB:NOPE not found at European Central Bank", 404), 404),
    (SDMXTooLargeError("The structure is larger than 32 MB"), 413),
    (SDMXImportLimitError("The structure needs more than 64 MB in all"), 413),
    (SDMXTimeoutError("European Central Bank took longer than 60 s"), 504),
    (SDMXFetchError("European Central Bank could not be reached: ConnectTimeout"), 502),
])
def test_provider_errors_map_to_http_statuses(monkeypatch, public_dns, error, status):
    def fetch(self, ref, **kwargs):
        raise error

    monkeypatch.setattr(SDMXClient, "fetch_structure", fetch)
    resp = _run(_post({"provider": "ecb", "dataflow": "NOPE"}))
    assert resp.status_code == status
    assert resp.json()["detail"] == str(error)


def test_imports_beyond_the_concurrency_cap_are_refused(monkeypatch, public_dns):
    """Imports cannot fill the shared thread pool: past the cap the server answers 503 at once."""
    monkeypatch.setattr(sdmx_reverse, "_import_slots", threading.BoundedSemaphore(1))
    started, release = threading.Event(), threading.Event()

    def slow_fetch(self, ref, **kwargs):
        started.set()
        release.wait(5)
        return parse_structure(build_structure_xml(), "DF_TEST")

    monkeypatch.setattr(SDMXClient, "fetch_structure", slow_fetch)
    first = threading.Thread(target=sdmx_reverse.sdmx_to_domain_model, args=("DF_TEST",),
                             kwargs={"provider": "ecb"})
    first.start()
    try:
        assert started.wait(5)
        resp = _run(_post({"provider": "ecb", "dataflow": "DF_TEST"}))
        assert resp.status_code == 503 and "Too many SDMX imports" in resp.json()["detail"]
    finally:
        release.set()
        first.join(5)
    assert _run(_post({"provider": "ecb", "dataflow": "DF_TEST"})).status_code == 200
