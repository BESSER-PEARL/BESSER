"""A small SDMX 2.1 REST client for structure messages (SDMX-ML), built on ``requests``.

It reads what a B-UML model needs: a dataflow with its data structure, concepts and codelists,
completed when the provider allows with the content constraint and hierarchical codelists.

``safe=True`` is the mode for servers that fetch on behalf of someone else (the web editor):
only ``https`` on port 443, only hosts that resolve to public addresses (checked again on every
redirect), a size cap streamed while reading, and a total deadline.
"""

import ipaddress
import re
import socket
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote, urljoin, urlsplit

import requests

from besser.utilities.sdmx_to_buml.structure import (
    DataStructure, dataflow_structure_ref, parse_codelists, parse_constraint, parse_hierarchy, parse_structure,
)

USER_AGENT = "BESSER-sdmx-to-buml (+https://github.com/BESSER-PEARL/BESSER)"
XML_ACCEPT = "application/vnd.sdmx.structure+xml;version=2.1, application/xml;q=0.9, */*;q=0.1"
DEFAULT_MAX_BYTES = 100 * 1024 * 1024
_MAX_REDIRECTS = 3
_MAX_EXTRA_CODELISTS = 30

# An SDMX id, agency or version: letters, digits and the separators SDMX allows (no "/" or "?").
_ID = re.compile(r"^[A-Za-z0-9_@$\-]+(\.[A-Za-z0-9_@$\-]+)*$")
_VERSION = re.compile(r"^(latest|\+|~|[0-9]+(\.[0-9]+){0,2}(\+)?)$")


class SDMXFetchError(RuntimeError):
    """The provider could not be reached, refused the query, or answered with something unusable.

    ``status`` is the HTTP status when there was one (404: the dataflow does not exist).
    """

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class SDMXTooLargeError(SDMXFetchError):
    """The answer is larger than the client's ``max_bytes``."""


@dataclass(frozen=True)
class SDMXProvider:
    """An SDMX 2.1 REST endpoint.

    ``references`` is the order in which the client asks for the dataflow's descendants: some
    providers reject ``all`` (Eurostat), others only return the data structure with it (.Stat Suite).
    """
    key: str
    name: str
    base_url: str
    agency: str = "all"
    references: tuple[str, ...] = ("descendants", "all")


# Public providers that answer SDMX 2.1 structure queries without a key.
PROVIDERS: dict[str, SDMXProvider] = {p.key: p for p in (
    SDMXProvider("eurostat", "Eurostat", "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1", "ESTAT"),
    SDMXProvider("eurostat-comext", "Eurostat Comext (detailed trade)",
                 "https://ec.europa.eu/eurostat/api/comext/dissemination/sdmx/2.1", "ESTAT"),
    SDMXProvider("ecb", "European Central Bank", "https://data-api.ecb.europa.eu/service", "ECB"),
    SDMXProvider("oecd", "OECD", "https://sdmx.oecd.org/public/rest", "all", ("all", "descendants")),
    SDMXProvider("lustat", "STATEC Luxembourg (LUSTAT)", "https://lustat.statec.lu/rest", "LU1",
                 ("all", "descendants")),
    SDMXProvider("ilo", "International Labour Organization", "https://sdmx.ilo.org/rest", "ILO"),
    SDMXProvider("bis", "Bank for International Settlements", "https://stats.bis.org/api/v1", "BIS"),
    SDMXProvider("unicef", "UNICEF", "https://sdmx.data.unicef.org/ws/public/sdmxapi/rest", "UNICEF"),
    SDMXProvider("abs", "Australian Bureau of Statistics", "https://data.api.abs.gov.au/rest", "ABS",
                 ("all", "descendants")),
    SDMXProvider("imf", "International Monetary Fund", "https://api.imf.org/external/sdmx/2.1", "all"),
    SDMXProvider("insee", "INSEE (France)", "https://bdm.insee.fr/series/sdmx", "FR1"),
    SDMXProvider("istat", "Istat (Italy)", "https://esploradati.istat.it/SDMXWS/rest", "IT1"),
    SDMXProvider("unsd", "UN Data", "https://data.un.org/ws/rest", "all"),
    SDMXProvider("bundesbank", "Deutsche Bundesbank", "https://api.statistiken.bundesbank.de/rest/metadata", "BBK"),
)}


@dataclass(frozen=True)
class DataflowRef:
    """A dataflow reference: agency (``None``: the provider's default), id and version."""
    id: str
    agency: Optional[str] = None
    version: str = "latest"


def parse_dataflow(text: str, agency: Optional[str] = None, version: Optional[str] = None) -> DataflowRef:
    """Read ``ID``, ``AGENCY,ID``, ``AGENCY,ID,VERSION`` or ``AGENCY:ID(VERSION)``.

    Raises:
        ValueError: when a part is not a valid SDMX id or version.
    """
    text = (text or "").strip()
    match = re.match(r"^([^:,()]+):([^:,()]+)\(([^()]*)\)$", text)
    if match:
        parts = [match.group(1), match.group(2), match.group(3)]
    else:
        parts = [p.strip() for p in text.split(",")]
    if len(parts) == 1:
        ref = DataflowRef(parts[0], agency, version or "latest")
    elif len(parts) == 2:
        ref = DataflowRef(parts[1], parts[0], version or "latest")
    elif len(parts) == 3:
        ref = DataflowRef(parts[1], parts[0], parts[2] or "latest")
    else:
        raise ValueError(f"Not a dataflow reference: {text!r} (use ID, AGENCY,ID or AGENCY,ID,VERSION)")
    if not ref.id or len(ref.id) > 120 or not _ID.match(ref.id):
        raise ValueError(f"Not a valid SDMX dataflow id: {ref.id!r}")
    if ref.agency is not None and (len(ref.agency) > 80 or not _ID.match(ref.agency)):
        raise ValueError(f"Not a valid SDMX agency id: {ref.agency!r}")
    if not _VERSION.match(ref.version):
        raise ValueError(f"Not a valid SDMX version: {ref.version!r}")
    return ref


def resolve_provider(source: "SDMXProvider | str") -> SDMXProvider:
    """A provider from its key (see :data:`PROVIDERS`) or from the base URL of an SDMX 2.1 REST API."""
    if isinstance(source, SDMXProvider):
        return source
    key = (source or "").strip()
    if key.lower() in PROVIDERS:
        return PROVIDERS[key.lower()]
    if re.match(r"^https?://", key, re.IGNORECASE):
        host = urlsplit(key).hostname or "custom"
        return SDMXProvider("custom", host, key.rstrip("/"))
    raise ValueError(f"Unknown SDMX provider {source!r}: use one of {', '.join(PROVIDERS)} or an https URL")


def check_public_url(url: str) -> None:
    """Refuse anything but ``https`` on port 443 to a host that resolves only to public addresses.

    Raises:
        ValueError: for any other URL (private, loopback, link-local, credentials in the URL...).
    """
    parts = urlsplit(url)
    if parts.scheme.lower() != "https":
        raise ValueError("Only https SDMX endpoints are allowed")
    if parts.username or parts.password:
        raise ValueError("Credentials in the URL are not allowed")
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("Invalid port in the URL") from exc
    if port not in (None, 443):
        raise ValueError("Only the default https port (443) is allowed")
    host = parts.hostname
    if not host:
        raise ValueError("The URL has no host")
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve host {host!r}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%", 1)[0])
        if not address.is_global:
            raise ValueError(f"Host {host!r} resolves to a non-public address")


class SDMXClient:
    """Reads SDMX-ML structure messages from one provider.

    Args:
        provider: a :class:`SDMXProvider`, a key of :data:`PROVIDERS` or an SDMX 2.1 base URL.
        timeout: seconds to connect and between bytes, and the total deadline of each request.
        max_bytes: largest answer accepted.
        safe: refuse non-public hosts, non-https URLs and off-host redirects (see the module doc).
        retries: attempts on connection errors, 429 and 5xx answers.
    """

    def __init__(self, provider: "SDMXProvider | str", timeout: float = 120, max_bytes: int = DEFAULT_MAX_BYTES,
                 safe: bool = False, retries: int = 2):
        self.provider = resolve_provider(provider)
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.safe = safe
        self.retries = max(1, retries)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        if safe:
            # The environment is kept: servers behind a TLS-inspecting proxy need its CA bundle.
            check_public_url(self.provider.base_url)

    def url(self, *path: str) -> str:
        """The URL of a resource below the provider's base URL (each part URL-quoted)."""
        return self.provider.base_url.rstrip("/") + "/" + "/".join(quote(p, safe="@$.,+~-_") for p in path)

    def get(self, *path: str, params: Optional[dict] = None, max_bytes: Optional[int] = None) -> bytes:
        """GET a structure resource and return its body.

        Raises:
            SDMXFetchError: network failure or an HTTP error (``status`` set).
            SDMXTooLargeError: the body is larger than ``max_bytes``.
        """
        url = self.url(*path)
        last: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                return self._get_once(url, params, max_bytes or self.max_bytes)
            except SDMXFetchError as exc:
                if exc.status is not None and exc.status not in (429, 500, 502, 503, 504):
                    raise
                last = exc
            except requests.RequestException as exc:
                last = SDMXFetchError(f"{self.provider.name} could not be reached: {exc.__class__.__name__}")
            if attempt + 1 < self.retries:
                time.sleep(min(2 ** attempt, 4))
        raise last  # type: ignore[misc]

    def _get_once(self, url: str, params: Optional[dict], max_bytes: int) -> bytes:
        deadline = time.monotonic() + self.timeout
        for _ in range(_MAX_REDIRECTS + 1):
            if self.safe:
                check_public_url(url)
            response = self.session.get(url, params=params, headers={"Accept": XML_ACCEPT}, stream=True,
                                        timeout=(min(self.timeout, 30), self.timeout), allow_redirects=False)
            try:
                if response.is_redirect and response.headers.get("Location"):
                    url, params = urljoin(response.url, response.headers["Location"]), None
                    continue
                if response.status_code >= 400:
                    raise SDMXFetchError(f"{self.provider.name} answered HTTP {response.status_code} for {url}",
                                         response.status_code)
                declared = response.headers.get("Content-Length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    raise SDMXTooLargeError(f"The structure from {self.provider.name} is larger than "
                                            f"{max_bytes // (1024 * 1024)} MB")
                chunks, size = [], 0
                for chunk in response.iter_content(chunk_size=65536):
                    size += len(chunk)
                    if size > max_bytes:
                        raise SDMXTooLargeError(f"The structure from {self.provider.name} is larger than "
                                                f"{max_bytes // (1024 * 1024)} MB")
                    if time.monotonic() > deadline:
                        raise SDMXFetchError(f"{self.provider.name} took longer than {self.timeout:.0f} s")
                    chunks.append(chunk)
                return b"".join(chunks)
            finally:
                response.close()
        raise SDMXFetchError(f"Too many redirects from {self.provider.name}")

    def fetch_structure(self, flow: "DataflowRef | str", lang: str = "en", with_constraint: bool = True,
                        with_hierarchies: bool = True) -> DataStructure:
        """The dataflow's structure, parsed, with its codelists, constraint and hierarchies when available.

        Raises:
            SDMXFetchError: the dataflow does not exist (``status`` 404) or the provider failed.
            ValueError: the provider's answer holds no data structure definition.
        """
        ref = parse_dataflow(flow) if isinstance(flow, str) else flow
        agency = ref.agency or self.provider.agency
        path = ["dataflow", agency, ref.id, ref.version]
        dsd, errors = None, []
        for references in self.provider.references:
            try:
                xml, path = self._get_dataflow(path, references, ref)
            except SDMXTooLargeError:
                raise
            except SDMXFetchError as exc:
                if exc.status == 404:
                    raise SDMXFetchError(f"Dataflow {agency}:{ref.id} not found at {self.provider.name}",
                                         404) from exc
                errors.append(str(exc))
                continue
            try:
                dsd = parse_structure(xml, ref.id, lang)
                break
            except ValueError:
                dsd = self._structure_of_dataflow_only(xml, ref, lang)
                if dsd is not None:
                    break
        if dsd is None:
            raise SDMXFetchError("; ".join(errors) or f"No data structure found for {agency}:{ref.id}")
        self._complete_codelists(dsd, lang)
        if with_constraint and not dsd.constraints:
            try:
                xml = self.get("contentconstraint", dsd.agency or agency, dsd.dataflow_id,
                               max_bytes=min(self.max_bytes, 20 * 1024 * 1024))
                dsd.constraints = parse_constraint(xml, dsd.dataflow_id)
            except (SDMXFetchError, ValueError):
                pass  # optional: roles fall back to the full codelists
        if with_hierarchies:
            for dim_id, (hcl_agency, hcl, version, hierarchy) in dsd.hier_context().items():
                try:
                    parents = parse_hierarchy(self.get("hierarchicalcodelist", hcl_agency, hcl, version), hierarchy)
                except (SDMXFetchError, ValueError):
                    continue
                if parents:
                    dsd.hierarchies[dim_id] = parents
        return dsd

    def _get_dataflow(self, path: list[str], references: str, ref: DataflowRef) -> tuple[bytes, list[str]]:
        """The dataflow message, and the path that worked."""
        try:
            return self.get(*path, params={"references": references}), path
        except SDMXFetchError as exc:
            if exc.status != 404 or len(path) != 4 or ref.version != "latest":
                raise
        # The version is optional in SDMX 2.1; some APIs (the Bundesbank's) only accept it left out.
        return self.get(*path[:3], params={"references": references}), path[:3]

    def _structure_of_dataflow_only(self, xml: bytes, ref: DataflowRef, lang: str) -> Optional[DataStructure]:
        """The provider sent the dataflow without its structure: follow the dataflow's reference."""
        found = dataflow_structure_ref(xml, ref.id, lang)
        if found is None:
            return None
        structure_ref, flow_id, flow_name = found
        try:
            dsd_xml = self.get("datastructure", structure_ref.agency or self.provider.agency, structure_ref.id,
                               structure_ref.version, params={"references": "children"})
            dsd = parse_structure(dsd_xml, ref.id, lang)
        except (SDMXFetchError, ValueError):
            return None
        dsd.dataflow_id, dsd.dataflow_name = flow_id or ref.id, flow_name or dsd.dataflow_name
        return dsd

    def _complete_codelists(self, dsd: DataStructure, lang: str) -> None:
        """Fetch the codelists a dimension refers to but the message did not include."""
        missing = [d for d in dsd.dimensions if d.codelist_ref and d.codelist_ref not in dsd.codelists]
        for dim in missing[:_MAX_EXTRA_CODELISTS]:
            source = dim.codelist_source
            if source is None or dim.codelist_ref in dsd.codelists:
                continue
            try:
                xml = self.get("codelist", source.agency or dsd.agency, source.id, source.version or "latest")
                dsd.codelists.update(parse_codelists(xml, lang))
            except (SDMXFetchError, ValueError):
                continue
