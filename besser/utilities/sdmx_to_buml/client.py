"""A small SDMX 2.1 REST client for structure messages (SDMX-ML), built on ``requests``.

It reads what a B-UML model needs: a dataflow with its data structure, concepts and codelists,
completed when the provider allows with the content constraint and hierarchical codelists.

Every :meth:`SDMXClient.fetch_structure` runs under one budget: a total deadline, a size cap per
message and for the whole import, and a number of requests.

``safe=True`` is the mode for servers that fetch on behalf of someone else (the web editor):

* only ``https`` on port 443, with no credentials, backslash, whitespace or control character in
  the URL, read with the same parser ``requests`` connects with;
* only hosts that resolve to public addresses, checked again on every redirect;
* the connection goes to the address that was checked (the host name is kept for SNI and the
  certificate check), so a DNS answer that changes between the check and the request
  (DNS rebinding) cannot reach another address;
* environment proxies are not used (they would connect on their own); the CA bundle of
  ``REQUESTS_CA_BUNDLE`` or ``CURL_CA_BUNDLE`` is, for servers behind a TLS-inspecting gateway.
"""

import ipaddress
import logging
import os
import re
import socket
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote, urljoin, urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.exceptions import HTTPError as Urllib3HTTPError
from urllib3.exceptions import LocationParseError
from urllib3.util import parse_url

from besser.utilities.sdmx_to_buml.structure import (
    DataStructure,
    dataflow_structure_ref,
    parse_codelists,
    parse_constraint,
    parse_hierarchy,
    parse_structure,
)

logger = logging.getLogger(__name__)

USER_AGENT = "BESSER-sdmx-to-buml (+https://github.com/BESSER-PEARL/BESSER)"
XML_ACCEPT = "application/vnd.sdmx.structure+xml;version=2.1, application/xml;q=0.9, */*;q=0.1"

MB = 1024 * 1024
# The largest structure message seen in practice is Eurostat Comext's DS-045409 (the CN8 product
# nomenclature, 23.6 MB in October 2026); 32 MB keeps headroom without letting one message use
# much more memory than that while it is parsed (about 15 times its size).
DEFAULT_MAX_BYTES = 32 * MB
# All the messages of one import (the dataflow, the constraint, missing codelists, hierarchies).
DEFAULT_MAX_TOTAL_BYTES = 64 * MB
# Seconds for one whole import, every request and retry included.
DEFAULT_TIMEOUT = 120.0
# Requests for one import: the dataflow (two reference modes, with and without a version), the
# constraint, up to _MAX_EXTRA_CODELISTS codelists and _MAX_HIERARCHIES hierarchies, with retries.
DEFAULT_MAX_REQUESTS = 50

_CONNECT_TIMEOUT = 30.0
_READ_CHUNK = 64 * 1024
_MAX_REDIRECTS = 3
_MAX_EXTRA_CODELISTS = 30
_MAX_HIERARCHIES = 10
_CONSTRAINT_MAX_BYTES = 20 * MB
_RETRY_STATUSES = (429, 500, 502, 503, 504)
_MAX_RETRY_WAIT = 4.0
_MAX_ID_LENGTH = 120
_MAX_AGENCY_LENGTH = 80

# An SDMX id, agency or version: letters, digits and the separators SDMX allows (no "/" or "?").
_ID = re.compile(r"^[A-Za-z0-9_@$\-]+(\.[A-Za-z0-9_@$\-]+)*$")
_VERSION = re.compile(r"^(latest|\+|~|[0-9]+(\.[0-9]+){0,2}(\+)?)$")
# Characters two URL parsers can read differently: refused anywhere in a URL in safe mode.
_UNSAFE_URL_CHARS = re.compile(r"[\\\s\x00-\x1f\x7f]")
# The NAT64 prefix maps IPv4 addresses (private ones too) into IPv6.
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
# One message for unresolvable and non-public hosts, so errors do not reveal internal names.
_NOT_PUBLIC = "The SDMX host cannot be reached from this server or is not a public address"


class SDMXFetchError(RuntimeError):
    """The provider could not be reached, refused the query, or answered with something unusable.

    ``status`` is the HTTP status when there was one (404: the dataflow does not exist).
    """

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class SDMXTooLargeError(SDMXFetchError):
    """One answer is larger than the client's ``max_bytes``."""


class SDMXImportLimitError(SDMXTooLargeError):
    """The import as a whole needs more bytes (``max_total_bytes``) or requests than its budget."""


class SDMXTimeoutError(SDMXFetchError):
    """The import took longer than the client's ``timeout``."""


# Limits of the whole import: an optional resource that hits one stops the import instead of being skipped.
_IMPORT_LIMITS = (SDMXImportLimitError, SDMXTimeoutError)


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
    if not ref.id or len(ref.id) > _MAX_ID_LENGTH or not _ID.match(ref.id):
        raise ValueError(f"Not a valid SDMX dataflow id: {ref.id!r}")
    if ref.agency is not None and (len(ref.agency) > _MAX_AGENCY_LENGTH or not _ID.match(ref.agency)):
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


def _is_public(address: "ipaddress.IPv4Address | ipaddress.IPv6Address") -> bool:
    if address.version == 6:
        if address.ipv4_mapped is not None:
            return address.ipv4_mapped.is_global
        if address in _NAT64:
            return False
    return address.is_global


def check_public_url(url: str) -> str:
    """Refuse anything but ``https`` on port 443 to a host that resolves only to public addresses.

    The URL is read with ``urllib3``'s parser, the one ``requests`` connects with, and must give
    the same host with the standard library's: two parsers that disagree are how a check is fooled.

    Returns:
        The public address the host resolves to: connect to it (not to the name) so the check holds.

    Raises:
        ValueError: for any other URL (private, loopback, link-local, credentials in the URL...).
    """
    if _UNSAFE_URL_CHARS.search(url):
        raise ValueError("The URL contains a backslash, whitespace or a control character")
    try:
        parsed = parse_url(url)
    except LocationParseError as exc:
        raise ValueError("Not a valid URL") from exc
    if (parsed.scheme or "").lower() != "https":
        raise ValueError("Only https SDMX endpoints are allowed")
    split = urlsplit(url)
    if parsed.auth or split.username or split.password:
        raise ValueError("Credentials in the URL are not allowed")
    if parsed.port not in (None, 443):
        raise ValueError("Only the default https port (443) is allowed")
    host = (parsed.host or "").strip("[]").lower()
    if not host:
        raise ValueError("The URL has no host")
    if host != (split.hostname or "").lower():
        raise ValueError("The URL's host is ambiguous")
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError) as exc:
        raise ValueError(_NOT_PUBLIC) from exc
    addresses = [ipaddress.ip_address(info[4][0].split("%", 1)[0]) for info in infos]
    if not addresses or not all(_is_public(a) for a in addresses):
        raise ValueError(_NOT_PUBLIC)
    return str(addresses[0])


def _ca_bundle() -> "str | bool":
    """The CA bundle ``requests`` would take from the environment, else its default."""
    return os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE") or True


class _PinnedAdapter(HTTPAdapter):
    """Connects to an IP address while checking the certificate of, and sending SNI for, a host name."""

    def __init__(self, host: str):
        self._host = host
        super().__init__(max_retries=0)

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        pool_kwargs["server_hostname"] = self._host
        pool_kwargs["assert_hostname"] = self._host
        super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)


def _size(n: int) -> str:
    """``33554432`` -> ``32 MB``, ``1500`` -> ``1500 bytes``."""
    return f"{n / MB:.0f} MB" if n >= MB else f"{n} bytes"


@dataclass
class _Budget:
    """What is left of one import: time, bytes and requests."""
    deadline: float
    bytes_left: int
    requests_left: int

    def remaining(self) -> float:
        return self.deadline - time.monotonic()


class SDMXClient:
    """Reads SDMX-ML structure messages from one provider.

    Args:
        provider: a :class:`SDMXProvider`, a key of :data:`PROVIDERS` or an SDMX 2.1 base URL.
        timeout: seconds for one whole :meth:`fetch_structure` (or one :meth:`get`), retries included.
        max_bytes: largest single answer accepted.
        safe: refuse non-public hosts and non-https URLs, and pin each connection (see the module doc).
        retries: attempts on connection errors, 429 and 5xx answers.
        max_total_bytes: bytes read by one import, all answers together.
        max_requests: requests made by one import.
    """

    def __init__(self, provider: "SDMXProvider | str", timeout: float = DEFAULT_TIMEOUT,
                 max_bytes: int = DEFAULT_MAX_BYTES, safe: bool = False, retries: int = 2,
                 max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES, max_requests: int = DEFAULT_MAX_REQUESTS):
        self.provider = resolve_provider(provider)
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_total_bytes = max(max_total_bytes, max_bytes)
        self.max_requests = max_requests
        self.safe = safe
        self.retries = max(1, retries)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._budget: Optional[_Budget] = None
        if safe:
            check_public_url(self.provider.base_url)

    def url(self, *path: str) -> str:
        """The URL of a resource below the provider's base URL.

        Raises:
            ValueError: a part is not an SDMX id or version (``..``, ``/`` or ``?`` could leave the API).
        """
        for part in path:
            if not (_ID.match(part) or _VERSION.match(part)):
                raise ValueError(f"Not a valid SDMX path part: {part!r}")
        return self.provider.base_url.rstrip("/") + "/" + "/".join(quote(p, safe="@$.,+~-_") for p in path)

    def _new_budget(self) -> _Budget:
        return _Budget(time.monotonic() + self.timeout, self.max_total_bytes, self.max_requests)

    def get(self, *path: str, params: Optional[dict] = None, max_bytes: Optional[int] = None) -> bytes:
        """GET a structure resource and return its body.

        Outside :meth:`fetch_structure`, the call has a budget of its own.

        Raises:
            SDMXFetchError: network failure or an HTTP error (``status`` set).
            SDMXTooLargeError: the body is larger than ``max_bytes``.
            SDMXImportLimitError: the import's byte or request budget is spent.
            SDMXTimeoutError: the import's deadline has passed.
        """
        budget = self._budget or self._new_budget()
        url = self.url(*path)
        last: Optional[SDMXFetchError] = None
        for attempt in range(self.retries):
            try:
                return self._get_once(url, params, min(max_bytes or self.max_bytes, self.max_bytes), budget)
            except _IMPORT_LIMITS:
                raise
            except SDMXFetchError as exc:
                if exc.status is not None and exc.status not in _RETRY_STATUSES:
                    raise
                last = exc
            wait = min(2.0 ** attempt, _MAX_RETRY_WAIT)
            if attempt + 1 < self.retries and budget.remaining() > wait:
                logger.warning("%s; retrying in %.0f s", last, wait)
                time.sleep(wait)
            elif attempt + 1 < self.retries:
                break
        raise last  # type: ignore[misc]

    def _get_once(self, url: str, params: Optional[dict], max_bytes: int, budget: _Budget) -> bytes:
        for _ in range(_MAX_REDIRECTS + 1):
            remaining = budget.remaining()
            if remaining <= 0:
                raise SDMXTimeoutError(f"{self.provider.name} took longer than {self.timeout:.0f} s")
            if budget.requests_left <= 0:
                raise SDMXImportLimitError(f"The structure from {self.provider.name} needs more than "
                                           f"{self.max_requests} requests")
            budget.requests_left -= 1
            try:
                response = self._send(url, params, remaining)
            except requests.RequestException as exc:
                raise self._network_error(exc, budget) from exc
            try:
                if response.is_redirect and response.headers.get("Location"):
                    url, params = urljoin(url, response.headers["Location"]), None
                    continue
                if response.status_code >= 400:
                    raise SDMXFetchError(f"{self.provider.name} answered HTTP {response.status_code} for {url}",
                                         response.status_code)
                return self._read_body(response, max_bytes, budget)
            finally:
                response.close()
        raise SDMXFetchError(f"Too many redirects from {self.provider.name}")

    def _send(self, url: str, params: Optional[dict], remaining: float) -> requests.Response:
        """One request, not following redirects; in safe mode, to the checked address only."""
        timeout = (min(_CONNECT_TIMEOUT, remaining), remaining)
        headers = {"Accept": XML_ACCEPT}
        if not self.safe:
            return self.session.get(url, params=params, headers=headers, stream=True, timeout=timeout,
                                    allow_redirects=False)
        address = check_public_url(url)
        parsed = parse_url(url)
        pinned = parsed._replace(host=f"[{address}]" if ":" in address else address).url
        with requests.Session() as session:
            # A fresh session per request: the pinned adapter belongs to this host and address only.
            session.trust_env = False
            session.mount("https://", _PinnedAdapter(parsed.host.strip("[]")))
            headers.update({"User-Agent": USER_AGENT, "Host": parsed.netloc})
            return session.get(pinned, params=params, headers=headers, stream=True, timeout=timeout,
                               allow_redirects=False, verify=_ca_bundle())

    def _network_error(self, exc: Exception, budget: _Budget) -> SDMXFetchError:
        if budget.remaining() <= 0 or isinstance(exc, (requests.Timeout, TimeoutError)):
            return SDMXTimeoutError(f"{self.provider.name} took longer than {self.timeout:.0f} s")
        return SDMXFetchError(f"{self.provider.name} could not be reached: {exc.__class__.__name__}")

    def _read_body(self, response: requests.Response, max_bytes: int, budget: _Budget) -> bytes:
        """The body, read a socket read at a time so the deadline and the size caps hold while it arrives."""
        limit = min(max_bytes, budget.bytes_left)
        declared = response.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise self._size_error(max_bytes, budget)
        chunks, size = [], 0
        sock = getattr(getattr(response.raw, "connection", None), "sock", None)
        while True:
            remaining = budget.remaining()
            if remaining <= 0:
                raise SDMXTimeoutError(f"{self.provider.name} took longer than {self.timeout:.0f} s")
            if sock is not None:
                sock.settimeout(remaining)  # a server sending one byte at a time cannot outlast the deadline
            try:
                chunk = response.raw.read1(_READ_CHUNK, decode_content=True)
            except (Urllib3HTTPError, OSError) as exc:
                raise self._network_error(exc, budget) from exc
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise self._size_error(max_bytes, budget)
            chunks.append(chunk)
        budget.bytes_left -= size
        return b"".join(chunks)

    def _size_error(self, max_bytes: int, budget: _Budget) -> SDMXTooLargeError:
        """The import's byte budget ran out, or this one answer is over its own cap."""
        if budget.bytes_left < max_bytes:
            return SDMXImportLimitError(f"The structure from {self.provider.name} needs more than "
                                        f"{_size(self.max_total_bytes)} in all", 413)
        return SDMXTooLargeError(f"The structure from {self.provider.name} is larger than {_size(max_bytes)}", 413)

    def fetch_structure(self, flow: "DataflowRef | str", lang: str = "en", with_constraint: bool = True,
                        with_hierarchies: bool = True) -> DataStructure:
        """The dataflow's structure, parsed, with its codelists, constraint and hierarchies when available.

        The whole call runs under one budget (``timeout``, ``max_total_bytes``, ``max_requests``).

        Raises:
            SDMXFetchError: the dataflow does not exist (``status`` 404) or the provider failed.
            SDMXTimeoutError, SDMXImportLimitError: the import ran out of time, bytes or requests.
            ValueError: the provider's answer holds no data structure definition, or not this dataflow.
        """
        ref = parse_dataflow(flow) if isinstance(flow, str) else flow
        self._budget = self._new_budget()
        try:
            dsd = self._fetch_dataflow(ref, lang)
            self._complete_codelists(dsd, lang)
            if with_constraint and not dsd.constraints:
                self._add_constraint(dsd, ref)
            if with_hierarchies:
                self._add_hierarchies(dsd)
            return dsd
        finally:
            self._budget = None

    def _fetch_dataflow(self, ref: DataflowRef, lang: str) -> DataStructure:
        agency = ref.agency or self.provider.agency
        path = ["dataflow", agency, ref.id, ref.version]
        errors: list[str] = []
        parse_error: Optional[ValueError] = None
        for references in self.provider.references:
            try:
                xml, path = self._get_dataflow(path, references, ref)
            except (SDMXTooLargeError, SDMXTimeoutError):
                raise
            except SDMXFetchError as exc:
                if exc.status == 404:
                    raise SDMXFetchError(f"Dataflow {agency}:{ref.id} not found at {self.provider.name}",
                                         404) from exc
                logger.warning("%s (references=%s)", exc, references)
                errors.append(str(exc))
                continue
            try:
                return parse_structure(xml, ref.id, lang)
            except ValueError as exc:
                parse_error = exc
                dsd = self._structure_of_dataflow_only(xml, ref, lang)
                if dsd is not None:
                    return dsd
                logger.warning("No data structure for %s:%s with references=%s: %s", agency, ref.id,
                               references, exc)
        if parse_error is not None and not errors:
            raise parse_error
        raise SDMXFetchError("; ".join(errors) or f"No data structure found for {agency}:{ref.id}")

    def _get_dataflow(self, path: list[str], references: str, ref: DataflowRef) -> tuple[bytes, list[str]]:
        """The dataflow message, and the path that worked."""
        try:
            return self.get(*path, params={"references": references}), path
        except SDMXFetchError as exc:
            if exc.status != 404 or len(path) != 4 or ref.version != "latest":
                raise
        # The version is optional in SDMX 2.1; some APIs (the Bundesbank's) only accept it left out.
        logger.info("%s has no %s with a version; asking without it", self.provider.name, "/".join(path))
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
        except _IMPORT_LIMITS:
            raise
        except (SDMXFetchError, ValueError) as exc:
            logger.warning("Could not read structure %s of dataflow %s: %s", structure_ref.id, ref.id, exc)
            return None
        dsd.dataflow_id, dsd.dataflow_name = flow_id or ref.id, flow_name or dsd.dataflow_name
        return dsd

    def _complete_codelists(self, dsd: DataStructure, lang: str) -> None:
        """Fetch the codelists a dimension refers to but the message did not include."""
        missing = [d for d in dsd.dimensions if d.codelist_ref and d.codelist_ref not in dsd.codelists]
        if len(missing) > _MAX_EXTRA_CODELISTS:
            logger.warning("%d codelists are missing from %s; fetching the first %d", len(missing),
                           dsd.dataflow_id, _MAX_EXTRA_CODELISTS)
        for dim in missing[:_MAX_EXTRA_CODELISTS]:
            source = dim.codelist_source
            if source is None or dim.codelist_ref in dsd.codelists:
                continue
            try:
                xml = self.get("codelist", source.agency or dsd.agency, source.id, source.version or "latest")
                dsd.codelists.update(parse_codelists(xml, lang))
            except _IMPORT_LIMITS:
                raise
            except (SDMXFetchError, ValueError) as exc:
                logger.warning("Codelist %s of dimension %s is unavailable, so it becomes a string: %s",
                               source.id, dim.id, exc)

    def _add_constraint(self, dsd: DataStructure, ref: DataflowRef) -> None:
        """The codes actually used, from the dataflow's content constraint (optional)."""
        try:
            xml = self.get("contentconstraint", dsd.agency or ref.agency or self.provider.agency, dsd.dataflow_id,
                           max_bytes=_CONSTRAINT_MAX_BYTES)
            dsd.constraints = parse_constraint(xml, dsd.dataflow_id)
        except _IMPORT_LIMITS:
            raise
        except (SDMXFetchError, ValueError) as exc:
            logger.info("No content constraint for %s (roles use the full codelists): %s", dsd.dataflow_id, exc)

    def _add_hierarchies(self, dsd: DataStructure) -> None:
        """Parents from the hierarchical codelists the dataflow's annotations name (optional)."""
        contexts = list(dsd.hier_context().items())
        if len(contexts) > _MAX_HIERARCHIES:
            logger.warning("%s names %d hierarchies; reading the first %d", dsd.dataflow_id, len(contexts),
                           _MAX_HIERARCHIES)
        for dim_id, (hcl_agency, hcl, version, hierarchy) in contexts[:_MAX_HIERARCHIES]:
            try:
                parents = parse_hierarchy(self.get("hierarchicalcodelist", hcl_agency, hcl, version), hierarchy)
            except _IMPORT_LIMITS:
                raise
            except (SDMXFetchError, ValueError) as exc:
                logger.warning("Hierarchy %s of dimension %s is unavailable: %s", hcl, dim_id, exc)
                continue
            if parents:
                dsd.hierarchies[dim_id] = parents
