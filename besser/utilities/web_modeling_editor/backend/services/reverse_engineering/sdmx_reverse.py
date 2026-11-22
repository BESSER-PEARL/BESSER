"""Reverse engineer an SDMX dataflow into a B-UML domain model, for the web editor.

The editor's "From Data" import offers this next to CSV/XLSX files. The backend fetches the
structure on the user's behalf, so it always uses the client's ``safe`` mode (https, public hosts
only, pinned connections) and bounds every import: see the ``sdmx_*`` fields of ``BackendSettings``
in ``settings.py`` (``BESSER_SDMX_TIMEOUT``, ``BESSER_SDMX_MAX_MB``, ``BESSER_SDMX_MAX_TOTAL_MB``,
``BESSER_SDMX_MAX_CONCURRENT_IMPORTS``, ``BESSER_SDMX_ALLOW_CUSTOM_URL``).
"""

__all__ = ["custom_urls_allowed", "sdmx_import_slots", "sdmx_providers", "sdmx_to_domain_model"]

import logging
import threading
from typing import Optional

from besser.BUML.metamodel.structural import DomainModel
from besser.utilities.sdmx_to_buml import (
    PROVIDERS,
    SDMXClient,
    SDMXFetchError,
    SDMXTimeoutError,
    SDMXTooLargeError,
    structure_to_buml,
)
from besser.utilities.sdmx_to_buml.builder import pascal
from besser.utilities.sdmx_to_buml.client import MB, parse_dataflow, resolve_provider
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError, UpstreamError
from besser.utilities.web_modeling_editor.backend.settings import get_settings

logger = logging.getLogger(__name__)

def sdmx_import_slots() -> threading.BoundedSemaphore:
    """A fresh set of ``BESSER_SDMX_MAX_CONCURRENT_IMPORTS`` import slots, for one app (built in the lifespan)."""
    return threading.BoundedSemaphore(get_settings().sdmx_max_concurrent_imports)


def sdmx_providers() -> list[dict]:
    """The preset providers, for the editor's provider list."""
    return [{"key": p.key, "name": p.name, "baseUrl": p.base_url, "agency": p.agency} for p in PROVIDERS.values()]


def custom_urls_allowed() -> bool:
    """Whether the editor may import from an SDMX API that is not a preset."""
    return get_settings().sdmx_allow_custom_url


def _upstream_error(exc: SDMXFetchError) -> UpstreamError:
    if isinstance(exc, SDMXTooLargeError):
        return UpstreamError(str(exc), 413)
    if isinstance(exc, SDMXTimeoutError):
        return UpstreamError(str(exc), 504)
    return UpstreamError(str(exc), 404 if exc.status == 404 else 502)


def sdmx_to_domain_model(
    dataflow: str,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    agency: Optional[str] = None,
    version: Optional[str] = None,
    model_name: Optional[str] = None,
    *,
    slots: threading.BoundedSemaphore,
) -> DomainModel:
    """Fetch a dataflow's structure (safe mode) and build its domain model.

    Exactly one of ``provider`` (a preset key) and ``base_url`` (an https SDMX 2.1 API) is given.
    ``slots`` are the app's import slots (``SdmxImportSlotsDep``).

    Raises:
        ValueError: invalid input (an unknown provider, a non-https or private URL, a bad dataflow id).
        UpstreamError: the provider failed (502), the dataflow does not exist (404), the answer is too
            large (413), the import took too long (504), or too many imports are running (503).
        ConversionError: the structure could not be turned into a model.
    """
    if bool(provider) == bool(base_url):
        raise ValueError("Give either a provider or a base URL")
    if provider and provider.lower() not in PROVIDERS:
        raise ValueError(f"Unknown SDMX provider {provider!r}")
    if base_url and not custom_urls_allowed():
        raise ValueError("Importing from a custom SDMX URL is disabled on this server")
    if base_url and not base_url.lower().startswith("https://"):
        raise ValueError("Only https SDMX endpoints are allowed")
    ref = parse_dataflow(dataflow, agency=agency or None, version=version or None)
    settings = get_settings()
    if not slots.acquire(blocking=False):
        logger.warning("SDMX import of %s refused: %d imports already running", ref.id,
                       settings.sdmx_max_concurrent_imports)
        raise UpstreamError("Too many SDMX imports are running on this server; try again in a minute", 503)
    try:
        client = SDMXClient(resolve_provider(provider or base_url), timeout=settings.sdmx_timeout_seconds,
                            max_bytes=int(settings.sdmx_max_mb * MB),
                            max_total_bytes=int(settings.sdmx_max_total_mb * MB), safe=True, retries=1)
        dsd = client.fetch_structure(ref)
    except SDMXFetchError as exc:
        raise _upstream_error(exc) from exc
    finally:
        slots.release()
    try:
        model, _ = structure_to_buml(dsd, model_name=pascal(model_name) if model_name else None)
    except (ValueError, TypeError) as exc:
        raise ConversionError(f"The structure of {ref.id} could not be turned into a class diagram: {exc}") from exc
    return model
