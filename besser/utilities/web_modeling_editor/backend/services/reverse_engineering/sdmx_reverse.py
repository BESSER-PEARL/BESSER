"""Reverse engineer an SDMX dataflow into a B-UML domain model, for the web editor.

The editor's "From data" import offers this next to CSV/XLSX files. The backend fetches the
structure on the user's behalf, so it always uses the client's ``safe`` mode (https, public hosts
only, bounded size and time). Limits can be tuned with environment variables:

* ``BESSER_SDMX_TIMEOUT``   seconds per request (default 60)
* ``BESSER_SDMX_MAX_MB``    largest structure message, in MB (default 64)
* ``BESSER_SDMX_ALLOW_CUSTOM_URL``  ``0`` to accept the preset providers only (default ``1``)
"""

__all__ = ["sdmx_to_domain_model", "sdmx_providers"]

import os
from typing import Optional

from besser.BUML.metamodel.structural import DomainModel
from besser.utilities.sdmx_to_buml import (
    PROVIDERS, SDMXClient, parse_dataflow, pascal, resolve_provider, structure_to_buml,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError

_TRUE = {"1", "true", "yes", "on"}


def _limit(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        return default
    return value if value > 0 else default


def sdmx_providers() -> list[dict]:
    """The preset providers, for the editor's provider list."""
    return [{"key": p.key, "name": p.name, "baseUrl": p.base_url, "agency": p.agency} for p in PROVIDERS.values()]


def custom_urls_allowed() -> bool:
    """Whether the editor may import from an SDMX API that is not a preset."""
    return os.environ.get("BESSER_SDMX_ALLOW_CUSTOM_URL", "1").strip().lower() in _TRUE


def sdmx_to_domain_model(
    dataflow: str,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    agency: Optional[str] = None,
    version: Optional[str] = None,
    model_name: Optional[str] = None,
) -> DomainModel:
    """Fetch a dataflow's structure (safe mode) and build its domain model.

    Exactly one of ``provider`` (a preset key) and ``base_url`` (an https SDMX 2.1 API) is given.

    Raises:
        ValueError: invalid input (an unknown provider, a non-https or private URL, a bad dataflow id).
        SDMXFetchError: the provider failed, the dataflow does not exist, or the answer is too large.
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
    client = SDMXClient(resolve_provider(provider or base_url), timeout=_limit("BESSER_SDMX_TIMEOUT", 60),
                        max_bytes=int(_limit("BESSER_SDMX_MAX_MB", 64) * 1024 * 1024), safe=True, retries=1)
    dsd = client.fetch_structure(ref)
    try:
        model, _ = structure_to_buml(dsd, model_name=pascal(model_name) if model_name else None)
    except (ValueError, TypeError) as exc:
        raise ConversionError(f"The structure of {ref.id} could not be turned into a class diagram: {exc}") from exc
    return model

