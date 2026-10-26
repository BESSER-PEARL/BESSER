"""Build a B-UML structural model from the structure of an SDMX dataflow.

SDMX is the ISO standard (ISO 17369) official statistics are published with: Eurostat, the ECB,
the OECD, the IMF, national statistical institutes. Every dataflow describes its data with a data
structure definition (dimensions, codelists, attributes, measure); this package reads it from the
provider's SDMX 2.1 REST API, or from an SDMX-ML file, and turns it into a validated
:class:`~besser.BUML.metamodel.structural.DomainModel`.

.. code-block:: python

    from besser.utilities.sdmx_to_buml import sdmx_to_buml

    model = sdmx_to_buml("eurostat", "une_rt_m")            # a preset provider
    model = sdmx_to_buml("ecb", "ECB,EXR,1.0")               # agency, id, version
    model = sdmx_to_buml("https://example.org/sdmx/rest", "DF_POP")  # any SDMX 2.1 API

See :mod:`.builder` for the mapping rules, :mod:`.roles` for the dimension roles the model's
metadata records and :mod:`.client` for the limits every import runs under.
"""

from typing import Iterable, Optional

from besser.BUML.metamodel.structural import DomainModel
from besser.utilities.sdmx_to_buml.builder import DEFAULT_ENUM_MAX, structure_to_buml
from besser.utilities.sdmx_to_buml.client import (
    DEFAULT_MAX_BYTES,
    DEFAULT_TIMEOUT,
    PROVIDERS,
    DataflowRef,
    SDMXClient,
    SDMXFetchError,
    SDMXImportLimitError,
    SDMXProvider,
    SDMXTimeoutError,
    SDMXTooLargeError,
)
from besser.utilities.sdmx_to_buml.structure import parse_structure

__all__ = [
    "PROVIDERS",
    "SDMXClient",
    "SDMXFetchError",
    "SDMXImportLimitError",
    "SDMXTimeoutError",
    "SDMXTooLargeError",
    "parse_structure",
    "sdmx_to_buml",
    "sdmx_xml_to_buml",
    "structure_to_buml",
]


def sdmx_to_buml(
    provider: "SDMXProvider | str",
    dataflow: "DataflowRef | str",
    model_name: Optional[str] = None,
    enum_max: int = DEFAULT_ENUM_MAX,
    prefix_hierarchy: Optional[Iterable[str]] = None,
    class_names: Optional[dict[str, str]] = None,
    lang: str = "en",
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = DEFAULT_MAX_BYTES,
    safe: bool = False,
) -> DomainModel:
    """Read a dataflow's structure from an SDMX 2.1 REST API and build its B-UML domain model.

    Args:
        provider: a key of :data:`PROVIDERS` (``"eurostat"``, ``"ecb"``, ``"oecd"``...) or the base
            URL of any SDMX 2.1 REST API.
        dataflow: ``ID``, ``AGENCY,ID``, ``AGENCY,ID,VERSION`` or ``AGENCY:ID(VERSION)``.
        model_name: name of the domain model; defaults to the dataflow id.
        enum_max: flat codelists with at most this many codes become enumerations, larger ones classes.
        prefix_hierarchy: dimension ids whose codes nest by prefix; ``None`` auto-detects them.
        class_names: names for the generated classes, keyed by dimension or codelist id.
        lang: language of the names read from the structure.
        timeout: seconds allowed for the whole import.
        max_bytes: largest structure message accepted (default 32 MB).
        safe: only https, public hosts, pinned connections (for servers fetching for others).

    Returns:
        DomainModel: the validated structural model.

    Raises:
        SDMXFetchError: the provider failed, the dataflow does not exist, or the import ran out of
            time (:class:`SDMXTimeoutError`) or size (:class:`SDMXTooLargeError`).
        ValueError: an invalid provider or dataflow reference, or an unusable structure message.
    """
    client = SDMXClient(provider, timeout=timeout, max_bytes=max_bytes, safe=safe)
    dsd = client.fetch_structure(dataflow, lang=lang)
    model, _ = structure_to_buml(dsd, model_name=model_name, enum_max=enum_max,
                                 prefix_hierarchy=prefix_hierarchy, class_names=class_names)
    return model


def sdmx_xml_to_buml(
    xml: "bytes | str",
    dataflow_id: Optional[str] = None,
    model_name: Optional[str] = None,
    enum_max: int = DEFAULT_ENUM_MAX,
    prefix_hierarchy: Optional[Iterable[str]] = None,
    class_names: Optional[dict[str, str]] = None,
    lang: str = "en",
) -> DomainModel:
    """Build the B-UML domain model of an SDMX-ML 2.1 structure message (a downloaded file).

    The message must hold the data structure definition and its codelists, as an SDMX API returns
    for ``dataflow/{agency}/{id}/latest?references=descendants``. The other arguments are those of
    :func:`sdmx_to_buml`.

    Raises:
        ValueError: the message is not SDMX-ML, holds no data structure, or holds dataflows but not
            ``dataflow_id``.
    """
    model, _ = structure_to_buml(parse_structure(xml, dataflow_id, lang), model_name=model_name,
                                 enum_max=enum_max, prefix_hierarchy=prefix_hierarchy, class_names=class_names)
    return model
