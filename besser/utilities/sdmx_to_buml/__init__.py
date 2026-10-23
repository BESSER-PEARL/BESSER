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

See :mod:`.builder` for the mapping rules and :mod:`.roles` for the dimension roles the model's
metadata records.
"""

from typing import Iterable, Optional

from besser.BUML.metamodel.structural import DomainModel
from besser.utilities.sdmx_to_buml.builder import (
    DimensionMapping, ImportMapping, literal_name, pascal, snake, structure_to_buml,
)
from besser.utilities.sdmx_to_buml.client import (
    PROVIDERS, DataflowRef, SDMXClient, SDMXFetchError, SDMXProvider, SDMXTooLargeError,
    check_public_url, parse_dataflow, resolve_provider,
)
from besser.utilities.sdmx_to_buml.roles import DimensionRole, detect_roles, find_total, infer_prefix_parents
from besser.utilities.sdmx_to_buml.structure import (
    Code, Codelist, Component, DataStructure, parse_codelists, parse_constraint, parse_hierarchy, parse_structure,
)

__all__ = [
    "sdmx_to_buml", "sdmx_xml_to_buml", "structure_to_buml",
    "PROVIDERS", "DataflowRef", "SDMXClient", "SDMXFetchError", "SDMXProvider", "SDMXTooLargeError",
    "check_public_url", "parse_dataflow", "resolve_provider",
    "DimensionMapping", "ImportMapping", "DimensionRole", "detect_roles", "find_total", "infer_prefix_parents",
    "literal_name", "pascal", "snake",
    "Code", "Codelist", "Component", "DataStructure",
    "parse_codelists", "parse_constraint", "parse_hierarchy", "parse_structure",
]


def sdmx_to_buml(
    provider: "SDMXProvider | str",
    dataflow: "DataflowRef | str",
    model_name: Optional[str] = None,
    enum_max: int = 50,
    prefix_hierarchy: Optional[Iterable[str]] = None,
    class_names: Optional[dict[str, str]] = None,
    lang: str = "en",
    timeout: float = 120,
    max_bytes: Optional[int] = None,
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
        timeout: seconds allowed for each request.
        max_bytes: largest structure message accepted (default 100 MB).
        safe: only https, public hosts and bounded answers (for servers fetching for others).

    Returns:
        DomainModel: the validated structural model.

    Raises:
        SDMXFetchError: the provider failed or the dataflow does not exist.
        ValueError: an invalid provider or dataflow reference, or an unusable structure message.
    """
    client = SDMXClient(provider, timeout=timeout, safe=safe,
                        **({"max_bytes": max_bytes} if max_bytes else {}))
    dsd = client.fetch_structure(dataflow, lang=lang)
    model, _ = structure_to_buml(dsd, model_name=model_name, enum_max=enum_max,
                                 prefix_hierarchy=prefix_hierarchy, class_names=class_names)
    return model


def sdmx_xml_to_buml(
    xml: "bytes | str",
    dataflow_id: Optional[str] = None,
    model_name: Optional[str] = None,
    enum_max: int = 50,
    lang: str = "en",
) -> DomainModel:
    """Build the B-UML domain model of an SDMX-ML 2.1 structure message (a downloaded file).

    The message must hold the data structure definition and its codelists, as an SDMX API returns
    for ``dataflow/{agency}/{id}/latest?references=descendants``.
    """
    model, _ = structure_to_buml(parse_structure(xml, dataflow_id, lang), model_name=model_name,
                                 enum_max=enum_max)
    return model
