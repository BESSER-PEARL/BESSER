From an SDMX dataflow to B-UML
==============================

Official statistics are published with `SDMX <https://sdmx.org>`_ (ISO 17369): Eurostat, the
European Central Bank, the OECD, the IMF, the ILO and many national statistics offices describe
every dataset (a *dataflow*) with a data structure definition: its dimensions, the codelists of
their values, its attributes and its measure. BESSER reads that description from the provider's
SDMX 2.1 REST API and builds a :doc:`../model_types/structural` from it. No LLM is involved: the
mapping is deterministic.

.. code-block:: python

    from besser.BUML.metamodel.structural import DomainModel
    from besser.utilities.sdmx_to_buml import sdmx_to_buml

    model: DomainModel = sdmx_to_buml("eurostat", "une_rt_m")        # a preset provider
    model = sdmx_to_buml("ecb", "ECB,EXR,1.0")                         # agency, id and version
    model = sdmx_to_buml("https://example.org/sdmx/rest", "DF_POP")   # any SDMX 2.1 REST API

The first argument is a preset provider (``eurostat``, ``eurostat-comext``, ``ecb``, ``oecd``,
``imf``, ``bis``, ``ilo``, ``unicef``, ``abs``, ``insee``, ``istat``, ``unsd``, ``bundesbank``,
``lustat``; see ``besser.utilities.sdmx_to_buml.PROVIDERS``) or the base URL of an SDMX 2.1 REST
API. The dataflow is ``ID``, ``AGENCY,ID``, ``AGENCY,ID,VERSION`` or ``AGENCY:ID(VERSION)``.

The importer also reads a downloaded SDMX-ML structure message (one that holds the data structure
and its codelists, as ``dataflow/{agency}/{id}/latest?references=descendants`` returns):

.. code-block:: python

    from besser.utilities.sdmx_to_buml import sdmx_xml_to_buml

    with open("une_rt_m_structure.xml", "rb") as f:
        model = sdmx_xml_to_buml(f.read(), "une_rt_m")

Mapping rules
-------------

* The dataflow becomes an ``Observation`` class with the time period, the observed value
  (``obs_value``, a float) and one optional string attribute per SDMX attribute (``OBS_FLAG``...).
* A dimension with a small, flat codelist (at most ``enum_max`` codes, 50 by default) becomes an
  enumeration and a typed attribute of ``Observation``. Numeric codes take a literal derived from
  their label (``1`` "IMPORT" becomes ``IMPORT``).
* A dimension with a larger or hierarchical codelist becomes a class with ``code`` (its identifier)
  and ``label``, linked to ``Observation`` by a 1 to 0..* association. Dimensions that share a
  codelist (a reporter and a partner country) share the class.
* A hierarchical codelist adds a ``parent``/``children`` self-association and a ``level``
  attribute. Hierarchies come from SDMX parent references, from .Stat Suite hierarchical codelists
  (``HIER_CONTEXT``) or from numeric codes nesting by prefix (``84`` > ``8471`` > ``847130``).
* A dimension that is not coded, or whose codelist the provider does not publish, becomes a string.

Each class carries :class:`~besser.BUML.metamodel.structural.Metadata`: an ``sdmx:`` URI of its
dataflow or codelist, and a description of each dimension's *role* (``frequency``, ``unit``,
``geo``, ``hierarchy`` or ``breakdown``), its total code (``_T``, ``TOTAL``, ``WORLD``...) and the
code to use when the dimension is not analysed, so that values are never summed twice. In the web
editor these descriptions show as comments next to the classes.

``structure_to_buml(dsd, ...)`` returns the model and an ``ImportMapping`` (which code maps to which
literal, which column to which attribute), for a loader that fills the database generated from the
model.

.. code-block:: python

    from besser.utilities.sdmx_to_buml import SDMXClient, structure_to_buml

    dsd = SDMXClient("eurostat", timeout=120).fetch_structure("une_rt_m")
    model, mapping = structure_to_buml(dsd, enum_max=30, class_names={"geo": "Country"})

From the web editor
-------------------

In the project hub, **From Data** offers two sources: files (CSV, XLSX) and **Statistics API
(SDMX)**. Pick a provider (or, where the server allows it, enter the https URL of an SDMX 2.1 API),
enter the dataflow, and the class diagram is created in a new project. The backend fetches on the
user's behalf with ``safe=True``:

* only https on port 443, with no credentials, backslash, whitespace or control character in the
  URL;
* only hosts that resolve to public addresses, checked again on every redirect, and each connection
  goes to the address that was checked (so a DNS answer that changes in between cannot redirect it);
* one deadline and one size budget for the whole import (60 s, 32 MB per message, 64 MB in all), and
  at most two imports at once;
* custom URLs are off unless ``BESSER_SDMX_ALLOW_CUSTOM_URL`` is on.

The ``BESSER_SDMX_*`` variables change these limits (see :doc:`../../web_editor_backend`).

.. note::

    Behind a proxy that inspects TLS, point ``REQUESTS_CA_BUNDLE`` to a bundle that holds both the
    public roots and the proxy's certificate authority. Safe mode does not use the environment's
    HTTP proxies.

Tested providers
----------------

The live sweep in ``tests/utilities/sdmx_to_buml/live_sweep.py`` imported 32 dataflows from 14
providers in October 2026 (Eurostat, Eurostat Comext, STATEC LUSTAT, the ECB, the OECD, the IMF, the BIS, the ILO,
UNICEF, the ABS, INSEE, Istat, UN Data and the Bundesbank) and runs the class-diagram generators
on each model (Python, Pydantic, SQLAlchemy, SQL, JSON schema, Java, RDF, Django, REST API and
backend). Providers change their APIs, so run it again before relying on a provider; it needs the
network and writes its report and the generated code to the folder given with ``--out``, outside
the repository:

.. code-block:: console

    python -m tests.utilities.sdmx_to_buml.live_sweep --out /tmp/sdmx_sweep

More details in the :doc:`API documentation <../../api/BUML/notations/sdmx2buml/sdmx>`.
