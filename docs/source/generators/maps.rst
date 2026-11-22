.. _maps:

Map Component Reference
=======================

The BESSER Map component generates an interactive, multi-layer map in the produced
React application. It is powered by `Leaflet <https://leafletjs.com/>`_ (used
directly, no wrapper library) with OpenStreetMap tiles — no API key is required.

A single ``Map`` can stack multiple **typed layers** on top of the same tile base,
each fetching data from its own domain class:

- A store-locator map combining point markers for each branch and a choropleth region
  layer to colour territories by sales volume is expressed as two layers on one map.
- A heatmap of incidents overlaid on GeoJSON district boundaries is two layers.

.. contents:: On this page
   :depth: 2
   :local:

Layer Types
-----------

Each layer has a ``layer_type`` that controls how its data is fetched and rendered.

.. list-table::
   :header-rows: 1
   :widths: 14 28 12 46

   * - ``layer_type``
     - Required columns
     - Optional columns
     - Renderer
   * - ``points``
     - ``latitude: float``, ``longitude: float``
     - ``label: str``
     - ``L.marker`` + popup for every row
   * - ``geojson``
     - ``geometry: str`` (GeoJSON string)
     - ``label: str``
     - ``L.geoJSON``; label shown in popup on click
   * - ``choropleth``
     - ``geometry: str``, ``value: float``
     - ``label: str``
     - ``L.geoJSON`` coloured by value + auto legend; gracefully falls back to
       plain GeoJSON when ``value_field`` is absent
   * - ``heatmap``
     - ``latitude: float``, ``longitude: float``
     - ``weight: float``
     - ``leaflet.heat``; weight controls intensity

.. note::
   Column names above are the **field names on the bound domain class**. The column
   does not have to be named exactly ``latitude`` — you map any ``Property`` to its
   role via ``latitude_field=``, ``geojson_field=``, etc. on ``MapLayer``.

Auto-detection
~~~~~~~~~~~~~~

When ``layer_type`` is ``None`` (the default), ``effective_layer_type()`` infers it
from the configured fields and, as a fallback, from the attribute names and types on
the bound class:

1. Explicit ``layer_type`` wins.
2. ``geojson_field`` **and** ``value_field`` set → ``choropleth``.
3. ``geojson_field`` set (no ``value_field``) → ``geojson``.
4. ``weight_field`` set → ``heatmap``.
5. ``latitude_field`` and ``longitude_field`` set → ``points``.
6. Inspect bound class attributes by name (exact, case-insensitive matches):

   - an attribute named ``geometry`` → ``geojson`` (or ``choropleth`` if an
     attribute named ``value`` also exists).
   - attributes named ``latitude``/``lat`` + ``longitude``/``lng``/``lon`` →
     ``points`` (or ``heatmap`` if an attribute named ``weight`` also exists).
7. Default: ``points``.

Metamodel
---------

``MapLayerType``
~~~~~~~~~~~~~~~~

.. code-block:: python

   from besser.BUML.metamodel.gui.dashboard import MapLayerType

   MapLayerType.points     # lat/lng markers
   MapLayerType.geojson    # GeoJSON polygons or lines
   MapLayerType.choropleth # value-coloured GeoJSON + legend
   MapLayerType.heatmap    # density heat layer

``MapLayer``
~~~~~~~~~~~~

A ``ViewComponent`` subclass that holds one data layer on a ``Map``.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Parameter
     - Default
     - Description
   * - ``name``
     - (required)
     - Unique layer name.
   * - ``layer_type``
     - ``None``
     - ``MapLayerType`` value, or ``None`` to use auto-detection.
   * - ``data_binding``
     - ``None``
     - ``DataBinding`` linking the layer to a domain ``Class``.
   * - ``latitude_field``
     - ``None``
     - ``Property`` whose value is the row's latitude (``points``/``heatmap``).
   * - ``longitude_field``
     - ``None``
     - ``Property`` whose value is the row's longitude (``points``/``heatmap``).
   * - ``label_field``
     - ``None``
     - ``Property`` shown in the popup on click (all types, optional).
   * - ``weight_field``
     - ``None``
     - ``Property`` used as the heat intensity (``heatmap``, optional).
   * - ``geojson_field``
     - ``None``
     - ``Property`` containing a GeoJSON string (``geojson``/``choropleth``).
   * - ``value_field``
     - ``None``
     - ``Property`` used to colour regions (``choropleth``).

All field setters validate that the assigned value is a ``Property`` instance (or
``None``). Assigning a non-``Property`` raises ``TypeError``.

``Map``
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Parameter
     - Default
     - Description
   * - ``name``
     - (required)
     - Unique component name.
   * - ``title``
     - ``None``
     - Optional heading displayed above the map.
   * - ``center_latitude``
     - ``0.0``
     - Initial map centre latitude (decimal degrees).
   * - ``center_longitude``
     - ``0.0``
     - Initial map centre longitude (decimal degrees).
   * - ``zoom``
     - ``10``
     - Initial zoom level (1 = world, 18 = street).
   * - ``layers``
     - ``[]``
     - List of ``MapLayer`` instances rendered on this map.

``WorldMap`` and ``LocationMap`` are thin subclasses of ``Map`` with no additional
parameters.

Worked Example
--------------

The following model produces a two-layer map: pin-drop markers for Luxembourg store
locations and a choropleth colouring each canton by population.

.. code-block:: python

   from besser.BUML.metamodel.structural import (
       Class, Property, DomainModel, FloatType, StringType,
   )
   from besser.BUML.metamodel.gui import GUIModel, Module, Screen, DataBinding
   from besser.BUML.metamodel.gui.dashboard import Map, MapLayer, MapLayerType

   # ── Domain model ────────────────────────────────────────────────────────────

   # Table 1: Store locations — point markers
   lat       = Property(name="latitude",   type=FloatType)
   lng       = Property(name="longitude",  type=FloatType)
   store_name = Property(name="name",      type=StringType)
   store = Class(name="Store", attributes={lat, lng, store_name})

   # Table 2: Canton polygons — choropleth
   geometry   = Property(name="geometry",   type=StringType)  # GeoJSON string
   population = Property(name="population", type=FloatType)
   canton_name = Property(name="name",      type=StringType)
   canton = Class(name="Canton", attributes={geometry, population, canton_name})

   domain = DomainModel(name="LuxembourgMap", types={store, canton})

   # ── Layers ───────────────────────────────────────────────────────────────────

   store_layer = MapLayer(
       name="stores",
       layer_type=MapLayerType.points,
       data_binding=DataBinding(name="store_binding", domain_concept=store),
       latitude_field=lat,
       longitude_field=lng,
       label_field=store_name,
   )

   canton_layer = MapLayer(
       name="cantons",
       layer_type=MapLayerType.choropleth,
       data_binding=DataBinding(name="canton_binding", domain_concept=canton),
       geojson_field=geometry,
       value_field=population,
       label_field=canton_name,
   )

   # ── Map component ────────────────────────────────────────────────────────────

   lux_map = Map(
       name="LuxMap",
       title="Luxembourg — Stores & Cantons",
       center_latitude=49.8153,
       center_longitude=6.1296,
       zoom=9,
       layers=[store_layer, canton_layer],
   )

   # ── GUI model ────────────────────────────────────────────────────────────────

   screen = Screen(name="MapScreen", description="", view_elements={lux_map},
                   is_main_page=True)
   module = Module(name="AppModule", screens={screen})
   gui = GUIModel(name="LuxApp", package="com.example.lux",
                  versionCode="1", versionName="1.0", modules={module})

Running ``WebAppGenerator(domain, gui, output_dir="output").generate()`` on this
model produces a React application with ``MapBlock.tsx`` that:

- Fetches rows from ``GET /store/`` and renders a pin for each store.
- Fetches rows from ``GET /canton/`` and colours each polygon by ``population``
  with a sequential white-to-red scale and an auto-generated legend.
- Both layers overlay the same OpenStreetMap tile base.

Generated ``MapBlock.tsx`` structure
-------------------------------------

The generated ``MapBlock.tsx`` drives **Leaflet directly** — there is no
``react-leaflet`` wrapper. The map instance lives in a ref, layers are plain
Leaflet layer groups, and popup text is always HTML-escaped. Each layer type
has its own section-comment banner so you can extend an individual renderer
without reading BESSER documentation:

.. code-block:: text

   MapBlock.tsx
   ├── Types: LayerConfig, MapConfig, MapBlockProps
   ├── fetchRows()          — robust fetch + row normaliser
   ├── escapeHtml()         — every dynamic value in a popup goes through this
   ├── choroplethColor()    — 5-step sequential white→red scale
   │
   ├── // ===== POINTS LAYER =====
   │   renderPointsLayer()     — fetches rows, L.marker + bindPopup
   │
   ├── // ===== GEOJSON LAYER =====
   │   renderGeoJsonLayer()    — fetches rows, L.geoJSON
   │
   ├── // ===== CHOROPLETH LAYER =====
   │   renderChoroplethLayer() — L.geoJSON + style fn + a legend added as
   │                             a new L.Control({position:"bottomright"})
   │
   ├── // ===== HEATMAP LAYER =====
   │   renderHeatLayer()       — dynamic import("leaflet.heat") + L.heatLayer()
   │
   ├── renderLayer()        — switch on layer.type, dispatching to the above;
   │                          falls back to the points renderer with a console.warn
   └── MapBlock (export)    — loops layers, static fallback when layers empty
       └── // ===== EXTENSION: CUSTOM MARKER ICONS / IMAGE POPUPS =====
           └── commented-out examples: L.icon(), an <img> in bindPopup(),
               and a per-layer colour override via L.divIcon()

Extending the generated ``MapBlock.tsx``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because section banners and commented extension blocks are inlined, you can make
most common customisations without touching any BESSER source:

**Custom marker icons** — uncomment and fill in the ``L.icon`` block in the
extension section at the bottom of the file, then swap it into the
``L.marker(...)`` call inside ``renderPointsLayer``:

.. code-block:: tsx

   const myIcon = L.icon({
     iconUrl: customIconPng,
     iconSize: [32, 32],
     iconAnchor: [16, 32],
     popupAnchor: [0, -32],
   });
   // then, inside renderPointsLayer, replace
   //   L.marker([lat, lng], { icon: defaultIcon })
   // with
   //   L.marker([lat, lng], { icon: myIcon })

The default pin is an ``L.divIcon`` built from an inline SVG, so recolouring
markers per layer needs no image asset at all — the extension section shows
that variant too.

**Per-choropleth colour scale** — replace ``choroplethColor()`` with your own
palette or a ``d3-scale`` interpolator.

**Image popups** — the commented ``bindPopup`` example in the extension section
shows how to display a thumbnail from a URL field. Run every dynamic value
through ``escapeHtml`` first: popups render HTML.

**Additional layer type** — add a new ``case 'mytype':`` to the ``switch`` in
``renderLayer()`` and a corresponding ``renderMytypeLayer()`` function above it.

Graceful degradation
~~~~~~~~~~~~~~~~~~~~

- A ``choropleth`` layer with no ``valueField`` automatically falls back to
  plain GeoJSON rendering and logs a ``console.warn``.
- Any layer with a misconfigured ``type`` falls back to ``renderPointsLayer``
  with a ``console.warn``.
- Parse errors in the geometry column are caught per-row and logged with a
  ``console.warn``, then skipped.
- No layer configuration → a static centre marker is shown.

npm Dependencies
----------------

The following packages are automatically added to the generated ``package.json``
when the GUI model contains at least one ``Map`` component:

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Package
     - Version
     - Purpose
   * - ``leaflet``
     - ``^1.9.4``
     - Core Leaflet mapping library (BSD-2-Clause), used directly.
   * - ``leaflet.heat``
     - ``^0.2.0``
     - Heat-map plugin (used when any layer is ``heatmap`` type).
   * - ``@types/leaflet``
     - ``^1.9.12``
     - TypeScript type definitions for Leaflet.
   * - ``@types/leaflet.heat``
     - ``^0.2.4``
     - TypeScript type definitions for leaflet.heat.

No API key is required. OpenStreetMap tiles are served by the public OSM tile CDN.

Editor (No-Code)
----------------

In the BESSER Web Modeling Editor the Map block lives in the **Charts** palette.

After dropping a ``Map`` onto a screen, the right-hand sidebar shows:

- **Title**, **Latitude**, **Longitude**, **Zoom** — static display settings.
- **Layers** — a repeatable panel (one row per layer) with:

  - A name field.
  - A **Type** dropdown: ``Points``, ``GeoJSON``, ``Choropleth``, ``Heat map``.
  - A **Data Source** dropdown (populated from the current class diagram).
  - Conditional **field selects** driven by the chosen type (latitude/longitude
    for points, geometry for GeoJSON/choropleth, etc.).

Use the **+** button to add a layer and **×** to remove one.  The layer list is
serialised as a JSON string in the ``map-layers`` component attribute, which the
BESSER backend parses into ``MapLayer`` instances on export.
