React Generator
===============

The React Generator produces a modern React frontend application from your GUI model. 
It is used internally by the :doc:`full_web_app` generator to create the frontend.
It is also one of the :doc:`Spec-Driven Agent's <../spec_driven_agent/index>`
generator tools, so an agentic run can produce this frontend and then customise
it further.

Overview
--------

The React Generator creates:

- **Pages**: React components for each screen in your GUI model
- **Components**: Reusable UI components (tables, forms, charts)
- **Contexts**: React contexts for state management
- **Routing**: Application navigation setup

Generated Components
--------------------

TableComponent
^^^^^^^^^^^^^^

Dynamic data tables with:

- Sorting and filtering
- Pagination
- Add/Edit modals with form validation
- Backend validation error display (keeps modal open on error)
- Support for all relationship types (1:1, 1:N, N:M)
- Lookup fields for related entities

MethodButton
^^^^^^^^^^^^

Executes class and instance methods with:

- Parameter input modal for methods with parameters
- Automatic type handling (int, float, bool, string, date, time)
- Detailed error display for 500 errors
- Modal stays open on error for retry
- Automatic table refresh after execution

Chart Components
^^^^^^^^^^^^^^^^

Visualization components including:

- Line charts
- Bar charts
- Pie charts
- Data binding to backend entities

MapBlock
^^^^^^^^

Interactive multi-layer map powered by `Leaflet <https://leafletjs.com/>`_ (used
directly, no wrapper library) with OpenStreetMap tiles (no API key required).
Each ``MapLayer`` on the metamodel ``Map`` component produces one rendered layer
in the generated output.  Popup labels come from database rows and are always
HTML-escaped.  Supported layer types:

- **points** — ``L.marker`` + popup per row (lat/lng columns required).
- **geojson** — ``L.geoJSON`` (geometry column required).
- **choropleth** — ``L.geoJSON`` with a value-driven fill colour + auto legend
  (geometry + value columns required; degrades to plain GeoJSON with a console
  warning when the value column is absent).
- **heatmap** — ``leaflet.heat`` heat layer (lat/lng columns required; weight
  column optional).

The generated ``MapBlock.tsx`` loops over the ``layers`` prop and dispatches to the
correct per-type renderer.  Each renderer is preceded by a section-comment banner
(``// ===== POINTS LAYER =====``, etc.) and extension blocks are left as commented
examples, so the output is self-documenting.  See :ref:`maps` for the full guide.

The following npm packages are automatically added to the generated ``package.json``
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

Usage
-----

The React Generator is typically used via the Full Web App Generator:

.. code-block:: python

    from besser.generators.web_app import WebAppGenerator
    
    gen = WebAppGenerator(domain_model, gui_model, output_dir="output")
    gen.generate()

Or standalone:

.. code-block:: python

    from besser.generators.react import ReactGenerator
    
    gen = ReactGenerator(domain_model, gui_model, output_dir="frontend")
    gen.generate()

Generated Structure
-------------------

.. code-block:: text

   frontend/
   ├── src/
   │   ├── components/
   │   │   ├── table/
   │   │   │   └── TableComponent.tsx
   │   │   ├── charts/
   │   │   ├── runtime/
   │   │   │   ├── TableBlock.tsx
   │   │   │   ├── ChartBlock.tsx
   │   │   │   ├── DataListBlock.tsx
   │   │   │   ├── MetricCardBlock.tsx
   │   │   │   └── MapBlock.tsx
   │   │   ├── InputComponents.tsx
   │   │   └── MethodButton.tsx
   │   ├── contexts/
   │   │   └── TableContext.tsx
   │   ├── pages/
   │   │   └── <PageName>.tsx
   │   ├── App.tsx
   │   ├── design.css
   │   └── index.tsx
   ├── public/
   ├── index.html
   ├── vite.config.ts
   ├── package.json
   └── tsconfig.json

One ``.tsx`` file is written per page in the GUI model, named after that page
(PascalCase), and ``App.tsx`` wires them into the router. ``MapBlock.tsx`` is
only emitted when the model contains a ``Map`` component.

Design stylesheet
-----------------

When the GUI model has a non-empty ``stylesheet`` (see :doc:`../buml_language/model_types/gui`),
it is written verbatim to ``src/design.css`` and imported from ``src/index.tsx``
after ``App``, so the design's class, ``:root``, ``@media`` and ``:hover`` rules
apply to the generated pages and win ties with the components' own CSS. Components
render their ``css_classes`` as ``className``; tables, charts and metric cards
forward it through their runtime blocks to the rendered root element.
Per-element ``Styling`` is still emitted as inline styles. Without a stylesheet
neither the file nor the import is generated.
