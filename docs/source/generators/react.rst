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

- Sorting and filtering; an enumeration column is filtered by choosing one of
  its literals
- Enumeration literals offered in the model's order (columns, Add/Edit dialog,
  association class fields, method parameters)
- Pagination
- Add/Edit modals with form validation; an empty field of an ``is_optional``
  attribute (``"nullable": true`` in ``formColumns``) is saved as ``null``, not
  ``""`` or ``0``
- A confirmation prompt before a row's delete icon removes the record
- Backend validation error display (keeps modal open on error)
- Support for all relationship types (1:1, 1:N, N:M)
- Lookup fields for related entities; the Add/Edit dialog offers only the
  relationships the backend ``<Entity>Create`` schema accepts (the same rule
  as the Pydantic generator), so the non-owning side of a 1:1 association
  has no field whose value the backend would drop
- Display-only form fields: a ``formColumns`` entry with ``"readOnly": true``
  is shown in the Add/Edit dialog but cannot be edited, is not validated as
  required, and is left out of the POST/PUT body (the server owns the value)

MethodButton
^^^^^^^^^^^^

Executes class and instance methods with:

- Parameter input modal for methods with parameters, each labelled by its
  name made readable (``min_pages`` shows as "Min pages"), as in the table dialog
- Automatic type handling (int, float, bool, string, date, time)
- Detailed error display for 500 errors; a method modeled without an
  implementation (the backend answers 501) shows "This action is not
  available yet."
- Modal stays open on error for retry
- A reload of the data shown on the page after execution (see `Data refresh`_)

CrudButton
^^^^^^^^^^

A ``Button`` whose ``actionType`` is ``Create``, ``Update`` or ``Delete``
acts through the table bound to its entity, reusing that table's dialog,
endpoint and selected row:

- **Create** opens the table's Add dialog (POST ``/<entity>/``).
- **Update** opens the Edit dialog on the table's selected row (PUT).
- **Delete** deletes the selected row (DELETE), after a confirmation prompt
  when ``confirmation_required`` is set.

The table is the button's ``instance_source`` or, failing that, the table
bound to the button's ``entity_class`` on the same screen. A create button
with no such table on its screen navigates to the first screen that has one
and opens its dialog there. Update and delete report "Select a row in the
table first" when nothing is selected; the confirmation prompt comes only once
the action can run. A CRUD button with no bound table is
rendered as a plain button.

FormBlock
^^^^^^^^^

A ``Form`` bound to a class (``Form.data_binding``) is rendered as a
``FormBlock`` around its fields. Submitting it POSTs ``/<class>/`` with the
value of every input bound to an attribute (typed from the attribute: int,
float, bool, text; an empty field is left out so the backend applies its
default), shows a success or error message (the backend's validation detail),
and clears the form on success. Fields keep their labels and their ``required``
flag; the submit button shows the form's ``submit_label`` and carries the classes
of the design's own submit button (kept on the form as ``data-submit-class``). An
unbound form is rendered as a plain ``<form>`` that does not submit.

A class with a mandatory to-one association (an order line that must reference
its order) cannot be created from its attributes alone. For each such
association end that no input of the form supplies, the form gets a
``LookupSelect``: a select of the related class's records that posts the chosen
record's key. An input or select named after a to-one association end (its
``name`` or ``data-field-name``) is bound to that end and rendered as the same
select. It reads its options the way the table's Add/Edit dialog does
(``runtime/lookup.ts``).

Navigation
^^^^^^^^^^

A ``Link`` or ``MenuItem`` whose URL is the route of one of the GUI's own screens
is opened through the router, so the app is not reloaded; any other URL, and a
link with ``target="_blank"``, stays a plain anchor. When the GUI comes from the
web editor, a link names its page by a ``data-navigate-to`` attribute or by a
site-relative ``href`` matching the page's route, name or id.

A ``<nav>`` bar's fixed ``height`` is generated as its ``min-height``, except for a
bar that scrolls inside its height (an ``overflow`` other than ``visible``), one
positioned ``fixed`` or ``absolute``, and one of height 0. On a screen
640 px wide or narrower, the bar and its direct children wrap their items onto
more lines (``src/index.css``), so its links stay on the screen and the bar grows
to fit them. The rule is in the ``base`` layer: any rule of the design, such as
a tab strip that scrolls sideways on a phone, takes precedence.

A page's route is the route set on the page in the web editor when it is URL-safe
(letters of any script, digits, ``_``, ``-`` and ``/``). Otherwise it is ``/`` and the page name
in lower case, with every run of characters other than letters (of any script),
digits and ``_`` as one hyphen and no hyphen at either end (``Orders & Returns``
gives ``/orders-returns``). A route another page already has is numbered (``-2``,
``-3``) in the order of the pages' ids, so reordering pages does not swap them.

Earlier versions used the name in lower case with each space as a hyphen and
ignored the route set in the editor. A page's route changes when that old route
held a character other than a letter, digit, ``_`` or ``-`` (``My.Page`` was
``/my.page``, now ``/my-page``), two hyphens in a row or a hyphen at either end
(a name with repeated, leading or trailing spaces), or when the page has a URL-safe
route of its own in the editor. Links written with an old route still open their
page; a bookmark of an old route opens the main page.

Data refresh
^^^^^^^^^^^^

Every successful write made through a runtime component (a table's dialog or
delete, a bound form, a method button) bumps one data version
(``runtime/dataVersion.ts``); tables, metric cards, charts, data lists and maps
reload when it changes. A record saved in one component therefore shows up in
the others on the page without a reload. Custom code that writes to the backend
calls ``notifyDataChanged()`` to get the same effect.

The table context's ``refreshTable`` and ``registerTableRefresh`` are deprecated:
generated tables no longer register a refresh function, ``refreshTable`` calls
``notifyDataChanged()`` (and a function custom code registered for that table).

Chart Components
^^^^^^^^^^^^^^^^

Visualization components including:

- Line charts
- Bar charts
- Pie charts
- Data binding to backend entities
- A series appears in the legend and tooltip under its ``label``; its ``name``
  is the key of its values in the chart data
- Aggregation: a series with an ``aggregation`` groups the records by its
  label field and shows one value per group (``count``, ``sum``, ``avg``,
  ``min``, ``max``, ``median``, ``first``, ``last``); a bound series without a
  value field counts the records per label
- Endpoint: a series (or the chart's own binding) whose binding has an
  ``endpoint`` fetches it instead of the class's CRUD list (``/<class>/``), as
  written: a relative path is resolved against ``VITE_API_URL``, and a filter
  applies to the rows it returns instead of becoming query parameters

Metric cards (``MetricCardBlock``) show their binding's ``aggregation`` over all
records; without one, the sum of the bound field, or the number of records when
no field is bound. No trend is computed: ``show_trend``, ``positive_color`` and
``negative_color`` are carried in the page's metric data but draw nothing, and
``MetricCardComponent`` takes no trend properties. A card whose binding has an ``endpoint`` fetches it instead
of the CRUD list, and applies the binding's filter to the rows it returns. The response contract (an array of rows, or an object holding
one under ``data``, ``results``, ``items``, ``records`` or ``list``) is described
with ``DataBinding`` in :doc:`../buml_language/model_types/gui`. Tables, maps and
forms always use the CRUD list.

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

Both generators take an optional ``app_name``: the application's name as its
users see it, shown in the browser tab and the README headings (the web editor
passes the project's name).

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
   │   │   │   ├── CrudButton.tsx
   │   │   │   ├── FormBlock.tsx
   │   │   │   ├── LookupSelect.tsx
   │   │   │   ├── lookup.ts
   │   │   │   ├── dataVersion.ts
   │   │   │   ├── humanize.ts
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
   ├── index.html
   ├── vite.config.ts
   ├── package.json
   └── tsconfig.json

One ``.tsx`` file is written per page in the GUI model, named after that page
(PascalCase), and ``App.tsx`` wires them into the router. ``MapBlock.tsx`` is
only emitted when the model contains a ``Map`` component, and
``components/AgentComponent.tsx`` with its ``public/img`` background only when
it contains an agent component.

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

A design is written against the browser's defaults, and the Tailwind reset the
app imports (``index.css``) removes them. ``design.css`` therefore starts with a
block in Tailwind's ``base`` layer that restores, inside the pages, the size and
weight of headings and the look of a plain button. The design's own rules and
utility classes take precedence over it.

Trace Links
-----------

After ``generate()``, ``generator.traced_files`` maps each file the generator wrote (path relative to the output
directory) to what it was generated from, for every GUI element that has an ``id``. Each page file
``src/pages/<Page>.tsx`` is generated as a whole from its screen; every component rendered in a page is a line
range (region) of its own, nested in its container's, and ``src/App.tsx`` records the import and the route of
each page as regions of the screen. The other files (the Jinja templates, copied runtime blocks and
assets) are file-level only. The generator writes no per-class files: the classes of the domain model only
appear in the pages through the data bindings of their components, which are not traced separately.

.. note::

   The web editor's GUI converter (``process_gui_diagram``) gives each screen its page id and each component the
   editor's trace records (tables, charts, metric cards, maps, agent components, forms, inputs, buttons, menus) its
   persistent ``attributes.id``, the ids the editor records them by. Text, images, links and layout containers get
   none: they render untraced, inside the region of the traced component that holds them, if any.

Recording traces does not change the generated files; see :doc:`traceable_generator`.
