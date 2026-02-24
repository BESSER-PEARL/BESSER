Utilities
=========

The ``besser.utilities`` package hosts cross-cutting helpers used by the SDK, generators,
and the web modeling editor backend. The most commonly used utilities are listed below.

.. toctree::

   utilities/serializer
   utilities/buml_code_builder

General Utilities
-----------------

* ``besser.utilities.utils`` provides shared helpers such as
  ``sort_by_timestamp`` and the ``ModelSerializer`` class.
* API reference: :doc:`Model serializer API <api/utilities/api_model_serializer>`.

Model Building Helpers
----------------------

These utilities help build BUML models from external sources:

* :doc:`Image to BUML <buml_language/model_building/image_to_buml>`
* :doc:`Knowledge Graph to BUML <buml_language/model_building/kg_to_buml>`

For the low-level API documentation, see:

* :doc:`Image-to-BUML API <api/BUML/notations/image2buml/image>`
* :doc:`KG-to-BUML API <api/BUML/notations/kg2buml/kg>`

Web Modeling Editor Backend Helpers
-----------------------------------

Backend utilities for the web modeling editor (JSON converters, validators, and
generator adapters) live under ``besser/utilities/web_modeling_editor/backend``.
Refer to :doc:`web_editor` and the contributor workflows in :doc:`contributing/index`
and :doc:`contributing/diagram_dsl_workflow` when extending the editor.

Debug Helpers
-------------

To generate a web app directly from a Web Modeling Editor JSON export, use the
debug CLI in the backend utilities:

.. code-block:: bash

   python -m besser.utilities.web_modeling_editor.backend.tools.generate_web_app_from_json path/to/export.json

The command expects both a ``ClassDiagram`` and ``GUINoCodeDiagram`` in the
exported project. Use ``--no-agent`` if you want to skip processing an
``AgentDiagram`` when present.
