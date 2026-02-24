Guide: Add a New DSL
========================

This guide covers the full lifecycle for extending B-UML with a new domain-specific branch and plugging it into the web
modeling editor.

For a full cross-repo checklist (editor package + webapp + backend), see
:doc:`diagram_dsl_workflow`.

1. Design the Metamodel
-----------------------

* Model the core concepts, relationships, and constraints first (UML class diagrams or plain sketches help).
* Implement the new elements inside ``besser/BUML`` by extending the existing metamodel packages.
* Keep naming consistent, reuse base classes where possible, and document the semantics in ``docs/source/buml_language``.

2. Implement Persistence and Validation
---------------------------------------

* Update serializers/deserializers so the new objects can be exported/imported alongside the rest of the BUML model.
* Add schema migrations or compatibility layers if the JSON/YAML representation changes.
* Cover business rules with unit tests under ``tests/buml`` (valid models) and ``tests/buml_invalid`` (error cases).

3. Extend Utilities and Converters
----------------------------------

* Update helpers under ``besser/utilities`` (diagram processors, converters, validators) to recognize the new DSL artifacts.
* Keep the converters symmetrical: JSON <-> BUML and BUML <-> JSON must support the same features.
* Document any new CLI entry points or scripts within ``docs/source/utilities``.

4. Integrate with the Web Modeling Editor
-----------------------------------------

The Web Modeling Editor (WME) frontend lives in the separate ``BESSER-WEB-MODELING-EDITOR`` repository and is vendored
here as a git submodule at ``besser/utilities/web_modeling_editor/frontend``. Frontend changes should be committed in
that repository, then the submodule pointer updated in BESSER.

* Decide the scope:

  - **Enable an existing diagram type** (already implemented in the editor package): wire it into the webapp project
    model, sidebar, and import/export flows. See
    ``packages/webapp/src/main/components/project/ADDING_NEW_DIAGRAM_TYPE.md`` in the WME repo.
  - **Add a brand-new diagram/DSL**: extend the editor package first (diagram type, element types, renderers, palette
    previews, translations, property editors), then wire it into the webapp.
* Frontend (WME repo): update the editor package and webapp to expose the new diagram type and UI affordances. Follow
  existing React/TypeScript patterns and add Storybook demos if available.
* Backend (BESSER): expose REST/WS endpoints, validation routes, and persistence logic under
  ``besser/utilities/web_modeling_editor/backend``. Align FastAPI/Flask schemas with the BUML definitions.
* Sync contracts: keep JSON element/relationship types, OpenAPI schemas, and TypeScript types consistent across the two
  repos so import/export and validation remain stable.
* Reference guide: `WME - Adding a New Diagram Type <https://besser.readthedocs.io/projects/besser-web-modeling-editor/en/latest/contributing/new-diagram-guide/index.html>`_.
* Sync the repos: after the WME change merges, update the submodule SHA in this repo and reference the WME commit/PR in
  your BESSER pull request.

5. Update Documentation and Examples
------------------------------------

* Describe the new DSL in ``docs/source/web_editor.rst`` (UI usage) and ``docs/source/api`` (service endpoints).
* Provide at least one runnable sample in ``docs/source/examples`` showing how to model with the new DSL and, if
  relevant, how generators consume it.
* Highlight migration advice or compatibility notes so existing users know how the change affects their projects.

6. Verify end-to-end behavior
-----------------------------

* Run the WME backend locally (``python besser/utilities/web_modeling_editor/backend/backend.py``; defaults to port
  9000).
* Run the WME frontend locally (``npm run start:webapp`` in the WME repo or submodule) and confirm the new palette
  items, properties, and serialization work against the backend.
* Run automated checks: ``python -m pytest`` for BESSER; ``npm run lint`` and ``npm run build:webapp`` for WME.

