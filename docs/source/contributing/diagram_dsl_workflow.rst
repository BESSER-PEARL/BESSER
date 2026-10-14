End-to-End: New Diagram Type + DSL
==================================

This guide explains the full workflow for adding a **new diagram type with a new DSL**
that must work across both repositories:

* **WME repo** (frontend diagramming library + webapp): `BESSER-WEB-MODELING-EDITOR`
* **BESSER repo** (backend + BUML): this repository

Use this when you need **new elements, new rendering, and new backend processing**.
If you only need to expose an existing UML diagram type in the webapp, follow the WME
checklist in ``packages/webapp/src/main/features/project/ADDING_NEW_DIAGRAM_TYPE.md``.

Decision Tree
-------------

1. **Diagram type already exists in the library package** (``packages/library``)?

   * Yes: only do the **webapp wiring** (sidebar, project model, import/export labels).
   * No: you must add **library package support** + **webapp wiring** + **backend processing**.

2. **New DSL / new semantics**?

   * Yes: update the BUML metamodel and backend converters in the BESSER repo.

Repository Boundaries (who owns what)
-------------------------------------

* **WME repo (frontend, diagram engine)**

  - Library package ``packages/library`` — React Flow + Zustand diagram engine
    (node components, edge renderers, palette previews, inspector panels)
  - Webapp ``packages/webapp`` — project model, sidebar, import/export, UI

* **BESSER repo (backend, BUML)**

  - BUML metamodel (new DSL concepts)
  - JSON <-> BUML converters
  - Validation and generation endpoints

Step 1: Extend the BUML Metamodel (BESSER repo)
-----------------------------------------------

1. Add new DSL concepts under ``besser/BUML`` (metamodel classes, relationships, constraints).
2. Update serializers/deserializers if the model format changes.
3. Add or update tests under ``tests/BUML`` and ``tests/BUML_invalid``.
4. Update docs under ``docs/source/buml_language``.

Step 2: Add the Diagram Type in the Library Package (WME repo)
--------------------------------------------------------------

These files live in the WME repo under ``packages/library``:

1. **Register diagram and element types**

   * ``packages/library/lib/types/DiagramType.ts`` (diagram-type enum)
   * ``packages/library/lib/types/nodes/NodeProps.ts`` (per-node data shapes)
   * ``packages/library/lib/edges/EdgeProps.ts`` (per-edge data shapes if you add new relationships)

2. **Create a new diagram folder for nodes and edges**

   * Create ``packages/library/lib/nodes/<yourDiagram>/`` and add the React Flow
     node components (``<NodeName>.tsx``) plus the ``index.ts`` that registers
     them under their v4 type strings.
   * Create ``packages/library/lib/edges/edgeTypes/<YourEdge>.tsx`` for any new
     edge types and add palette previews in
     ``packages/library/lib/components/svgs/nodes/<yourDiagram>/``.

3. **Wire registries**

   * ``packages/library/lib/nodes/<yourDiagram>/index.ts`` (per-diagram node map)
   * ``packages/library/lib/edges/edgeTypes/index.ts`` (edge type map)
   * ``packages/library/lib/components/inspectors/<yourDiagram>/index.ts`` (inspector panels)
   * ``packages/library/lib/components/popovers/PopoverManager.tsx`` (popover routing)
   * ``packages/library/lib/utils/versionConverter.ts`` (v3 → v4 migration cases
     if you need to lift legacy fixtures)

4. **Translations**

   * Inspector / palette labels are inline in the component sources;
     localization currently lives at the webapp layer
     (``packages/webapp/src/main/i18n/``).

Tip: Follow patterns from existing diagram folders
(e.g. ``packages/library/lib/nodes/classDiagram``).

Step 3: Wire the Diagram into the Webapp (WME repo)
---------------------------------------------------

Once the editor package can render the diagram, expose it in the webapp.
Follow the checklist in:

``packages/webapp/src/main/features/project/ADDING_NEW_DIAGRAM_TYPE.md``

That checklist covers:

* Project model types (``types/project.ts``)
* Sidebar entries (``DiagramTypeSidebar.tsx``)
* Export labels, settings badges, import handling
* Default project creation for the new diagram

Step 4: Add Backend Processing (BESSER repo)
--------------------------------------------

The BESSER backend uses a **modular router architecture**. The application factory
(``backend.py``) registers middleware and includes routers from ``backend/routers/``.
Endpoints are organized by concern:

- ``routers/generation_router.py`` — code generation
- ``routers/conversion_router.py`` — BUML import/export
- ``routers/validation_router.py`` — diagram validation
- ``routers/deployment_router.py`` — deployment integration

The BESSER backend must understand the JSON produced by the editor:

1. **JSON -> BUML**

   * Add a processor in
     ``besser/utilities/web_modeling_editor/backend/services/converters/json_to_buml/``
   * Register it in
     ``besser/utilities/web_modeling_editor/backend/services/converters/json_to_buml/__init__.py``
     and re-export it from
     ``besser/utilities/web_modeling_editor/backend/services/converters/__init__.py``.
   * Add or update endpoints in the appropriate router (e.g., ``routers/generation_router.py``
     for generation, ``routers/conversion_router.py`` for export).
   * If the diagram is part of a project payload, update
     ``besser/utilities/web_modeling_editor/backend/services/converters/json_to_buml/project_converter.py``
     to include the new diagram type.

2. **BUML -> JSON** (if import/export back to the editor is required)

   * Add a converter in
     ``besser/utilities/web_modeling_editor/backend/services/converters/buml_to_json/``
   * Update
     ``besser/utilities/web_modeling_editor/backend/services/converters/buml_to_json/project_converter.py``
     to include the new diagram in project JSON outputs.

3. **Validation**

   * Update validation logic and OCL checks under
     ``besser/utilities/web_modeling_editor/backend/services/validators`` as needed.

4. **Error handling**

   * Use the ``@handle_endpoint_errors("endpoint_name")`` decorator from
     ``routers/error_handler.py`` on new endpoints. It maps custom exceptions
     (``ConversionError``, ``ValidationError`` → 400; ``GenerationError`` → 500) to
     consistent HTTP responses.

Step 5: Sync Contracts Across Repos
-----------------------------------

Keep the following consistent between WME and BESSER:

* Diagram type strings (`UMLDiagramType`)
* Element and relationship type names
* JSON schema structure (element fields, relationship fields)
* Property panel fields vs backend expectations

Step 6: End-to-End Verification
-------------------------------

1. Start the BESSER backend:

   .. code-block:: bash

      python besser/utilities/web_modeling_editor/backend/backend.py

2. Start the WME webapp:

   .. code-block:: bash

      npm run dev

3. Create a new project using your diagram type.
4. Add elements and relationships, edit properties, and save.
5. Trigger generation or validation and confirm the backend accepts the JSON.

Step 7: Sync Repos and Open PRs
-------------------------------

1. Commit WME changes in the WME repo.
2. Update the submodule SHA in the BESSER repo:

   .. code-block:: bash

      cd besser/utilities/web_modeling_editor/frontend
      git fetch
      git checkout <wme-commit-or-branch>
      cd ../../../..
      git add besser/utilities/web_modeling_editor/frontend

3. Commit BESSER changes and link the two PRs.

Common Pitfalls
---------------

* The webapp sidebar shows the diagram but the editor package does not know it.
* JSON type strings in the backend do not match the editor package.
* Backend is not running on ``http://localhost:9000/besser_api`` when you test.
* Existing projects do not include the new diagram slot (create a new project).
