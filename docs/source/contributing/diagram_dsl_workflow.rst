End-to-End: New Diagram Type + DSL
==================================

This guide explains the full workflow for adding a **new diagram type with a new DSL**
that must work across both repositories:

* **WME repo** (frontend + editor package): `BESSER-WEB-MODELING-EDITOR`
* **BESSER repo** (backend + BUML): this repository

Use this when you need **new elements, new rendering, and new backend processing**.
If you only need to expose an existing UML diagram type in the webapp, follow the WME
checklist in ``packages/webapp/src/main/components/project/ADDING_NEW_DIAGRAM_TYPE.md``.

Decision Tree
-------------

1. **Diagram type already exists in the editor package** (``packages/editor``)?

   * Yes: only do the **webapp wiring** (sidebar, project model, import/export labels).
   * No: you must add **editor package support** + **webapp wiring** + **backend processing**.

2. **New DSL / new semantics**?

   * Yes: update the BUML metamodel and backend converters in the BESSER repo.

Repository Boundaries (who owns what)
-------------------------------------

* **WME repo (frontend, diagram engine)**

  - Editor package (diagram types, elements, rendering, palette, property panels)
  - Webapp (project model, sidebar, import/export, UI)

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

Step 2: Add the Diagram Type in the Editor Package (WME repo)
-------------------------------------------------------------

These files live in the WME repo under ``packages/editor``:

1. **Register diagram and element types**

   * ``packages/editor/src/main/packages/diagram-type.ts``
   * ``packages/editor/src/main/uml-element-type.ts``
   * ``packages/editor/src/main/uml-relationship-type.ts`` (if you add new relationships)

2. **Create a new diagram package folder**

   * Create ``packages/editor/src/main/packages/<your-diagram>/``
   * Add element classes (model), React components (rendering), and palette previews.

3. **Wire registries**

   * ``packages/editor/src/main/packages/components.ts`` (component map)
   * ``packages/editor/src/main/packages/uml-elements.ts`` (element map)
   * ``packages/editor/src/main/packages/uml-relationships.ts`` (relationship map)
   * ``packages/editor/src/main/packages/compose-preview.ts`` (palette previews)
   * ``packages/editor/src/main/packages/popups.ts`` (property panels)
   * ``packages/editor/src/main/components/create-pane/create-pane.tsx`` (diagram type selection)

4. **Translations**

   * ``packages/editor/src/main/i18n/en.json`` (and other locales if needed)

Tip: Follow patterns from existing diagram packages (e.g. ``uml-class-diagram``).

Step 3: Wire the Diagram into the Webapp (WME repo)
---------------------------------------------------

Once the editor package can render the diagram, expose it in the webapp.
Follow the checklist in:

``packages/webapp/src/main/components/project/ADDING_NEW_DIAGRAM_TYPE.md``

That checklist covers:

* Project model types (``types/project.ts``)
* Sidebar entries (``DiagramTypeSidebar.tsx``)
* Export labels, settings badges, import handling
* Default project creation for the new diagram

Step 4: Add Backend Processing (BESSER repo)
--------------------------------------------

The BESSER backend must understand the JSON produced by the editor:

1. **JSON -> BUML**

   * Add a processor in
     ``besser/utilities/web_modeling_editor/backend/services/converters/json_to_buml/``
   * Register it in
     ``besser/utilities/web_modeling_editor/backend/services/converters/json_to_buml/__init__.py``
     and re-export it from
     ``besser/utilities/web_modeling_editor/backend/services/converters/__init__.py``.
   * Call it from the relevant ``backend.py`` endpoints (generation, export, validation).
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

      npm run start:webapp

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
