Guide: Add a New Sub-DSL
========================

This guide covers the full lifecycle for extending B-UML with a new domain-specific branch and plugging it into the web
modeling editor.

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

* Update helpers under ``besser/utilities`` (diagram processors, converters, validators) to recognize the new
  sub-DSL artifacts.
* Keep the converters symmetrical: JSON <-> BUML and BUML <-> JSON must support the same features.
* Document any new CLI entry points or scripts within ``docs/source/utilities``.

4. Integrate with the Web Modeling Editor
-----------------------------------------

* Frontend: add the visual palette elements, properties panels, and canvas behaviors in
  ``besser/utilities/web_modeling_editor/frontend``. Follow the existing component patterns (React/TypeScript) and add
  Storybook demos if available.
* Backend: expose REST/WS endpoints, validation routes, and persistence logic under
  ``besser/utilities/web_modeling_editor/backend``. Align FastAPI/Flask schemas with the BUML definitions.
* Sync contracts: update shared type definitions so frontend and backend stay compatible (e.g., OpenAPI schemas, TS types).

5. Update Documentation and Examples
------------------------------------

* Describe the new DSL in ``docs/source/web_editor.rst`` (UI usage) and ``docs/source/api`` (service endpoints).
* Provide at least one runnable sample in ``docs/source/examples`` showing how to model with the new DSL and, if
  relevant, how generators consume it.
* Highlight migration advice or compatibility notes so existing users know how the change affects their projects.

