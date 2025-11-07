Where to Contribute
===================

The repository is organized around four main pillars. Use this guide to understand the scope of each pillar and to
follow best practices when you open a pull request.

B-UML Language Core (``besser/BUML``)
-------------------------------------

* Own the metamodel: new concepts and relationships start in this folder. Keep naming consistent with the rest of B-UML.
* Update serialization: whenever you add or modify model elements, extend the serializers/deserializers so BUML models
  round-trip cleanly.
* Protect behavior with tests: place unit tests under ``tests/buml`` and cover both valid and invalid scenarios.
* Document everything: reflect the changes in ``docs/source/buml_language.rst`` and any other affected guides to help
  users reason about the new constructs.

Platform Utilities (``besser/utilities``)
-----------------------------------------

* Reuse before reinventing: common converters, validation helpers, and processors should live here so other parts of the
  platform can import them.
* Keep scripts idempotent: utilities are often used in CI and automation, so make sure rerunning them is safe.
* Explain usage: every new tool needs a short README section plus an entry in ``docs/source/utilities.rst`` outlining
  invocation examples and configuration flags.

Code Generators (``besser/generators``)
---------------------------------------

* Mirror the existing layout: split templates, transformers, and configuration objects into separate modules so the
  generator stays maintainable.
* Stay deterministic: generators should produce the same result for the same model; avoid timestamp-based file names
  unless explicitly required.
* Provide samples: include fixtures under ``examples`` or ``tests/generators`` that demonstrate the expected output.
* Wire documentation: annotate ``docs/source/generators`` with usage instructions, prerequisites, and troubleshooting
  tips.

Web Modeling Editor Backend (``besser/utilities/web_modeling_editor/backend``)
------------------------------------------------------------------------------

* Synchronize with the frontend: for every API change, communicate with the UI maintainers and update TypeScript types
  or schemas as needed.
* Cover converters: backend services contain JSON <-> B-UML converters for diagrams; extend both directions and add tests
  whenever you introduce a new concept.
* Respect validation contracts: make sure FastAPI/Flask models, Pydantic schemas, and business rules stay aligned with
  the BUML definitions.
* Document endpoints: update ``docs/source/web_editor.rst`` and ``docs/source/api`` whenever you add, rename, or delete
  API endpoints so integrators can keep up.

Contribution Checklist
----------------------

Regardless of the area:

1. Discuss the idea first (issue tracker or community chat) to avoid duplicated effort.
2. Write or update automated tests that demonstrate the new behavior.
3. Run the relevant commands locally (``pytest``, ``npm test``, ``make docs``, etc.) and ensure they pass.
4. Describe migrations, breaking changes, and manual steps clearly in your pull request description.
