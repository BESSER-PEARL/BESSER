Guide: Create a New Generator
=============================

Use this playbook when you want to add support for a new technology stack (e.g., FastAPI, Spring Boot, Vue) in BESSER.

1. Define the Scope
-------------------

* Identify the target runtime, deployment model, and tooling (framework versions, package managers, Docker images).
* Document the mapping rules between B-UML constructs and generated artifacts inside ``docs/source/generators`` so the
  community can review the expected behavior.

2. Bootstrap the Module
-----------------------

* Copy the minimal scaffold from an existing generator (folder structure, setup module, test layout) and rename all
  packages, entry points, and CLI hooks.
* Keep reusable helpers inside ``besser/utilities``. Only generator-specific code should live under your new folder in
  ``besser/generators``.

3. Implement Transformations
----------------------------

* Build a clear pipeline: BUML model -> intermediate representation -> rendered files.
* Respect the generator service interface so the component can be triggered from both the CLI and the web modeling
  editor backend.
* Store templates separately (Jinja2, f-strings, etc.) and favor small, composable functions to simplify testing.

4. Add Regression Tests
-----------------------

* Place unit and integration tests under ``tests/generators/<your-generator>``.
* Use small BUML fixture models to test mappings in isolation and full-generation scenarios.
* Validate both structure (e.g., class names, endpoints) and content (e.g., business logic snippets, configuration files).

5. Document and Demo
--------------------

* Extend ``docs/source/examples.rst`` (or add a new section) with a walkthrough that starts from a BUML model and ends
  with the generated artifact running locally.
* List prerequisites (installed toolchains, environment variables, Docker images) and troubleshooting hints.
* Attach screenshots, logs, or GIFs if they make reviewer validation easier.
