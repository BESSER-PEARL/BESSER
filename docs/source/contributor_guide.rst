Contributor Guide
=================

This guide complements :doc:`contributing/index` and the root-level
``CONTRIBUTING.md`` file. It provides a deeper walkthrough of the repository
layout, development workflows, and quality expectations so that new
contributors can become productive quickly.

Project Overview
----------------

BESSER builds upon the B-UML metamodel to generate deployable applications.
Most contributions fall into one of these areas:

* **B-UML metamodel** (:mod:`besser.BUML.metamodel`): defines the core modeling
  constructs that generators consume.
* **Generators** (:mod:`besser.generators`): translate models into code for
  specific stacks (e.g., Django, SQLAlchemy, Flutter).
* **Utilities** (:mod:`besser.utilities`): cross-cutting helpers shared across
  the metamodel and generators. This package also exposes services that power
  the backend of the web modeling editor (REST APIs, validation helpers,
  deployment glue).
* **Examples and tests** (:mod:`tests`): executable specifications that document
  expected behavior and protect against regressions.
* **Documentation** (:mod:`docs`): user- and contributor-focused guidance built
  with Sphinx and published to Read the Docs.

Repository Layout
-----------------

``besser/``
    Python packages that expose the BESSER APIs. Subpackages mirror major
    feature areas and collectively implement the backend of the web modeling
    editor:

    * ``BUML/`` – Metamodel definitions, notations, and tooling.
    * ``generators/`` – Code generators implementing
      :class:`~besser.generators.generator_interface.GeneratorInterface`.
    * ``utilities/`` – Reusable helpers, configuration objects, backend
      for the web modeling editor (e.g., logging, template handling, REST endpoints).

``docs/``
    Source files for the documentation site. ReStructuredText lives in
    ``docs/source/`` and static assets in ``docs/source/_static``.

``tests/``
    Pytest suites covering the metamodel, generators, utilities, and sample
    applications. Tests often double as executable examples.


``docker-compose.yml`` and ``Dockerfile``
    Container definitions that reproduce the full stack locally, useful for
    verifying generator output or end-to-end flows.

Frontend assets for the BESSER Web Modeling Editor live in the separate
`BESSER-WEB-MODELING-EDITOR <https://github.com/BESSER-PEARL/BESSER-WEB-MODELING-EDITOR>`_
repository (vendored here as the ``besser/utilities/web_modeling_editor/frontend``
submodule). Contribute UI updates there; this repository focuses on the backend
APIs, metamodel, generators, and utilities consumed by that frontend.

Getting Started
---------------

1. Fork the repository and clone your fork locally.
2. Create and activate a Python 3.10+ virtual environment.
3. Install dependencies:

   .. code-block:: bash

      python -m venv venv
      venv\Scripts\activate          # Windows
      source venv/bin/activate       # Linux / macOS
      pip install -r requirements.txt
4. Install the documentation extras with ``pip install -r docs/requirements.txt``
   if you plan to edit the docs.
5. Run an example, such as ``python tests/BUML/metamodel/structural/library/library.py``,
   to validate your setup.

Web Modeling Editor Frontend Setup
----------------------------------

If you want to run or extend the Web Modeling Editor (WME) frontend, use the
WME repository directly or the git submodule inside this repo.
Prerequisites: Node.js 20+ and npm.

**Using the submodule:**

1. Initialize the submodule: ``git submodule update --init --recursive``.
2. Install dependencies: ``cd besser/utilities/web_modeling_editor/frontend && npm install``.
3. Start the backend API (from the repo root): ``python besser/utilities/web_modeling_editor/backend/backend.py``.
4. Start the webapp dev server: ``npm run dev`` (still in the frontend folder).

The Vite dev server runs on http://localhost:8080 and expects the backend at
http://localhost:9000/besser_api in development mode.

**Using the standalone WME repo:**

1. Clone `BESSER-WEB-MODELING-EDITOR <https://github.com/BESSER-PEARL/BESSER-WEB-MODELING-EDITOR>`_.
2. Run ``npm install`` and ``npm run dev``.
3. Start the BESSER backend as above if you need live API integration.

Development Workflow
--------------------

* Create a topic branch from ``development`` (the integration branch) for each
  logical change. **Do not branch from or target** ``master`` **directly** —
  ``master`` tracks released versions, while ``development`` is where features
  are integrated before a release.
* Name branches with a prefix that matches the type of change, mirroring our
  Conventional Commits style:

  * ``feature/<short-description>`` — new features (e.g.,
    ``feature/add-django-generator``)
  * ``fix/<short-description>`` — bug fixes (e.g.,
    ``fix/ocl-parser-null-handling``)
  * ``docs/<short-description>`` — documentation-only changes (e.g.,
    ``docs/update-contributor-guide``)
  * ``refactor/<short-description>`` — restructuring without behavior changes
  * ``test/<short-description>`` — test additions or improvements
  * ``chore/<short-description>`` — tooling, CI, or dependency maintenance

  Use lowercase, hyphen-separated descriptive names. Avoid personal prefixes
  (``armen/...``) and avoid umbrella branches that bundle unrelated changes.
* Keep commits focused and descriptive. Use Conventional Commit prefixes
  (``feat:``, ``fix:``, ``docs:``, ``refactor:``, ``test:``, ``chore:``) in
  short, imperative subjects. Favor incremental commits over monolithic ones so
  reviewers can follow the reasoning.
* Update or add tests alongside code changes.
* Update the documentation whenever you introduce new behaviors, options, or
  workflows.
* Run automated checks locally before pushing (tests, linting, docs build).

Coordinating Changes Across BESSER and WME
------------------------------------------

If a change impacts both the BESSER backend and the WME frontend (for example,
adding a new DSL with graphical notation):

1. Implement and commit the WME changes in the WME repository.
2. Implement and commit the BESSER changes in this repository.
3. Update the git submodule pointer in this repo to the new WME commit.
4. Link the two pull requests so reviewers can merge them in the correct order.

For the WME-side steps, see the WME contributor guide:
`Adding a New Diagram Type <https://besser.readthedocs.io/projects/besser-web-modeling-editor/en/latest/contributing/new-diagram-guide/index.html>`_.
For the full cross-repo checklist, see :doc:`contributing/diagram_dsl_workflow`.

Testing and Quality Checks
--------------------------

* Run the entire test suite with ``python -m pytest`` from the repository root.
* Use ``python -m pytest <path>`` to target specific directories or modules
  while iterating.
* Prefer reusing shared fixtures from ``tests/conftest.py`` (e.g.,
  ``library_book_author_model``, ``employee_self_assoc_model``) instead of
  duplicating test models. Additional domain-specific fixtures live in
  ``tests/generators/conftest.py``.
* Keep assertions focused on observable behavior—avoid over-specifying
  implementation details.

Documentation Workflow
----------------------

* Documentation is authored in reStructuredText. Follow the style demonstrated
  in existing files under ``docs/source/``.
* Use ``make html`` inside the ``docs`` directory to build the site locally and
  validate formatting before opening a pull request.
* Place reusable images in ``docs/source/_static`` and screenshots or diagrams in
  ``docs/source/img``. Reference them with relative paths inside the docs.
* Cross-link related pages using ``:doc:``, ``:mod:``, or ``:ref:`` roles to help
  readers navigate.

Submitting Your Change
----------------------

* **Open pull requests against** ``development``, **not** ``master``.
  Maintainers merge ``development`` into ``master`` as part of the release
  process; contributors should not target ``master`` directly.
* Ensure your branch is rebased on the latest ``development`` before opening a
  pull request.
* Fill in the pull request template, describing the change, tests executed, and
  any additional context reviewers should know.
* Respond promptly to review feedback. Discussions in pull requests are a
  collaborative space to reach the best solution.
* Once a pull request is approved and checks pass, a maintainer will merge it
  according to the governance process.

Need Help?
----------

If you are unsure where to start, open a discussion in the issue tracker or
reach out via info@besser-pearl.org. We are happy to mentor new contributors and
pair on first issues.
