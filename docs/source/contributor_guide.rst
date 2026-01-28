Contributor Guide
=================

This guide complements :doc:`contributing` and the root-level
``CONTRIBUTING.md`` file. It provides a deeper walkthrough of the repository
layout, development workflows, and quality expectations so that new
contributors can become productive quickly.

.. contents:: On this page
   :local:
   :depth: 2

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

``setup_environment.py``
    A helper script that creates a virtual environment, installs dependencies,
    and sets the ``PYTHONPATH`` so the ``besser`` package resolves correctly in
    editors.

``docker-compose.yml`` and ``Dockerfile``
    Container definitions that reproduce the full stack locally, useful for
    verifying generator output or end-to-end flows.

Frontend assets for the BESSER Web Modeling Editor live in the separate
`BESSER_WME_standalone <https://github.com/BESSER-PEARL/BESSER_WME_standalone>`_
repository (vendored here as the ``besser/utilities/web_modeling_editor``
submodule). Contribute UI updates there; this repository focuses on the backend
APIs, metamodel, generators, and utilities consumed by that frontend.

Getting Started
---------------

1. Fork the repository and clone your fork locally.
2. Create and activate a Python 3.10+ virtual environment.
3. Run ``python setup_environment.py`` to install dependencies and configure
   ``PYTHONPATH``.
4. Install the documentation extras with ``pip install -r docs/requirements.txt``
   if you plan to edit the docs.
5. Run an example, such as ``python tests/BUML/metamodel/structural/library/library.py``,
   to validate your setup.

Development Workflow
--------------------

* Create a feature branch from ``main`` for each logical change.
* Keep commits focused and descriptive. Favor incremental commits over monolithic
  ones so reviewers can follow the reasoning.
* Update or add tests alongside code changes.
* Update the documentation whenever you introduce new behaviors, options, or
  workflows.
* Run automated checks locally before pushing (tests, linting, docs build).

Testing and Quality Checks
--------------------------

* Run the entire test suite with ``python -m pytest`` from the repository root.
* Use ``python -m pytest <path>`` to target specific directories or modules
  while iterating.
* Prefer fixtures and example models located under ``tests/`` when extending
  coverage instead of duplicating assets.
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

* Ensure your branch is rebased on the latest ``main`` before opening a pull
  request.
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
