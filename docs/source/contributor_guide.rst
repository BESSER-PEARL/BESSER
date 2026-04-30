Contributor Guide
=================

This guide complements :doc:`contributing/index` and the root-level
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

Ways to Contribute
------------------

You don't need to be a Python expert to help. Useful contributions include:

* **Code** — new generators, metamodel improvements, bug fixes, performance
  work, refactors.
* **Documentation** — clarifying existing pages, fixing typos, writing
  tutorials, adding examples to ``docs/source/``.
* **Tests** — improving coverage, adding regression tests, hardening the
  metamodel and converters.
* **Issue triage** — reproducing reported bugs, narrowing down causes,
  suggesting workarounds.
* **Feedback** — sharing how you use BESSER, what is missing, what is
  confusing. Open a discussion or issue.
* **Examples** — contributing new sample models or use cases to
  `BESSER-examples <https://github.com/BESSER-PEARL/BESSER-examples>`_.

If you are looking for a starting point, browse issues labeled
`good first issue <https://github.com/BESSER-PEARL/BESSER/labels/good%20first%20issue>`_
or `help wanted <https://github.com/BESSER-PEARL/BESSER/labels/help%20wanted>`_.

Repository Layout
-----------------

``besser/``
    Python packages that expose the BESSER APIs. Subpackages mirror major
    feature areas and collectively implement the backend of the web modeling
    editor:

    * ``BUML/metamodel/`` – Metamodel definitions (structural, state machine,
      GUI, agents, quantum, etc.).
    * ``BUML/notations/`` – Concrete-syntax parsers (PlantUML, OCL,
      mockup-to-BUML, …).
    * ``generators/`` – Code generators implementing
      :class:`~besser.generators.generator_interface.GeneratorInterface`.
    * ``utilities/`` – Reusable helpers, configuration objects, backend for
      the web modeling editor (REST endpoints, validation, deployment glue).
    * ``utilities/web_modeling_editor/frontend/`` – Frontend git submodule
      (separate repository).

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
repository (vendored here as the
``besser/utilities/web_modeling_editor/frontend`` submodule). Contribute UI
updates there; this repository focuses on the backend APIs, metamodel,
generators, and utilities consumed by that frontend.

Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~

* **Python 3.10, 3.11, or 3.12** (CI tests against all three).
* **Git**, including submodule support.
* **Node.js 20+** *(only if you plan to run the web modeling editor frontend
  locally)*.

Fork, clone, and initialize
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Fork the repository to your own GitHub account.
2. Clone your fork (with submodules so the frontend is available):

   .. code-block:: bash

      git clone --recurse-submodules https://github.com/<your-username>/BESSER.git
      cd BESSER

   If you already cloned without ``--recurse-submodules``:

   .. code-block:: bash

      git submodule update --init --recursive

3. Add the upstream remote so you can keep your fork in sync:

   .. code-block:: bash

      git remote add upstream https://github.com/BESSER-PEARL/BESSER.git

Create a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m venv venv

   # Activate it
   source venv/bin/activate           # Linux / macOS
   venv\Scripts\activate              # Windows (cmd / PowerShell)

Install dependencies
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -r requirements.txt
   pip install -e .                   # editable install of the BESSER package

Optional but recommended:

.. code-block:: bash

   pip install -r docs/requirements.txt   # Sphinx + theme for building the docs

Some test modules require extra packages that are **not** in
``requirements.txt``: ``torch``, ``tensorflow`` (for ``tests/generators/nn/``),
and ``openpyxl`` (for the spreadsheet importer test). CI installs them;
locally, install them only if you intend to touch those areas.

Verify the install
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python tests/BUML/metamodel/structural/library/library.py
   python -m pytest -k library

If both run cleanly, you are ready to go.

Web Modeling Editor Frontend Setup
----------------------------------

If you want to run or extend the Web Modeling Editor (WME) frontend, use the
WME repository directly or the git submodule inside this repo. Prerequisites:
Node.js 20+ and npm.

**Using the submodule:**

1. Initialize the submodule: ``git submodule update --init --recursive``.
2. Install dependencies:
   ``cd besser/utilities/web_modeling_editor/frontend && npm install``.
3. Start the backend API (from the repo root):
   ``python besser/utilities/web_modeling_editor/backend/backend.py``.
4. Start the webapp dev server: ``npm run dev`` (still in the frontend
   folder).

The Vite dev server runs on http://localhost:8080 and expects the backend at
http://localhost:9000/besser_api in development mode.

**Using the standalone WME repo:**

1. Clone `BESSER-WEB-MODELING-EDITOR <https://github.com/BESSER-PEARL/BESSER-WEB-MODELING-EDITOR>`_.
2. Run ``npm install`` and ``npm run dev``.
3. Start the BESSER backend as above if you need live API integration.

Development Guidelines
----------------------

Code style
~~~~~~~~~~

* Follow **PEP 8** with 4-space indentation and a **120-character** line
  limit (configured in ``pyproject.toml``).
* Use ``snake_case`` for functions/variables, ``PascalCase`` for classes,
  ``UPPER_CASE`` for constants.
* Add type hints for public APIs and write descriptive docstrings.
* Order imports: standard library → third-party → local.
* Additional formatting, naming, and AI-assistant rules are in
  ``.github/copilot-instructions.md`` (also available as ``.cursorrules`` for
  Cursor users).

Linting
~~~~~~~

We use **Ruff** for linting (run automatically in CI):

.. code-block:: bash

   pip install ruff
   ruff check .

Fix lint warnings before opening a PR. CI will fail on lint errors.

Working in core packages
~~~~~~~~~~~~~~~~~~~~~~~~

* **Generators** (``besser/generators/``) — inherit from
  :class:`~besser.generators.generator_interface.GeneratorInterface`; place
  Jinja2 templates in ``generators/<name>/templates/``; register the
  generator in
  ``besser/utilities/web_modeling_editor/backend/config/generators.py`` if it
  should be exposed via the web editor.
* **Metamodel** (``besser/BUML/metamodel/``) — keep changes
  backward-compatible where possible; mirror existing patterns (private
  fields with validating setters, ``Element → NamedElement → …`` hierarchy).
* **Notations** (``besser/BUML/notations/``) — use ANTLR grammars and the
  listener pattern, mirroring the existing PlantUML and OCL parsers.
* **Backend** (``besser/utilities/web_modeling_editor/backend/``) — the
  FastAPI app uses a modular router architecture (routers, middleware,
  services, models). Prefer extending an existing module over creating
  bespoke helpers.
* **Bidirectional converters** — if you support a feature in
  ``json_to_buml/``, add the symmetric path in ``buml_to_json/``. Round-trips
  must be lossless.

Validation strategy
~~~~~~~~~~~~~~~~~~~

Three validation layers run together; respect each in the right place:

1. **Construction** — setter checks in metamodel classes (fail fast).
2. **Metamodel** — ``.validate()`` returns
   ``{success, errors, warnings}``.
3. **Constraints** — OCL constraints evaluated against the model.

Collect validation errors instead of throwing on the first one, so users see
the full picture.

Testing and Quality Checks
--------------------------

Run the full suite before pushing:

.. code-block:: bash

   python -m pytest

While iterating, target the area you are changing:

.. code-block:: bash

   python -m pytest tests/generators -k sqlalchemy
   python -m pytest tests/BUML/metamodel/structural -k library

If you do **not** have ``torch``, ``tensorflow``, or ``openpyxl`` installed
locally, skip the modules that import them so collection does not fail:

.. code-block:: bash

   python -m pytest tests/ \
     --ignore=tests/generators/nn \
     --ignore=tests/utilities/web_modeling_editor/backend/test_spreadsheet_import.py

Guidelines:

* **Add or update tests** alongside any behavior change.
* **Reuse fixtures** from ``tests/conftest.py`` (e.g.,
  ``library_book_author_model``, ``employee_self_assoc_model``,
  ``player_team_domain_model``) and ``tests/generators/conftest.py`` instead
  of duplicating models.
* **Test round-trips** for converters — ``JSON → BUML → JSON`` must be the
  identity.
* **Validate behavior**, not implementation details. Assertions on observable
  outputs age better than assertions on internal structure.

Documentation Workflow
----------------------

Documentation lives in ``docs/source/`` and is written in reStructuredText.
Keep it in sync with code: a new generator, endpoint, metamodel concept, or
workflow should be reflected in the docs.

Build locally to check formatting:

.. code-block:: bash

   cd docs
   make html                # Linux / macOS
   make.bat html            # Windows

Open ``docs/build/html/index.html`` to preview.

Common pages to update:

* ``buml_language.rst`` — metamodel additions
* ``generators.rst`` — new generators
* ``web_editor.rst`` — backend API changes
* ``utilities.rst`` — new utilities
* ``contributor_guide.rst`` / ``ai_assistant_guide.rst`` — workflow changes

Place images in ``docs/source/img/`` or ``docs/source/_static/``. Cross-link
related pages using ``:doc:``, ``:mod:``, or ``:ref:`` roles to help readers
navigate.

Branching and Commits
---------------------

Branch naming
~~~~~~~~~~~~~

Create a topic branch off **``development``** (the integration branch). Use
a prefix that matches the type of change:

.. list-table::
   :header-rows: 1
   :widths: 15 50 35

   * - Prefix
     - Use for
     - Example
   * - ``feature/``
     - New features or capabilities
     - ``feature/add-django-generator``
   * - ``fix/``
     - Bug fixes
     - ``fix/ocl-parser-null-handling``
   * - ``docs/``
     - Documentation-only changes
     - ``docs/update-contributor-guide``
   * - ``refactor/``
     - Code restructuring without behavior changes
     - ``refactor/split-backend-routers``
   * - ``test/``
     - Test additions or improvements
     - ``test/state-machine-validation``
   * - ``chore/``
     - Tooling, CI, dependency bumps, repo maintenance
     - ``chore/bump-fastapi``

Use **lowercase, hyphen-separated, descriptive** names. Avoid personal
prefixes (``armen/...``) and avoid umbrella branches that bundle unrelated
changes.

Commit conventions
~~~~~~~~~~~~~~~~~~

We follow `Conventional Commits <https://www.conventionalcommits.org/>`_.
Use one of these prefixes in your commit subject:

* ``feat:`` — a new feature
* ``fix:`` — a bug fix
* ``docs:`` — documentation-only changes
* ``refactor:`` — code change that neither fixes a bug nor adds a feature
* ``test:`` — adding or fixing tests
* ``chore:`` — tooling, CI, dependency, or repo-maintenance changes

Keep subjects short, imperative, and lowercase, e.g.::

   feat: add quantum circuit generator
   fix: handle null cardinality in OCL parser
   docs: clarify branch naming in CONTRIBUTING

Group related work into logically scoped commits — reviewers should be able
to follow the reasoning commit by commit.

Creating Pull Requests
----------------------

Target branch
~~~~~~~~~~~~~

.. important::

   **Open all pull requests against** ``development``\ **, not** ``master``.

   The ``master`` branch tracks released versions. The ``development`` branch
   is where features are integrated and validated before release.
   Maintainers periodically merge ``development`` into ``master`` as part of
   the release process — contributors should not target ``master`` directly.

When opening the PR on GitHub, the **base branch** dropdown defaults to
``master``. **Change it to** ``development`` before clicking *Create pull
request*::

   base repository: BESSER-PEARL/BESSER     base: development   ←  change this!
   head repository: <your-fork>/BESSER      compare: feature/your-change

If you forget, no harm done — just edit the base branch on the existing PR
(GitHub allows changing it after creation). Maintainers will also flag it.

Step-by-step
~~~~~~~~~~~~

1. **Sync your fork and branch off** ``development``:

   .. code-block:: bash

      git checkout development
      git pull upstream development
      git checkout -b feature/your-change

2. **Commit your work** using the Conventional Commits prefixes. Make focused
   commits — reviewers should be able to follow your reasoning commit by
   commit.

3. **Run quality checks locally** before pushing:

   .. code-block:: bash

      python -m pytest                # tests
      ruff check .                    # lint
      cd docs && make html && cd ..   # if you touched the docs

4. **Push your branch:**

   .. code-block:: bash

      git push -u origin feature/your-change

5. **Open the PR on GitHub** with ``development`` as the base branch (see
   above). If the work is in progress and you want early feedback, open it
   as a **draft PR** — maintainers will review when you mark it ready.

6. **Rebase or merge** ``development`` into your branch before requesting
   final review, so the PR reflects an up-to-date base:

   .. code-block:: bash

      git fetch upstream
      git rebase upstream/development
      git push --force-with-lease

7. **Respond to review feedback.** Review is a collaborative conversation,
   not a gate. Push fixup commits during review and let the maintainer
   squash on merge if appropriate.

Writing a good PR description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A great PR description answers four questions: **What changed? Why? How was
it tested? What should reviewers pay attention to?** Use this template (the
PR template in this repo follows the same structure):

.. code-block:: markdown

   ## What

   A 1–3 sentence summary of the change. Stay concrete — name the files,
   generators, endpoints, or metamodel concepts you touched.

   ## Why

   The motivation. Link the issue (`Closes #123`), or describe the bug,
   limitation, or use case driving the change. Reviewers should not have
   to guess the intent.

   ## How

   A brief explanation of the approach, *especially* for non-obvious design
   choices. Mention alternatives you considered and why you discarded them.
   Skip this section for trivial fixes.

   ## Testing

   - Unit tests added/updated: `tests/path/to/test_thing.py`
   - Manual checks: e.g., "ran `python tests/.../library.py` and inspected
     output", or "tested in the web editor: created a class diagram with
     inheritance, exported it, re-imported, round-trip preserved"
   - Edge cases considered: …

   ## Screenshots / Recordings

   (For UI or output changes. Drag-and-drop images or GIFs into the PR.)

   ## Follow-ups / Known limitations

   Anything intentionally out of scope for this PR, or known gaps to address
   later. Linking a follow-up issue is even better.

   ## Checklist

   - [ ] Targeted `development` (not `master`)
   - [ ] Tests added or updated
   - [ ] Docs updated (`docs/source/...`) if user-facing behavior changed
   - [ ] Lint passes (`ruff check .`)
   - [ ] Frontend submodule pointer updated (if a cross-repo change)

Tips
~~~~

* **Title matters.** Use a Conventional Commits-style title that mirrors
  your main commit, e.g. ``feat: add quantum circuit generator``. Reviewers
  scan PR lists by title.
* **Be specific, not generic.** "Fix bug" is not a PR description. "Fix
  null-pointer in OCL parser when constraint references an unset attribute"
  is.
* **Show, don't tell.** For UI work, include before/after screenshots. For
  generator work, paste a snippet of the generated output. For metamodel
  work, paste the model that exercises the change.
* **Call out anything risky.** Backward-incompatible changes, schema
  migrations, performance trade-offs, dependencies bumped — flag them
  explicitly so reviewers don't miss them.
* **Keep it focused.** One PR = one logical change. If you find yourself
  writing "this PR also …", split it.

Example (good)
~~~~~~~~~~~~~~

   **Title:** ``feat: add JSONSchema generator with array & enum support``

   **What.** Adds a new ``JSONSchemaGenerator`` under
   ``besser/generators/jsonschema/``, registered in the backend generator
   registry. Supports primitive types, enums, arrays (via multiplicity),
   and nested associations.

   **Why.** Closes #412. Several users asked for JSON Schema export to
   drive validation in non-Python services.

   **How.** Walks the ``DomainModel``, emitting one schema file per
   ``Class``. Reuses the existing primitive-type mapping from
   ``SQLGenerator`` (extracted into ``besser/utilities/type_mapping.py``).
   Templates use Jinja2, consistent with other generators.

   **Testing.** Added ``tests/generators/jsonschema/`` with three fixtures
   (library, enums, self-association). Round-trip-validated outputs against
   the ``jsonschema`` Python package. Manually exported from the web editor.

   **Follow-ups.** OneOf/AnyOf for inheritance is out of scope — tracked in
   #418.

Review and merging
~~~~~~~~~~~~~~~~~~

* All PRs must pass automated checks (tests on Python 3.10/3.11/3.12, Ruff
  lint, frontend lint+build, CodeQL).
* At least one maintainer review is required, per the
  `governance rules <https://github.com/BESSER-PEARL/BESSER/blob/master/GOVERNANCE.md>`_.
* Maintainers choose the merge strategy (squash, rebase, or merge commit)
  based on the change.

Coordinating Changes Across BESSER and WME
------------------------------------------

If a change impacts both the BESSER backend and the WME frontend (for
example, adding a new DSL with graphical notation):

1. Implement and commit the **frontend** change in the WME repository,
   targeting its ``develop`` branch.
2. Implement and commit the **backend** change in this repository, targeting
   ``development``.
3. Update the git submodule pointer in this repo to the new WME commit:

   .. code-block:: bash

      cd besser/utilities/web_modeling_editor/frontend
      git pull origin develop
      cd ../../../..
      git add besser/utilities/web_modeling_editor/frontend
      git commit -m "chore: bump frontend submodule"

4. Link the two pull requests in their descriptions and note the intended
   **merge order**.

For the WME-side steps, see the WME contributor guide:
`Adding a New Diagram Type <https://besser.readthedocs.io/projects/besser-web-modeling-editor/en/latest/contributing/new-diagram-guide/index.html>`_.
For the full cross-repo checklist, see
:doc:`contributing/diagram_dsl_workflow`.

Need Help?
----------

If you are unsure where to start, open a discussion in the issue tracker or
reach out via info@besser-pearl.org. We are happy to mentor new
contributors and pair on first issues.
