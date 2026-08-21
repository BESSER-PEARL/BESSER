Contributing
============

Thank you for your interest in contributing to BESSER! This section covers
everything you need to get started, from setting up your environment to
submitting a pull request.

.. tip::
   Not sure where to start? Check :doc:`areas` to find the part of the codebase
   that matches your interests, or browse
   `open issues <https://github.com/BESSER-PEARL/BESSER/issues>`_ labeled
   ``good first issue``.

Quick Links
-----------

- :doc:`../contributor_guide` — Full setup, workflow, and quality guide
- :doc:`areas` — Where to contribute (metamodel, generators, utilities, backend)
- :doc:`create_generator` — Step-by-step guide for adding a new code generator
- :doc:`create_dsl` — Step-by-step guide for adding a new modeling language
- :doc:`diagram_dsl_workflow` — End-to-end workflow for new diagram types (frontend + backend)
- :doc:`../ai_assistant_guide` — Guidelines for AI-assisted contributions

Code Style
----------

- **Python**: PEP 8, 4-space indentation, 120-character line limit
- **Linting**: The CI pipeline uses `Ruff <https://docs.astral.sh/ruff/>`_ with
  rules E, W, F enabled. Run locally before pushing:

  .. code-block:: bash

     pip install ruff
     ruff check besser/

- **Naming**: ``snake_case`` for functions and variables, ``PascalCase`` for classes,
  ``UPPER_CASE`` for constants
- **Commits**: Use `Conventional Commits <https://www.conventionalcommits.org/>`_
  (``feat:``, ``fix:``, ``refactor:``, ``docs:``, ``test:``)

Branching and PR Workflow
-------------------------

1. Create a feature branch from ``master`` (e.g., ``feature/add-rdf-generator``).
2. Make focused, incremental commits.
3. Run tests: ``python -m pytest``
4. Run linting: ``ruff check besser/``
5. Build docs (if changed): ``cd docs && make html``
6. Push and open a pull request against ``master``.
7. Fill in the PR description with: what changed, why, how to test, any follow-up work.
8. Respond to review feedback — reviews are collaborative.

CI will automatically run tests on Python 3.10/3.11/3.12, Ruff linting, and
(for frontend changes) npm lint + build.

.. toctree::
   :maxdepth: 1
   :caption: Guides

   areas
   ../contributor_guide
   ../ai_assistant_guide
   create_generator
   create_dsl
   diagram_dsl_workflow

Our Team
--------

Maintainers:

* `Ivan Alfonso <https://github.com/ivan-alfonso>`_
* `Armen Sulejmani <https://github.com/ArmenSl>`_
* `Jordi Cabot <https://github.com/jcabot>`_

We want to recognize the work of the contributors:

* `Aaron Conrardy <https://github.com/Aran30>`_
* `Fitash Ul-Haq <https://github.com/FitashUlHaq>`_
* `Marcos Gomez <https://github.com/mgv99>`_
* `Atefeh Nirumand <https://github.com/AtefehNirumandJazi>`_
* `Nadia Daoudi <https://github.com/DaoudiNadia>`_
* `Fedor Chikhachev <https://github.com/FChikh>`_
* `Alisa Vorokhta <https://github.com/Vorokhalice>`_

Contact
-------

You can reach us at: info@besser-pearl.org

Website: https://besser-pearl.github.io/website/
