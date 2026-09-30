Install the Python library
==========================

BESSER requires Python **3.11 or newer**. CI tests Python **3.11 and 3.12**;
use one of those versions for the documented setup. Optional dependencies may
have additional platform or Python version constraints.

Install from PyPI
-----------------

Create a virtual environment:

.. code-block:: console

   python -m venv .venv

Activate it in your shell:

.. code-block:: powershell

   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1

.. code-block:: bash

   # Linux or macOS
   source .venv/bin/activate

Install the latest published release:

.. code-block:: console

   python -m pip install besser
   python -c "from importlib.metadata import version; print(version('besser'))"

Continue with :doc:`start/first-model` to generate your first source file.

Install from source
-------------------

Use this path to test the integration branch or contribute a change:

.. code-block:: console

   git clone --branch development https://github.com/BESSER-PEARL/BESSER.git
   cd BESSER
   python -m venv .venv

Activate the environment as shown above, then install the checkout:

.. code-block:: console

   python -m pip install -e .

An editable installation makes the package importable from your scripts;
you do not need to configure ``PYTHONPATH`` for ordinary library use.

Optional dependencies
---------------------

.. code-block:: console

   # OpenAI and Anthropic clients for the Spec-Driven Agent
   python -m pip install "besser[agents]"

   # Neural-network dependencies
   python -m pip install "besser[nn]"

The hosted editor backend has its own requirements. Its runtime setup is
covered by :doc:`contributor_guide` and the
`editor's local deployment guide <https://besser.readthedocs.io/projects/besser-web-modeling-editor/en/latest/user-guide/deploy_locally.html>`_.

If installation fails, check :doc:`troubleshooting` and include the Python
version and complete install error when reporting an issue.
