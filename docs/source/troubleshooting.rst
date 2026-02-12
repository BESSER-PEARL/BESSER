Troubleshooting
===============

This page collects common problems that users encounter when installing,
configuring, or running BESSER.  Most of them are **not** bugs in BESSER
itself but rather environment or tooling issues.  If your problem is not
listed here, feel free to
`open an issue <https://github.com/BESSER-PEARL/BESSER/issues>`_.

.. contents:: On this page
   :local:
   :depth: 2


Installation Issues
-------------------

Permission denied when installing packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom**

.. code-block:: text

   ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied: ...

**Cause**

You are installing into a system-managed Python instead of a virtual
environment, or you do not have write access to the target directory.

**Fix**

Create and activate a virtual environment first:

.. code-block:: console

   $ python -m venv .venv

   # Linux / macOS
   $ source .venv/bin/activate

   # Windows (PowerShell)
   $ .\.venv\Scripts\Activate.ps1

   # Windows (cmd)
   $ .venv\Scripts\activate.bat

   $ pip install besser

.. tip::
   If you are on Linux and get ``Permission denied`` when creating the venv
   directory itself, make sure you own the target folder or use ``sudo chown``
   on it.

Python 3.13 is not supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom**

Build errors or missing wheels when running ``pip install besser`` on
Python 3.13.

**Cause**

Some BESSER dependencies have not yet released Python 3.13-compatible
wheels.

**Fix**

Use **Python 3.10** or **3.12** instead.  You can manage multiple Python
versions with `pyenv <https://github.com/pyenv/pyenv>`_ (Linux/macOS) or
the `Python Launcher for Windows <https://docs.python.org/3/using/windows.html#launcher>`_.


IDE Configuration
-----------------

``ModuleNotFoundError: No module named 'besser'`` inside VS Code or Cursor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom**

Running a script from the editor's integrated terminal fails with:

.. code-block:: text

   ModuleNotFoundError: No module named 'besser'

even though ``pip install -e .`` succeeded.

**Cause**

The IDE is using a different Python interpreter (e.g., the system Python)
instead of the virtual environment where BESSER is installed, or the
``PYTHONPATH`` is not set.

**Fix**

1. Open the Command Palette (``Ctrl+Shift+P`` / ``Cmd+Shift+P``) and run
   **Python: Select Interpreter**.
2. Pick the interpreter inside your virtual environment
   (e.g., ``.venv/bin/python`` or ``.venv\Scripts\python.exe``).

.. note::
   If you installed BESSER with ``pip install besser``, steps 1 and 2 are
   usually enough.  The steps below only apply when you are working with
   the **cloned source code**.

3. When working with the BESSER source code (i.e., you cloned the repository
   and run scripts directly from the repo), the ``PYTHONPATH`` must include
   the repository root so Python can find the ``besser`` package.  You have
   two options:

   **Option A** — add a ``.env`` file at the project root:

   .. code-block:: text

      PYTHONPATH=${workspaceFolder}

   **Option B** — set it permanently in your VS Code **User or Workspace
   settings** (``settings.json``).  This injects the variable into every
   integrated terminal:

   .. code-block:: jsonc

      // Windows
      "terminal.integrated.env.windows": {
          "PYTHONPATH": "C:\\Users\\<you>\\path\\to\\BESSER"
      }

      // Linux
      "terminal.integrated.env.linux": {
          "PYTHONPATH": "/home/<you>/path/to/BESSER"
      }

      // macOS
      "terminal.integrated.env.osx": {
          "PYTHONPATH": "/Users/<you>/path/to/BESSER"
      }

   Replace the path with the absolute path to your cloned repository.

   .. tip::
      After changing this setting you need to **open a new terminal**
      (or restart VS Code) for it to take effect.  Existing terminals
      keep the old environment.

Folder path errors when running Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom**

.. code-block:: text

   FileNotFoundError: [Errno 2] No such file or directory: '...'

or relative-import errors when executing a script.

**Cause**

The working directory of the terminal does not match the expected project
root.  VS Code and Cursor sometimes open the terminal in a sub-folder or
the user's home directory.

**Fix**

* Verify the working directory with ``pwd`` (Linux/macOS) or ``Get-Location``
  (PowerShell) before running the script.
* In VS Code, set ``"python.terminal.executeInFileDir": true`` in your
  workspace settings so scripts run from their own folder.
* Alternatively, always use absolute paths or ``pathlib.Path(__file__).parent``
  in your scripts to resolve files relative to the script location.

PowerShell execution policy blocks activation scripts (Windows)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom**

.. code-block:: text

   .\.venv\Scripts\Activate.ps1 : File ... cannot be loaded because running
   scripts is disabled on this system.

or:

.. code-block:: text

   ... is not digitally signed. You cannot run this script on the current
   system.

**Cause**

The default PowerShell execution policy on Windows (``Restricted``) blocks
all ``.ps1`` scripts, including the virtual-environment activation script.

**Fix**

*Option 1* — Change the policy for your user account (persistent, recommended):

.. code-block:: powershell

   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

*Option 2* — Change the policy for the current PowerShell session only
(temporary):

.. code-block:: powershell

   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

*Option 3* — If you only hit this in VS Code's integrated terminal, add the
following to your ``settings.json``:

.. code-block:: jsonc

   "terminal.integrated.env.windows": {
       "PSExecutionPolicyPreference": "RemoteSigned"
   }

After applying any of these options, open a **new** terminal and retry
activating the virtual environment.

.. note::
   ``RemoteSigned`` allows locally created scripts to run but requires
   downloaded scripts to be signed.  For development purposes this is
   the safest permissive policy.


Port already in use
^^^^^^^^^^^^^^^^^^^

**Symptom**

.. code-block:: text

   ERROR: Ports are not available: listen tcp 0.0.0.0:8000: bind: address already in use

**Cause**

Another process (or a previous ``docker-compose up`` session) is already
listening on the same port.

**Fix**

Stop the previous containers first:

.. code-block:: console

   $ docker-compose down

Or find and stop the process occupying the port:

.. code-block:: console

   # Linux / macOS
   $ lsof -i :8000
   $ kill <PID>

   # Windows (PowerShell)
   $ Get-NetTCPConnection -LocalPort 8000
   $ Stop-Process -Id <PID>

Docker Compose is not recognized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom**

.. code-block:: text

   'docker-compose' is not recognized as an internal or external command

**Cause**

Docker Compose is not installed or not on the system ``PATH``.

**Fix**

* Install `Docker Desktop <https://www.docker.com/products/docker-desktop/>`_,
  which bundles Docker Compose.
* If you already have Docker installed, try the newer ``docker compose``
  (with a space) syntax — it is built into recent Docker CLI versions:

  .. code-block:: console

     $ docker compose up --build


Docker & Containers
-------------------

Docker build fails behind a corporate proxy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom**

``pip install`` or ``npm install`` inside a Dockerfile hangs or times out.

**Cause**

The Docker build context does not inherit your host's proxy settings.

**Fix**

Pass the proxy as a build argument:

.. code-block:: console

   $ docker-compose build \
       --build-arg HTTP_PROXY=$HTTP_PROXY \
       --build-arg HTTPS_PROXY=$HTTPS_PROXY

Or add the following to your ``Dockerfile`` before the ``RUN`` commands:

.. code-block:: dockerfile

   ARG HTTP_PROXY
   ARG HTTPS_PROXY
   ENV http_proxy=$HTTP_PROXY
   ENV https_proxy=$HTTPS_PROXY

Docker runs out of disk space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom**

.. code-block:: text

   no space left on device

**Cause**

Old containers, images, and volumes accumulate over time.

**Fix**

.. code-block:: console

   $ docker system prune -a --volumes

.. warning::
   This removes **all** unused images, containers, and volumes.  Make sure
   you do not need any of them before running this command.


Code Generators
---------------

Generated code has import errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom**

Running the generated backend or frontend fails immediately with an
``ImportError`` or a missing-module error.

**Cause**

Dependencies were not installed in the generated project.

**Fix**

.. code-block:: console

   # Backend (FastAPI / SQLAlchemy)
   $ cd <output>/backend
   $ pip install -r requirements.txt

   # Frontend (React)
   $ cd <output>/frontend
   $ npm install

Generator output is not deterministic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom**

Running the same generator twice produces files with different content
(e.g., reordered imports or attributes).

**Cause**

Some Python collection types (``set``, ``dict``) do not guarantee iteration
order across runs.

**Fix**

Set the ``PYTHONHASHSEED`` environment variable to a fixed value:

.. code-block:: console

   $ PYTHONHASHSEED=0 python my_generator_script.py


Still Stuck?
------------

If none of the above solves your problem:

1. Search the `existing issues <https://github.com/BESSER-PEARL/BESSER/issues>`_
   to see if someone has already reported it.
2. If not, open a new issue with:

   * Your OS and Python version.
   * The full error traceback.
   * The command you ran and the expected vs. actual behavior.
   * Any relevant model files (anonymized if needed).
