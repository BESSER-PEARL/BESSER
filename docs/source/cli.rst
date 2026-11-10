Command-line interface
======================

``pip install besser`` installs the ``besser`` command. It validates B-UML
models, runs the :doc:`code generators <generators>` and converts model files,
without writing a Python script for each step. ``python -m besser`` runs the
same command.

.. code-block:: console

   $ besser --help
   $ besser --version
   besser 8.9.3

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Command
     - What it does
   * - ``besser validate <model>``
     - Runs ``validate()`` on every model in the file.
   * - ``besser generators``
     - Lists the generators, the models each reads and its options.
   * - ``besser generate <generator> <model> -o <dir>``
     - Runs one generator and prints the files it wrote.
   * - ``besser convert <in> -o <out>``
     - Converts a B-UML ``.py`` file to an editor JSON project, or back.

Every command takes ``--json``, before or after the command name, for
:ref:`machine-readable output <cli-json>`.

Model files
-----------

A ``<model>`` is one of:

* a **B-UML Python file** (``.py``), such as the one written in
  :doc:`start/first-model` or exported as B-UML by the editor;
* an **editor JSON export** (``.json``): a project, or a single diagram.

.. warning::

   A ``.py`` model is Python code. ``validate``, ``generate`` and ``convert``
   **execute** it, with your permissions, to load the models it defines. Only
   pass files you trust, as you would with ``python model.py``.

   Nothing but the file you name is executed. Its
   ``if __name__ == "__main__":`` block does not run, and its directory is not
   added to the import path, so it imports installed packages only (set
   ``PYTHONPATH`` yourself to import your own modules). Everything else at the
   top level of the file does run: a script that calls a generator there
   generates when it is loaded.

   A ``.json`` model is parsed as data and never executed.

From a ``.py`` file, the command takes the models of the ``Project`` the file
defines. Without a ``Project``, it takes every model bound to a top-level name
(``DomainModel``, ``ObjectModel``, ``GUIModel``, ``Agent``, ``StateMachine``,
``QuantumCircuit``, ``NN``, ``BPMNModel`` and the deployment, component,
knowledge-graph and feature models).

Reading a ``.json`` file uses the Web Modeling Editor's converters, which need
the editor backend's packages. They are not part of the core install; the
command prints the ``pip install -r .../requirements.txt`` line to run when
they are missing.

``besser validate``
-------------------

.. code-block:: console

   $ besser validate library.py
   library.py: 1 model
     library_model (DomainModel): ok
   OK: 0 errors, 0 warnings

Each model is checked with its own ``validate()`` method (see
:doc:`buml_language/model_types/structural` for the rules of a domain model).
Errors and warnings are listed under the model they belong to:

.. code-block:: console

   $ besser validate broken.py
   broken.py: 1 model
     broken_model (DomainModel): 1 error, 0 warnings
       error: Attribute 'title' of class 'Book' has type 'Text', which is not in the domain model 'Broken'. ...
   FAILED: 1 error, 0 warnings

Warnings do not fail the command. Pass ``--strict`` to exit with status 1 on
warnings too.

A file that cannot be loaded (it is missing, raises while it runs, or holds no
model) is reported with exit status 2, with the line of the file that raised:

.. code-block:: console

   $ besser validate crash.py
   error: crash.py: line 2: ZeroDivisionError: division by zero

The command runs ``validate()`` only. It does not evaluate OCL constraints on
an object model, which the editor's validation also does.

``besser generators``
---------------------

Lists every generator of the registry (``besser/generators/registry.py``, the
list the editor and the Spec-Driven Agent also use): its id, the models it
reads in brackets, what it generates, and its options the way they are typed.

.. code-block:: console

   $ besser generators
   Generators (id, the models it reads, what it generates):

   python  [domain]  plain Python classes
   java  [domain]  Java classes
   pydantic  [domain]  Pydantic models
       --option backend=true|false  (default: false)
           Generate 'Create' suffixed models for API request bodies
   ...
   sqlalchemy  [domain]  SQLAlchemy ORM models
       --option dbms=sqlite|postgresql|mysql|mssql|mariadb|oracle  (default: sqlite)
           Target database system
   ...
   react  [domain, gui]  React frontend

The model kinds are ``domain`` (a ``DomainModel``, the class diagram), ``gui``,
``object``, ``agent``, ``quantum``, ``nn``, ``bpmn``, ``deployment`` (the cloud
infrastructure the Terraform generator provisions) and ``uml_deployment`` (a
UML deployment diagram).

``besser generators --json`` gives the same list with each option as a JSON
Schema property (``type``, ``enum``, ``default``, ``description``).

``besser generate``
-------------------

.. code-block:: console

   $ besser generate sqlalchemy library.py -o out --option dbms=postgresql
   Generated 1 file in out with the 'sqlalchemy' generator:
     out/sql_alchemy.py

The command:

#. loads the model file and picks the one model of each kind the generator
   reads;
#. validates those models, and stops with exit status 1 if one has errors
   (warnings are printed and do not stop it);
#. runs the generator into the ``-o`` directory, which is created if missing;
#. prints the files the run created or rewrote.

An option is a keyword argument of the generator's constructor, given as
``--option name=value``; repeat ``--option`` for several. The value is read as
the type ``besser generators`` shows: ``true`` or ``false``, an integer, a
string, or a comma-separated list:

.. code-block:: console

   $ besser generate django library.py -o site --option project_name=library_site --option app_name=books
   $ besser generate rest_api library.py -o api --option http_methods=GET,POST

An unknown generator, an unknown option, a value of the wrong type, or a model
file without the kind of model the generator reads (``react`` needs a ``gui``
model next to the ``domain`` model) is exit status 2, and nothing is written.

When the file holds several models of a kind the generator reads, the command
does not guess: keep one in the file, or define a ``Project(...)`` naming the
models to use.

The web editor runs its generators through the same function,
``besser.generators.runner.run_generator``, so a generator behaves the same
from the command line, from Python and from the editor.

``besser convert``
------------------

Converts between the two model file formats. The direction follows the file
extensions:

.. code-block:: console

   $ besser convert library.py -o library.json     # B-UML -> editor project
   Converted library.py (1 model) to library.json
   $ besser convert library.json -o library.py     # editor export -> B-UML

``library.json`` has the format of an editor project export, so the editor
imports it as a project. The ``.py`` output is the B-UML code the editor's own
export writes. An existing output file is overwritten.

Both directions reuse the editor's converters, so they share their limits:

* The editor has no diagram for a Terraform ``deployment`` model or a feature
  model; converting to JSON leaves them out and says so in a warning.
* A lone model takes the default title of its diagram (a ``DomainModel`` named
  ``Library`` comes back as ``Class_Diagram``); its classes, attributes and
  associations keep their names.
* A model written in Python has no layout: the converter computes the
  positions of its elements.

.. _cli-json:

JSON output and exit status
---------------------------

With ``--json``, standard output is exactly one JSON object. These keys are
on every command and are stable:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Value
   * - ``ok``
     - ``true`` when the exit status is 0.
   * - ``errors``
     - List of messages. For ``validate``, each starts with the model's name.
   * - ``warnings``
     - List of messages; they never make ``ok`` false, except with ``--strict``.
   * - ``files``
     - The files the command wrote, as paths under the ``-o`` you gave.
   * - ``command``
     - The command that ran (empty when the arguments did not parse).

Each command adds its own: ``validate`` a ``models`` list (``name``, ``kind``,
``type``, ``validated``, ``errors``, ``warnings`` per model), ``generators`` a
``generators`` list, ``generate`` the ``generator``, ``model`` and
``output_dir``, ``convert`` the ``input`` and ``output``.

.. code-block:: console

   $ besser validate library.py --json
   {
     "ok": true,
     "command": "validate",
     "errors": [],
     "warnings": [],
     "files": [],
     "model": "library.py",
     "models": [
       {
         "name": "library_model",
         "kind": "domain",
         "type": "DomainModel",
         "validated": true,
         "errors": [],
         "warnings": []
       }
     ]
   }

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Exit status
     - Meaning
   * - 0
     - Success.
   * - 1
     - The command ran and failed: validation errors (or warnings with
       ``--strict``), a generator that raised, a conversion that failed.
   * - 2
     - The command could not run: bad arguments, an unknown generator or
       option, a model file that does not load. A usage error is also a JSON
       object when ``--json`` is given.

Standard output carries the result only. Whatever a model file or a generator
prints while it runs, and the log messages of BESSER, go to standard error.
Without ``--json``, error and warning lines of ``generate`` and ``convert`` go
to standard error too.

Use from a coding agent or a script
-----------------------------------

The command is meant to be driven by AI coding agents (Claude Code, Cursor,
Codex, ...) and CI jobs as much as by people: no prompts, stable exit
statuses, and JSON on request. A typical loop for an agent that writes a
B-UML model:

.. code-block:: console

   $ besser generators --json                    # what can be generated, with which options
   $ besser validate model.py --json             # fix every entry of "errors", then repeat
   $ besser generate backend model.py -o app --json    # "files" lists what to read next
   $ besser convert model.py -o model.json       # open the model in the editor

In CI, ``besser validate model.py --strict`` fails the job on any error or
warning.

The `BESSER Skills <https://github.com/BESSER-PEARL/besser-skills>`_ teach an
agent to write B-UML; the command gives it a way to check and run what it
wrote.
