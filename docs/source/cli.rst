Command-line interface
======================

``pip install besser`` installs the ``besser`` command. It validates and
summarises B-UML models, runs the :doc:`code generators <generators>`, brings
generated code up to a changed model without losing hand edits, and converts
model files, without writing a Python script for each step. ``python -m besser``
runs the same command.

.. code-block:: console

   $ besser --help
   $ besser --version
   besser 8.9.3

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Command
     - What it does
   * - ``besser new <file.py>``
     - Writes a starter model to edit.
   * - ``besser validate <model>...``
     - Runs ``validate()`` on every model in each file.
   * - ``besser info <model>``
     - Summarises the models of a file and names the generators that read it.
   * - ``besser generators``
     - Lists the generators, the models each reads and its options.
   * - ``besser generate <generator> <model> -o <dir>``
     - Runs one generator into an empty directory and prints the files it wrote.
   * - ``besser regenerate <generator> <model> -o <dir>``
     - Brings generated code up to the model and keeps the edits made in it.
   * - ``besser convert <in> -o <out>``
     - Converts a B-UML ``.py`` file to an editor JSON project, or back.

Every command takes these options, before or after the command name:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Option
     - Effect
   * - ``--json``
     - :ref:`Machine-readable output <cli-json>`: one JSON object on standard output.
   * - ``-q``, ``--quiet``
     - Prints errors only: no report, no warnings, and nothing of what a model
       file or a generator prints. The exit status tells the outcome. With
       ``--json`` the JSON object is still printed, whole.
   * - ``-v``, ``--verbose``
     - Prints the Python traceback of a model file that does not load, or of a
       generator that fails, on standard error, before the one-line error.

Model files
-----------

A ``<model>`` is one of:

* a **B-UML Python file** (``.py``), such as the one written in
  :doc:`start/first-model` or exported as B-UML by the editor;
* an **editor JSON export** (``.json``): a project, or a single diagram;
* ``-``, to read either of them from **standard input**
  (``cat model.py | besser validate -``). The content says which it is: an
  editor export is a JSON object, anything else is run as a B-UML script.
  Messages name it ``<stdin>``.

.. warning::

   A ``.py`` model is Python code. ``validate``, ``info``, ``generate``,
   ``regenerate`` and ``convert`` **execute** it, with your permissions, to load
   the models it defines. Only pass files you trust, as you would with
   ``python model.py``. The same holds for a script given on standard input.

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

``besser new``
--------------

.. code-block:: console

   $ besser new model.py
   Wrote a starter model to model.py. Next:
     besser validate model.py
     besser info model.py
     besser generate python model.py -o out

Writes a small, commented domain model (a library, its books and their
authors) that validates without warnings and that every generator reading a
domain model runs on. Edit it into your own. The file must be a ``.py`` file;
an existing file is kept unless you pass ``--force``.

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

Several files are checked in one run; a shell glob covers a folder:

.. code-block:: console

   $ besser validate models/*.py
   models/library.py: 1 model
     library_model (DomainModel): ok
   models/shop.py: 1 model
     shop_model (DomainModel): ok
   OK: 0 errors, 0 warnings

The exit status is that of the worst file: 2 when one does not load (the
others are still validated), else 1 when one has errors. With several files,
each message of ``errors`` and ``warnings`` starts with the file it is about.
The command takes files, not a folder: it would have to execute every ``.py``
file found in it, and only the files you name are executed.

The command runs ``validate()`` only. It does not evaluate OCL constraints on
an object model, which the editor's validation also does.

``besser info``
---------------

.. code-block:: console

   $ besser info library.py
   library.py: 1 model
     library_model (DomainModel 'LibraryModel'): 2 classes, 1 association
       class Library: name: str
       class Book: title: str, pages: int
       association holds: library: Library [1] -- books: Book [0..*]
   Generators that read this model: python, java, pydantic, test_case, django, spring, backend, sqlalchemy, sql, supabase, jsonschema, rdf, rest_api

Each model is listed under the name the other commands call it (the variable
it is bound to in a ``.py`` file, the diagram's title in an editor export),
with its class and, in quotes, its own name when that differs. Then come the
counts of what it holds (classes, objects, screens, states, ...) and, for a
domain model, its classes with their attributes, methods and parents, its
enumerations and its associations, in the order they were defined. The last
line names the generators whose models the file holds.

The model is not validated: a model with errors is summarised too.
``besser info --json`` gives ``models`` (``name``, ``kind``, ``type``,
``model_name``, ``counts`` and, for a domain model, ``classes``,
``enumerations`` and ``associations``) and ``generators``.

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

The list ends with the generators ``besser regenerate`` works with.

``besser generators --json`` gives the same list with each option as a JSON
Schema property (``type``, ``enum``, ``default``, ``description``), and
``supports_regenerate`` (``true`` or ``false``) for each generator.

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
#. runs the generator into the ``-o`` directory, which is created if missing
   and must be empty;
#. prints the files the run created or rewrote.

``generate`` overwrites what it generates, so it refuses (exit status 2) a
directory that is not empty, and nothing is written. ``--force`` writes into it
anyway, overwriting the files of the same names. To update generated code
after a model change and keep the edits made in it, use
:ref:`besser regenerate <cli-regenerate>`.

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

When the file holds several models of a kind the generator reads (two domain
models in a script, an editor project with several class diagrams), the
command does not guess. Name the one to use with ``--use``, by the name
``besser info`` lists it under or by the model's own name; repeat ``--use`` to
name one of each kind:

.. code-block:: console

   $ besser generate python project.json -o out
   error: project.json holds 2 domain models (Library, Notes) and the 'python' generator reads one: name it with --use NAME
   $ besser generate python project.json -o out --use Notes

The web editor runs its generators through the same function,
``besser.generators.runner.run_generator``, so a generator behaves the same
from the command line, from Python and from the editor.

.. _cli-regenerate:

``besser regenerate``
---------------------

``regenerate`` brings a directory of generated code up to the model and keeps
the edits made in it. It is :doc:`generator.regenerate() <generators/regenerating>`
from the command line: the generator runs in a scratch folder, and its output
is merged, three ways, with what it generated last time and with the files as
they are now. It needs the merge library:

.. code-block:: console

   $ pip install "besser[regen]"

The first run, into an empty or missing directory, generates the code and
starts the record of it, ``.besser_generated.json``, in that directory. Keep
the record with the code (commit it):

.. code-block:: console

   $ besser regenerate python library.py -o out
   Regenerated out with the 'python' generator (1 written):
     written  out/classes.py

Edit the generated files, change the model, and run it again. Here a comment
was added at the top of ``classes.py`` and the model gained an attribute:

.. code-block:: console

   $ besser regenerate python library.py -o out
   Regenerated out with the 'python' generator (1 merged):
     merged  out/classes.py
   warning: the model's changes were merged into your edits; a clean merge can still break, so run your tests: out/classes.py

Each file is reported as ``written`` (new, or nobody edited it), ``merged``
(your edits and the model's changes), ``conflict`` or ``deleted`` (the model
no longer produces it and nobody edited it). When nothing changes, the command
says the directory is up to date. Warnings name the files left as they were:
edited files the model no longer produces, files you deleted, files that
cannot be merged as text. :doc:`generators/regenerating` has the full table.

**Conflicts.** When an edit and the model changed the same lines, the file
gets conflict markers around those lines, the command lists it and exits with
status 1:

.. code-block:: console

   $ besser regenerate python library.py -o out
   Regenerated out with the 'python' generator (1 conflict):
     conflict  out/classes.py
   error: out/classes.py: your edit and the model changed the same lines. Resolve the conflict markers ('<<<<<<< your edit' ... '>>>>>>> generated from your model'), then run regenerate again

Edit the file so the markers are gone. Until then the next ``regenerate``
changes nothing and exits with status 2.

**A directory written by** ``besser generate`` has no record, so edits in it
cannot be told from generated code and ``regenerate`` refuses it (status 2).
Adopt it once with ``--adopt`` and the model it was generated from, before
you change the model. This changes no file:

.. code-block:: console

   $ besser generate python library.py -o out
   $ besser regenerate python library.py -o out --adopt
   Recorded out as the 'python' generator's output for library.py; no file was changed. 'besser regenerate' now keeps the edits made in it.

Files that differ from that model's output are listed in a warning and count
as your edits from then on. If you plan to edit the generated code, start
with ``regenerate`` instead of ``generate``.

``regenerate`` takes the same ``--option`` and ``--use`` as ``generate``. Give
the same options on every run: a different option changes the output like a
model change does.

The command is refused with status 2, and nothing is written, when:

* the generator does not take part (``besser generators`` lists those that
  do): its output differs from run to run for one model. Supabase, Flutter,
  Terraform, the agent generator and Docker Compose do not;
* ``pygit2`` is not installed: the error gives the ``pip install`` line above;
* the directory holds the output of another generator, a Spec-Driven Agent
  app, or conflict markers of the last run.

``besser generate`` and a regenerated directory: ``generate`` refuses a
directory holding ``.besser_generated.json`` and prints the ``regenerate``
command to run. With ``--force`` it overwrites the hand edits and removes the
record, which described the output it replaced, and says so in a warning.

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
export writes. An existing output file is overwritten. With ``-`` as the
input, the direction follows the output's extension and what standard input
holds.

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
     - List of messages. For ``validate``, each starts with the model's name
       (after the file's, when several files are given).
   * - ``warnings``
     - List of messages; they never make ``ok`` false, except with ``--strict``.
   * - ``files``
     - The files the command wrote, as paths under the ``-o`` you gave. For
       ``regenerate``: the written, merged and conflicted files.
   * - ``command``
     - The command that ran (empty when the arguments did not parse).

Each command adds its own:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Keys
   * - ``validate``
     - ``model_files`` (the files given) and ``models``: per model its
       ``file``, ``name``, ``kind``, ``type``, ``validated``, ``errors`` and
       ``warnings``.
   * - ``info``
     - ``model``, ``models`` and ``generators``.
   * - ``generators``
     - ``generators``: per generator its ``id``, ``summary``, ``category``,
       ``models``, ``options`` and ``supports_regenerate``.
   * - ``generate``
     - ``generator``, ``model`` and ``output_dir``.
   * - ``regenerate``
     - Those of ``generate``, ``adopted``, and one list of paths per outcome:
       ``written``, ``merged``, ``conflicted``, ``deleted``,
       ``kept_removed_from_model``, ``not_restored``, ``not_merged`` and
       ``differs_from_model`` (the fields of ``RegenerationReport``).
   * - ``new``
     - ``model``: the file written.
   * - ``convert``
     - ``input`` and ``output``.

.. code-block:: console

   $ besser validate library.py --json
   {
     "ok": true,
     "command": "validate",
     "errors": [],
     "warnings": [],
     "files": [],
     "model_files": [
       "library.py"
     ],
     "models": [
       {
         "file": "library.py",
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
       ``--strict``), a generator that raised, a conversion that failed,
       conflicts left by ``regenerate``.
   * - 2
     - The command could not run: bad arguments, an unknown generator or
       option, a model file that does not load, an output directory that
       ``generate`` or ``regenerate`` refuses. Nothing was generated. A usage
       error is also a JSON object when ``--json`` is given.

Standard output carries the result only. Whatever a model file or a generator
prints while it runs, and the log messages of BESSER, go to standard error.
Without ``--json``, error and warning lines go to standard error too, except
those ``validate`` lists under each model in its report.

Use from a coding agent or a script
-----------------------------------

The command is meant to be driven by AI coding agents (Claude Code, Cursor,
Codex, ...) and CI jobs as much as by people: no prompts, stable exit
statuses, and JSON on request. A typical loop for an agent that writes a
B-UML model:

.. code-block:: console

   $ besser new model.py                         # a starter model to edit
   $ besser generators --json                    # what can be generated, with which options
   $ besser validate model.py --json             # fix every entry of "errors", then repeat
   $ besser info model.py --json                 # what the model holds, which generators read it
   $ besser regenerate backend model.py -o app --json  # "files" lists what to read next
   $ besser convert model.py -o model.json       # open the model in the editor

Generating with ``regenerate`` from the first run lets the agent, or you, edit
the generated code and still change the model later: the next ``regenerate``
keeps those edits, and ``conflicted`` lists the files to resolve when it exits
with status 1. ``besser validate -`` checks a model from standard input before
it is written to a file.

In CI, ``besser validate model.py --strict`` fails the job on any error or
warning.

The `BESSER Skills <https://github.com/BESSER-PEARL/besser-skills>`_ teach an
agent to write B-UML; the command gives it a way to check and run what it
wrote.
