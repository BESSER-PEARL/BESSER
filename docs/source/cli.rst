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
   * - ``besser init [folder]``
     - Sets a project up for a coding agent: a starter model, a ``besser.toml``,
       the agent guide (``AGENTS.md``, ``CLAUDE.md``) and ``.ignore`` rules.
   * - ``besser new <file> [--kind KIND]``
     - Writes a starter model to edit, of any model kind (``--list`` names them),
       and the agent guide beside it.
   * - ``besser guide``
     - Prints the rules a coding agent follows in a BESSER project.
   * - ``besser schema [topic]``
     - Describes the formats: the B-UML classes of a model kind, the editor
       project JSON Schema, the ``--json`` output, ``besser.toml``.
   * - ``besser validate <model|dir>...``
     - Runs ``validate()`` on every model in each file or directory;
       ``--ocl`` checks the OCL constraints too.
   * - ``besser info <model>``
     - Summarises the models of a file and names the generators that read it.
   * - ``besser generators``
     - Lists the generators, the models each reads and its options.
   * - ``besser generate <generator> <model> -o <dir>``
     - Runs one generator into an empty directory, prints the files it wrote
       and records them for ``regenerate``. ``--keep-edits`` is ``regenerate``.
   * - ``besser regenerate <generator> <model> -o <dir>``
     - Brings generated code up to the model and keeps the edits made in it.
   * - ``besser build [output]...``
     - Generates every output listed in ``besser.toml``: ``generate`` the first
       time, ``regenerate`` afterwards.
   * - ``besser apply``
     - ``build``, then the ``test`` command of ``besser.toml``: the one command
       to run after a model change.
   * - ``besser convert <in> -o <out>``
     - Converts a B-UML ``.py`` file to an editor JSON project, or back, and
       imports PlantUML, draw.io, CSV, RDF/OWL, SDMX and Python code as models.
   * - ``besser completion bash|zsh|fish``
     - Prints a shell completion script.
   * - ``besser mcp``
     - Serves the commands as tools of an MCP server on standard input and
       output, for an AI agent (``besser[mcp]``).
   * - ``besser --version``
     - Prints the installed BESSER version.

``generate``, ``regenerate``, ``build`` and ``convert`` take ``--dry-run``: they print the
files they would write or change, and write nothing.

Every command takes these options, before or after the command name:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Option
     - Effect
   * - ``--json``
     - :ref:`Machine-readable output <cli-json>`: one JSON object on standard output.
   * - ``-q``, ``--quiet``
     - Drops the diagnostics: notes, warnings, BESSER's log warnings and
       whatever a model file or a generator prints, and the report of a command
       that checks or writes files (``validate``, ``generate``, ``regenerate``,
       ``convert``, ``new``). Errors are still printed, and so is the output of
       a command whose output is the result: ``completion`` (so
       ``eval "$(besser completion bash -q)"`` works), ``generators``,
       ``info``, ``schema``, ``new --list`` and ``--version``. A ``validate --strict`` run that fails on
       warnings alone prints ``error: N warnings (--strict)`` and the warnings.
       With ``--json`` the JSON object is still printed, whole.
   * - ``-v``, ``--verbose``
     - Prints the whole report even when standard output is not a terminal
       (see below), and the Python traceback of a model file that does not
       load, or of a generator that fails, on standard error, before the
       one-line error.

.. _cli-terse:

When standard output is not a terminal (a coding agent, a script, a CI log),
the output is **terse**: a command that writes or checks files prints a
one-line summary instead of its report, then its warnings and errors, and no
notes; what a model file or a generator prints is dropped. A failure keeps its
report (``validate`` lists every error with its file and line), and the output
of ``info``, ``generators``, ``schema`` and ``guide`` is printed whole, since it
is what was asked for. ``BESSER_OUTPUT=full`` prints the whole report anyway,
``BESSER_OUTPUT=terse`` the summary on a terminal too; ``--json`` is always
complete.

.. code-block:: console

   $ besser build | cat
   OK: 1 output, 0 failed: [backend] Generated 10 files in backend with the 'backend' generator

.. _cli-model-files:

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
   ``python model.py``. The same holds for a script given on standard input:
   do not pipe a download you have not read into ``besser validate -``.

   Nothing but the files you name is executed (and those ``besser.toml`` names
   for ``build``, and, with
   ``validate --exec-py``, the ``.py`` files found in a directory you name). A file's
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

Reading a ``.json`` file uses the Web Modeling Editor's converters. Besides
the core install they import ``deep-translator``, ``Pygments`` and ``bocl``
(the B-OCL evaluator, ``besser[ocl]``), not the backend's web server (FastAPI,
Starlette, uvicorn) or its other packages. All three are in the editor
backend's requirements file, and when one is missing the command prints the
``pip install -r .../requirements.txt`` line that installs them.

.. _cli-several-diagrams:

Several models of a kind
~~~~~~~~~~~~~~~~~~~~~~~~

A generator reads one model of each kind it needs. When the file holds several
of a kind, ``--use NAME`` names the one to read, by the name ``besser info``
lists it under or by the model's own name (an editor diagram's model is named
after its title).

Without ``--use``, an editor project picks for you: the editor stores which
diagram of each type is open (``currentDiagramIndices``), and the command reads
that diagram's model and says so in a note (``notes`` and ``used`` in the JSON
output). ``besser info`` marks that model ``(current in the editor)``
(``"current": true`` in its JSON):

.. code-block:: console

   $ besser generate python project.json -o out
   note: project.json holds 2 domain models (Library, Shop); using Shop, the project's current ClassDiagram (--use NAME picks another)
   $ besser generate python project.json -o out --use Library

A ``.py`` file, or a project that stores no current diagram for that type, has
no such default: the command lists the models and asks for ``--use``. So does a
project whose current diagram is out of range or empty, or one with two diagrams
of a type that share an id (the converter then makes one model of both, so the
models cannot be told apart by diagram), and the error says why. ``besser info``
says the same in a note whenever a file holds several models of a kind and the
project names no current one, so it shows beforehand what ``generate`` will ask:

.. code-block:: console

   $ besser info project.json
   note: project.json holds 2 domain models and names no current one: its currentDiagramIndices has no ClassDiagram entry; generators need --use NAME

When ``--use NAME`` names several models (two diagrams with the same title),
the project's current diagram decides if it is one of them, and a note names
the diagram read. Otherwise the name is ambiguous: the error lists the diagrams
by type, index and id, and the way out is to give them different titles in the
editor.

``besser new``
--------------

.. code-block:: console

   $ besser new model.py
   Wrote a starter domain model to model.py, and the agent guide: AGENTS.md, CLAUDE.md, .ignore. Next:
     besser validate model.py
     besser info model.py
     besser generate python model.py -o out
     besser schema domain          the classes to write it with

Writes a small, commented model that validates without warnings, to edit into
your own. Without ``--kind`` it is a domain model (a library, its books and
their authors) that every generator reading a domain model runs on.
``--kind`` picks another kind of model, and ``--list`` names them:

.. code-block:: console

   $ besser new --list
   Kinds (besser new FILE --kind KIND; FILE is a .py file, or a .json file for the editor):
     domain          a class diagram: a library, its books and their authors
     object          an object diagram: instances of a small domain model
     state_machine   a state machine: the life of an order
     agent           a conversational agent: intents, states and replies
     gui             a GUI model over a domain model: screens that list its classes
     nn              a neural network: a small image classifier
     quantum         a quantum circuit: a Bell state
     bpmn            a BPMN process: check an order, then ship or reject it
     deployment      a cloud deployment: a containerized application on a cluster
     uml_deployment  a UML deployment diagram: where an application's parts run
     component       a UML component diagram: components and their interfaces
     kg              a knowledge graph: classes, individuals and the edges between them
     feature         a feature model: the features of a product line
     project         a project: several models in one file
   $ besser new machine.py --kind state_machine

A starter that needs another model holds it too: the ``object`` and ``gui``
starters define the domain model they refer to. Each one names, in its header,
a generator that runs on it when one reads its kind.

A ``.py`` file is a B-UML script, the format to write and edit. A ``.json``
file is the same starter as an editor project (``besser new model.json --kind
bpmn``), which needs the :ref:`backend packages <cli-model-files>`; the editor
has no diagram for the ``deployment`` and ``feature`` kinds, which are ``.py``
only. An existing file is kept unless you pass ``--force``.

Beside the model, ``new`` also writes what a :ref:`coding agent <cli-agents>`
needs: the guide ``besser guide`` prints, as a block of ``AGENTS.md``; a
``CLAUDE.md`` that imports it (``@AGENTS.md``); and a block of ``.ignore`` that
keeps the generated folders of ``besser.toml`` out of searches. What people
wrote in those files is kept: each block sits between markers and is replaced
in place, and a file already up to date is not rewritten. ``--no-agent-files``
writes the model alone.

.. _cli-init:

``besser init``
---------------

.. code-block:: console

   $ besser init shop
   Set up shop: wrote shop/model.py, shop/besser.toml, shop/AGENTS.md, shop/CLAUDE.md, shop/.ignore.
   Next: edit model.py, then run 'besser apply' (validate, build, test).

Sets a folder (default: the current one) up as a project a coding agent can
work in: the starter model of ``--kind`` as ``model.py``, a ``besser.toml``
that builds it (a ``backend`` API into ``backend/`` for a class diagram, the
generator ``new`` suggests for another kind) with a commented ``test`` line to
fill in, and the agent files ``new`` writes. A ``model.py`` or ``besser.toml``
that exists is kept, and a note says so. ``.ignore`` then reads:

.. code-block:: text

   # besser:begin
   # Generated by BESSER (besser.toml): a black box, see AGENTS.md.
   /backend/*
   !/backend/extensions.py
   **/.besser_generated.json
   # besser:end

ripgrep reads ``.ignore``, and so do the tools that run it (Claude Code's Grep
tool), ``fd`` and ``ag``; git does not, so generated code can still be committed. The
hand-written ``extensions.py`` is not ignored. ``besser build`` keeps the block
in step with the outputs of ``besser.toml``; an ``.ignore`` without the block
is left alone.

``besser guide``
----------------

Prints the guide ``new`` and ``init`` write to ``AGENTS.md``, in about 2 KB: the
rules of the :ref:`frugal workflow <cli-agents>` and what the ``backend``
generator's API does, so that an agent needs neither the generated code nor
this documentation. ``--json`` puts it in ``guide``.

``besser schema``
-----------------

Prints what a model is written with, so that nobody has to guess a class or an
argument name. Without a topic it lists the topics:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Topic
     - What it prints
   * - a model kind (``domain``, ``state_machine``, ...)
     - The package to import from, the classes it defines with their
       constructor arguments, types and defaults, the values it exports
       (``StringType``, ...), the editor diagram of the kind and the generators
       that read it. It is read from the metamodel when the command runs, so
       it is the API of the BESSER that is installed.
   * - ``project``
     - The ``Project`` class of a ``.py`` file and the JSON Schema of an editor
       project file, from the model the editor backend validates it with. The
       schema fixes the project's own keys; what a diagram's ``model`` holds
       is free-form there, and ``besser new model.json --kind KIND`` writes
       an example of each.
   * - ``result``
     - The JSON Schema of the object every command prints with ``--json``
       (see :ref:`cli-json`).
   * - ``config``
     - An example ``besser.toml`` (see :ref:`besser build <cli-build>`).

.. code-block:: console

   $ besser schema state_machine
   state_machine: a StateMachine, from besser.BUML.metamodel.state_machine
     A state machine model.
     a starter to edit: besser new model.py --kind state_machine
     in editor JSON: a StateMachineDiagram of the project ('besser schema project'); besser new model.json --kind state_machine writes one
     generators that read it: none

   Classes (from besser.BUML.metamodel.state_machine import ...), with their constructor arguments:
     ...
     Event(name: str)
         The representation of an event (i.e., external or internal stimuli or input) that may cause the transition of state in a state machine.
     State(sm: StateMachine, name: str, initial: bool = False, final: bool = False)
         A state of a state machine.
     StateMachine(name: str)
         A state machine model.

With ``--json``, a kind's ``description`` gives each class's ``name``,
``signature``, ``doc``, ``bases``, ``abstract`` and ``parameters`` (``name``,
``type``, ``default``, ``required``).

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

In a run of several files it is listed in its place among them
(``crash.py: does not load: line 2: ZeroDivisionError: division by zero``), and
the others are still validated.

The errors of an editor project itself, such as a diagram that references a
diagram the project does not hold, are listed under the file and fail the run
like a model's errors. (``besser info`` shows them as warnings: it summarises,
and does not fail on what ``validate`` checks.)

Several files are checked in one run, and a directory is searched for model
files, recursively and in sorted order:

.. code-block:: console

   $ besser validate models
   models/library.json: 1 model
     Library (DomainModel): ok
   models/nested/book.json: 1 model
     Library (DomainModel): ok
   skipped: models/package.json: not an editor export (no 'diagrams' or 'model' object)
   skipped: models/shop.py: a .py model is executed when loaded
   OK: 0 errors, 0 warnings
   note: 1 .py file skipped; pass --exec-py to load the .py files found in a directory

What a directory search checks:

* Its ``.json`` files that are editor exports. Other JSON files
  (``package.json``, ``tsconfig.json``) are skipped as not a model.
* Its ``.py`` files only with ``--exec-py``, since loading a ``.py`` model
  executes it. Without the flag they are listed as skipped and not run.
* A file found that loads and holds no model (a helper ``.py`` file, a project
  whose diagrams are all empty) is skipped.
* Not hidden directories (``.git``, ``.venv``, ...) nor ``venv``, ``env``,
  ``node_modules``, ``__pycache__`` and ``site-packages``, which hold
  third-party files.
* Not what a symbolic link leads to outside the directory. A link to a file
  inside it counts (once, with the file); a linked directory is not entered,
  and a link to nothing, or a link loop, is skipped.

A file you name on the command line is checked as named, whether it comes
before or after a directory that holds it: a ``.py`` file is loaded without
``--exec-py``, and a file that is not a model, or holds none, is an error, not
skipped. A file reached twice (named, and found in a directory, or in a
directory named twice) is checked, or listed as skipped, once.

A run with nothing to check, such as an empty directory, exits with status 2
and says what was looked for:

.. code-block:: console

   $ besser validate emptydir
   error: nothing to validate: no editor .json export found in emptydir (.py models in a directory are loaded with --exec-py)

The exit status is that of the worst file: 2 when one does not load (the
others are still validated), else 1 when one has errors (or warnings, with
``--strict``). With several files, or a directory, each message of ``errors``
and ``warnings`` starts with the file it is about. With ``--json``, the
command adds:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Value
   * - ``model_files``
     - The files checked, directories expanded (the skipped ones are not in it).
   * - ``models``
     - Per model of every file: ``file``, ``name``, ``kind``, ``type``,
       ``validated``, ``errors`` and ``warnings``.
   * - ``results``
     - Per file checked: ``file``, ``ok``, ``exit_code``, ``errors``,
       ``warnings`` (each starting with the model's name) and its ``models``.
       A file that does not load has exit code 2, its load error and no models.
   * - ``skipped``
     - Per file found and not checked: ``file`` and ``reason``.
   * - ``diagnostics``
     - Every error and warning of the run with where it is: ``severity``
       (``error`` or ``warning``), ``file``, ``model`` (absent for a file that
       does not load and for a project's own errors), ``message`` and, when it
       is known, ``line``.

``line`` is the line of a ``.py`` model file to edit: the line a file fails
at when it does not load, and otherwise the line where the element a message
names is defined (its ``name="..."`` argument), when exactly one line defines
it. The text report ends such a message with ``(model.py, line 39)``. An
editor JSON file has no lines: its messages name the element.

.. _cli-ocl:

OCL constraints
~~~~~~~~~~~~~~~

``--ocl`` also checks the OCL constraints of the domain models, as the editor's
validation does: every constraint (invariants, pre- and postconditions) is
parsed, and the invariants are evaluated on each object model of the same file
that instantiates the domain model (when no domain model of the file matches
its objects' classes, on the file's only domain model). A constraint that does
not parse is an error of the domain model, and so is one the editor JSON
converter dropped because it does not parse; an invariant an object model
violates is an error of the object model. They start with ``OCL constraint``
and the constraint:

.. code-block:: console

   $ besser validate books.py --ocl
   books.py: 2 models
     book_model (DomainModel): 1 error, 0 warnings
       error: OCL constraint [Book inv broken]: 'context Book inv broken: self.pages >' - line 1, column 37: Expression ends unexpectedly — the operand on the right is missing.
     shelf (ObjectModel): 1 error, 0 warnings
       error: OCL constraint [Book inv positive_pages]: 'context Book inv positive_pages: self.pages > 0' - Constraint violation: Evaluates to False
   FAILED: 2 errors, 0 warnings
   note: OCL: 2 constraints checked, invariants evaluated on 1 object model

The note counts the constraints checked, those that do not parse included, over
every file of the run. An invariant the evaluator cannot handle yet is reported
as a warning on the object model, not as a violation.

The evaluator works by building Python source from the constraint and the
object model's values and calling ``eval()`` on it, so an object model whose
values are crafted runs code when its invariants are evaluated. Parsing is
safe. The invariants are therefore evaluated only where running the file's code
was already allowed:

* on the object models of a ``.py`` model, which was executed to load it;
* on those of an editor ``.json`` export only with ``--exec-py``. Without it,
  the constraints are still parsed, and the note says that the invariants were
  not evaluated:

.. code-block:: console

   $ besser validate books.json --ocl
   books.json: 2 models
     Class_Diagram (DomainModel): ok
     Object_Diagram (ObjectModel): ok
   OK: 0 errors, 0 warnings
   note: OCL: 1 constraint checked; invariants not evaluated on 1 object model of editor JSON: the OCL evaluator runs Python built from the models' values; pass --exec-py to evaluate them

The check needs the B-OCL evaluator (``pip install "besser[ocl]"``) and the
packages that reading a ``.json`` model needs (see `Model files`_), even for a
``.py`` model. When they are missing the constraints are not checked, and a
warning says what to install: the run does not fail, except with ``--strict``.

The check is opt-in, since it needs those optional packages.

``besser info``
---------------

.. code-block:: console

   $ besser info library.py
   library.py: 1 model
     library_model (DomainModel 'LibraryModel'): 2 classes, 1 association, 1 constraint
       class Library: name: str
       class Book: title: str, pages: int, isbn: str? = 'none'
       association holds: library: Library [1] -- books: Book [0..*]
       constraint pages_positive on Book: context Book inv pages_positive: self.pages > 0
       backend routes per class X: POST|GET /x/, GET|PUT|DELETE /x/{x_id}/, /x/bulk|count|paginated|search/
   Generators that read this model: python, java, pydantic, test_case, django, spring, backend, sqlalchemy, sql, supabase, jsonschema, rdf, rest_api

Each model is listed under the name the other commands call it (the variable
it is bound to in a ``.py`` file, the diagram's title in an editor export),
with its class and, in quotes, its own name when that differs. Then come the
counts of what it holds (classes, objects, screens, states, ...) and, for a
domain model, its classes with their attributes (``?`` marks an optional one,
``= value`` its default), methods and parents, its enumerations, its
associations and its constraints, in the order they were defined, then the
routes the ``backend`` generator writes for each class (and for each method).
The last line names the generators whose models the file holds. It is the
model in about 1 KB: a coding agent reads it instead of the model file or the
generated code.

The model is not validated: a model with errors is summarised too, and the
errors of an editor project itself (a dangling reference between its diagrams)
are warnings here, so ``info`` exits with status 0 on them. ``besser validate``
is the command that fails on them. The model an editor project marks current is
shown ``(current in the editor)``, and a note says when the file holds several
models of a kind with none current (see :ref:`cli-several-diagrams`).
``besser info --json`` gives ``models`` (``name``, ``kind``, ``type``,
``model_name``, ``current``, ``counts`` and, for a domain model, ``classes``
(each attribute with ``name``, ``type``, ``optional`` and ``default``),
``enumerations``, ``associations``, ``constraints`` (``name``, ``context``,
``expression``) and ``backend_endpoints`` (class name -> ``"METHOD /path/"``
for create, read, update, delete and each method); for the other kinds ``elements``, the
names of what the model holds per collection, as in ``states: created, paid``)
and ``generators``.

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

.. _cli-generate:

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
#. records what it wrote in ``.besser_generated.json``, in that directory, when
   the generator works with ``besser regenerate`` (``besser generators`` says
   which do), and says so in a note;
#. prints the files the run created or rewrote. A generator that writes no
   file has failed (exit status 1): it logged what it could not generate,
   on standard error.

``-o`` names a directory: a file there, or at one of its parents, is refused
with exit status 2 before anything runs (``out is a file: -o takes a
directory``).

``generate`` overwrites what it generates, so it refuses (exit status 2) a
directory that is not empty, and nothing is written. ``--force`` writes into it
anyway, overwriting the files of the same names, and records what it wrote in
place of the old record. Files the generator does not produce are left alone.
When the directory holds files, the refusal also gives the ``besser regenerate
... --adopt`` command that records a directory generated without a record, so
that ``regenerate`` keeps its edits.

The record is what :ref:`besser regenerate <cli-regenerate>` merges against: to
update generated code after a model change and keep the edits made in it, run
``besser regenerate`` (or ``besser generate --keep-edits``, the same command)
on the directory. The note gives that command with the ``--option`` and
``--use`` of the run, quoted for the shell, so that it reads the same model
with the same options; when ``pygit2`` is not installed, it ends with
``(needs pip install "besser[regen]")``:

.. code-block:: console

   $ besser generate sqlalchemy library.py -o out --option dbms=postgresql
   note: recorded the output in out/.besser_generated.json: 'besser regenerate sqlalchemy library.py -o out --option dbms=postgresql' brings it up to a changed model and keeps your edits
   Generated 1 file in out with the 'sqlalchemy' generator:
     out/sql_alchemy.py

``--use`` is given for each kind the file holds several models of, or for
every kind when you gave ``--use``. Keep the record with the code (commit it).
Writing it needs no extra package; only the merge needs ``pygit2``.
``"recorded"`` in the JSON output says whether it was written. It is not
written for a generator that does not take part in ``regenerate``, nor, with a
warning, when the directory rules it out (a Spec-Driven Agent app).

``--keep-edits`` makes ``generate`` run ``besser regenerate`` with the same
arguments: the output, the exit status and the JSON (``"command":
"regenerate"``) are those of ``regenerate``. It cannot be combined with
``--force``, which overwrites edits (exit status 2).

**Dry run.** ``--dry-run`` runs the generator in a scratch folder and lists
what a real run would write, each file ``new``, ``changed`` (it differs from
the file in the directory) or ``unchanged``. Nothing is written, and the ``-o``
directory is not created. The record is bookkeeping, not output: it is not
listed, and a note says the run would write it:

.. code-block:: console

   $ besser generate python library.py -o out --dry-run
   note: would record the output in out/.besser_generated.json: 'besser regenerate python library.py -o out' would then bring it up to a changed model and keep your edits
   Dry run: nothing was written.
   The 'python' generator would write 1 file in out (1 new):
     new  out/classes.py

A dry run is refused where the run would be: ``-o`` naming a file, a
directory that is not empty without ``--force``, a model with validation
errors, an unknown option. With ``--json``, ``files`` is empty,
``would_write`` maps each path to its status (its paths are the ``files`` of
the real run), and ``recorded`` says whether the run would write the record.
For a generator whose output differs from run to run, a file can read as
``changed`` when the model is not, and a note says so.

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
An editor project whose diagrams reference elements or diagrams it does not
hold (a component realizing a class that is not in the project), the errors
``besser validate`` lists under the project, is exit status 1: ``nothing was
generated: the project's diagrams reference elements or diagrams it does not
hold``.

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
changes nothing and exits with status 2:

.. code-block:: console

   $ besser regenerate python library.py -o out
   error: out/classes.py still holds the conflict markers of the last regenerate: resolve them ('<<<<<<< your edit' ... '>>>>>>> generated from your model'), then run regenerate again.

**A directory written by** ``besser generate`` has its record, so
``regenerate`` merges into it from the first run. A directory generated without
one (by an older BESSER, or by other means) cannot have its edits told from
generated code, and ``regenerate`` refuses it (status 2):

.. code-block:: console

   $ besser regenerate python library.py -o out
   error: out holds generated files (classes.py) but no .besser_generated.json, so your edits cannot be told from generated code. Record it once with the model those files came from: 'besser regenerate python <that model> -o out --adopt', then regenerate with the changed model.

Adopt it once with ``--adopt`` and the model it was generated from, before you
change the model. This changes no file:

.. code-block:: console

   $ besser regenerate python library.py -o out --adopt
   Recorded out as the 'python' generator's output for library.py; no file was changed. 'besser regenerate' now keeps the edits made in it.

Files that differ from that model's output are listed in a warning and count
as your edits from then on. ``--adopt`` on a directory that has its record is
refused: ``out already has .besser_generated.json: there is nothing to adopt;
run it without --adopt``.

**Dry run.** ``--dry-run`` runs the same merge on a scratch copy of the
directory and reports what the run would do, with the same lists and exit
status (1 when the merge would write conflict markers), and its warnings in
the future tense, starting with ``dry run:``. The directory is left byte for
byte as it was. With ``--json``, ``files`` is empty and ``would_write`` maps
each file the run would change to ``written``, ``merged``, ``conflicted`` or
``deleted``, the key every dry run has:

.. code-block:: console

   $ besser regenerate python library.py -o out --dry-run
   Dry run: nothing was written.
   Regenerating out with the 'python' generator would change 1 file (1 merged):
     merged  out/classes.py
   warning: dry run: the model's changes would be merged into your edits; a clean merge can still break, so run your tests: out/classes.py

``generate --keep-edits`` is this command; see :ref:`besser generate <cli-generate>`.

The record names the BESSER version that wrote it (``besser_version``). When
another version regenerates the directory, a note says so (``out was generated
with BESSER 8.9.2 and this is BESSER 8.9.3``): a generator can change between
versions, and a file then changes without the model having changed.

``regenerate`` takes the same ``--option`` and ``--use`` as ``generate``. Give
the same options on every run: a different option changes the output like a
model change does.

The command is refused with status 2, and nothing is written, when:

* the generator does not take part (``besser generators`` lists those that
  do): its output differs from run to run for one model. Supabase, Flutter,
  Terraform, the agent generator and Docker Compose do not;
* ``pygit2`` is not installed: the error gives the ``pip install`` line above;
* ``-o`` names a file, or a path under one (``out is a file: -o takes a
  directory``);
* the directory holds the output of another generator, a Spec-Driven Agent
  app, or conflict markers of the last run.

``besser generate`` and a regenerated directory: ``generate`` refuses a
directory holding ``.besser_generated.json`` and prints the ``regenerate``
command to run. With ``--force`` it overwrites the hand edits and writes a
fresh record of what it wrote.

.. _cli-build:

``besser build``
----------------

A project usually has several outputs, each with its generator, directory and
options. ``besser.toml`` lists them once, and ``besser build`` generates them
all (``besser schema config`` prints this example):

.. code-block:: toml

   # besser.toml: what 'besser build' generates. Paths are relative to this file.

   model = "model.py"              # the model file every output reads, unless it names its own
   test = "python -m pytest -q"    # optional: what 'besser apply' runs after the build (no shell)

   [output.api]                    # one table per output; 'besser build api' builds this one alone
   generator = "backend"           # a generator id ('besser generators' lists them)
   dir = "generated/api"           # where its code goes
   options = { http_methods = ["GET", "POST"] }   # the generator's options, typed

   [output.db]
   generator = "sqlalchemy"
   dir = "generated/db"
   model = "other_model.py"        # optional: this output's own model file
   use = ["library_model"]         # optional: the model to read when the file holds several of a kind
   options = { dbms = "postgresql" }
   overwrite = false               # optional: true regenerates from scratch, discarding edits made in dir

.. code-block:: console

   $ besser build
   [api] backend: model.py -> generated/api: ok
     Generated 11 files in generated/api with the 'backend' generator:
       ...
   [db] sqlalchemy: model.py -> generated/db: ok
     Generated 1 file in generated/db with the 'sqlalchemy' generator:
       generated/db/sql_alchemy.py
   OK: 2 outputs, 0 failed

Each output is built as the two commands above build it:

* into a missing or empty directory, as :ref:`besser generate <cli-generate>`;
* into a directory that holds something, as :ref:`besser regenerate
  <cli-regenerate>`: the model's changes are merged into the edits made there;
* with ``overwrite = true``, as ``besser generate --force``, every time. A
  generator that does not support ``regenerate`` needs it for a second build,
  and the error says so.

``besser build api db`` builds the outputs named; ``-c FILE`` reads another
file than ``besser.toml`` in the current directory; ``--dry-run`` writes
nothing and reports what each output would write. An output that fails does
not stop the others, and the exit status is the worst output's. An unknown
key, a value of the wrong type or a missing ``generator``, ``dir`` or model is
exit status 2, naming the file and the table. With ``--json``, ``outputs``
holds, per output, its ``name`` and the whole object of the ``generate`` or
``regenerate`` run that built it; ``errors``, ``warnings`` and ``notes`` repeat
their messages, each starting with the output's name, and ``files`` lists all
the files written. When the project's ``.ignore`` has the block
:ref:`besser init <cli-init>` writes, ``build`` updates it to the outputs of the
file (and lists it in ``files``).

Every file a generator writes for ``build``, ``generate`` and ``regenerate``
starts with a comment that says it is generated and where hand-written code goes
instead, for the languages whose comments BESSER knows (Python, TypeScript,
JavaScript, Java, SQL, CSS, YAML, ...; not JSON or plain text):

.. code-block:: python

   # Generated by BESSER: edit the model, not this file. Custom code goes in extensions.py.

``.besser_generated.json`` in the output directory lists every generated file.
The ``backend`` generator (and the backend of ``web_app``) loads
``extensions.py`` from beside ``main_api.py``: see
:doc:`generators/backend`. For the other generators, hand-written code lives
outside the output directory and imports the generated code. Either way it
survives every ``build``. A directory generated before the header existed
adopts (``regenerate --adopt``) without reading the header as an edit.

.. _cli-apply:

``besser apply``
----------------

.. code-block:: console

   $ besser apply
   OK: 1 output, 0 failed: [backend] backend is up to date with model.py: no file changed; tests: 12 passed in 0.41s

The one command to run after a model change: ``besser build`` for every output
(each model is validated first, and nothing is generated from a model with
errors), then, when the build succeeded, the ``test`` command line of
``besser.toml``, run from its folder without a shell. It prints one line when
everything passes; when the tests fail, the last 40 lines of their output, and
exit status 1. Without a ``test`` key it builds and notes that no test ran;
``-c FILE`` reads another configuration file. ``--json`` is the object of
``build`` with ``command`` ``apply`` and ``test``: its ``command``,
``exit_code`` and ``output_tail``, or ``null`` when no test ran.

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
holds. ``-o`` names a file: a directory there, or a file at one of its parents,
is exit status 2 (``out/library.json cannot be created: out is a file``).

**Importing another notation.** The same command turns a file that is not
B-UML into a model, written as ``.py`` or ``.json``. The notation follows the
input's extension, or ``--from`` when the extension does not say. These files
are read as data by BESSER's importers: nothing in them is executed, and no
import calls a language model or the network.

.. list-table::
   :header-rows: 1
   :widths: 14 26 60

   * - ``--from``
     - Extensions
     - What is read
   * - ``plantuml``
     - ``.plantuml``, ``.puml``, ``.pu``
     - A :doc:`PlantUML class diagram <buml_language/model_building/plantuml_structural>`,
       as a domain model. A syntax error is exit status 2 with its line and column.
   * - ``drawio``
     - ``.drawio``
     - A draw.io (diagrams.net) UML class diagram, as a domain model.
   * - ``csv``
     - ``.csv``
     - CSV files with a header row, as a domain model: a class per file, its
       columns as attributes, and an association where a column names another
       file's id (``author_id``). The only format that takes several inputs.
   * - ``rdf``
     - ``.ttl``, ``.owl``, ``.rdf``, ``.nt``, ``.n3``
     - An OWL / RDF / SHACL file, as a knowledge graph.
   * - ``sdmx``
     - (name it)
     - An SDMX-ML 2.1 structure message, a downloaded ``.xml`` file, as a
       domain model.
   * - ``code``
     - (name it; a directory)
     - Python code: its SQLAlchemy models, Django models, Pydantic models and
       dataclasses, as a domain model. The sources are parsed, not run.

.. code-block:: console

   $ besser convert library.plantuml -o library.py
   Converted library.plantuml (1 model) to library.py
   $ besser convert author.csv book.csv -o books.py
   $ besser convert src/ -o model.py --from code
   $ besser convert ontology.ttl -o graph.py                     # a knowledge graph
   $ besser convert ontology.ttl -o model.py --class-diagram     # its class diagram
   $ besser convert graph.py -o graph.ttl                        # back to RDF

``--class-diagram`` writes, for each knowledge graph of the input (imported, or
of a ``.py`` or ``.json`` model file), the class diagram derived from it:
classes, attributes, associations and OCL constraints. A knowledge graph is
also written as RDF: ``-o`` with ``.ttl`` (Turtle), ``.owl`` or ``.rdf``
(RDF/XML). An import is not validated: run ``besser validate`` on the result.

``--dry-run`` converts into a scratch file, so it fails and warns as the real
conversion would, and says whether the output would be created, overwritten,
or (for a ``.py`` output) left as it is because it holds the same; a ``.json``
output gets new ids on every conversion, so it is always ``changed``. Nothing
is written (``would_write`` in the JSON output):

.. code-block:: console

   $ besser convert library.py -o library.json --dry-run
   Dry run: nothing was written.
   Converting library.py (1 model) would create library.json

Both directions reuse the editor's converters, so they share their limits:

* The editor has no diagram for a Terraform ``deployment`` model or a feature
  model; converting to JSON leaves them out and says so in a warning.
* A lone model takes the default title of its diagram (a ``DomainModel`` named
  ``Library`` comes back as ``Class_Diagram``); its classes, attributes and
  associations keep their names.
* A model written in Python has no layout: the converter computes the
  positions of its elements.
* What a diagram cannot draw is not in the JSON: a state machine transition's
  condition (``with_condition``), a quantum circuit's classical registers, an
  agent's configuration properties. Keep the ``.py`` file as the source when a
  model uses them.

``besser completion``
---------------------

Prints a completion script for bash, zsh or fish on standard output. Load it
from the shell's start-up file:

.. code-block:: console

   $ eval "$(besser completion bash)"     # in ~/.bashrc
   $ eval "$(besser completion zsh)"      # in ~/.zshrc, after compinit
   $ besser completion fish > ~/.config/fish/completions/besser.fish

The script completes the command names (zsh and fish show each one's help),
every option of the command being typed, the values of an option with fixed
choices, the generator ids after ``generate`` and ``regenerate``, model files
(``.py`` and ``.json`` first, and directories), a directory after ``-o`` of
``generate`` and ``regenerate`` and a file after ``-o`` of ``convert``.

It is generated from the command's own argument parser, so it follows the
installed version of BESSER. The generator ids are written into the script:
print it again after installing a version that adds a generator. With
``--json`` the object holds ``shell`` and ``script``, the text printed without
it.

.. _cli-json:

JSON output and exit status
---------------------------

With ``--json``, standard output is exactly one JSON object. ``besser schema
result`` prints its JSON Schema, and the test suite checks every command's
output against it. These keys are on every command and are stable:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Value
   * - ``ok``
     - ``true`` when the exit status is 0.
   * - ``command``
     - The command that ran, ``version`` for ``--version``. A usage error names
       the command it is about; ``command`` is empty only when the arguments
       did not parse and name no command.
   * - ``exit_code``
     - The exit status: 0, 1 or 2 (see below).
   * - ``errors``
     - List of messages. For ``validate``, each starts with the model's name
       (after the file's, when several files are given).
   * - ``warnings``
     - List of messages; they never make ``ok`` false, except with ``--strict``.
   * - ``notes``
     - List of messages about what the command chose on its own: which model of
       several it read, that it recorded its output, which ``.py`` files a
       directory search skipped. Without ``--json`` they are ``note:`` lines on
       standard error; ``--quiet`` drops them.
   * - ``files``
     - The files the command wrote, as paths relative to the working directory
       (absolute when the path you gave was), under ``-o`` for ``generate`` and
       ``regenerate``. For ``regenerate``: the written, merged and conflicted
       files. Always empty with ``--dry-run``.

Each command adds its own:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Keys
   * - ``validate``
     - ``model_files``: the files checked, directories expanded; ``results``:
       per file its ``file``, ``ok``, ``exit_code``, ``errors``, ``warnings``
       and ``models``; ``models``: every file's models in one list, per model
       its ``file``, ``name``, ``kind``, ``type``, ``validated``, ``errors``
       and ``warnings``; ``skipped``: per file a directory search left out,
       its ``file`` and ``reason``; ``diagnostics``: every error and warning
       with its ``severity``, ``file``, ``model``, ``message`` and ``line``.
   * - ``info``
     - ``model``, ``models`` (each with ``current``: the diagram the editor
       project marks as current; for a domain model also ``constraints`` and
       ``backend_endpoints``) and ``generators``.
   * - ``generators``
     - ``generators``: per generator its ``id``, ``summary``, ``category``,
       ``models``, ``options`` and ``supports_regenerate``.
   * - ``generate``
     - ``generator``, ``model``, ``output_dir``, ``dry_run`` and ``recorded``
       (whether ``.besser_generated.json`` was written), ``used`` (model kind ->
       the name of the model read); with ``--dry-run``, ``would_write``: path ->
       ``new``, ``changed`` or ``unchanged``, the record left out (``recorded``
       then says whether the run would write it).
   * - ``regenerate``
     - ``generator``, ``model``, ``output_dir``, ``dry_run``, ``used``,
       ``adopted``, and one list of paths per outcome:
       ``written``, ``merged``, ``conflicted``, ``deleted``,
       ``kept_removed_from_model``, ``not_restored``, ``not_merged`` and
       ``differs_from_model`` (the fields of ``RegenerationReport``); with
       ``--dry-run``, ``would_write``: path -> ``written``, ``merged``,
       ``conflicted`` or ``deleted``.
   * - ``new``
     - ``model``, the file written, and its ``kind``; with ``--list``,
       ``kinds``: per kind its ``kind``, ``summary``, ``generators`` and
       ``editor_diagram``.
   * - ``schema``
     - ``topic``; ``topics`` without one; ``description`` for a model kind
       (and ``project``); ``schema`` for ``project`` and ``result``;
       ``example`` for ``config``.
   * - ``build``
     - ``config``, ``dry_run`` and ``outputs``: per output its ``name`` and
       the object of the run that built it.
   * - ``apply``
     - The keys of ``build``, and ``test``: ``command``, ``exit_code`` and
       ``output_tail`` (the last lines of its output), or ``null``.
   * - ``init``
     - ``folder`` and ``kind``.
   * - ``guide``
     - ``guide``: the text ``AGENTS.md`` gets.
   * - ``convert``
     - ``input`` (the first of ``inputs``), ``inputs``, ``output``,
       ``dry_run`` and ``models`` (per model written its ``name``, ``kind``
       and ``type``); with ``--dry-run``, ``would_write``, as for ``generate``.
   * - ``completion``
     - ``shell`` and ``script``.
   * - ``--version``
     - ``version`` (``command`` is ``version``).
   * - ``mcp``
     - None. Standard output is the protocol while it serves, and nothing is
       printed after the session ends. A failure before serving (the MCP SDK
       is not installed, ``--root`` is not a directory) is the usual
       ``"ok": false`` object, exit status 2.

``generate --keep-edits`` prints what ``regenerate`` prints, ``command``
included. A model file is ``model`` on every command that reads one, except
``model_files`` (a list) on ``validate``, which reads several, and ``input``
and ``output`` on ``convert``.

.. code-block:: console

   $ besser validate library.py --json
   {
     "ok": true,
     "command": "validate",
     "exit_code": 0,
     "errors": [],
     "warnings": [],
     "notes": [],
     "files": [],
     "model_files": ["library.py"],
     "models": [
       {"file": "library.py", "name": "library_model", "kind": "domain", "type": "DomainModel",
        "validated": true, "errors": [], "warnings": []}
     ],
     "results": [
       {"file": "library.py", "ok": true, "exit_code": 0, "errors": [], "warnings": [],
        "models": [{"file": "library.py", "name": "library_model", "...": "as in models"}]}
     ],
     "skipped": []
   }

(Shortened: the command prints one key per line.)

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

.. _cli-mcp:

``besser mcp``
--------------

``besser mcp`` runs a `Model Context Protocol <https://modelcontextprotocol.io>`_
server on standard input and output, so an MCP client (an AI coding agent)
calls the commands as tools instead of running them in a shell. It needs the
MCP SDK:

.. code-block:: console

   $ pip install "besser[mcp]"
   $ besser mcp --root .

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Option
     - Effect
   * - ``--root DIR``
     - The directory the tools read and write (default: the current directory).
   * - ``--exec-py``
     - Lets the tools load B-UML ``.py`` model files, which executes them, and
       lets ``validate`` with ``ocl`` evaluate the invariants of editor JSON (see
       :ref:`OCL constraints <cli-ocl>`). Without it the tools load editor
       ``.json`` exports only and only parse their OCL constraints.

The tools are ``new``, ``schema``, ``validate``, ``info``, ``generators``,
``generate``, ``regenerate`` and ``convert`` (``besser build`` is not one: a
client passes each output's arguments itself). Each takes the command's arguments by name
(``generate``: ``generator``, ``model``, ``output_dir``, and optionally
``options``, ``use``, ``force``, ``dry_run``; ``validate`` takes ``paths``, a
list, where the others take one ``model``) and returns the command's
:ref:`--json object <cli-json>`, as structured content and as JSON text; the
result is flagged as an error when ``ok`` is false. ``options`` is an object,
``{"dbms": "postgresql"}``, or a list of ``"name=value"`` strings as
``--option`` takes them. ``tools/list`` gives each tool's input schema.

The object is worded for the client: its paths (``files``, ``model``,
``output_dir``, ``model_files``, the keys of ``would_write`` and the paths in
messages) are relative to ``--root``, as the client gives them, and a message
that names a command-line flag names the tool argument instead (``force:
true`` for ``--force``, the ``use`` argument for ``--use NAME``, the
``regenerate`` tool with its arguments as JSON for ``'besser regenerate ...'``).
A symbolic link loop in a folder is skipped, as the command skips it.

The server confines the tools:

* every path is resolved, ``..`` and symbolic links included, and refused
  unless it is inside ``--root``; so is a folder holding a symbolic link that
  leads out of it;
* ``-`` (standard input) is not a model file: standard input is the protocol;
* a ``.py`` model is refused unless the server was started with ``--exec-py``;
* ``generate`` refuses a non-empty ``output_dir`` unless ``force`` is true,
  as the command does.

.. warning::

   With ``--exec-py``, ``--root`` is not a security boundary. A ``.py`` model
   runs inside the server process, with your permissions, and can read, write
   or run anything you can; the confinement above only covers the paths the
   tools are given. Start the server with ``--exec-py`` only on model files you
   trust.

Everything a model file or a generator prints goes to the server's standard
error. Tools run one at a time. To add the server to Claude Code:

.. code-block:: console

   $ claude mcp add besser -- besser mcp --root .

Other clients take the same command in their MCP configuration, for example:

.. code-block:: json

   {"mcpServers": {"besser": {"command": "besser", "args": ["mcp", "--root", "."]}}}

What the command leaves out
---------------------------

The command covers every model kind of B-UML, every registered generator and
every importer that reads a file as data. It leaves out, on purpose:

* everything that calls a language model: the :doc:`Spec-Driven Agent
  <spec_driven_agent/index>`, image and mockup to model, the LLM-assisted knowledge
  graph clean-up. A coding agent that drives the command is the model writer;
* the textual grammars whose reader builds Python code from the file and
  executes it (the :doc:`deployment <buml_language/model_building/deployment_grammar>`
  and :doc:`neural network <buml_language/model_building/nn_grammar>` grammars):
  write those models in B-UML (``besser new --kind deployment``, ``--kind nn``);
* fetching an SDMX structure from a provider's API (``convert --from sdmx``
  reads a downloaded file) and XLSX workbooks (export the sheet as CSV);
* what the editor's server does for a signed-in user: deploying to GitHub or
  Render, rendering a diagram as SVG.

.. _cli-agents:

Using BESSER from a coding agent
--------------------------------

The command is built for AI coding agents (Claude Code, Cursor, Codex, ...)
as much as for people and CI jobs. The agent is the modeller: it writes the
B-UML model itself, and the command is its toolbox around the model. There is
no command that calls a language model. Every command asks nothing, reads
standard input only for ``-``, exits with 0, 1 or 2 with the same meaning
everywhere, names the file, the element and the line of an error, and takes
``--json`` (:ref:`one documented schema <cli-json>`).

What costs an agent is not the command but what it reads: every file, help
text and command output stays in its context and is read again on each later
turn of the session. In an internal experiment, an agent (Claude Opus) built a
small application (7 entities, a FastAPI backend and a React frontend) in 11
steps, an initial build and 10 change requests, once by hand and once with
BESSER. Following the earlier guidance of this page (``validate --json`` after
each edit, then ``build --json``; learn the metamodel from ``besser schema``
and ``besser new``; edit the generated code), the BESSER run used **7.13M
tokens, 1.69 times the 4.21M of hand-coding**: most of the difference went to
reading generated code back, BESSER's documentation and sources, and the model
file. Following the rules below, it used **3.27M tokens, 22% fewer than
hand-coding**, for the same quality. One run per arm, on one
small application: read the numbers as an indication, not a benchmark.

The rules are what ``besser init`` (and ``besser new``) write to ``AGENTS.md``,
and what ``besser guide`` prints:

1. **Generated code is a black box.** The agent does not open, search or edit
   the generated folders, nor ``.besser_generated.json``. ``.ignore`` keeps them
   out of ripgrep's searches, and every generated file says in its first line
   that it is generated. What the generated ``backend`` does (its routes, the
   request bodies, which OCL constraints it enforces) is in the guide.
2. **No documentation, sources or help.** The guide is the whole of what the
   agent needs; reading this page or BESSER's code costs more than the guide.
3. **``besser info model.py`` instead of the model file**: the classes,
   attributes (optional, defaults), enumerations, associations, constraints and
   the API's routes in about 1 KB. To change the model, the agent greps the
   lines it needs and edits those.
4. **One command per model change: ``besser apply``.** It validates, builds
   every output of ``besser.toml`` and runs its ``test`` command, and prints
   one line when all of it passes. No separate ``validate``, no ``--json``;
   output is :ref:`terse <cli-terse>` off a terminal anyway.
5. **Hand-written code never goes into a generated file**: the backend's goes
   in ``extensions.py``, which ``main_api.py`` loads (see
   :doc:`generators/backend`); anything else lives outside the generated
   folders and imports them. A build then never meets an edit, so it never
   conflicts, and ``--force`` or ``overwrite = true`` are never needed.
6. **Trust the build**: check behaviour with tests, not by reading the output.

.. code-block:: console

   $ besser init                    # model.py, besser.toml, AGENTS.md, CLAUDE.md, .ignore
   $ # edit model.py; fill in 'test' in besser.toml
   $ besser apply
   OK: 1 output, 0 failed: [backend] Generated 10 files in backend with the 'backend' generator; tests: 4 passed in 0.52s
   $ besser info model.py           # the model and its routes, when the agent needs them
   $ # write extensions.py and its tests; after each model change:
   $ besser apply

The other commands remain for what the loop does not cover: ``--dry-run``
before a risky change, ``regenerate`` to keep edits made in generated code
when there are some, ``besser schema KIND`` and ``besser new --kind KIND``
the first time the agent writes a kind of model, ``convert`` to hand the model
to the editor, and ``besser validate -`` to check a model from standard input
before it is written to a file. An agent with MCP support can use
:ref:`besser mcp <cli-mcp>` instead of a shell.

In CI, ``besser validate models/ --strict`` fails the job on any error or
warning in the editor exports under ``models/`` (add ``--exec-py`` for the
``.py`` models there, and ``--ocl`` for their OCL constraints).

The `BESSER Skills <https://github.com/BESSER-PEARL/besser-skills>`_ teach an
agent to write B-UML; the command gives it a way to check and run what it
wrote.
