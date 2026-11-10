Regenerating After a Model Change
=================================

``generate()`` writes a generator's output and overwrites what was there, so hand
edits in the output are lost when you run it again. ``regenerate()`` brings the folder
up to your changed model and keeps those edits:

.. code-block:: python

    from besser.generators.java_classes import JavaGenerator

    JavaGenerator(library_model, output_dir="app").regenerate()   # the first run

    # ... edit the files in app/, change the model ...

    report = JavaGenerator(new_library_model, output_dir="app").regenerate()
    print(report.merged, report.conflicted)

It needs an explicit ``output_dir``, which also holds the record, and the optional merge library:

.. code-block:: console

    python -m pip install "besser[regen]"

``generate()`` is unchanged, and you never have to use ``regenerate()``.

From a terminal, :ref:`besser regenerate <cli-regenerate>` does the same for a model file:

.. code-block:: console

    besser regenerate java library.py -o app

To generate and start the record without pygit2, call ``record_output`` instead of
``generate()``; ``besser generate`` does this for every generator that supports
``regenerate()``, so a later ``regenerate()`` merges into the folder without adopting it:

.. code-block:: python

    from besser.generators.regeneration import record_output

    record_output(JavaGenerator(library_model, output_dir="app"))   # needs no pygit2

How it works
------------

``regenerate()`` runs the generator in a scratch folder, then merges three versions of
each file: what it generated last time, the file as it is now, and the new output.
What it generated last time is recorded in ``.besser_generated.json`` in the output
folder, so keep that file with your code. The record also names the BESSER version that
wrote it (``besser_version``): a generator can change between versions, so the ``besser``
command notes when another version regenerates the folder. Files the generator does not
produce are never read or changed.

Every edit is kept or flagged, never overwritten:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - The file
     - What happens, and where the report says so
   * - Nobody edited it
     - It takes the new output (``written``).
   * - You edited it, and the model changes other lines
     - Both are kept (``merged``).
   * - You edited it, and the model changes the same lines
     - Conflict markers go in the file, around those lines only (``conflicted``).
   * - You edited it, and the model no longer produces it
     - It stays (``kept_removed_from_model``).
   * - Nobody edited it, and the model no longer produces it
     - It is removed (``deleted``).
   * - You deleted it, and the model changed it
     - It stays deleted (``not_restored``).
   * - It holds text that is not UTF-8, or binary data, and clashes with the model
     - It stays as you left it (``not_merged``).

Files that are not text, such as images, are written once, and never merged or removed: one the
model drops stays, and is listed in ``kept_removed_from_model``. Each file is replaced in one step,
so a full disk cannot leave half of it behind; run ``regenerate()`` again afterwards.

.. warning::

   A clean merge can still break your code. An edit that uses an attribute the model
   just removed merges without a conflict and then fails at runtime. Run your tests
   after a regeneration.

Resolving conflicts
-------------------

A conflict looks like one from ``git merge``, with ``<<<<<<< your edit`` and ``>>>>>>> generated
from your model`` around the lines. Edit the file so the markers are gone. Until then the next
``regenerate()`` raises ``RegenerationError`` and changes nothing. It and the report are in
``besser.generators.regeneration`` (:doc:`API <../api/generators/api_regeneration>`).

Folders generated earlier
-------------------------

A folder holding generated files but no record is refused, because edits cannot be told from
generated code. Adopt it once, using the model those files came from. This changes no file, and
``report.differs_from_model`` lists what differs from that model: those differences count as
your edits from then on, so a long list usually means it was the wrong model.

.. code-block:: python

    JavaGenerator(library_model, output_dir="app").regenerate(adopt=True)

    JavaGenerator(new_library_model, output_dir="app").regenerate()

On the command line this is ``besser regenerate java library.py -o app --adopt``.

Limits
------

- **Renamed classes are not followed.** The old file stays, with your edits, and the new
  one appears beside it. The report lists both, as ``kept_removed_from_model`` and
  ``written``. Move your edits across by hand. A rename that changes only the case of a
  file name is followed.
- **One folder per generator.** A folder recorded by another generator is refused, because
  merging into it would delete that generator's files.
- **Spec-Driven Agent apps are refused.** They keep their own record of who wrote each
  line, so change their model through a :doc:`modify run <../spec_driven_agent/regeneration>`.
- **Some generators do not take part**, because their output is not the same on every
  run for one model: Supabase (a timestamp in the file name), Flutter, Terraform, the BESSER
  Agentic Framework generator and Docker Compose. Neither does a ``BackendGenerator`` set to push
  a Docker image, which would not hold your edits, or a ``WebAppGenerator`` with a
  personalized agent, which calls an LLM. Calling ``regenerate()`` on these raises
  ``RegenerationError``.
- **Options go on the generator.** ``regenerate()`` runs ``generate()`` with no arguments, so
  give ``SQLAlchemyGenerator`` its ``dbms`` and ``ocl_check_constraints`` when you create it.
- **Line endings.** A rewritten file keeps CRLF only if every line used it; a file with mixed
  line endings comes back with LF.
- **Editor downloads have no record.** The web editor's ZIP does not include one. See below
  for keeping edits across downloads with git.

Keeping edits across editor downloads
-------------------------------------

Commit each download to its own branch, and work on another. Unpack the first download into an
empty folder, then:

.. code-block:: console

    git init -b generated
    git add -A
    git commit -m "Generated code"
    git switch -c main

Work on ``main`` as usual. When you download a new version, replace the contents of the
``generated`` branch with it, so that files of removed classes go away, and merge it:

.. code-block:: console

    git switch generated
    git rm -rqf .
    # unpack the new download here
    git add -A
    git commit -m "Generated code after the model change"
    git switch main
    git merge generated

git keeps your edits and marks the conflicts.

Your own generator
------------------

A generator takes part by setting ``regenerable = True``, once its output is
byte-identical on every run for one model. Add it to
``tests/generators/regenerable_builders.py``; ``tests/generators/test_regenerable_generators.py``
then checks this across hash seeds. If one value has to differ per run, override
``_keep_generated_values`` to carry it over, as ``DjangoGenerator`` does for ``SECRET_KEY``. If that
value is a secret, also override ``_without_secrets`` to keep it out of the record.
