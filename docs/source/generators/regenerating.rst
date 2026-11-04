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

It needs the optional merge library:

.. code-block:: console

    python -m pip install "besser[regen]"

``generate()`` is unchanged, and you never have to use ``regenerate()``.

How it works
------------

``regenerate()`` runs the generator in a scratch folder, then merges three versions of
each file: what it generated last time, the file as it is now, and the new output.
What it generated last time is recorded in ``.besser_generated.json`` in the output
folder, so keep that file with your code. Files the generator does not produce are
never read or changed.

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
   * - You edited it into text that is not UTF-8, and it clashes with the model
     - It stays as you left it (``not_merged``).

Files that are not text, such as images, are written once and never merged.

.. warning::

   A clean merge can still break your code. An edit that uses an attribute the model
   just removed merges without a conflict and then fails at runtime. Run your tests
   after a regeneration.

Resolving conflicts
-------------------

A conflict looks like one from ``git merge``. Edit the file so the markers are gone. Until
then the next ``regenerate()`` raises ``RegenerationError`` and changes nothing.

Folders generated earlier
-------------------------

A folder that holds files but no record is refused, because edits cannot be told from
generated code. Adopt it once, using the model those files came from. This changes no file:

.. code-block:: python

    JavaGenerator(library_model, output_dir="app").regenerate(adopt=True)

    JavaGenerator(new_library_model, output_dir="app").regenerate()

Limits
------

- **Renamed classes are not followed.** The old file stays, with your edits, and the new
  one appears beside it. The report lists both, as ``kept_removed_from_model`` and
  ``written``. Move your edits across by hand.
- **Spec-Driven Agent apps are refused.** They keep their own record of who wrote each
  line, so change their model through a :doc:`modify run <../spec_driven_agent/regeneration>`.
- **Some generators do not take part**, because their output is not the same on every
  run for one model: Supabase (a timestamp in the file name), Flutter, the BESSER
  Agentic Framework generator and Docker Compose. Calling ``regenerate()`` on one raises
  ``RegenerationError``.
- **Editor downloads have no record.** The web editor's ZIP does not include one. To keep
  edits across downloads, commit each generated version to its own git branch, merge that
  branch into your working branch, and let git find the conflicts.

Your own generator
------------------

A generator takes part by setting ``regenerable = True``, once its output is
byte-identical on every run for one model. Add it to
``tests/generators/regenerable_builders.py``, which checks this across hash seeds. If one
value has to differ per run, override ``_keep_generated_values`` to carry it over, as
``DjangoGenerator`` does for ``SECRET_KEY``.
