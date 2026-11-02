Runs, budgets and modify mode
=============================

Budgets and caps
----------------

Every run is bounded by three soft caps, each clamped to a server-side hard cap
so a client can never exceed them. All six values are overridable per
deployment through ``BESSER_LLM_*`` environment variables (see
:doc:`configuration`); the figures below are the shipped defaults, and the live
values are exposed by ``GET /besser_api/spec-driven/config``.

.. list-table::
   :header-rows: 1
   :widths: 34 22 22

   * - Cap
     - Default
     - Hard limit
   * - ``max_cost_usd``
     - 5.0
     - 5.0
   * - ``max_runtime_seconds``
     - 2400
     - 2400
   * - ``max_turns``
     - 120
     - 150

When a run hits a cap it still returns whatever it has produced so far — the
cap is surfaced as a non-terminal warning before the final result, not as a
hard failure.

The caps are checked at turn boundaries: cancellation first, then elapsed time,
then spend — and the cost check runs *before* the next request is sent, so a
run cannot overshoot its cap by one more billable call. A single warning is
logged at 80% of the cost cap.

Only the paid LLM calls count against the budget: the planning pass, the
customize loop, and any Phase 3 fix turns. The deterministic scaffold, the
validators, and the download are free.

.. _spec-driven-durable-runs:

Durable runs
------------

A run is owned by the server, not by the browser connection. A background task
consumes the run to completion; every SSE frame is assigned a monotonically
increasing sequence number and stored before any subscriber sees it. That makes
runs survive a dropped connection:

- ``GET /besser_api/spec-driven/runs/{run_id}`` returns the run's lifecycle
  metadata (never the request or the API key).
- ``GET /besser_api/spec-driven/runs/{run_id}/events?after=N`` replays every
  frame after sequence ``N`` and then follows the live producer. Native
  ``EventSource`` clients can use the ``Last-Event-ID`` header instead; the
  larger of the two cursors wins, so a stale query value can't replay
  duplicates. A resumed run drops the previous attempt's frames but keeps
  numbering from where it stopped, so a cursor from that attempt replays the
  whole new attempt.

.. code-block:: bash

   # What happened to this run?
   curl https://<host>/besser_api/spec-driven/runs/<runId>

   # Replay everything after sequence 42, then follow the live stream
   curl -N -H 'Accept: text/event-stream' \
     'https://<host>/besser_api/spec-driven/runs/<runId>/events?after=42'

Losing *every* subscriber starts a grace period (default 5 minutes) before the
producer is cooperatively stopped — which bounds unattended BYOK spend. A
server restart marks still-running records ``interrupted``;
``POST /besser_api/spec-driven/resume/{run_id}`` can then continue them from
their checkpoint when the client re-supplies the request body. Resume relies on
the checkpoint written after every tool-use turn (see :doc:`how_it_works`): it
refuses to resume against a project whose fingerprint has changed, and seeds
the already-spent cost so a crash-and-resume cycle cannot double the bill.

.. _spec-driven-modify-mode:

Modifying a previous run
------------------------

A run can also be **incremental**. With ``mode: "modify"`` and a
``base_run_id`` naming a previous, still-downloadable run, the new run's
workspace is seeded from that run's generated files and edited in place instead
of being rebuilt from scratch. Phase 2 is never skipped for a modify run — you
asked for a change, so a change is made. When the base has expired the runner
warns and falls back to normal from-scratch generation rather than failing the
request.

Every run records the class diagram it was given in the run's
``.besser_recipe.json`` as ``model.user_model``: classes (abstract, parents,
attributes with their type and optional/required), enumerations with their
literals, and associations with their ends and multiplicities. A modify run
compares the current diagram with it before any LLM call. When anything
changed, the agent receives the list of changes, for example
``Booking.bookingNumber type str -> int; Person.phone removed; association
handledBy: Booking.handledBy[1..1] - ... -> Booking.handledBy[0..1] - ...``,
in its project inventory and as a Phase 2 issue, with the instruction that the
new model wins. In a fix run it follows the reported failure. The user's own
request text is not changed. The list covers classes (added, removed, renamed,
abstract, parents), attributes (added, removed, renamed, type,
optional/required), enumeration literals and association ends; methods and
OCL constraints are not compared. Elements are matched as described in
:ref:`spec-driven-preview-model-change`, so with editor ids a rename reads as
``class Book renamed to Publication``. A base run recorded before this existed
only names its classes, and those include classes the agent added itself, so
it gets no change list (logged as a warning).

.. _spec-driven-regeneration:

Bringing generated files up to a changed model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the model changed and the app's backend came from ``generate_fastapi_backend``,
the generator runs again on the new model before Phase 2, and its output is
merged into the app file by file. The base of the merge is the generator's
output as it was written, stored in ``.besser_generator_output.json`` when
Phase 1 generates the app and after each regeneration:

- A file still equal to the stored output (untouched) becomes the new output,
  byte for byte.
- An edited file gets a three-way merge: base = the stored output, ours = the
  file now, theirs = the new output. When the edits and the model change touch
  neither the same nor adjacent lines, the file gets the model change and keeps
  the edits.
  Line endings do not count as edits (on Windows the generator writes some files
  with CRLF, and an editor may convert them); the merged file keeps its own.
- A file whose edits overlap the model change is kept as it is. The agent gets,
  per file, the generator's change to apply by hand. A file is never left half
  merged or with conflict markers. Such a file keeps its old merge base, so if
  the agent leaves the change out, the next model change brings it back instead
  of reading its absence as an edit.
- A file the new model adds (the router of a new class) is written. A
  generated file deleted since stays deleted.
- A file the new model no longer produces is deleted when untouched, unless
  another file of the app still names it.
- A renamed class's files (``routers/tag.py`` and ``routers/tag_methods.py``
  for ``Tag`` renamed to ``Label``) move to the new name, edits included. Renames come from the model change
  list (matched by editor id), not from file similarity: git's rename detection
  misses a renamed class once its file was edited, and could pair a removed
  class's file with an added one.

The merge uses `pygit2 <https://www.pygit2.org/>`_ (``pip install
besser[agents]``; pinned in the backend requirements). Without it, or without
a usable stored output (an app built before the output was stored, an output
over 1.5 MB, which is not stored, or a store that is malformed, larger than
that, or names a path outside ``backend/``), the run falls back to
the line attribution in ``.besser_attribution.json``: only a file whose lines
are all still the generator's is replaced, every edited file is left to the
agent, and a warning says why. Packaging steps recorded as generator
contributors (the secret scrub, the frontend repair) do not make a file look
untouched.

The store holds the generated code itself (the app already contains it) and,
attached at packaging, the class diagram the output was generated from: the
diagram the run received, or the model-synced one when the run added classes.
Only a store written in that run gets a diagram. The next modify run generates
the diagram again:

- With the same BESSER, the result must be the stored files, or the store is
  not used (it was edited by hand) and the run falls back to the attribution.
  The check leaves out the files an earlier run left to the agent, whose base is
  older on purpose.
- After a BESSER upgrade, the stored output differs from the new one by the
  generator's own changes, and those conflict with edits near them. An edited
  file is merged against the new BESSER's output for the stored diagram
  instead, so it gets the model change but keeps the older generator's code
  (untouched files get both, and the agent is told). Without the stored
  diagram, edited files are merged against the older output, and a warning
  says so.

Measured offline on 42 FastAPI runs and 7 kinds of model change (adding,
removing or retyping an attribute, loosening a multiplicity, renaming or adding
a class): the change needed no hand edit in 78% of the cases, against 31% when
only untouched files were regenerated, and no cleanly merged app failed to boot,
create records or pass the conformance check below. A class rename still needs
the agent when the file's edits touch the renamed lines.

The stored output travels with the app: a modify run's seed, the download ZIP
and the GitHub push carry it, beside the recipe and the attribution, and a
continue-from-GitHub run reads it back. The ZIP holds the three files at its
root. A push deletes them from the repository when the run no longer has them,
so a stale base never outlives its run.

The agent sees the result under "Regenerated files" in its inventory and as a
Phase 2 issue; lines quoted from files are marked as file content. The trace
records a ``regeneration`` event with the ``method`` (``merge`` or
``attribution``), the files replaced, added, merged, moved and removed, the
files left to the agent, the BESSER version an upgrade came from and any
warning, which the agent's note repeats; also when regeneration is skipped,
with the reason. An app with
neither a stored output nor a trustworthy attribution is not regenerated: a
warning is logged and the run continues as before. Other generators are not
regenerated yet.

A generator file an earlier run's LLM edited stays marked ``llm_modified`` in
the recipe of every later modify run, until regeneration replaces it with the
generator's output.

After Phase 2 a deterministic check compares the FastAPI backend
(``sql_alchemy.py`` and the ``<Class>Create`` schemas in
``pydantic_classes.py``) with the changed attributes. A removed attribute still
defined, an added one missing, a type still declared as the old one, or an
optionality the code contradicts is a ``model conformance:`` blocker for the
Phase 3 fix loop (see :doc:`validation`). For another stack the check is
reported as not run.

The same comparison is available before the run: ``/preview`` returns it as
``model_change`` (see :ref:`spec-driven-preview-model-change`), so a client can
ask whether to update the existing app or start fresh when the model changed a
lot.

A repository that was pushed to GitHub by a previous run can be brought back
the same way: ``POST /besser_api/spec-driven/import-github-run`` re-imports it
as a run whose ``run_id`` can then seed a ``mode: "modify"`` run. See
:doc:`api` for both request shapes.

.. _spec-driven-agent-trace:

Who wrote which lines (Agent Trace)
-----------------------------------

Every run ships an `Agent Trace <https://agent-trace.dev/>`_ record (spec
version ``0.1``, MIME type ``application/vnd.agent-trace.record+json``) at
``.agent-trace/<run id>.json``, in the download ZIP and in the GitHub push. It
lists each text file of the delivered tree with line ranges and who wrote them:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Contributor
     - Lines
   * - ``ai`` with ``model_id``
     - Written or modified by the LLM. ``model_id`` is ``provider/model``, the
       model that served the run (after any fallback switch). The
       conversation's ``related`` entry ``urn:besser:run:<run id>`` names the
       run, so a range can be traced back to the run's events.
   * - ``unknown`` (generator)
     - Unchanged from a deterministic BESSER generator's output. ``related``
       names the generator and BESSER version
       (``urn:besser:generator:django@<version>``). Deterministic packaging
       fixes (the frontend build configuration, secret redaction) are
       attributed the same way.
   * - ``human``
     - On a run that continues from a previous run or a GitHub import: lines a
       person changed in between.
   * - ``unknown`` (no generator)
     - Code carried over from a repository pushed before attribution existed.

Ranges refer to the files as delivered and carry a ``content_hash``
(``sha256:``) so moved code can be found again. ``metadata["org.besser-pearl"]``
holds the run id, the base generator, the model id and line counts per kind.
The ``done`` event carries the AI-written share (``aiShare``): AI lines overall
and for this run, and per file.

Attribution compares files, it does not replay edits: whenever a generator
writes a file, its lines are recorded (as short hashes) in
``.besser_attribution.json``; at packaging time every line that differs from
that record goes to the LLM. A line keeps its contributor however often the
file was edited, as long as the line itself is unchanged. The record holds
hashes only, never content, and goes into the GitHub push and the download ZIP
beside the recipe, so a continue-from-GitHub run can tell a person's edits from
the AI's. Binary files
and files over 2 MB are not attributed. In files longer than 5,000 lines only
the unchanged lines at the start and the end keep their contributor: a change
in the middle gives every line from the first change to the last one to the
new contributor (a full line diff of such files is too slow).

When attribution cannot be trusted (the state file is unreadable, or a
generator's update failed), the run ships without a record and without
``.besser_attribution.json``, rather than with a record that credits someone
else's lines to the AI; the next run on that code then starts from
``unknown``. The record carries no ``vcs`` revision:
it is part of the commit it would have to name. Older traces in the repository
are kept, and their ranges refer to the revision they were written for.
