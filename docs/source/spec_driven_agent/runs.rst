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

Only the paid LLM calls count against the budget: generator selection, the
planning pass, a modify run's model sync, the customize loop, requirement
extraction and judging, and any Phase 3 fix turns. The deterministic scaffold, the
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
  whole new attempt. A finished run's events are deleted after
  ``BESSER_LLM_DOWNLOAD_TTL_SECONDS``; its status record stays for 24 hours,
  as long as its workspace can be resumed, so numbering continues after that too.
  Once its events are deleted, ``/events`` and ``events.json`` answer ``410``
  (Gone) while the status endpoint still answers; after the 24 hours all three
  answer ``404``.

.. code-block:: bash

   # What happened to this run?
   curl https://<host>/besser_api/spec-driven/runs/<runId>

   # Replay everything after sequence 42, then follow the live stream
   curl -N -H 'Accept: text/event-stream' \
     'https://<host>/besser_api/spec-driven/runs/<runId>/events?after=42'

Losing *every* subscriber starts a grace period (default 5 minutes) before the
producer is cooperatively stopped — which bounds unattended BYOK spend. A
client polling ``events.json`` counts as present: the run is stopped only when
no stream is open *and* no poll arrived within the grace period. When
the event store itself fails mid-run (a full disk, a lock held past its
timeout), or the backend stops, the run's worker is cancelled at its next turn
boundary, and its concurrency slot is freed only once the worker has stopped;
the run is recorded ``failed`` (``interrupted`` on shutdown). A
server restart marks still-running records ``interrupted`` and ends their event
log with a terminal ``error`` frame (code ``INTERNAL``) whose ``resumeAvailable``,
like the status record's, says whether a checkpoint is on disk;
``POST /besser_api/spec-driven/resume/{run_id}`` can then continue them from
their checkpoint when the client re-supplies the request body. Resume relies on
the checkpoint written after every tool-use turn (see :doc:`how_it_works`): it
refuses to resume against a project whose fingerprint has changed, and seeds
the already-spent cost so a crash-and-resume cycle cannot double the bill.
The resumed run continues the interrupted one: the same checklist (shown again
in its prompt), requirements, generator-file tags, routed stack and record of
successful writes, a modify run's model change and seed-run findings, and a
fresh model-contract check. In the recipe's history it replaces the interrupted
attempt's entry instead of adding one.

A resume that fails (a changed project, a rejected key, a provider error)
leaves the interrupted run's workspace, checkpoint and download in place, so
it can be resumed again with the right input until the 24-hour temp cleanup
removes it. Likewise a run that fails or is cancelled after writing a
checkpoint keeps its workspace, so an abandoned run that reports
``resumeAvailable: true`` can be resumed.

.. _spec-driven-modify-mode:

Modifying a previous run
------------------------

A run can also be **incremental**. With ``mode: "modify"`` and a
``base_run_id`` naming a previous, still-downloadable run, the new run's
workspace is seeded from that run's generated files and edited in place instead
of being rebuilt from scratch. Phase 2 is never skipped for a modify run — you
asked for a change, so a change is made. When the base has expired the runner
warns and falls back to normal from-scratch generation rather than failing the
request. The editor avoids that for a project linked to GitHub: it starts the run
from the repository instead (see :ref:`web-editor-github-coming-back`).

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

Bringing generated files up to a changed model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the model changed and the app's backend came from ``generate_fastapi_backend``,
the generator runs again on the new model before Phase 2, and its output is
merged into the app file by file, like ``git merge``: untouched files become the
new output, edited files keep their edits and get the model change, and only
what cannot be merged is left to the agent, with a precise note. The base of
the merge is the generator's output as it was written, stored in
``.besser_generator_output.json``, which travels with the app in the seed, the
ZIP and the GitHub push. See :doc:`regeneration` for how it works, the records
involved, worked examples and the measured results.

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

A push keeps the run history in the repository's ``.besser/`` folder: an index
(``.besser/README.md``) and one file per run under ``.besser/runs/``, of which
the newest 50 are kept. ``AGENTS.md`` gets one fixed line that points there, so
it never grows; a recap section an earlier version left in ``AGENTS.md`` is
moved to ``.besser/previous-recap.md``. Layout and rules:
:ref:`web-editor-github-coming-back`.

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

.. _spec-driven-trace-view:

The run's trace in the editor
-----------------------------

The editor's run card shows what a run did, from data the run already records,
both while it runs and afterwards (see :doc:`../web_editor`):

- **Steps.** Each phase row lists its tool calls (files read, written, edited,
  commands run) and how long the phase took, how many LLM turns it used and,
  once the run has ended, its estimated cost. The figures come from the run
  totals each ``phase`` event carries at its start (see :doc:`api`).
  ``validate`` starts with Phase 3, so a long first check (``npm``, ``tsc``)
  counts as validation, not as customization.
- **Findings and repairs.** The ``validate`` row reads as check, fix attempt,
  re-check: the blockers each check found (from the ``findings`` events),
  the tool calls of the fix attempt that followed, and on each re-check which
  blockers were fixed, which still fail and which are new. When a run goes
  back to a better earlier tree, the last round says so and lists what the
  delivered app still has.
- **Who wrote which lines.** *View files* reads the run's Agent Trace record
  (see :ref:`spec-driven-agent-trace`) from the ZIP and marks each line as
  written by the AI, from a BESSER generator, edited by a person between runs,
  or unattributed, with per-file counts and the AI share of all lines. Each
  range's ``content_hash`` is checked: a file that no longer matches its record
  shows no attribution rather than a wrong one. The record is written when the
  run is packaged, so this view exists once the run has finished.

Everything except the files comes from the run's events, which are stored as
they are sent (see :ref:`spec-driven-durable-runs`): a page reloaded during a
run replays them and shows the same timeline. The files come from the download,
so *View files* works while the server keeps it.

.. _spec-driven-run-record:

The run record
--------------

Packaging adds a ``run`` section to the run's ``.besser_recipe.json``, so a
downloaded folder or a pushed repository says what kind of run produced it. Its
keys, always in this order:

- ``mode`` (``generate`` or ``modify``) and ``mode_reason``: ``fresh project``,
  ``edit of a previous result``, ``previous result expired, rebuilt from
  scratch``, ``previous result could not be copied, rebuilt from scratch`` or
  ``resumed``; ``base_run_id`` when the request named a previous run.
- ``target`` (the selector's kind: ``web_app``, ``backend`` or ``other``, or
  ``null``), ``target_decided_by`` (``selector``, ``approved plan`` or ``not
  decided``), ``frontend_requested`` (whether the run had to ship a web
  frontend, from the target or, without one, from the request's wording) and
  ``generator``.
- ``key_tier`` (``own key``, ``free`` or ``sponsored``, never the key) and
  ``models``: the public model labels of the models that ran, by the rule the
  recipe's ``llm_model`` follows (the route name for the keyless tiers and the
  gateways).
- ``limits`` (``time_seconds``, ``cost_usd``, ``turns``), ``stopped_by``
  (``finished``, ``time limit``, ``cost limit``, ``turn limit``, ``provider
  error``, ``internal error``, ``cancelled``, ``repeated edits did not land`` or
  ``handed off to validation``), ``stop_message`` (the run card's sentence for
  that ending, without its variable parts) and ``outcome``.
- ``warnings``: the warnings shown to the user during the run (at most eight),
  each as a fixed sentence. A warning whose text can quote an error is left out.
- ``duration_seconds``, ``phase_seconds`` (``select``, ``generate``, ``plan``,
  ``customise``, ``validate``; ``validate`` starts with Phase 3 and includes its
  fix loop) and ``estimated_cost_usd``.
- ``counts``: ``turns``, ``tool_calls``, ``files_written``, ``files_edited``,
  ``blockers``.
- ``lines``: the lines of the delivered files by author, as in the run's Agent
  Trace record: ``total``, ``generator``, ``ai``, ``human``, ``unknown``, and
  ``ai_this_run`` (the part of ``ai`` this run wrote). ``null`` when the run has
  no Agent Trace record.
- ``besser_version``, ``besser_build`` and ``request_chars`` (the request itself
  is ``instructions``).

The section is built from the run's own counters, never from a recipe on disk: a
modify run that fails before writing its recipe still has the seeded base's,
so packaging also replaces the recipe's ``usage`` with this run's. The section
holds fixed values and fixed sentences only, never an error's text, so it holds
no key, no user e-mail, no server path and no host name. ``BESSER_GENERATION.md`` shows
its mode, end and model in the *This run* section.

The worker logs one line when a run starts and one when it ends, with the same
names: ``spec-driven run start run_id=... mode=... base_run_id=... key_tier=...
model=... generator=...`` (``generator=auto`` unless the request bound one; the
target is decided later) and ``spec-driven run end run_id=... outcome=...
duration_seconds=... blockers=... estimated_cost_usd=... input_tokens=...
output_tokens=... cache_creation_tokens=... cache_read_tokens=... lines_total=...
lines_generator=... lines_ai=... lines_human=... lines_unknown=...
lines_ai_this_run=... archive_written=...``: the token counts under the recipe's
``usage`` names, the line counts as in ``lines``, and ``-`` for a value the run
does not have.
``outcome`` is ``completed``, ``completed_with_blockers``, ``cancelled``,
``budget_stop`` or ``failed``. Neither line carries the request or the key.
They are written by the ``besser.spec_driven.runs`` logger, which the backend
gives a handler of its own at startup: its other ``INFO`` records have none.

.. _spec-driven-recipe-size:

The recipe's size
-----------------

Packaging reads ``.besser_recipe.json`` back from the run's workspace and sends
it in the ``done`` event, so it refuses one over 256 KiB: the run's own tools can
write to the workspace, and a planted or runaway file is not read into memory.
A run's recipe grows with the work it did (every tool call and API workflow), so
the orchestrator keeps it under 128 KiB when it writes it, which leaves room for
the sections packaging adds. A recipe within that budget is written in full.
Over it, these cuts apply in order until it fits:

1. Each tool call's ``input`` keeps its ``path`` and values of at most 80
   characters that are not lists or objects. The trace holds every call with its
   input as the recipe recorded it, and ``.besser_tool_inputs.jsonl`` the whole
   input of every write tool.
2. API workflows lose their request and response bodies (``json``, ``text``);
   methods, paths, statuses and failures stay.
3. ``tool_calls`` keeps only the last call of each file-changing tool
   (``write_file``, ``modify_file``, ``replace_file_lines``, ``delete_file``) on
   each file, so the files the run changed are all still listed.
4. Each API workflow keeps its id, each request's method and path, and its
   status.

``tool_calls_count`` and every other total stay true, and nothing a later reader
uses is cut: the model, ``output_files``, ``generator_used``, ``target``,
``validation_issues``, ``requirements``, ``tasks``, ``history`` and the
acceptance matrix. A cut recipe has a ``recipe_shortened`` section,
``{bytes_before, budget_bytes, tool_call_inputs_cut, api_bodies_cut,
tool_calls_dropped, api_workflows_cut}``, and the cut is logged as a warning.
The history keeps the last ten runs, each with at most 300 characters of its
request and 20 files; entries read from an earlier recipe are cut the same way.

Packaging builds the ``verification`` report from the recipe the orchestrator
wrote in this run. When the run ended before writing one, it reads the file;
a file it cannot read (over the limit, or not a JSON object) is left as it is,
and the report says the run's record could not be read rather than reporting
nothing to verify.
