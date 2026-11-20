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

Work is budgeted in executed tool calls, not in LLM turns: ``max_turns``
allows four tool calls per turn, for the customize loop (Phase 2) and the
repair loop (Phase 3) together, so a model that makes one call per turn gets
the same work as one that batches four (480 calls with the default 120). The
same multiple of turns (480) is kept only as a guard against a runaway loop.

Phase 2 leaves Phase 3 a quarter of each budget: it stops with a quarter of the
runtime cap and of the cost cap left, and, with auto-fix on, a quarter of the
tool calls (Phase 2 may make 360 of the default 480). Phase 3 uses that share
for its checks, the requirements judge and the repair loop (with auto-fix off,
the time and spend shares are kept for the checks and the judge). A Phase 2
that uses its share of any of the three hands off to validation, and its
trace's Phase 2 ``phase_exit`` names the share in ``stop_detail`` (``time share
used``, ``cost share used`` or ``tool-call share used``). A handed-off run
completes only when validation finds nothing left and no checklist item is
still open or blocked: an item no check closes is reported as not confirmed,
not repaired, so the run is reported as handed off, never as plainly
completed. Only a Phase 2 that reaches the cap itself (a Phase 1 that ran past
it, or one reply that spent past it) stops as ``timeout`` or ``cost_cap``.
Phase 3's time share is held back while Phase 2 runs and released when Phase 2
returns, so the formatting between the phases has it too. The quarter is ``_PHASE3_BUDGET_SHARE``,
sized from recorded runs: where Phase 3 repaired, it made a median quarter of
the run's tool calls, and a quarter of the 2400 s cap covers most recorded
validation phases.

Verifying is not stalling. Phase 2 hands off early (``stop_detail`` starts
``inspection handoff``) only after 20 tool calls in a row that changed no source
and returned nothing new: a read of new content, a checklist or test change, or
any other successful result not seen before (digits and UUIDs are ignored, so a
new port, id or timestamp does not make a repeated command new; a failed call
or a command that exits non-zero is never new). The model is first shown a
reminder, with its open checklist items, and the handoff comes no earlier than
the next turn; a source change starts a new stretch, with its own reminder.
After 40 calls without a source change it is reminded to implement.

The model is told what is left. When half, three quarters and nine tenths of
Phase 2's share (of the tool calls, the time or the spend, whichever is
furthest along) are used, a short reminder after that turn's tool results gives
the tool calls, minutes and, for a priced model, dollars left before validation
and repair take over, and asks the model to record what it cannot finish as
blocked. It is never part of the system prompt, which stays byte-identical so
the provider's prompt cache keeps serving it.

When a run hits a cap it still returns whatever it has produced so far — the
cap is surfaced as a non-terminal warning before the final result, not as a
hard failure.

The caps are checked at turn boundaries: tool calls first, then cancellation,
then elapsed time, then spend — and the cost check runs *before* the next
request is sent, so a run cannot overshoot its cap by one more billable call. A
single warning is logged at 80% of the cost cap. While Phase 2 runs, its LLM
calls and commands treat the start of Phase 3's time share as the cap.

The runtime cap and cancellation also hold *inside* a turn. LLM calls and the
agent's own commands stop 30 seconds before the cap. Every LLM call of a run (the customize and repair turns, generator
selection, planning, model sync, requirement extraction and judging) gets as its
timeout that time left, never more than its usual timeout, and is not started
with less than 20 seconds left: requirements a judgment would have checked are
reported as not checked for lack of time, an unknown rather than a failure. The
provider SDKs apply a timeout to each wait for data, not to a whole response, so
a streaming call is also closed when the time left runs out; its partial reply,
a half-written tool call included, is dropped. A call stopped by the cap or a
cancellation is reported as such, not as a provider error, and never moves the
run to a fallback model; a provider error (an exhausted quota, an unknown model)
stays one. Every command a run starts (``run_command``,
``install_dependencies``, the scaffold's install, the validators, the build and
type checks) ends by the cap at the latest, and a check or a tool cut by the cap
says so rather than reporting its own timeout. Those last 30 seconds are left to
formatting the output and to Phase 3's checks, which start while at least 10
seconds remain before the cap, so a repair whose turns ran out of time is still
checked, not thrown away. Writing the recipe, report and checkpoint and packing
the download start no command and take seconds; they may end a few seconds past
the cap, which the watchdog's grace (``BESSER_LLM_WATCHDOG_GRACE_SECONDS``)
covers.

A cancellation — from ``POST /besser_api/spec-driven/cancel/{run_id}``, a closed
connection, the abandonment timer or the runtime watchdog — takes effect at
once: the command running at that moment is killed with every process it
started (its whole sandbox on Linux, its process tree on Windows and macOS) and
the tool reports ``Command stopped: the run was cancelled``; a streaming LLM
response is closed and its socket shut down. A non-streaming LLM call that is
already in flight still runs to its end, which its timeout bounds. The run then
writes its checkpoint, recipe and report as it does at any other stop.

Only the paid LLM calls count against the budget: generator selection, the
planning pass, a modify run's model sync, the customize loop, requirement
extraction and judging, and any Phase 3 fix turns. The deterministic scaffold, the
validators, and the download are free. A spent cost cap or an exhausted provider
quota therefore stops only those calls: Phase 3 still runs its checks and the
boot-and-create probe while time is left, so a defect the run left behind is
reported as a blocker, and the requirements judge and the repair turns are
skipped.

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
timeout), or the backend stops, the run's worker is cancelled (as above, the
command it is running is stopped at once), and its concurrency slot is freed only once the worker has stopped;
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

.. _spec-driven-run-evidence:

What the server keeps of a run
------------------------------

The recipe (``.besser_recipe.json``) ships in the download, so it is bounded:
over 128 KiB it is shortened, and its ``recipe_shortened`` section counts what
was cut and says so in ``note``, for example *kept 5 of 109 tool calls*. The run
log (``.besser_trace.jsonl``) stays in the workspace and is deleted with it,
after ``BESSER_LLM_DOWNLOAD_TTL_SECONDS``.

The server keeps a run's full record only when the operator asks for it, by
setting ``BESSER_INCIDENT_LOG_DIR`` (a blank value counts as unset). Each run
started through the backend then appends to
``<BESSER_INCIDENT_LOG_DIR>/runs/<run id>.jsonl`` as it goes, so a run that is
cancelled or crashes keeps what it did up to then, and a shortened recipe adds
*full record kept server-side under run <run id>* to its note. Files are kept
``BESSER_LLM_RUN_EVIDENCE_RETENTION_DAYS`` days (30); ``0`` turns the record
off (a negative value counts as ``0``) and then the cleanup removes nothing, so
older files stay until the operator deletes them. A run driven from Python with
``LLMOrchestrator`` writes none unless it passes ``run_evidence``, a callable
that receives each record. The record is written only with the run log
(``BESSER_LLM_ENABLE_TRACING``, on by default).

**What it contains.** One JSON line for every record of the run log, with
``phase`` added. That is personal data of the user and of what the agent read:

- the first 500 characters of the user's request (in ``run_start``) and the
  requirements read from it (in ``requirements``);
- every file the agent wrote or edited, whole up to 64 KiB (``write_file``
  content, ``modify_file`` old and new text);
- up to 16 KiB of every tool result, the text the model read: the contents of
  files it read, including files of an imported repository and files the user
  uploaded, command and test output, API responses;
- each tool call's turn, ``batch_index`` (its place in the turn's calls),
  ``success``, ``status`` and ``seconds`` (how long the tool took), the
  validation findings, the phases and their token counters, and each
  compaction or eviction of the history.

It does not contain the conversation with the LLM (the prompts and the model's
messages).

- **Caps.** Any one string is kept up to 64 KiB (its first and last 32 KiB
  beyond that). A tool result is kept up to 16 KiB, its first and last 8 KiB.
- **Redaction.** Every string is redacted before it is cut: first the user's
  own key and every token or password the server holds (the LLM tiers' tokens,
  the demo token, ...), then the rules of the run's events and of a generation
  trace: provider tokens, private keys, ``.env``-style assignments, credentials
  written as quoted literals in code (``apiKey = "..."``), URL passwords
  (``https://user:password@host``), ``Authorization``, ``Cookie``,
  ``Set-Cookie`` and ``X-...-Key`` / ``X-...-Token`` headers, token fields and
  JWTs, and ``test_api`` header values. A tool result in JSON is redacted with
  its text unescaped, so the line rules see real lines.
- **Access.** A run's shell never sees the folder: the sandbox masks
  ``BESSER_INCIDENT_LOG_DIR``. Whoever can read the folder on the host can read
  the records.
- **Size.** Measured on production runs, a record is about 0.2 to 1.6 MB
  (0.7 MB on average).

While a run is working, each warning its worker logs carries
``run_id=<run id>``, as the ``spec-driven run start`` and ``run end`` lines do.

.. _spec-driven-modify-mode:

Modifying a previous run
------------------------

A run can also be **incremental**. With ``mode: "modify"`` and a
``base_run_id`` naming a previous, still-downloadable run, the new run's
workspace is seeded from that run's generated files and edited in place instead
of being rebuilt from scratch. Phase 2 is never skipped for a modify run — you
asked for a change, so a change is made.

The base's workspace is kept for ``BESSER_LLM_DOWNLOAD_TTL_SECONDS`` (3 hours by
default) and its ZIP download for ``BESSER_LLM_ARCHIVE_TTL_SECONDS`` (24 hours).
Once the workspace is gone, the run is seeded from the download instead: it holds
the same source tree and the run files a later run reads (``.besser_recipe.json``,
``.besser_attribution.json``, ``.besser_generator_output.json``), and it is
unpacked with the checks of an uploaded archive (no path outside the workspace, no
links, size and file-count limits; of a file's mode only its execute bits are
kept, so ``mvnw`` or a ``*.sh`` script stays executable). The GitHub source record
of a run started from or pushed to a repository (``.besser_github_source.json``,
the commit and the files it held) is not in the download; it is kept beside it and
restored, so a later push still knows which commit the files came from. It is still
a modify run. A download that cannot be unpacked is treated like a failed copy: the
run warns and rebuilds; one removed while being opened counts as expired. Of a ZIP
uploaded through ``/spec-driven/import-generated-code`` only the user's files are
used, never BESSER run records inside it.

When neither is left (or the base's output was a single file, not a ZIP) the runner
warns and falls back to normal from-scratch generation rather than failing the
request; the recipe's ``run.mode_reason`` says ``previous result expired, rebuilt
from scratch`` and the ``done`` event's ``mode`` is ``create``. A client can learn
this before starting: ``/preview`` with ``mode: "modify"`` and the ``base_run_id``
returns ``base_available`` (``false`` when the run would rebuild). The editor avoids
the rebuild for a project linked to GitHub: it starts the run from the repository
instead (see :ref:`web-editor-github-coming-back`).

Validation of a modify run is proportional to its request: it keeps the previous
run's requirement verdicts, judges only the requirements of the messages since
that run and those whose files changed, and reports a requirement the previous run
had left open as still open rather than as a blocker of this run, unless the new
messages ask to finish it (see
:ref:`requirements of a modify run <spec-driven-modify-requirements>`). The build, type-check, boot and runtime
gates apply as in any run.

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

.. _spec-driven-run-trace:

The run's trace
---------------

A run is recorded in the project's trace (:doc:`../buml_language/model_types/traceability`)
like a traced generation when the request sets ``traceabilityEnabled`` (the project
records traces; opt-in, off by default): the ``done`` event carries a trace-file fragment
(``trace``, ``projectId`` null, ``schemaVersion`` 6), and the same JSON ships in
the download ZIP as ``.besser_trace_model.json``. Otherwise the run builds no trace:
``trace`` is ``null``, the ZIP has no ``.besser_trace_model.json`` and no requirement
item has a ``traceRef``. A run keeps three records of
itself, with three names: the *run log* (``.besser_trace.jsonl``, for debugging,
see :doc:`how_it_works`), the *Agent Trace record* (who wrote which lines, below)
and this *run trace*, a ``TraceModel``. The run trace is built at the end of the
run from the other two, the recipe and the delivered files
(``besser/spec_driven_agent/state/trace_model.py``), so it is the same after a
resumed run.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Element
     - What it records
   * - ``CodeGeneration`` ``spec-driven:<run id>``
     - The run, performed by the ``Agent`` "BESSER Spec-Driven Agent" (with the
       BESSER version) and the ``Person`` named by ``actorName``. Its status is
       ``SUCCEEDED`` when the customization loop finished (unresolved findings
       are in its log records), ``CANCELLED`` or ``TIMED_OUT`` when it was cut
       short by a cancellation or the time cap, ``FAILED`` otherwise (provider
       error, cost or turn cap).
   * - Reads
     - One root ``ModelArtifact`` per diagram the run read (the diagram id as
       element id, the B-UML metaclass as type), the instructions as a
       ``PromptArtifact``, and the requirements the editor sent, by their own ids.
   * - ``CodeGenerator`` tool calls
     - One per generator run: Phase 1, a generator tool the LLM called, a modify
       run's regeneration. ``RuleEvidence`` links go from the models it read to
       its files and, for the generators that record line ranges, from each
       element to its lines.
   * - ``LLM`` tool calls
     - One per agent turn (Phases 2 and 3), with the model that served it, the
       provider, and the tokens of that turn, from its start to its LLM response
       (input counts cache reads and writes; a context compaction before the
       request is part of the turn). The LLM calls a phase made outside its turns
       (generator selection, planning, Phase 3 validation, the judge, the runtime
       gate's checks) are one more call when they used tokens. A turn's tool
       calls (arguments without file bodies), validation findings, gate outcomes
       and the run's final findings are its log records, at most 40 per call and
       500 characters each, redacted.
   * - Code versions
     - Every file version the run produced, with its SHA-256 (content kept up to
       1 MB per file, 4 MB in all, with its credentials redacted as in a traced
       generation). A file the LLM wrote is linked from the prompt
       with ``AIEvidence``, at the line ranges the attribution gives to this run's
       LLM; a generated file the LLM then edited is a new version whose
       ``revisionOfId`` is the generator's version, which keeps the generator's links.
   * - ``RequirementExtraction`` ``spec-driven:<run id>:requirements``
     - Only when the run extracted the requirements itself: one LLM call with the
       extraction's tokens, writing one ``RequirementArtifact`` per requirement
       (id ``<run id>:R<n>``), each linked from the prompt.
   * - Judge's verdicts
     - One LLM call whose log records are the verdicts. A requirement found
       implemented is linked to the cited line (or file) with ``AIEvidence`` resting
       on its verdict and confidence 1, one partly implemented with confidence 0.5;
       the confidence's explanation is the verdict. A requirement found missing, or
       that no verdict settled, has no link: as for a model element, a requirement
       without a link is one the code does not realize, and its verdict says why.

A run that fails before its ``done`` event sends no trace; the editor records it
from its stream under the same operation id. Known limits: a modify run's edit of a
file from the previous run is a new version without ``revisionOfId`` (the previous
version lives in the earlier run's trace); the judge's tokens are counted in the
phase's call for the LLM calls outside the turns, not on the call holding its verdicts.

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
       (``urn:besser:generator:django@<version>``). Files and lines the
       harness writes itself (the ``.gitignore``, the frontend build
       configuration, Dockerfiles, ``.dockerignore`` files and README of the
       scaffold repair, secret redaction) are attributed the same way, under
       the name of the step that wrote them, and so is a package manager's lock
       file (``package-lock.json``, ``yarn.lock``, ``poetry.lock``, ...)
       whoever ran the install (``package_manager``).
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
that record goes to the LLM, except the lines a harness step or a package
manager wrote. A line keeps its contributor however often the
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
  error``, ``internal error``, ``cancelled``, ``repeated edits did not land``,
  ``output limit`` or ``handed off to validation``), ``stop_message`` (the run
  card's sentence for that ending, without its variable parts) and ``outcome``.
  ``turn limit`` means the customization loop made every tool call of the
  budget (auto-fix off, so no share was kept) or reached the runaway guard on
  turns; ``time limit`` and ``cost limit`` mean the cap itself, never only
  Phase 2's share of it; ``output limit`` means the model's replies were cut off at its output limit
  more often than the four recoveries a run allows. A handoff stays ``handed off
  to validation`` while a checklist item is still open or blocked after
  validation, and its ``outcome`` is then never ``completed``. A run whose
  customization was skipped because the scaffold already covered the request
  ended ``finished``.
- ``warnings``: the warnings shown to the user during the run (at most eight),
  each as a fixed sentence. A warning whose text can quote an error is left out.
- ``duration_seconds``, ``phase_seconds`` (``select``, ``generate``, ``plan``,
  ``customise``, ``validate``; ``validate`` starts with Phase 3 and includes its
  fix loop) and ``estimated_cost_usd``.
- ``counts``: ``turns``, ``tool_calls``, ``files_written``, ``files_edited``,
  ``blockers`` (the unresolved blockers the run left, also when a limit or a
  cancel cut it short; the end log line carries the same count) and
  ``not_confirmed`` (findings that are not confirmed, see
  :ref:`spec-driven-not-confirmed`; never blockers; a dependency install the
  agent left undone counts here, not in ``blockers``). A finished run whose only
  findings are not confirmed ends ``completed``, and its ``stop_message`` says how
  many could not be confirmed.
- ``not_verified``: each class of checks that left no evidence on the delivered
  app, as ``{"check", "reason", "configured_off"}`` with a fixed reason:
  ``requirements`` (not checked against the request), ``runtime`` (never
  started), ``records`` (no record could be created through the API) or
  ``build`` (the frontend was never built). ``configured_off`` is true when the
  server has that check switched off; such an entry is a property of the server
  and changes neither the outcome nor the ``stop_message``. A finished run with
  another entry and no blocker ends ``completed_unverified``, and its
  ``stop_message`` names them: ``The app was generated, but it was not checked
  against your request (too little time was left).``
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
duration_seconds=... blockers=... not_confirmed=... not_verified=... checks_off=... estimated_cost_usd=...
input_tokens=...
output_tokens=... cache_creation_tokens=... cache_read_tokens=... lines_total=...
lines_generator=... lines_ai=... lines_human=... lines_unknown=...
lines_ai_this_run=... archive_written=...``: the token counts under the recipe's
``usage`` names, the line counts as in ``lines``, and ``-`` for a value the run
does not have (``not_verified`` and ``checks_off`` list classes, comma-separated).
``outcome`` is ``completed``, ``completed_unverified``, ``completed_with_blockers``, ``cancelled``,
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
