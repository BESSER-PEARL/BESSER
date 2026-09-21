How it works
============

A run is a three-phase pipeline (``LLMOrchestrator.run`` in
``besser/spec_driven_agent/orchestrator.py``).

The three phases
----------------

**Phase 1 — deterministic generation.**
   A deterministic BESSER generator is selected and run against your B-UML
   model to produce a correct scaffold (ORM models, schemas, CRUD routes,
   pages). This is exactly the machinery documented under
   :doc:`../generators` — the agent drives those generators rather than
   replacing them. Only generators whose required models are present are
   offered, and an explicitly approved target from the preview screen is
   honoured as-is — which also saves one paid LLM call.

   Two smaller steps bracket it. When no deterministic generator fits the
   target stack (Next.js, Rust, Kotlin/Spring), a *stack-metadata* step writes
   the minimal build-config files that stack needs instead. Afterwards, the
   Phase-1 output is validated so any pre-existing issue is handed to Phase 2
   as context rather than discovered later.

**Phase 2 — LLM customization loop.**
   The main agent loop. The LLM works through a constrained
   :doc:`tool surface <tools>` — reading and writing files, querying the domain
   model, invoking further generators, and tracking its own work through a
   ``task_list`` checklist — making surgical, scoped changes on top of the
   scaffold. Before the loop starts, a cheap *planning* call analyses the gap
   between the scaffold and the request and produces the task list; when it
   judges the scaffold already sufficient, Phase 2 is skipped entirely.

**Phase 3 — validation and bounded auto-fix.**
   The workspace is snapshotted, then swept by a set of static validators.
   Findings are classified by severity, and **blocker**-level issues drive a
   bounded repair loop: up to five rounds of (LLM fix turns → re-validate). The
   loop stops early when blockers reach zero, gives up after two consecutive
   rounds without progress, and rolls back to the pre-Phase-3 snapshot if the
   fixes made blockers *worse* — so a failed repair can never cost you the
   Phase 2 work. Non-blocker findings are recorded but never burn fix turns.
   See :doc:`validation` for the full list of checks, the severity model, and
   how to turn the repair half on or off.

The whole run streams progress over Server-Sent Events, so the editor shows the
phase timeline, the LLM's tool calls, a live cost/runtime meter, and the
streamed text as it happens. The stream reports five coarse phases:
``select``, ``generate``, ``gap``, ``customize``, and ``validate``. The event
schema is documented in :doc:`api`.

A run, end to end
-----------------

Suppose your project contains a class diagram with ``Book``, ``Author``, and
``Member`` (and their associations), and you ask:

   *"Generate a FastAPI backend for this model with JWT authentication and a
   Dockerfile."*

The run unfolds like this:

#. **select / generate** — the :doc:`FastAPI (Backend) generator
   <../generators/backend>` runs and produces a model-faithful baseline:
   SQLAlchemy models for ``Book`` / ``Author`` / ``Member``, Pydantic schemas,
   and a router per resource.
#. **gap** — the planner notes the request needs two things the template
   doesn't provide: JWT auth and a Dockerfile.
#. **customize** — the LLM adds them on top of the scaffold, e.g.::

      main_api.py             # slim app — now wires the auth router in
      database.py             # engine / session / get_db
      sql_alchemy.py          # Book, Author, Member — from your diagram
      pydantic_classes.py
      bal_stdlib.py
      routers/
      ├── book.py             # CRUD, now behind auth
      ├── author.py
      └── member.py
      auth.py                 # NEW — JWT issue/verify, password hashing
      deps.py                 # NEW — get_current_user dependency
      Dockerfile              # NEW
      requirements.txt        # + python-jose, + passlib

#. **validate** — Phase 3 sweeps the result. Say the LLM referenced
   ``SECRET_KEY`` in ``deps.py`` without importing it: ``ruff`` reports
   ``F821``, which is a blocker, and the fix loop repairs it before the ZIP is
   handed back.

The domain classes come from your diagram — the model stays the source of
truth; only the auth and container scaffolding are LLM-authored. The ``done``
event carries a three-way authorship split over the final tree (untouched
generator output / generator output the LLM modified / LLM-authored), so you
can see exactly how much of the app each side carried.

How a run stays on the rails
----------------------------

An agent loop fails in characteristic ways — it fills its context, it edits the
same file forever, it declares victory with work outstanding, it gets cut off
mid-write. Phase 2 carries a specific guard for each. They matter when you are
reading a trace and wondering why a run did what it did.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Guard
     - Behaviour
   * - Batched tool calls
     - Up to four tool calls from one turn execute in parallel, *except* that
       calls writing to the same path are grouped and run sequentially in the
       order the model emitted them — otherwise two edits to one file in one
       turn would both read the pre-turn content and the second would clobber
       the first.
   * - Context compaction
     - Above ``BESSER_LLM_COMPACT_THRESHOLD`` estimated tokens (default
       80,000) everything but the last six messages is replaced by a summary:
       a model recap, a tool-call histogram, and the lists of files written and
       read. The cut never lands between a tool call and its result. For models
       with a known small context window the threshold is lowered to that
       window minus a 16,000-token reserve, so a small model compacts instead
       of overflowing.
   * - Truncation recovery
     - When a response is cut off at the output-token limit, the partial turn
       is discarded rather than executed — nothing lands half-written on disk —
       and the model is told to emit a smaller turn. At most four such
       recoveries per run; after that the run stops with a resumable
       ``api_error`` rather than burning the cost cap. The budget is a
       per-run total that does NOT reset after a clean turn, so it is sized
       against the whole turn budget, not against a single bad patch.
   * - Per-file modify loop
     - Three consecutive ``modify_file`` calls on the same path that all
       fail to match inject a reminder to read the file and copy
       ``old_text`` verbatim, never to rewrite it from memory. Successful
       edits do not count; it fires once per path, and any other tool in
       between breaks the streak.
   * - Checklist gate
     - The run does not finish while ``task_list`` items are open: an
       ``end_turn`` with open items is sent back with the list, twice — four
       times when an open item carries a machine check. Some items are verified
       before they are accepted (the "build the frontend" item checks that
       frontend files exist); an item whose check fails three times is recorded
       as *blocked* and stops holding the gate, so a task the model cannot
       satisfy can no longer livelock the run. An item the user did not ask
       for is closed honestly with ``task_list(action='drop', id=N,
       reason=...)`` rather than marked done.
   * - Per-write diagnostics
     - Every file the model writes is parsed immediately — ``ast`` plus
       pyflakes' undefined-name checks for Python, a structural scanner for
       TS/TSX/JS/JSX, and the respective parser for JSON, YAML and TOML — and
       any finding comes back in the same tool result, while the file is still
       in context. An edit that would turn a file the harness can parse into
       one it cannot is refused outright and never reaches disk, in those same
       languages. Data-contract violations ride along the same way. Toggle with
       ``BESSER_LLM_PER_WRITE_DIAGNOSTICS`` (default on).
   * - Checkpointing
     - A checkpoint is written to ``.besser_checkpoint.json`` in the workspace
       after every tool-use turn, and deleted only when Phase 2 ends cleanly. A
       run cut short by an error, a cap, or a cancellation therefore leaves one
       behind for :ref:`resume <spec-driven-durable-runs>`, which refuses to
       resume against a project whose fingerprint has changed and seeds the
       already-spent cost so a crash-and-resume cycle cannot double the bill.
       Toggle with ``BESSER_LLM_ENABLE_CHECKPOINTING`` (default on).
   * - Tracing
     - Every phase transition, turn, tool call, cost update, compaction,
       snapshot, rollback and validation finding is appended to
       ``.besser_trace.jsonl`` in the workspace. It is best-effort — a failed
       trace write never fails a run. Toggle with
       ``BESSER_LLM_ENABLE_TRACING`` (default on).
   * - Secret redaction
     - Credential-shaped values are stripped at both boundaries where output
       leaves the process: every SSE frame is redacted before it is sent, and
       the workspace is swept before the artifact is packaged — a populated
       ``.env`` is deleted outright, a template ``.env.example`` is kept. The
       count of findings is recorded in the run's recipe.

Every ``BESSER_LLM_*`` switch named above is collected in
:doc:`configuration`.
